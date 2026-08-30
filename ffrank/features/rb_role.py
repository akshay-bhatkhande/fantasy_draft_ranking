"""RB committee / expected offensive snap share (informational).

Combines historical snap %, carry share (run-game role), current depth-chart competition,
team backfield structure, and camp-buzz notes into:

  * expected_snap_pct  -- projected share of team offensive snaps (0-100)
  * rb_role_label      -- Bellcow / Lead (soft committee) / Committee share / ...
  * rb_committee_flag  -- Clear / Lean committee / Lean bellcow / Clear bellcow
  * rb_committee_note  -- short audit string

Does NOT feed Composite Z-Score directly. Expected snap % IS applied in STEP 3b as an
RB-only workload multiplier on Final Projected PPG (see rb_workload_multiplier).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from config import weights as W

COMMITTEE_NOTE_RE = re.compile(
    r"\b(committee|timeshare|time[- ]share|platoon|tandem|split(?:ting)? (?:carries|reps|work|the backfield)|"
    r"reduce(?:d)? (?:his )?workload|bigger role than|first[- ]team reps)\b",
    re.I,
)
BELLCOW_NOTE_RE = re.compile(
    r"\b(bell[- ]?cow|workhorse|every[- ]down|clear RB1|lead (?:back|role)|feature back|"
    r"top back|clear .{0,20}top back|majority of (?:the )?carries)\b",
    re.I,
)
SNAP_RANGE_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*[-–to]+\s*(\d{1,2}(?:\.\d)?)\s*%\s*(?:of\s+)?(?:offensive\s+)?snaps?",
    re.I,
)
SNAP_SINGLE_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*%\s*(?:of\s+)?(?:offensive\s+)?snaps?",
    re.I,
)

OUT_COLS = [
    "expected_snap_pct",
    "rb_role_label",
    "rb_committee_flag",
    "rb_committee_note",
    "rb_team_structure",
    "hist_snap_pct",
    "carry_share_pct",
]


def rb_workload_multiplier(
    expected_snap_pct,
    position: str,
    *,
    limited_sample: bool = False,
    camp_buzz_multiplier: float = 1.0,
) -> float:
    """STEP 3b: scale RB Final PPG by expected snap share vs a par RB1 workload.

    Softens for limited-sample backs and when camp buzz is already lifting PPG, so a
    one-season feature back with +2 camp (e.g. Skattebo) cannot stack two full lifts.
    """
    if position != "RB":
        return 1.0
    if expected_snap_pct is None or (isinstance(expected_snap_pct, float) and np.isnan(expected_snap_pct)):
        return 1.0
    try:
        frac = float(expected_snap_pct) / 100.0
    except (TypeError, ValueError):
        return 1.0
    if frac <= 0:
        return 1.0
    par = float(W.RB_WORKLOAD_PAR_SNAP)
    lo, hi = W.RB_WORKLOAD_MULT_BOUNDS
    mult = float(np.clip(frac / par, lo, hi))

    if limited_sample and mult != 1.0:
        shrink = float(W.RB_WORKLOAD_LIMITED_SAMPLE_SHRINK)
        mult = 1.0 + (mult - 1.0) * shrink

    try:
        camp = float(camp_buzz_multiplier)
    except (TypeError, ValueError):
        camp = 1.0
    if camp > 1.0 and mult > 1.0:
        camp_shrink = float(W.RB_WORKLOAD_CAMP_LIFT_SHRINK)
        mult = 1.0 + (mult - 1.0) * camp_shrink

    return float(mult)


def parse_camp_snap_pct(note: str | float | None) -> float | None:
    """Extract an explicit snap-% projection from a camp note, if present."""
    if note is None or (isinstance(note, float) and np.isnan(note)):
        return None
    text = str(note).strip()
    if not text:
        return None
    m = SNAP_RANGE_RE.search(text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return float(np.clip(((lo + hi) / 2.0) / 100.0, 0.0, 1.0))
    m = SNAP_SINGLE_RE.search(text)
    if m:
        return float(np.clip(float(m.group(1)) / 100.0, 0.0, 1.0))
    return None


def player_snap_share_by_season(
    snaps: pd.DataFrame,
    min_snaps: int | None = None,
) -> pd.DataFrame:
    """Mean offensive snap share per player-season, using only games the player actually played.

    Games with fewer than `min_snaps` offensive snaps are dropped so ramp-up / injury-exit
    cameos do not dilute a featured role down to a false committee average.
    """
    min_snaps = W.RB_SNAP_MIN_SNAPS_FOR_GAME if min_snaps is None else min_snaps
    cols = ["player_id", "season", "offense_pct", "games_used"]
    if snaps.empty:
        return pd.DataFrame(columns=cols)

    sn = snaps.copy()
    sn["offense_pct"] = pd.to_numeric(sn.get("offense_pct"), errors="coerce")
    sn["offense_snaps"] = pd.to_numeric(sn.get("offense_snaps"), errors="coerce")
    id_col = "pfr_player_id" if "pfr_player_id" in sn.columns else None
    if id_col is None:
        return pd.DataFrame(columns=cols)

    played = sn[sn["offense_snaps"].fillna(0) >= float(min_snaps)].copy()
    if played.empty:
        return pd.DataFrame(columns=cols)

    out = (
        played.groupby([id_col, "season"], as_index=False)
        .agg(offense_pct=("offense_pct", "mean"), games_used=("week", "nunique"))
        .rename(columns={id_col: "player_id"})
    )
    return out


def _rb_weekly_concurrent_snaps(sn: pd.DataFrame) -> pd.DataFrame:
    """Per team-week concurrent RB1/RB2 offensive snap shares."""
    weekly_rows = []
    for (team, week), g in sn.groupby(["team", "week"]):
        top = g.sort_values("offense_pct", ascending=False)["offense_pct"].tolist()
        rb1 = float(top[0]) if top else np.nan
        rb2 = float(top[1]) if len(top) > 1 else 0.0
        if pd.isna(rb1):
            continue
        weekly_rows.append({"team": team, "week": week, "rb1": rb1, "rb2": rb2})
    return pd.DataFrame(weekly_rows)


def player_end_of_season_snap_share(
    snaps: pd.DataFrame,
    prior_season: int,
    last_n_weeks: int | None = None,
    min_snaps: int | None = None,
) -> pd.DataFrame:
    """Mean offensive snap share over the last N weeks of prior_season (played games only)."""
    last_n_weeks = W.RB_SNAP_TREND_LATE_WEEKS if last_n_weeks is None else last_n_weeks
    min_snaps = W.RB_SNAP_MIN_SNAPS_FOR_GAME if min_snaps is None else min_snaps
    cols = ["player_id", "offense_pct"]
    if snaps.empty:
        return pd.DataFrame(columns=cols)

    sn = snaps.copy()
    sn["season"] = pd.to_numeric(sn.get("season"), errors="coerce")
    sn = sn[sn["season"] == prior_season]
    pos = sn.get("position", pd.Series(index=sn.index, dtype=object)).astype(str).str.upper()
    sn = sn[pos.eq("RB")].copy()
    sn["offense_pct"] = pd.to_numeric(sn.get("offense_pct"), errors="coerce")
    sn["offense_snaps"] = pd.to_numeric(sn.get("offense_snaps"), errors="coerce")
    sn["week"] = pd.to_numeric(sn.get("week"), errors="coerce")
    id_col = "pfr_player_id" if "pfr_player_id" in sn.columns else None
    if id_col is None or sn.empty:
        return pd.DataFrame(columns=cols)

    played = sn[sn["offense_snaps"].fillna(0) >= float(min_snaps)].dropna(subset=["week"])
    if played.empty:
        return pd.DataFrame(columns=cols)

    max_week = int(played["week"].max())
    cutoff = max_week - int(last_n_weeks) + 1
    late = played[played["week"] >= cutoff]
    if late.empty:
        return pd.DataFrame(columns=cols)

    out = (
        late.groupby(id_col, as_index=False)
        .agg(offense_pct=("offense_pct", "mean"))
        .rename(columns={id_col: "player_id"})
    )
    return out


def team_backfield_snap_trend(snaps: pd.DataFrame, prior_season: int) -> pd.DataFrame:
    """Early vs late concurrent RB1/RB2 snap shares; flag backfields trending committee."""
    cols = [
        "team",
        "rb_trend_committee",
        "rb_trend_rb1_early",
        "rb_trend_rb1_late",
        "rb_trend_rb2_early",
        "rb_trend_rb2_late",
    ]
    if snaps.empty or prior_season is None:
        return pd.DataFrame(columns=cols)

    sn = snaps.copy()
    sn["season"] = pd.to_numeric(sn.get("season"), errors="coerce")
    sn = sn[sn["season"] == prior_season]
    pos = sn.get("position", pd.Series(index=sn.index, dtype=object)).astype(str).str.upper()
    sn = sn[pos.eq("RB")].copy()
    sn["offense_pct"] = pd.to_numeric(sn["offense_pct"], errors="coerce")
    sn["offense_snaps"] = pd.to_numeric(sn.get("offense_snaps"), errors="coerce")
    sn["week"] = pd.to_numeric(sn.get("week"), errors="coerce")
    sn = sn[sn["offense_snaps"].fillna(0) > 0].dropna(subset=["offense_pct", "team", "week"])
    if sn.empty:
        return pd.DataFrame(columns=cols)

    weekly = _rb_weekly_concurrent_snaps(sn)
    if weekly.empty:
        return pd.DataFrame(columns=cols)

    early_cut = int(W.RB_SNAP_TREND_EARLY_WEEKS)
    max_week = int(weekly["week"].max())
    late_cut = max_week - int(W.RB_SNAP_TREND_LATE_WEEKS) + 1
    rows = []
    for team, g in weekly.groupby("team"):
        early = g[g["week"] <= early_cut]
        late = g[g["week"] >= late_cut]
        if early.empty or late.empty:
            continue
        rb1_early = float(early["rb1"].mean())
        rb1_late = float(late["rb1"].mean())
        rb2_early = float(early["rb2"].mean())
        rb2_late = float(late["rb2"].mean())
        trending = (
            (rb2_late - rb2_early) >= float(W.RB_SNAP_TREND_RB2_RISE_MIN)
            and (rb1_early - rb1_late) >= float(W.RB_SNAP_TREND_RB1_FALL_MIN)
        )
        rows.append(
            {
                "team": team,
                "rb_trend_committee": trending,
                "rb_trend_rb1_early": rb1_early,
                "rb_trend_rb1_late": rb1_late,
                "rb_trend_rb2_early": rb2_early,
                "rb_trend_rb2_late": rb2_late,
            }
        )
    return pd.DataFrame(rows)


def team_rb_snap_structure(snaps: pd.DataFrame, prior_season: int) -> pd.DataFrame:
    """Last-season RB backfield structure from CONCURRENT weekly snap leaders.

    Season-average leaderboards falsely label injury handoffs as committees (Tracy ~54% over
    15 games vs Skattebo ~52% over 8 games looks like a 50/50 split, but they mostly did not
    share the field). Weekly RB1/RB2 shares measure the live committee.
    """
    cols = ["team", "rb_team_structure", "team_rb1_snap", "team_rb2_snap", "team_rb_gap"]
    if snaps.empty or prior_season is None:
        return pd.DataFrame(columns=cols)

    sn = snaps.copy()
    sn = sn[pd.to_numeric(sn.get("season"), errors="coerce") == prior_season]
    if sn.empty:
        return pd.DataFrame(columns=cols)
    pos = sn.get("position", pd.Series(index=sn.index, dtype=object)).astype(str).str.upper()
    sn = sn[pos.eq("RB")].copy()
    if sn.empty:
        return pd.DataFrame(columns=cols)

    sn["offense_pct"] = pd.to_numeric(sn["offense_pct"], errors="coerce")
    sn["offense_snaps"] = pd.to_numeric(sn.get("offense_snaps"), errors="coerce")
    # Only count RBs who actually played that week.
    sn = sn[sn["offense_snaps"].fillna(0) > 0].dropna(subset=["offense_pct", "team", "week"])
    if sn.empty:
        return pd.DataFrame(columns=cols)

    weekly = _rb_weekly_concurrent_snaps(sn)
    if weekly.empty:
        return pd.DataFrame(columns=cols)
    weekly["gap"] = weekly["rb1"] - weekly["rb2"]
    rows = []
    for team, g in weekly.groupby("team"):
        rb1 = float(g["rb1"].mean())
        rb2 = float(g["rb2"].mean())
        gap = float(g["gap"].mean())
        if rb1 >= W.RB_TEAM_BELLCOW_TOP_SNAP and gap >= 0.20:
            structure = "Bellcow"
        elif rb1 <= W.RB_TEAM_COMMITTEE_TOP_SNAP or gap <= 0.15:
            structure = "Committee"
        else:
            structure = "Soft committee"
        rows.append(
            {
                "team": team,
                "rb_team_structure": structure,
                "team_rb1_snap": rb1,
                "team_rb2_snap": rb2,
                "team_rb_gap": gap,
            }
        )
    return pd.DataFrame(rows)


def _depth_prior(rank: float, structure: str, competitors: float) -> float:
    table = (
        W.RB_DEPTH_SNAP_PRIORS_BELLCOW
        if structure == "Bellcow"
        else W.RB_DEPTH_SNAP_PRIORS_COMMITTEE
    )
    if pd.isna(rank):
        base = table.get(2, 0.25)
    else:
        r = int(rank)
        if r in table:
            base = table[r]
        elif r > max(table):
            base = min(table.values()) * 0.5
        else:
            base = table.get(1, 0.55)
    # Crowded rooms compress the starter and slightly lift the committee.
    if pd.notna(competitors) and competitors >= 4:
        if pd.notna(rank) and int(rank) == 1:
            base *= 0.92
        elif pd.notna(rank) and int(rank) == 2:
            base = min(base * 1.08, 0.40)
    return float(np.clip(base, 0.03, 0.85))


def _role_label(expected: float, rank: float, structure: str) -> str:
    if expected >= 0.65 and (pd.isna(rank) or int(rank) == 1) and structure == "Bellcow":
        return "Bellcow"
    if expected >= 0.55 and (pd.isna(rank) or int(rank) <= 1):
        return "Lead (soft committee)"
    if expected >= 0.38:
        return "Committee share"
    if expected >= 0.22:
        return "Committee RB2"
    if expected >= 0.12:
        return "Change-of-pace / RB3"
    return "Depth / handcuff"


def _committee_flag(structure: str, expected: float, rank: float, competitors: float, note_hit: str) -> str:
    is_rb1 = pd.isna(rank) or int(rank) == 1
    if note_hit == "committee" and not (is_rb1 and expected >= 0.55):
        return "Committee likely"
    if structure == "Committee" and not (is_rb1 and expected >= 0.58):
        return "Committee likely"
    if is_rb1 and expected >= 0.62 and structure == "Bellcow":
        return "Bellcow likely"
    if is_rb1 and expected >= 0.55:
        return "Lean bellcow"
    if structure == "Committee" or note_hit == "committee":
        return "Committee likely"
    if structure == "Soft committee" or (pd.notna(competitors) and competitors >= 4 and not is_rb1):
        return "Lean committee"
    if expected < 0.50 and is_rb1:
        return "Committee likely"
    return "Unclear"


def compute_rb_roles(df: pd.DataFrame, snaps: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """Add RB role / expected snap columns; non-RBs get empty/NaN values."""
    out = df.copy()
    for c in OUT_COLS:
        if c not in out.columns:
            out[c] = np.nan if c.endswith("_pct") or c.startswith("hist") or c.startswith("carry") else ""

    prior_season = target_season - 1
    structure = team_rb_snap_structure(snaps, prior_season)
    if not structure.empty and "team" in out.columns:
        out = out.drop(columns=[c for c in structure.columns if c != "team" and c in out.columns], errors="ignore")
        out = out.merge(structure, on="team", how="left")
    else:
        out["rb_team_structure"] = out.get("rb_team_structure", "Unknown")
        out["team_rb1_snap"] = np.nan
        out["team_rb2_snap"] = np.nan
        out["team_rb_gap"] = np.nan

    trend = team_backfield_snap_trend(snaps, prior_season)
    if not trend.empty and "team" in out.columns:
        out = out.drop(columns=[c for c in trend.columns if c != "team" and c in out.columns], errors="ignore")
        out = out.merge(trend, on="team", how="left")

    recent_snaps = player_end_of_season_snap_share(snaps, prior_season)
    if not recent_snaps.empty and "pfr_id" in out.columns:
        recent_snaps = recent_snaps.rename(columns={"player_id": "pfr_id", "offense_pct": "snap_share_recent"})
        out = out.drop(columns=["snap_share_recent"], errors="ignore")
        out = out.merge(recent_snaps, on="pfr_id", how="left")

    rb_mask = out.get("position", pd.Series(index=out.index)).eq("RB")
    if not rb_mask.any():
        return out

    for idx in out.index[rb_mask]:
        row = out.loc[idx]
        structure_label = row.get("rb_team_structure") or "Unknown"
        if pd.isna(structure_label) or structure_label == "":
            structure_label = "Unknown"

        rank = pd.to_numeric(row.get("depth_chart_rank"), errors="coerce")
        comps = pd.to_numeric(row.get("depth_chart_competitors"), errors="coerce")
        hist = pd.to_numeric(row.get("snap_share"), errors="coerce")
        recent = pd.to_numeric(row.get("snap_share_recent"), errors="coerce")
        carry = pd.to_numeric(row.get("carry_share"), errors="coerce")
        camp = int(pd.to_numeric(row.get("camp_buzz_score"), errors="coerce") or 0)
        note = row.get("camp_buzz_note") or ""
        is_rb1 = pd.isna(rank) or int(rank) == 1
        team_trending = bool(row.get("rb_trend_committee"))

        if team_trending and is_rb1 and structure_label == "Bellcow":
            structure_label = "Soft committee"

        prior = _depth_prior(rank, structure_label, comps)
        # Evidence blend: trust history when the player actually played meaningful snaps.
        # Carry share is kept light for featured backs — pass-down specialists behind a lead
        # back (Tracy on NYG) pull carry/snap apart without meaning a true committee.
        if pd.notna(hist) and hist >= 0.55:
            w_hist, w_prior, w_carry = 0.70, 0.20, 0.10
        elif pd.notna(hist) and hist >= 0.15:
            w_hist, w_prior, w_carry = 0.50, 0.30, 0.20
        elif pd.notna(hist):
            w_hist, w_prior, w_carry = 0.25, 0.50, 0.25
        else:
            w_hist, w_prior, w_carry = 0.0, 0.70, 0.30
        carry_v = float(carry) if pd.notna(carry) else prior
        hist_v = float(hist) if pd.notna(hist) else prior
        if pd.notna(recent) and pd.notna(hist):
            decline = float(hist) - float(recent)
            if decline >= float(W.RB_SNAP_TREND_PLAYER_DECLINE_MIN) and (
                team_trending or decline >= 2 * float(W.RB_SNAP_TREND_PLAYER_DECLINE_MIN)
            ):
                b = float(W.RB_SNAP_TREND_RECENT_BLEND)
                hist_v = (1.0 - b) * float(hist) + b * float(recent)
        expected = w_hist * hist_v + w_prior * prior + w_carry * carry_v

        # Camp score bump.
        expected += W.RB_CAMP_SNAP_BUMP.get(camp, 0.0)

        note_hit = ""
        # Prefer bellcow cues over committee cues when both appear (common in "clear RB1
        # but committee on passing downs" notes).
        if BELLCOW_NOTE_RE.search(str(note)):
            note_hit = "bellcow"
            expected += 0.03
        elif COMMITTEE_NOTE_RE.search(str(note)):
            note_hit = "committee"
            expected -= 0.04

        parsed = parse_camp_snap_pct(note)
        if parsed is not None:
            b = W.RB_CAMP_SNAP_NOTE_BLEND
            expected = (1.0 - b) * expected + b * parsed
            note_hit = note_hit or "camp snap%"

        expected = float(np.clip(expected, 0.03, 0.85))

        out.at[idx, "expected_snap_pct"] = round(expected * 100.0, 1)
        out.at[idx, "_rb_rank"] = rank
        out.at[idx, "_rb_comps"] = comps
        out.at[idx, "_rb_note_hit"] = note_hit
        out.at[idx, "hist_snap_pct"] = round(float(hist) * 100.0, 1) if pd.notna(hist) else np.nan
        out.at[idx, "carry_share_pct"] = round(float(carry) * 100.0, 1) if pd.notna(carry) else np.nan
        out.at[idx, "rb_team_structure"] = structure_label
        bits = []
        if pd.notna(hist):
            bits.append(f"hist snaps {hist * 100:.0f}% (games played, ≥{W.RB_SNAP_MIN_SNAPS_FOR_GAME} snaps)")
        if pd.notna(recent) and pd.notna(hist) and float(hist) - float(recent) >= float(W.RB_SNAP_TREND_PLAYER_DECLINE_MIN):
            bits.append(f"late-season snaps {recent * 100:.0f}% (last {W.RB_SNAP_TREND_LATE_WEEKS} wks)")
        if team_trending and is_rb1:
            rb2_early = pd.to_numeric(row.get("rb_trend_rb2_early"), errors="coerce")
            rb2_late = pd.to_numeric(row.get("rb_trend_rb2_late"), errors="coerce")
            if pd.notna(rb2_early) and pd.notna(rb2_late):
                bits.append(
                    f"backfield trending committee (RB2 {rb2_early * 100:.0f}%→{rb2_late * 100:.0f}%)"
                )
        if pd.notna(carry):
            bits.append(f"carry share {carry * 100:.0f}%")
        if pd.notna(rank):
            bits.append(f"depth RB{int(rank)}")
        if pd.notna(comps):
            bits.append(f"{int(comps)} RBs listed")
        base_structure = row.get("rb_team_structure") or "Unknown"
        if pd.isna(base_structure) or base_structure == "":
            base_structure = "Unknown"
        if structure_label != base_structure:
            bits.append(f"team {str(base_structure).lower()} → {structure_label.lower()} (late-season trend)")
        else:
            bits.append(f"team {structure_label.lower()} (weekly concurrent)")
        if pd.notna(row.get("team_rb1_snap")):
            bits.append(
                f"2025 weekly RB1/RB2 snaps {float(row['team_rb1_snap']) * 100:.0f}/"
                f"{float(row.get('team_rb2_snap') or 0) * 100:.0f}%"
            )
        if parsed is not None:
            bits.append(f"camp note ~{parsed * 100:.0f}% snaps")
        elif note_hit:
            bits.append(f"camp suggests {note_hit}")
        out.at[idx, "rb_committee_note"] = "; ".join(bits)

    # Finalize labels from expected snaps (no team-wide renormalize — that was crushing
    # true lead backs when a former starter still had a high historical snap rate as RB2).
    for idx in out.index[rb_mask]:
        expected = float(pd.to_numeric(out.at[idx, "expected_snap_pct"], errors="coerce") or 0) / 100.0
        rank = out.at[idx, "_rb_rank"] if "_rb_rank" in out.columns else np.nan
        comps = out.at[idx, "_rb_comps"] if "_rb_comps" in out.columns else np.nan
        note_hit = out.at[idx, "_rb_note_hit"] if "_rb_note_hit" in out.columns else ""
        structure_label = out.at[idx, "rb_team_structure"] or "Unknown"
        out.at[idx, "rb_role_label"] = _role_label(expected, rank, structure_label)
        out.at[idx, "rb_committee_flag"] = _committee_flag(structure_label, expected, rank, comps, note_hit)

    out = out.drop(columns=[c for c in ("_rb_rank", "_rb_comps", "_rb_note_hit") if c in out.columns])
    return out
