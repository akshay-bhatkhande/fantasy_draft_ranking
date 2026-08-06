"""STEP 2 -- Situational Context Score (10% of the Composite Z-Score).

Scored inputs (equal-weighted average of their z-scores):
  * offensive line quality           -- PFR yards-before-contact per attempt + sack rate allowed
  * offensive scheme / pace          -- plays per game + pass rate over expected
  * QB quality / stability           -- team QB EPA per play (pass-catchers only)
  * coordinator / coaching change    -- scored by the incoming coach's historical usage
                                        tendency for that position (bell-cow vs committee)
  * team implied point total         -- from the published schedule's Vegas spread/total
  * full-season strength of schedule -- opponent quality, inverted so easier is better
  * depth chart competition          -- crowded competition for touches lowers the score

Informational only, deliberately NOT scored and NOT in any multiplier:
  * weeks 15-17 SOS (fantasy playoff schedule)
  * bye week
  * dome versus outdoor/cold-weather

Directionality note: pass rate over expected is good for pass-catchers and QBs but bad for
running backs, so its sign is flipped for RBs rather than being applied blindly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import weights as W

PLAYOFF_WEEKS = (15, 16, 17)


def _z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    sd = s.std(ddof=0)
    if not sd or np.isnan(sd):
        return pd.Series(0.0, index=s.index).where(s.notna())
    return (s - s.mean()) / sd


# --------------------------------------------------------------------------------------
# Team-level metrics
# --------------------------------------------------------------------------------------


def team_offense_metrics(pbp: pd.DataFrame, adv_rush: pd.DataFrame) -> pd.DataFrame:
    """Per-team offensive metrics from the most recent completed season(s).

    Returns team, plays_per_game, pass_oe, qb_epa_per_play, sack_rate_allowed,
    stuffed_run_rate, ybc_att.
    """
    if pbp.empty:
        return pd.DataFrame(columns=["team"])

    df = pbp.copy()
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    for col in ("qb_dropback", "sack", "rush_attempt", "pass_attempt", "qb_epa", "pass_oe", "yards_gained"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["posteam"].notna()]
    is_play = df["play_type"].isin(["pass", "run"])
    plays = df[is_play]

    games = plays.groupby("posteam")["game_id"].nunique()
    n_plays = plays.groupby("posteam").size()

    agg = pd.DataFrame({"plays": n_plays, "games": games})
    agg["plays_per_game"] = agg["plays"] / agg["games"].replace(0, np.nan)

    if "pass_oe" in plays.columns:
        agg["pass_oe"] = plays.groupby("posteam")["pass_oe"].mean()
    else:
        agg["pass_oe"] = np.nan

    dropbacks = df[df["qb_dropback"].eq(1)]
    agg["qb_epa_per_play"] = dropbacks.groupby("posteam")["qb_epa"].mean() if "qb_epa" in df.columns else np.nan
    sacks = dropbacks.groupby("posteam")["sack"].sum()
    n_drop = dropbacks.groupby("posteam").size()
    agg["sack_rate_allowed"] = sacks / n_drop.replace(0, np.nan)

    runs = df[df["rush_attempt"].eq(1)]
    stuffed = runs[runs["yards_gained"].le(0)].groupby("posteam").size()
    agg["stuffed_run_rate"] = (stuffed / runs.groupby("posteam").size().replace(0, np.nan)).fillna(0)

    agg = agg.reset_index().rename(columns={"posteam": "team"})

    # Yards before contact per attempt is the cleanest free signal for run blocking; it is
    # only available from the PFR mirror, aggregated to team level weighted by attempts.
    if not adv_rush.empty and {"tm", "ybc_att", "att"}.issubset(adv_rush.columns):
        r = adv_rush.copy()
        r["ybc_att"] = pd.to_numeric(r["ybc_att"], errors="coerce")
        r["att"] = pd.to_numeric(r["att"], errors="coerce").fillna(0)
        r = r[r["ybc_att"].notna() & (r["att"] > 0)]
        if not r.empty:
            r["wsum"] = r["ybc_att"] * r["att"]
            team_ybc = r.groupby("tm").apply(
                lambda g: g["wsum"].sum() / g["att"].sum() if g["att"].sum() else np.nan,
                include_groups=False,
            )
            agg = agg.merge(
                team_ybc.rename("ybc_att").reset_index().rename(columns={"tm": "team"}),
                on="team",
                how="left",
            )
    if "ybc_att" not in agg.columns:
        agg["ybc_att"] = np.nan
    return agg


def team_strength(pbp: pd.DataFrame) -> pd.Series:
    """Overall team strength (EPA per play, offense minus defense) for SOS calculations."""
    if pbp.empty:
        return pd.Series(dtype=float)
    df = pbp.copy()
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    df["epa"] = pd.to_numeric(df.get("epa"), errors="coerce")
    df = df[df["play_type"].isin(["pass", "run"]) & df["epa"].notna()]
    off = df.groupby("posteam")["epa"].mean()
    dfn = df.groupby("defteam")["epa"].mean()
    return (off - dfn).dropna()


def team_implied_totals(schedules: pd.DataFrame, season: int) -> pd.DataFrame:
    """Vegas implied points per team from the published schedule.

    nflverse's spread_line is the home team's line (positive = home favoured), so:
        implied home total = total_line/2 + spread_line/2
        implied away total = total_line/2 - spread_line/2
    This is why no odds API or key is needed -- the schedule release already carries lines.
    """
    if schedules.empty:
        return pd.DataFrame(columns=["team", "implied_total_per_game", "games_with_lines"])

    s = schedules[schedules["season"] == season].copy()
    if "game_type" in s.columns:
        s = s[s["game_type"] == "REG"]
    s["spread_line"] = pd.to_numeric(s.get("spread_line"), errors="coerce")
    s["total_line"] = pd.to_numeric(s.get("total_line"), errors="coerce")
    s = s[s["total_line"].notna() & s["spread_line"].notna()]
    if s.empty:
        return pd.DataFrame(columns=["team", "implied_total_per_game", "games_with_lines"])

    home = pd.DataFrame(
        {"team": s["home_team"], "implied": s["total_line"] / 2 + s["spread_line"] / 2}
    )
    away = pd.DataFrame(
        {"team": s["away_team"], "implied": s["total_line"] / 2 - s["spread_line"] / 2}
    )
    both = pd.concat([home, away], ignore_index=True)
    out = both.groupby("team")["implied"].agg(["mean", "count"]).reset_index()
    return out.rename(columns={"mean": "implied_total_per_game", "count": "games_with_lines"})


def home_roof_lookup(schedules_history: pd.DataFrame) -> dict[str, str]:
    """Most common home-stadium roof type per team, from completed seasons.

    Needed because the upcoming season's schedule release often has `roof` unpopulated (in
    the 2026 release, 43 games including every Arizona home game). Falling back to the
    team's historical home roof keeps the dome/outdoor note correct instead of silently
    labelling retractable-roof teams as outdoor.
    """
    if schedules_history.empty or "roof" not in schedules_history.columns:
        return {}
    s = schedules_history.dropna(subset=["roof", "home_team"])
    if s.empty:
        return {}
    modes = s.groupby("home_team")["roof"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    return {k: v for k, v in modes.items() if v is not None}


def team_schedule_context(
    schedules: pd.DataFrame,
    season: int,
    strength: pd.Series,
    roof_fallback: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Bye week, full-season SOS, weeks 15-17 SOS and dome share, per team.

    SOS is expressed as mean opponent strength; the situational score inverts it so that an
    easier schedule is a positive, while the raw value stays visible for auditing.
    """
    cols = ["team", "bye_week", "sos_full", "sos_playoffs", "dome_share", "opponents"]
    if schedules.empty:
        return pd.DataFrame(columns=cols)
    roof_fallback = roof_fallback or {}

    s = schedules[schedules["season"] == season].copy()
    if "game_type" in s.columns:
        s = s[s["game_type"] == "REG"]
    if s.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    home = s[["week", "home_team", "away_team", "roof"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    home["is_home"] = True
    away = s[["week", "away_team", "home_team", "roof"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    away["is_home"] = False
    long = pd.concat([home, away], ignore_index=True)

    all_weeks = set(range(1, int(s["week"].max()) + 1))
    for team, group in long.groupby("team"):
        played = set(group["week"].astype(int))
        bye = sorted(all_weeks - played)
        opp_strength = group["opponent"].map(strength)
        playoff_mask = group["week"].astype(int).isin(PLAYOFF_WEEKS)
        home_rows = group[group["is_home"]]
        # Fill an unpopulated roof from the team's historical home stadium before judging.
        roof_filled = home_rows["roof"].fillna(roof_fallback.get(team))
        dome_games = int(roof_filled.isin(["dome", "closed"]).sum())
        home_games = max(int(group["is_home"].sum()), 1)
        rows.append(
            {
                "team": team,
                "bye_week": bye[0] if bye else None,
                "sos_full": opp_strength.mean(),
                "sos_playoffs": opp_strength[playoff_mask].mean(),
                "dome_share": dome_games / home_games,
                "opponents": len(group),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Coaching change / usage tendency
# --------------------------------------------------------------------------------------


def coach_by_team_season(schedules: pd.DataFrame) -> pd.DataFrame:
    """Head coach per team per season, from the schedule's home_coach/away_coach columns."""
    if schedules.empty or "home_coach" not in schedules.columns:
        return pd.DataFrame(columns=["team", "season", "coach"])
    s = schedules.copy()
    home = s[["season", "home_team", "home_coach"]].rename(
        columns={"home_team": "team", "home_coach": "coach"}
    )
    away = s[["season", "away_team", "away_coach"]].rename(
        columns={"away_team": "team", "away_coach": "coach"}
    )
    both = pd.concat([home, away], ignore_index=True).dropna(subset=["coach"])
    return (
        both.groupby(["team", "season"])["coach"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
    )


def positional_concentration(weekly: pd.DataFrame) -> pd.DataFrame:
    """Per team-season, how concentrated each position's usage is on its top player.

    High values mean the offence featured one player (a bell-cow back, an alpha receiver);
    low values mean a committee. This is the historical tendency the coaching-change input
    scores against.
    """
    if weekly.empty:
        return pd.DataFrame(columns=["team", "season", "position", "concentration"])
    w = weekly.copy()
    w["carries"] = pd.to_numeric(w.get("carries"), errors="coerce").fillna(0)
    w["targets"] = pd.to_numeric(w.get("targets"), errors="coerce").fillna(0)
    w["touches"] = w["carries"] + w["targets"]
    w = w[w["position"].isin(["QB", "RB", "WR", "TE"])]

    by_player = w.groupby(["team", "season", "position", "player_id"], as_index=False)["touches"].sum()
    rows = []
    for (team, season, pos), grp in by_player.groupby(["team", "season", "position"]):
        total = grp["touches"].sum()
        if total <= 0:
            continue
        rows.append(
            {
                "team": team,
                "season": season,
                "position": pos,
                "concentration": grp["touches"].max() / total,
            }
        )
    return pd.DataFrame(rows)


def coach_tendency(
    schedules_history: pd.DataFrame, weekly_history: pd.DataFrame
) -> pd.DataFrame:
    """Historical positional concentration per coach, used to score coaching changes."""
    coaches = coach_by_team_season(schedules_history)
    conc = positional_concentration(weekly_history)
    if coaches.empty or conc.empty:
        return pd.DataFrame(columns=["coach", "position", "coach_concentration"])
    merged = conc.merge(coaches, on=["team", "season"], how="inner")
    return (
        merged.groupby(["coach", "position"], as_index=False)["concentration"]
        .mean()
        .rename(columns={"concentration": "coach_concentration"})
    )


def coaching_change_scores(
    schedules_target: pd.DataFrame,
    schedules_history: pd.DataFrame,
    weekly_history: pd.DataFrame,
    target_season: int,
) -> pd.DataFrame:
    """Per team+position coaching-change score.

    A team that kept its coach scores 0 (neutral): there is no *change* to price in. A team
    with a new coach is scored by that coach's historical tendency to concentrate usage at
    the position, centred so an average-tendency hire is also roughly neutral.
    """
    coaches = coach_by_team_season(pd.concat([schedules_target, schedules_history], ignore_index=True))
    if coaches.empty:
        return pd.DataFrame(columns=["team", "position", "coach_change_score", "coach_change_note"])

    target = coaches[coaches["season"] == target_season][["team", "coach"]].rename(columns={"coach": "new_coach"})
    prior_season = target_season - 1
    prior = coaches[coaches["season"] == prior_season][["team", "coach"]].rename(columns={"coach": "prior_coach"})
    merged = target.merge(prior, on="team", how="left")
    merged["changed"] = merged["new_coach"] != merged["prior_coach"]

    tendency = coach_tendency(schedules_history, weekly_history)
    rows = []
    for pos in ("QB", "RB", "WR", "TE"):
        pos_tend = tendency[tendency["position"] == pos]
        mean_conc = pos_tend["coach_concentration"].mean()
        lookup = dict(zip(pos_tend["coach"], pos_tend["coach_concentration"]))
        for r in merged.itertuples():
            if not r.changed:
                rows.append(
                    {"team": r.team, "position": pos, "coach_change_score": 0.0, "coach_change_note": ""}
                )
                continue
            conc = lookup.get(r.new_coach)
            if conc is None or pd.isna(mean_conc):
                rows.append(
                    {
                        "team": r.team,
                        "position": pos,
                        "coach_change_score": 0.0,
                        "coach_change_note": f"New coach {r.new_coach}: insufficient data on {pos} usage tendency",
                    }
                )
                continue
            rows.append(
                {
                    "team": r.team,
                    "position": pos,
                    "coach_change_score": conc - mean_conc,
                    "coach_change_note": (
                        f"New coach {r.new_coach}: historical {pos} usage concentration "
                        f"{conc:.2f} vs league avg {mean_conc:.2f}"
                    ),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Depth chart competition
# --------------------------------------------------------------------------------------


def depth_chart_competition(depth: pd.DataFrame) -> pd.DataFrame:
    """Workload-security score from the current depth chart.

    Being listed first with few credible bodies behind you is secure; being third on a
    crowded depth chart is not. Higher score = more secure. Returned per gsis_id.
    """
    cols = ["player_id", "depth_chart_rank", "depth_chart_competitors", "competition_score"]
    if depth.empty or "pos_abb" not in depth.columns:
        return pd.DataFrame(columns=cols)

    d = depth[depth["pos_abb"].isin(["QB", "RB", "WR", "TE", "FB"])].copy()
    if d.empty or "gsis_id" not in d.columns:
        return pd.DataFrame(columns=cols)
    d["pos_rank"] = pd.to_numeric(d["pos_rank"], errors="coerce")
    d = d.dropna(subset=["gsis_id", "pos_rank"])

    counts = d.groupby(["team", "pos_abb"])["gsis_id"].transform("count")
    d["depth_chart_competitors"] = counts
    # Rank dominates (being the starter is what matters), with a smaller penalty for the
    # sheer number of bodies competing at the position.
    d["competition_score"] = -(d["pos_rank"] - 1.0) - 0.25 * (counts - 1.0)
    out = d.rename(columns={"gsis_id": "player_id", "pos_rank": "depth_chart_rank"})
    return out[cols].drop_duplicates(subset=["player_id"])


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def compute_situational(
    player_teams: pd.DataFrame,
    pbp_recent: pd.DataFrame,
    adv_rush: pd.DataFrame,
    weekly_history: pd.DataFrame,
    schedules_target: pd.DataFrame,
    schedules_history: pd.DataFrame,
    depth: pd.DataFrame,
    target_season: int,
) -> pd.DataFrame:
    """Build the Situational Context Score plus the informational-only columns.

    Args:
        player_teams: player_id, position, team for the upcoming season.

    Returns one row per player with situational_value (equal-weighted mean of the applicable
    component z-scores), each component's z-score for auditing, and the informational
    columns (bye week, playoff SOS, dome share).
    """
    out = player_teams.copy()

    team_metrics = team_offense_metrics(pbp_recent, adv_rush)
    strength = team_strength(pbp_recent)
    implied = team_implied_totals(schedules_target, target_season)
    sched_ctx = team_schedule_context(
        schedules_target, target_season, strength, roof_fallback=home_roof_lookup(schedules_history)
    )
    changes = coaching_change_scores(schedules_target, schedules_history, weekly_history, target_season)
    competition = depth_chart_competition(depth)

    # --- team-level z-scores -----------------------------------------------------------
    tm = team_metrics.copy()
    if not tm.empty:
        # O-line quality: good run blocking (high yards before contact) and pass protection
        # (low sack rate, few stuffed runs).
        parts = []
        if tm["ybc_att"].notna().any():
            parts.append(_z(tm["ybc_att"]))
        parts.append(-_z(tm["sack_rate_allowed"]))
        parts.append(-_z(tm["stuffed_run_rate"]))
        tm["z_oline_quality"] = pd.concat(parts, axis=1).mean(axis=1)
        tm["z_pace"] = _z(tm["plays_per_game"])
        tm["z_pass_oe"] = _z(tm["pass_oe"])
        tm["z_qb_quality"] = _z(tm["qb_epa_per_play"])
    team_z = tm[["team", "z_oline_quality", "z_pace", "z_pass_oe", "z_qb_quality"]] if not tm.empty else pd.DataFrame(columns=["team"])

    if not implied.empty:
        implied = implied.copy()
        implied["z_implied_team_total"] = _z(implied["implied_total_per_game"])
    if not sched_ctx.empty:
        sched_ctx = sched_ctx.copy()
        # Inverted: facing weaker opponents is a positive for the player.
        sched_ctx["z_strength_of_schedule"] = -_z(sched_ctx["sos_full"])

    out = out.merge(team_z, on="team", how="left")
    if not implied.empty:
        out = out.merge(implied[["team", "implied_total_per_game", "z_implied_team_total"]], on="team", how="left")
    if not sched_ctx.empty:
        out = out.merge(
            sched_ctx[["team", "bye_week", "sos_full", "sos_playoffs", "dome_share", "z_strength_of_schedule"]],
            on="team",
            how="left",
        )
    if not changes.empty:
        out = out.merge(changes, on=["team", "position"], how="left")
    if not competition.empty:
        out = out.merge(competition, on="player_id", how="left")

    # Depth-chart competition z-scores normally across the player population.
    out["z_depth_chart_competition"] = _z(out["competition_score"]) if "competition_score" in out.columns else np.nan

    # Coaching change needs special handling. Most teams keep their coach, so the raw score
    # is a big spike of zeros with a few non-zero values; z-scoring that whole distribution
    # gave a real hire a |z| near 4, which then dominated the equal-weighted situational
    # average. Instead: scale by the spread among teams that ACTUALLY changed coach, leave
    # unchanged teams at exactly 0, and clip so this input can nudge but never dominate.
    if "coach_change_score" in out.columns:
        raw = pd.to_numeric(out["coach_change_score"], errors="coerce").fillna(0.0)
        changed = raw != 0.0
        sd = raw[changed].std(ddof=0) if changed.any() else 0.0
        scaled = (raw / sd) if sd and not np.isnan(sd) else raw * 0.0
        out["z_coordinator_tendency"] = scaled.clip(-2.0, 2.0)
    else:
        out["z_coordinator_tendency"] = np.nan

    # Pass rate over expected helps pass-catchers and QBs, and hurts running backs.
    is_rb = out["position"].eq("RB")
    pass_oe_signed = out.get("z_pass_oe", pd.Series(np.nan, index=out.index)).copy()
    pass_oe_signed[is_rb] = -pass_oe_signed[is_rb]
    out["z_scheme_pace"] = pd.concat(
        [out.get("z_pace", pd.Series(np.nan, index=out.index)), pass_oe_signed], axis=1
    ).mean(axis=1)

    component_to_col = {
        "oline_quality": "z_oline_quality",
        "scheme_pace": "z_scheme_pace",
        "qb_quality": "z_qb_quality",
        "coordinator_tendency": "z_coordinator_tendency",
        "implied_team_total": "z_implied_team_total",
        "strength_of_schedule": "z_strength_of_schedule",
        "depth_chart_competition": "z_depth_chart_competition",
    }

    # Equal-weighted average of applicable components only. Components that do not apply to
    # a position (QB quality for a QB) or are unavailable are skipped, not zero-filled --
    # zero-filling would drag a player spuriously toward the positional mean.
    values = pd.Series(np.nan, index=out.index, dtype=float)
    used_counts = pd.Series(0, index=out.index, dtype=int)
    for comp, col in component_to_col.items():
        if col not in out.columns:
            continue
        applicable = out["position"].isin(W.SITUATIONAL_APPLICABILITY.get(comp, ()))
        vals = pd.to_numeric(out[col], errors="coerce").where(applicable)
        values = values.add(vals.fillna(0.0), fill_value=0.0)
        used_counts = used_counts.add(vals.notna().astype(int), fill_value=0)

    out["situational_value"] = (values / used_counts.replace(0, np.nan)).astype(float)
    out["situational_components_used"] = used_counts

    def _detail(row) -> str:
        bits = []
        for comp, col in component_to_col.items():
            if col in out.columns and pd.notna(row.get(col)) and row["position"] in W.SITUATIONAL_APPLICABILITY.get(comp, ()):
                bits.append(f"{comp}={row[col]:+.2f}")
        return ", ".join(bits) if bits else "insufficient data"

    out["situational_detail"] = out.apply(_detail, axis=1)
    return out
