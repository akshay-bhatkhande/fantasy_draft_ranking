"""STEP 2 -- Efficiency (15% of the Composite Z-Score).

  RB     YPRR (receiving) + yards after contact + broken tackle rate, equal weights
  WR/TE  YPRR + (actual TD rate - expected TD rate), equal weights
  QB     YPA + (actual TD rate - expected TD rate) + sack rate avoided, equal weights

Routes run: computed for real from nflverse participation data, not proxied. participation
publishes `offense_players` (semicolon-delimited gsis ids for all 11 offensive players on
the play) on 100% of rows for 2023-2025, and every regular-season dropback in play-by-play
has a matching participation row. A route is therefore counted as "eligible receiver
(WR/TE/RB/FB) on the field for a qb_dropback".

Two honest caveats, both surfaced on the sheet rather than hidden:

* Being on the field for a dropback includes pass-blocking assignments, so this slightly
  overcounts true routes. Measured targets-per-pass-snap for 2024 was 0.193 for WRs and
  0.144 for TEs (both close to published targets-per-route, implying roughly a 5%
  overcount) but 0.147 for RBs, implying RB pass snaps overstate true routes by roughly
  25-35% because backs stay in to block. Since STEP 2 z-scores YPRR within the position
  pool, a systematic proportional bias largely cancels.
* If a participation release is missing for a season, YPRR falls back to a pass-snap proxy
  and is labelled as such.

Yards after contact and broken tackles come from the PFR advanced-stats mirror, which joins
on pfr_id rather than gsis_id. Expected TDs come from the ffopportunity model, so we are not
hand-rolling an expected-points model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import weights as W
from .weighted_ppg import recency_weighted_mean

ROUTE_ELIGIBLE_POSITIONS = {"WR", "TE", "RB", "FB"}


# --------------------------------------------------------------------------------------
# Routes run
# --------------------------------------------------------------------------------------


def compute_routes(participation: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Real routes run per player-season, from participation joined to pbp dropbacks.

    Returns player_id, season, routes, plus routes_source = "participation".
    """
    if participation.empty or pbp.empty:
        return pd.DataFrame(columns=["player_id", "season", "routes", "routes_source"])

    pb = pbp.copy()
    pb["qb_dropback"] = pd.to_numeric(pb["qb_dropback"], errors="coerce")
    dropbacks = pb[(pb.get("season_type") == "REG") & pb["qb_dropback"].eq(1)][
        ["game_id", "play_id", "season"]
    ].drop_duplicates()
    if dropbacks.empty:
        return pd.DataFrame(columns=["player_id", "season", "routes", "routes_source"])

    part = participation.rename(columns={"nflverse_game_id": "game_id"})
    keep = [c for c in ("game_id", "play_id", "offense_players", "offense_positions") if c in part.columns]
    part = part[keep].copy()

    # Align join key dtypes: participation stores play_id as float, pbp as float/int.
    dropbacks["play_id"] = pd.to_numeric(dropbacks["play_id"], errors="coerce")
    part["play_id"] = pd.to_numeric(part["play_id"], errors="coerce")

    joined = dropbacks.merge(part, on=["game_id", "play_id"], how="inner")
    if joined.empty:
        return pd.DataFrame(columns=["player_id", "season", "routes", "routes_source"])

    joined = joined[joined["offense_players"].notna() & joined["offense_positions"].notna()]
    ids = joined["offense_players"].str.split(";")
    pos = joined["offense_positions"].str.split(";")
    seasons = joined["season"].to_numpy()

    records: dict[tuple[str, int], int] = {}
    for id_list, pos_list, season in zip(ids, pos, seasons):
        if id_list is None or pos_list is None or len(id_list) != len(pos_list):
            continue
        for pid, ppos in zip(id_list, pos_list):
            if ppos in ROUTE_ELIGIBLE_POSITIONS and pid:
                key = (pid, int(season))
                records[key] = records.get(key, 0) + 1

    if not records:
        return pd.DataFrame(columns=["player_id", "season", "routes", "routes_source"])

    out = pd.DataFrame(
        [{"player_id": k[0], "season": k[1], "routes": v} for k, v in records.items()]
    )
    out["routes_source"] = "participation"
    return out


def routes_proxy(weekly: pd.DataFrame, snaps: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Fallback route estimate for seasons with no participation release.

    Estimates pass snaps as offensive snaps times the team's pass rate. Strictly a
    degradation path: labelled routes_source = "pass-snap proxy" so the sheet never
    presents it as measured routes.
    """
    if snaps.empty or pbp.empty:
        return pd.DataFrame(columns=["player_id", "season", "routes", "routes_source"])

    pb = pbp.copy()
    pb["qb_dropback"] = pd.to_numeric(pb["qb_dropback"], errors="coerce").fillna(0)
    pb["rush_attempt"] = pd.to_numeric(pb.get("rush_attempt"), errors="coerce").fillna(0)
    team_rate = (
        pb[pb.get("season_type") == "REG"]
        .groupby(["posteam", "season"])
        .agg(dropbacks=("qb_dropback", "sum"), plays=("play_id", "count"))
        .reset_index()
    )
    team_rate["pass_rate"] = team_rate["dropbacks"] / team_rate["plays"].replace(0, np.nan)

    sn = snaps.copy()
    sn["offense_snaps"] = pd.to_numeric(sn["offense_snaps"], errors="coerce").fillna(0)
    agg = sn.groupby(["pfr_player_id", "season", "team"], as_index=False)["offense_snaps"].sum()
    agg = agg.merge(
        team_rate.rename(columns={"posteam": "team"})[["team", "season", "pass_rate"]],
        on=["team", "season"],
        how="left",
    )
    agg["routes"] = agg["offense_snaps"] * agg["pass_rate"]
    agg["routes_source"] = "pass-snap proxy"
    return agg.rename(columns={"pfr_player_id": "pfr_id"})[
        ["pfr_id", "season", "routes", "routes_source"]
    ]


# --------------------------------------------------------------------------------------
# Efficiency components
# --------------------------------------------------------------------------------------


def _season_receiving(weekly: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "receiving_yards": "rec_yards",
        "receptions": "receptions",
        "targets": "targets",
        "receiving_tds": "rec_tds",
    }
    present = {k: v for k, v in cols.items() if k in weekly.columns}
    if not present:
        return pd.DataFrame(columns=["player_id", "season"])
    w = weekly.copy()
    for col in present:
        w[col] = pd.to_numeric(w[col], errors="coerce").fillna(0.0)
    return (
        w.groupby(["player_id", "season"], as_index=False)[list(present)]
        .sum()
        .rename(columns=present)
    )


def _season_passing_rushing(weekly: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "passing_yards": "pass_yards",
        "attempts": "attempts",
        "passing_tds": "pass_tds",
        "sacks_suffered": "sacks_taken",
        "carries": "carries",
        "rushing_yards": "rush_yards",
        "rushing_tds": "rush_tds",
    }
    present = {k: v for k, v in cols.items() if k in weekly.columns}
    if not present:
        return pd.DataFrame(columns=["player_id", "season"])
    w = weekly.copy()
    for col in present:
        w[col] = pd.to_numeric(w[col], errors="coerce").fillna(0.0)
    return (
        w.groupby(["player_id", "season"], as_index=False)[list(present)]
        .sum()
        .rename(columns=present)
    )


def _expected_tds(ff_opp: pd.DataFrame) -> pd.DataFrame:
    """Actual-minus-expected TD rate per player-season from the ffopportunity model."""
    if ff_opp.empty:
        return pd.DataFrame(columns=["player_id", "season", "td_diff_per_opp"])

    df = ff_opp.copy()
    id_col = next((c for c in ("player_id", "gsis_id", "player_gsis_id") if c in df.columns), None)
    if id_col is None:
        return pd.DataFrame(columns=["player_id", "season", "td_diff_per_opp"])

    # ffopportunity publishes season as a string; the stats frames use int. Normalise or the
    # merge fails outright on mismatched key dtypes.
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df.dropna(subset=["season"])
    df["season"] = df["season"].astype(int)

    actual_cols = [c for c in ("rec_touchdown", "rush_touchdown", "pass_touchdown") if c in df.columns]
    exp_cols = [c for c in ("rec_touchdown_exp", "rush_touchdown_exp", "pass_touchdown_exp") if c in df.columns]
    if not actual_cols or not exp_cols:
        # Fall back to the model's fantasy-point differential, which embeds TD luck.
        if "total_fantasy_points_diff" in df.columns:
            g = df.groupby([id_col, "season"], as_index=False)["total_fantasy_points_diff"].mean()
            return g.rename(columns={id_col: "player_id", "total_fantasy_points_diff": "td_diff_per_opp"})
        return pd.DataFrame(columns=["player_id", "season", "td_diff_per_opp"])

    for c in actual_cols + exp_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    agg = df.groupby([id_col, "season"], as_index=False)[actual_cols + exp_cols].sum()
    agg["td_diff_per_opp"] = agg[actual_cols].sum(axis=1) - agg[exp_cols].sum(axis=1)
    return agg.rename(columns={id_col: "player_id"})[["player_id", "season", "td_diff_per_opp"]]


def compute_efficiency_score(
    weekly: pd.DataFrame,
    pbp: pd.DataFrame,
    participation: pd.DataFrame,
    snaps: pd.DataFrame,
    adv_rush: pd.DataFrame,
    adv_rec: pd.DataFrame,
    ff_opp: pd.DataFrame,
    positions: pd.Series,
    pfr_ids: pd.Series,
    target_season: int,
) -> pd.DataFrame:
    """STEP 2 efficiency: one recency-weighted sub-value per player, ready to be z-scored.

    Components with too small a sample are dropped from the player's average rather than
    filled with a noisy number (see MIN_ROUTES_FOR_YPRR and friends in config/weights.py).
    """
    routes = compute_routes(participation, pbp)
    routes_source = "participation"
    if routes.empty:
        proxy = routes_proxy(weekly, snaps, pbp)
        routes_source = "pass-snap proxy"
        if not proxy.empty:
            pfr_to_gsis = {v: k for k, v in pfr_ids.dropna().items()}
            proxy["player_id"] = proxy["pfr_id"].map(pfr_to_gsis)
            routes = proxy.dropna(subset=["player_id"])[
                ["player_id", "season", "routes", "routes_source"]
            ]

    receiving = _season_receiving(weekly)
    passrush = _season_passing_rushing(weekly)
    td_diff = _expected_tds(ff_opp)

    per_season = receiving.merge(passrush, on=["player_id", "season"], how="outer")
    if not routes.empty:
        per_season = per_season.merge(
            routes[["player_id", "season", "routes"]], on=["player_id", "season"], how="left"
        )
    else:
        per_season["routes"] = np.nan
    per_season = per_season.merge(td_diff, on=["player_id", "season"], how="left")

    # YPRR -- suppressed below the minimum route sample.
    per_season["yprr"] = np.where(
        per_season["routes"].fillna(0) >= W.MIN_ROUTES_FOR_YPRR,
        per_season["rec_yards"] / per_season["routes"].replace(0, np.nan),
        np.nan,
    )

    # QB rate stats -- suppressed below the minimum dropback sample.
    attempts = per_season.get("attempts", pd.Series(np.nan, index=per_season.index))
    sacks = per_season.get("sacks_taken", pd.Series(0.0, index=per_season.index)).fillna(0.0)
    dropbacks = attempts.fillna(0) + sacks
    enough_qb = dropbacks >= W.MIN_DROPBACKS_FOR_QB_EFF
    per_season["ypa"] = np.where(enough_qb, per_season.get("pass_yards") / attempts.replace(0, np.nan), np.nan)
    # "Sack rate avoided" is expressed so higher is better, matching the other components.
    per_season["sack_rate_avoided"] = np.where(enough_qb, 1.0 - (sacks / dropbacks.replace(0, np.nan)), np.nan)

    # TD rate versus expected, normalised per opportunity so volume does not dominate.
    opportunities = per_season.get("targets", pd.Series(0.0, index=per_season.index)).fillna(0.0) + per_season.get(
        "carries", pd.Series(0.0, index=per_season.index)
    ).fillna(0.0) + attempts.fillna(0.0)
    per_season["td_rate_vs_expected"] = np.where(
        opportunities >= W.MIN_TARGETS_FOR_TD_RATE,
        per_season["td_diff_per_opp"] / opportunities.replace(0, np.nan),
        np.nan,
    )

    # PFR advanced stats: yards after contact per attempt and broken tackle rate.
    adv = _merge_pfr(adv_rush, adv_rec)
    if not adv.empty:
        gsis_by_pfr = {v: k for k, v in pfr_ids.dropna().items()}
        adv["player_id"] = adv["pfr_id"].map(gsis_by_pfr)
        adv = adv.dropna(subset=["player_id"])
        per_season = per_season.merge(
            adv[["player_id", "season", "yards_after_contact", "broken_tackle_rate"]],
            on=["player_id", "season"],
            how="left",
        )
    else:
        per_season["yards_after_contact"] = np.nan
        per_season["broken_tackle_rate"] = np.nan

    component_cols = [
        "yprr", "yards_after_contact", "broken_tackle_rate",
        "td_rate_vs_expected", "ypa", "sack_rate_avoided",
    ]
    weighted = recency_weighted_mean(per_season, component_cols, target_season)
    if weighted.empty:
        return pd.DataFrame(columns=["player_id", "efficiency_value", "efficiency_detail", "routes_source"])

    weighted = weighted.copy()
    weighted["position"] = weighted["player_id"].map(positions)
    weighted["routes_source"] = routes_source

    total_routes = (
        routes.groupby("player_id", as_index=False)["routes"].sum() if not routes.empty else pd.DataFrame(columns=["player_id", "routes"])
    )
    weighted = weighted.merge(total_routes.rename(columns={"routes": "total_routes"}), on="player_id", how="left")

    values = pd.Series(np.nan, index=weighted.index, dtype=float)
    details = pd.Series("", index=weighted.index, dtype=object)

    for pos, sub_weights in W.EFFICIENCY_SUB_WEIGHTS.items():
        mask = weighted["position"] == pos
        if not mask.any():
            continue
        usable = {k: v for k, v in sub_weights.items() if k in weighted.columns}
        if not usable:
            continue
        block = weighted.loc[mask, list(usable)].apply(pd.to_numeric, errors="coerce")
        w_series = pd.Series(usable, dtype=float)

        # Each component is z-scored inside the position before being combined, because the
        # raw units are incommensurable (YPRR ~2.0 vs broken tackle rate ~0.15); an
        # unnormalised weighted sum would let the largest-scaled component dominate.
        z_block = block.apply(lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) else s * 0.0)
        # Backstop against a surviving small-sample outlier dominating the average.
        z_block = z_block.clip(-W.EFFICIENCY_Z_CLIP, W.EFFICIENCY_Z_CLIP)
        avail = z_block.notna()
        wmat = avail.mul(w_series, axis=1)
        wsum = wmat.sum(axis=1)
        values.loc[mask] = z_block.fillna(0.0).mul(w_series, axis=1).sum(axis=1) / wsum.replace(0, np.nan)
        details.loc[mask] = block.apply(
            lambda r: ", ".join(f"{c}={r[c]:.3f}" for c in block.columns if pd.notna(r[c])) or "insufficient data",
            axis=1,
        )

    weighted["efficiency_value"] = values
    weighted["efficiency_detail"] = details
    return weighted


def _merge_pfr(adv_rush: pd.DataFrame, adv_rec: pd.DataFrame) -> pd.DataFrame:
    """Combine PFR rushing and receiving advanced stats into per-player-season efficiency."""
    frames = []
    if not adv_rush.empty and {"pfr_id", "season"}.issubset(adv_rush.columns):
        r = adv_rush.copy()
        r["yac_att"] = pd.to_numeric(r.get("yac_att"), errors="coerce")
        r["brk_tkl"] = pd.to_numeric(r.get("brk_tkl"), errors="coerce")
        r["att"] = pd.to_numeric(r.get("att"), errors="coerce")
        r["rush_btr"] = r["brk_tkl"] / r["att"].replace(0, np.nan)
        # Suppress rate stats built on a handful of carries. Without this a 5-carry sample is
        # treated as equal evidence to a 300-carry season, and small denominators produce
        # extreme rates that survive z-scoring as huge outliers.
        thin = r["att"].fillna(0) < W.MIN_CARRIES_FOR_RUSH_EFF
        r.loc[thin, ["yac_att", "rush_btr"]] = np.nan
        frames.append(r[["pfr_id", "season", "yac_att", "rush_btr"]].rename(columns={"yac_att": "rush_yac_att"}))
    if not adv_rec.empty and {"pfr_id", "season"}.issubset(adv_rec.columns):
        c = adv_rec.copy()
        c["yac_r"] = pd.to_numeric(c.get("yac_r"), errors="coerce")
        c["brk_tkl"] = pd.to_numeric(c.get("brk_tkl"), errors="coerce")
        c["rec"] = pd.to_numeric(c.get("rec"), errors="coerce")
        c["rec_btr"] = c["brk_tkl"] / c["rec"].replace(0, np.nan)
        thin_rec = c["rec"].fillna(0) < W.MIN_RECEPTIONS_FOR_REC_EFF
        c.loc[thin_rec, ["yac_r", "rec_btr"]] = np.nan
        frames.append(c[["pfr_id", "season", "yac_r", "rec_btr"]].rename(columns={"yac_r": "rec_yac_r"}))

    if not frames:
        return pd.DataFrame(columns=["pfr_id", "season", "yards_after_contact", "broken_tackle_rate"])

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["pfr_id", "season"], how="outer")

    rush_yac = out.get("rush_yac_att", pd.Series(np.nan, index=out.index))
    rec_yac = out.get("rec_yac_r", pd.Series(np.nan, index=out.index))
    out["yards_after_contact"] = rush_yac.fillna(rec_yac)
    rush_btr = out.get("rush_btr", pd.Series(np.nan, index=out.index))
    rec_btr = out.get("rec_btr", pd.Series(np.nan, index=out.index))
    out["broken_tackle_rate"] = rush_btr.fillna(rec_btr)
    return out[["pfr_id", "season", "yards_after_contact", "broken_tackle_rate"]]


def yprr_label(routes_source: str) -> str:
    """Column label that is honest about how routes were counted."""
    if routes_source == "participation":
        return "YPRR (routes from participation; RB routes are pass-snap-based)"
    return "YPRR (pass-snap proxy - participation unavailable)"
