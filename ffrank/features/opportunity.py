"""STEP 2 -- Opportunity share (25% of the Composite Z-Score).

Each position gets its own definition of "opportunity", folded into ONE sub-value per
player before it is z-scored inside the position group:

  RB     carry share 60% + red-zone carry share 25% + target share 15%
  WR/TE  target share 50% + air yards share 30% + red-zone target share 20%
  QB     dropback share + designed rush share + red-zone pass attempt share, equal weights

Shares are computed per season as player-over-team, then recency-weighted with STEP 1's
55/30/15 scheme so opportunity and PPG use one consistent recency convention.

Red-zone counts come from play-by-play (yardline_100 <= 20); base target and air-yards
shares come from nflverse's weekly stats, which already publish them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import weights as W
from .weighted_ppg import recency_weighted_mean

RED_ZONE_YARDLINE = 20

# Play-by-play columns needed by this module (kept explicit so the pipeline can request a
# narrow pbp frame instead of all ~370 columns).
PBP_COLUMNS = [
    "season", "week", "game_id", "play_id", "season_type", "posteam",
    "play_type", "qb_dropback", "qb_scramble", "rush_attempt", "pass_attempt", "sack",
    "passer_player_id", "rusher_player_id", "receiver_player_id",
    "yardline_100", "air_yards", "yards_gained", "receiving_yards", "passing_yards",
    "pass_touchdown", "rush_touchdown", "complete_pass", "interception",
]


def _safe_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return (numerator / denom).replace([np.inf, -np.inf], np.nan)


def _player_team_counts(
    pbp: pd.DataFrame,
    player_col: str,
    mask: pd.Series,
    label: str,
    player_games: pd.Series | None = None,
) -> pd.DataFrame:
    """Share of team opportunity per GAME PLAYED, per player-season.

    Computed as (player events per game he played) / (team events per game the team played),
    NOT as raw season totals. This matters as much here as it does in STEP 1, and for the same
    reason: a player who misses half the season accumulates half the raw team share, so a
    totals-based share punishes an injury-shortened season a second time on top of the
    Expected Games Played discount in STEP 3c.

    The size of the error is not marginal. In 2025 Omarion Hampton took 124 carries in 9 games
    while the Chargers ran 411 times across 17. Season totals give him a 0.302 carry share
    against Kyren Williams' 0.566 -- but per game played they are 0.570 and 0.566, essentially
    identical. The totals view invented a 26-point gap that did not exist.

    The team denominator uses the same mask, so red-zone carry share is genuinely "this
    player's red-zone carries per game over his team's red-zone carries per game".
    """
    sub = pbp[mask & pbp[player_col].notna()]
    if sub.empty:
        return pd.DataFrame(columns=["player_id", "season", label])

    player = (
        sub.groupby([player_col, "season", "posteam"], dropna=True)
        .size()
        .reset_index(name="player_count")
        .rename(columns={player_col: "player_id"})
    )
    masked_team = pbp[mask & pbp["posteam"].notna()]
    team = (
        masked_team.groupby(["posteam", "season"], dropna=True)
        .agg(team_count=("play_id", "size"), team_games=("game_id", "nunique"))
        .reset_index()
    )
    merged = player.merge(team, on=["posteam", "season"], how="left")
    merged["team_rate"] = _safe_share(merged["team_count"], merged["team_games"])

    # Total events across every team the player appeared for that season.
    totals = merged.groupby(["player_id", "season"], as_index=False)["player_count"].sum()

    # For a traded player, blend his teams' per-game rates weighted by how much of his own
    # volume came with each, so the denominator reflects the offences he actually played in.
    merged["weighted_rate"] = merged["team_rate"] * merged["player_count"]
    blended = merged.groupby(["player_id", "season"], as_index=False).agg(
        weighted_rate=("weighted_rate", "sum"), weight=("player_count", "sum")
    )
    blended["team_rate"] = _safe_share(blended["weighted_rate"], blended["weight"])

    out = totals.merge(blended[["player_id", "season", "team_rate"]], on=["player_id", "season"])

    if player_games is not None:
        games = pd.MultiIndex.from_frame(out[["player_id", "season"]]).map(player_games)
        out["games"] = pd.to_numeric(pd.Series(games, index=out.index), errors="coerce")
    else:
        out["games"] = np.nan
    # Fall back to the team's game count when games played is unknown, which reduces to the old
    # totals-based behaviour rather than dropping the player entirely.
    out["games"] = out["games"].where(out["games"] > 0)

    player_rate = _safe_share(out["player_count"], out["games"])
    out[label] = _safe_share(player_rate, out["team_rate"])
    return out[["player_id", "season", label]]


def compute_opportunity_inputs(pbp: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """Per player-season opportunity components, before recency weighting."""
    if pbp.empty:
        return pd.DataFrame(columns=["player_id", "season"])

    df = pbp.copy()
    for col in ("qb_dropback", "qb_scramble", "rush_attempt", "pass_attempt", "yardline_100", "sack"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]

    in_red_zone = df["yardline_100"].le(RED_ZONE_YARDLINE)
    is_designed_run = df["rush_attempt"].eq(1) & df["qb_scramble"].ne(1)
    is_dropback = df["qb_dropback"].eq(1)
    is_target = df["pass_attempt"].eq(1) & df["receiver_player_id"].notna()

    # Games actually played, from the same source STEP 1 uses, so both steps agree on the
    # denominator and neither double-counts missed time.
    player_games = None
    if not weekly.empty and {"player_id", "season", "week"}.issubset(weekly.columns):
        player_games = (
            weekly.groupby(["player_id", "season"])["week"].nunique().rename("games")
        )

    parts = [
        _player_team_counts(df, "rusher_player_id", is_designed_run, "carry_share", player_games),
        _player_team_counts(df, "rusher_player_id", is_designed_run & in_red_zone, "rz_carry_share", player_games),
        _player_team_counts(df, "receiver_player_id", is_target & in_red_zone, "rz_target_share", player_games),
        _player_team_counts(df, "passer_player_id", is_dropback, "dropback_share", player_games),
        _player_team_counts(df, "passer_player_id", is_dropback & in_red_zone, "rz_pass_attempt_share", player_games),
    ]

    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["player_id", "season"], how="outer")

    # QB designed rush share: QB carries that are NOT scrambles, over team designed runs.
    # A QB's carry_share already captures this, so it is reused rather than recomputed.
    out["designed_rush_share"] = out["carry_share"]

    # Base target share / air yards share are already published per week by nflverse; average
    # them across the player's games in a season.
    if not weekly.empty:
        w = weekly.copy()
        for col in ("target_share", "air_yards_share"):
            if col in w.columns:
                w[col] = pd.to_numeric(w[col], errors="coerce")
        share_cols = [c for c in ("target_share", "air_yards_share") if c in w.columns]
        if share_cols:
            weekly_shares = (
                w.groupby(["player_id", "season"], dropna=True)[share_cols].mean().reset_index()
            )
            out = out.merge(weekly_shares, on=["player_id", "season"], how="outer")

    return out


def compute_opportunity_score(
    pbp: pd.DataFrame,
    weekly: pd.DataFrame,
    positions: pd.Series,
    target_season: int,
) -> pd.DataFrame:
    """STEP 2 opportunity: one recency-weighted sub-value per player, ready to be z-scored.

    Args:
        positions: Series mapping player_id -> position, used to pick each player's
            position-specific sub-weight table.

    Returns player_id, opportunity_value, opportunity_detail, and the raw components for
    auditing on the sheet.
    """
    per_season = compute_opportunity_inputs(pbp, weekly)
    if per_season.empty:
        return pd.DataFrame(columns=["player_id", "opportunity_value", "opportunity_detail"])

    component_cols = [
        "carry_share", "rz_carry_share", "target_share", "air_yards_share",
        "rz_target_share", "dropback_share", "designed_rush_share", "rz_pass_attempt_share",
    ]
    present = [c for c in component_cols if c in per_season.columns]
    weighted = recency_weighted_mean(per_season, present, target_season)
    if weighted.empty:
        return pd.DataFrame(columns=["player_id", "opportunity_value", "opportunity_detail"])

    weighted = weighted.copy()
    weighted["position"] = weighted["player_id"].map(positions)

    values, details = fold_opportunity(weighted, weighted["position"])
    weighted["opportunity_value"] = values
    weighted["opportunity_detail"] = details
    return weighted


def fold_opportunity(
    frame: pd.DataFrame, position_series: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Fold the per-position sub-components into ONE opportunity value, plus an audit string.

    Shared by the live scoring path and by the historical role-prior derivation, so a prior is
    always expressed in exactly the same units as the value it will be blended with.
    """
    values = pd.Series(np.nan, index=frame.index, dtype=float)
    details = pd.Series("", index=frame.index, dtype=object)

    for pos, sub_weights in W.OPPORTUNITY_SUB_WEIGHTS.items():
        mask = position_series == pos
        if not mask.any():
            continue
        usable = {k: v for k, v in sub_weights.items() if k in frame.columns}
        if not usable:
            continue
        block = frame.loc[mask, list(usable)].apply(pd.to_numeric, errors="coerce")
        w_series = pd.Series(usable, dtype=float)
        # Renormalise over components that actually exist for each player, so a missing
        # component means "average of what we have", not "treated as zero opportunity".
        avail = block.notna()
        wmat = avail.mul(w_series, axis=1)
        wsum = wmat.sum(axis=1)
        values.loc[mask] = (block.fillna(0.0).mul(w_series, axis=1).sum(axis=1) / wsum.replace(0, np.nan))
        details.loc[mask] = block.apply(
            lambda r: ", ".join(f"{c}={r[c]:.3f}" for c in block.columns if pd.notna(r[c])), axis=1
        )

    return values, details


# --------------------------------------------------------------------------------------
# Projected-role blending
# --------------------------------------------------------------------------------------


def derive_role_opportunity_priors(
    depth_history: pd.DataFrame,
    pbp_history: pd.DataFrame,
    weekly_history: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """What opportunity share does a given depth-chart slot historically earn?

    Built by joining each season's opening depth chart to that season's realised opportunity
    value, so the answer is empirical rather than assumed. Monotonicity is enforced across
    ranks: a lower slot cannot be expected to out-earn a higher one, which corrects small-sample
    inversions (a 16-player QB3 bucket came out above QB2 before this).
    """
    if depth_history.empty or pbp_history.empty:
        return {}

    frames = []
    for season, dc_season in depth_history.groupby("season"):
        pbp_season = pbp_history[pbp_history["season"] == season]
        weekly_season = weekly_history[weekly_history["season"] == season]
        if pbp_season.empty:
            continue
        per_season = compute_opportunity_inputs(pbp_season, weekly_season)
        if per_season.empty:
            continue
        # Positions come from the depth chart itself, so the prior does not depend on a
        # current-season roster map that would not cover players from past seasons.
        pos_map = dc_season.drop_duplicates(subset=["player_id"]).set_index("player_id")["position"]
        per_season = per_season.copy()
        per_season["position"] = per_season["player_id"].map(pos_map)
        per_season = per_season[per_season["position"].notna()]
        if per_season.empty:
            continue
        vals, _ = fold_opportunity(per_season, per_season["position"])
        per_season["opportunity_value"] = vals
        merged = dc_season[["player_id", "position", "depth_chart_rank"]].merge(
            per_season[["player_id", "opportunity_value"]], on="player_id", how="inner"
        )
        frames.append(merged)

    if not frames:
        return {}
    allm = pd.concat(frames, ignore_index=True)
    allm = allm[allm["opportunity_value"].notna()]

    priors: dict[tuple[str, int], float] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        sub = allm[allm["position"] == pos]
        running = None
        for rank in range(1, W.ROLE_PRIOR_MAX_RANK + 1):
            at_rank = sub[sub["depth_chart_rank"] == rank]["opportunity_value"]
            if len(at_rank) < W.ROLE_PRIOR_MIN_SAMPLE:
                continue
            value = float(at_rank.median())
            # Enforce non-increasing expectation as you go down the depth chart.
            if running is not None:
                value = min(value, running)
            running = value
            priors[(pos, rank)] = value
    return priors


def blend_role_expectation(
    opportunity_value: pd.Series,
    positions: pd.Series,
    depth_chart_rank: pd.Series,
    snap_share: pd.Series,
    priors: dict[tuple[str, int], float],
    evidence_k: float | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Blend historical opportunity with what the player's CURRENT depth-chart slot earns.

    The composite is otherwise ~65% backward-looking, while a player's current role entered only
    as one of seven inputs in a 10%-weighted component -- roughly 1.4%. So a back promoted to
    RB1 was still priced as the backup he used to be, and a backup who filled in for an injured
    starter was priced as a starter.

    Weighting is by EVIDENCE, not by rank: w_hist = snap_share / (snap_share + k). A player with
    a full starter's workload behind him is trusted on his own record; a player with a 20% snap
    share has a history that says little about how a starting role would look, so he is pulled
    toward the empirical prior for the slot he now occupies. That cuts both ways by design.

    Returns (blended value, weight given to history, audit note).
    """
    evidence_k = W.ROLE_BLEND_EVIDENCE_K if evidence_k is None else evidence_k

    hist = pd.to_numeric(opportunity_value, errors="coerce")
    rank = pd.to_numeric(depth_chart_rank, errors="coerce")
    evidence = pd.to_numeric(snap_share, errors="coerce").clip(lower=0.0)

    prior = pd.Series(
        [priors.get((p, int(r))) if pd.notna(r) and pd.notna(p) else None for p, r in zip(positions, rank)],
        index=hist.index,
        dtype=float,
    )

    w_hist = evidence / (evidence + evidence_k)
    w_hist = w_hist.where(evidence.notna(), 0.0)
    # With no prior available there is nothing to blend toward.
    w_hist = w_hist.where(prior.notna(), 1.0)
    # With no history at all, lean entirely on the prior.
    w_hist = w_hist.where(hist.notna(), 0.0)

    blended = w_hist * hist.fillna(0.0) + (1.0 - w_hist) * prior.fillna(0.0)
    blended = blended.where(hist.notna() | prior.notna())

    if W.ROLE_BLEND_DIRECTION == "up_only":
        # Never drag a player below his own measured record. See the config note: a player with a
        # real snap share has better evidence about himself than a rank-average prior does.
        blended = blended.where(hist.isna() | (blended >= hist), hist)

    note = pd.Series("", index=hist.index, dtype=object)
    moved = blended.notna() & hist.notna() & (blended - hist).abs().gt(0.02)
    for idx in hist.index[moved]:
        direction = "up" if blended.at[idx] > hist.at[idx] else "down"
        note.at[idx] = (
            f"Opportunity adjusted {direction} from {hist.at[idx]:.3f} to {blended.at[idx]:.3f}: "
            f"depth chart lists him {positions.at[idx]}{int(rank.at[idx])} (slot typically earns "
            f"{prior.at[idx]:.3f}) and his own snap share of {evidence.at[idx]:.2f} carries "
            f"{w_hist.at[idx]:.0%} of the weight"
        )
    return blended, w_hist, note
