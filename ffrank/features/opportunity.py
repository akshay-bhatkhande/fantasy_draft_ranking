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
) -> pd.DataFrame:
    """Count events per player-season and per team-season, then convert to a share.

    The team denominator is computed on the same mask, so e.g. red-zone carry share is
    genuinely "this player's red-zone carries over his team's red-zone carries".
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
    team = (
        pbp[mask & pbp["posteam"].notna()]
        .groupby(["posteam", "season"], dropna=True)
        .size()
        .reset_index(name="team_count")
    )
    merged = player.merge(team, on=["posteam", "season"], how="left")
    merged[label] = _safe_share(merged["player_count"], merged["team_count"])

    # A player traded mid-season appears with two teams; sum his shares so the total
    # reflects his full-season opportunity rather than only his last stop.
    return merged.groupby(["player_id", "season"], as_index=False)[label].sum()


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

    parts = [
        _player_team_counts(df, "rusher_player_id", is_designed_run, "carry_share"),
        _player_team_counts(df, "rusher_player_id", is_designed_run & in_red_zone, "rz_carry_share"),
        _player_team_counts(df, "receiver_player_id", is_target & in_red_zone, "rz_target_share"),
        _player_team_counts(df, "passer_player_id", is_dropback, "dropback_share"),
        _player_team_counts(df, "passer_player_id", is_dropback & in_red_zone, "rz_pass_attempt_share"),
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

    values = pd.Series(np.nan, index=weighted.index, dtype=float)
    details = pd.Series("", index=weighted.index, dtype=object)

    for pos, sub_weights in W.OPPORTUNITY_SUB_WEIGHTS.items():
        mask = weighted["position"] == pos
        if not mask.any():
            continue
        usable = {k: v for k, v in sub_weights.items() if k in weighted.columns}
        if not usable:
            continue
        block = weighted.loc[mask, list(usable)].apply(pd.to_numeric, errors="coerce")
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

    weighted["opportunity_value"] = values
    weighted["opportunity_detail"] = details
    return weighted
