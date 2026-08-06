"""STEP 1 -- Weighted PPG (the per-game baseline, in real fantasy points).

Recency-weighted average points-per-game using ONLY games actually played. Season totals
divided by 17 are never used: that would punish an injury-shortened season twice, since
missed games are already handled separately as a literal games count in STEP 3c.

Default weights: last season 55%, two seasons ago 30%, three seasons ago 15%. Players with
fewer than three seasons of data (rookies, second-year players) get the available weights
redistributed proportionally and are flagged "limited sample".

Output is a real points number (e.g. 16.2 points/game), not an index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import weights as W

FANTASY_COL = "fantasy_points_ppr"


def season_ppg_table(weekly: pd.DataFrame) -> pd.DataFrame:
    """Per player-season points-per-game over games actually played.

    Shared building block: STEP 1 weights these, and curves.py reuses the same table for
    the age-curve regression and the empirical contract-year lift so all three are
    consistent about what "a season's PPG" means.
    """
    if weekly.empty:
        return pd.DataFrame(columns=["player_id", "season", "games_played", "season_ppg", "total_points"])

    df = weekly.copy()
    df[FANTASY_COL] = pd.to_numeric(df[FANTASY_COL], errors="coerce").fillna(0.0)

    grouped = (
        df.groupby(["player_id", "season"], dropna=True)
        .agg(
            games_played=("week", "nunique"),
            total_points=(FANTASY_COL, "sum"),
            position=("position", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
            team=("team", lambda s: s.dropna().iloc[-1] if s.notna().any() else None),
            player_name=("player_display_name", lambda s: s.dropna().iloc[-1] if s.notna().any() else None),
        )
        .reset_index()
    )
    grouped = grouped[grouped["games_played"] >= W.MIN_GAMES_FOR_SEASON]
    grouped["season_ppg"] = grouped["total_points"] / grouped["games_played"]
    return grouped


def compute_weighted_ppg(
    weekly: pd.DataFrame,
    target_season: int,
    recency_weights: dict[int, float] | None = None,
) -> pd.DataFrame:
    """STEP 1: recency-weighted PPG per player, with proportional reweighting.

    Args:
        weekly: weekly player stats covering the lookback seasons.
        target_season: the season being drafted. "Seasons ago" is measured from this.
        recency_weights: {seasons_ago: weight}; defaults to 55/30/15.

    Returns one row per player with weighted_ppg, the seasons that contributed, the games
    behind each, and a limited_sample flag.
    """
    recency_weights = recency_weights or W.RECENCY_WEIGHTS
    per_season = season_ppg_table(weekly)
    if per_season.empty:
        return pd.DataFrame(
            columns=[
                "player_id", "player_name", "position", "weighted_ppg", "seasons_used",
                "total_games_played", "limited_sample", "ppg_detail",
            ]
        )

    per_season = per_season.copy()
    per_season["seasons_ago"] = target_season - per_season["season"]
    per_season = per_season[per_season["seasons_ago"].isin(recency_weights.keys())]
    if per_season.empty:
        return pd.DataFrame(
            columns=[
                "player_id", "player_name", "position", "weighted_ppg", "seasons_used",
                "total_games_played", "limited_sample", "ppg_detail",
            ]
        )

    per_season["raw_weight"] = per_season["seasons_ago"].map(recency_weights).astype(float)

    # Proportional redistribution: divide each contributing season's nominal weight by the
    # total nominal weight actually present for that player. A player with only last season
    # available gets 0.55/0.55 = 1.0 on it rather than a spuriously deflated average.
    weight_sum = per_season.groupby("player_id")["raw_weight"].transform("sum")
    per_season["applied_weight"] = per_season["raw_weight"] / weight_sum
    per_season["weighted_contribution"] = per_season["season_ppg"] * per_season["applied_weight"]

    max_seasons = len(recency_weights)

    def _detail(group: pd.DataFrame) -> str:
        bits = [
            f"{int(r.season)}: {r.season_ppg:.1f} ppg in {int(r.games_played)}g (w={r.applied_weight:.2f})"
            for r in group.sort_values("season", ascending=False).itertuples()
        ]
        return "; ".join(bits)

    agg = (
        per_season.groupby("player_id")
        .agg(
            weighted_ppg=("weighted_contribution", "sum"),
            seasons_used=("season", "nunique"),
            total_games_played=("games_played", "sum"),
            last_season_games=("games_played", "last"),
        )
        .reset_index()
    )

    details = per_season.groupby("player_id", group_keys=False).apply(_detail, include_groups=False)
    agg["ppg_detail"] = agg["player_id"].map(details)

    # Latest known name/position/team from the most recent contributing season.
    latest = (
        per_season.sort_values("season")
        .groupby("player_id")
        .agg(player_name=("player_name", "last"), position=("position", "last"), stat_team=("team", "last"))
        .reset_index()
    )
    agg = agg.merge(latest, on="player_id", how="left")

    agg["limited_sample"] = agg["seasons_used"] < max_seasons
    agg["weighted_ppg"] = agg["weighted_ppg"].replace([np.inf, -np.inf], np.nan)
    return agg


def recency_weighted_mean(
    per_season: pd.DataFrame,
    value_cols: list[str],
    target_season: int,
    key: str = "player_id",
    recency_weights: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Recency-weight any per-player-per-season metric with STEP 1's weighting scheme.

    Reused by the STEP 2 opportunity/efficiency features so that a player's opportunity
    share is weighted the same way his PPG is -- one recency convention across the model,
    with the same proportional redistribution when seasons are missing.
    """
    recency_weights = recency_weights or W.RECENCY_WEIGHTS
    if per_season.empty:
        return pd.DataFrame(columns=[key, *value_cols])

    df = per_season.copy()
    df["seasons_ago"] = target_season - df["season"]
    df = df[df["seasons_ago"].isin(recency_weights.keys())]
    if df.empty:
        return pd.DataFrame(columns=[key, *value_cols])

    df["raw_weight"] = df["seasons_ago"].map(recency_weights).astype(float)

    out = {}
    for col in value_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        # Renormalise per column so a metric missing in one season does not shrink the
        # average toward zero; only seasons where the metric exists carry weight.
        w = df["raw_weight"].where(vals.notna(), 0.0)
        wsum = w.groupby(df[key]).transform("sum")
        applied = (w / wsum).where(wsum > 0, 0.0)
        contrib = (vals.fillna(0.0) * applied).groupby(df[key]).sum()
        has_any = vals.notna().groupby(df[key]).any()
        out[col] = contrib.where(has_any)

    result = pd.DataFrame(out).reset_index().rename(columns={"index": key})
    if key not in result.columns:
        result = result.rename(columns={result.columns[0]: key})
    return result


def limited_sample_note(row) -> str:
    """Notes text explaining a reweighted STEP 1 average."""
    if not row.get("limited_sample"):
        return ""
    raw = row.get("seasons_used")
    n = 0 if raw is None or pd.isna(raw) else int(raw)
    if n <= 0:
        return "Limited sample: no qualifying prior NFL seasons (no STEP 1 history)"
    word = "season" if n == 1 else "seasons"
    return f"Limited sample: only {n} {word} of data; STEP 1 weights redistributed proportionally"
