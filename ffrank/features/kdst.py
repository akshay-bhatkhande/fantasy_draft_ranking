"""Kickers and team defenses -- deliberately minimal.

These should be drafted last or streamed, so the full STEP 1-4 VORP pipeline is skipped for
them on purpose. A simple recency-weighted PPG rank is sufficient and is all this module
produces.

One wrinkle: nflverse's fantasy_points_ppr column is 0 for kickers, because it does not score
kicking. So kicker points are computed here from the field-goal distance buckets using this
league's kicker scoring table. Team-defense points are likewise assembled from team defensive
counting stats plus points allowed from the schedule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import weights as W
from config.league import DST_SCORING, K_SCORING
from .weighted_ppg import recency_weighted_mean

# FG distance buckets published by nflverse, mapped to this league's scoring tiers.
FG_BUCKETS = {
    "fg_made_0_19": "fg_made_0_39",
    "fg_made_20_29": "fg_made_0_39",
    "fg_made_30_39": "fg_made_0_39",
    "fg_made_40_49": "fg_made_40_49",
    "fg_made_50_59": "fg_made_50_plus",
    "fg_made_60_": "fg_made_50_plus",
}


def kicker_weekly_points(weekly: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly kicker fantasy points from FG/PAT detail."""
    if weekly.empty:
        return pd.DataFrame(columns=["player_id", "season", "week", "k_points"])
    k = weekly[weekly["position"] == "K"].copy()
    if k.empty:
        return pd.DataFrame(columns=["player_id", "season", "week", "k_points"])

    points = pd.Series(0.0, index=k.index)
    for col, tier in FG_BUCKETS.items():
        if col in k.columns:
            points += pd.to_numeric(k[col], errors="coerce").fillna(0.0) * K_SCORING[tier]
    for col, key in (("fg_missed", "fg_missed"), ("pat_made", "pat_made"), ("pat_missed", "pat_missed")):
        if col in k.columns:
            points += pd.to_numeric(k[col], errors="coerce").fillna(0.0) * K_SCORING[key]

    k["k_points"] = points
    return k[["player_id", "season", "week", "k_points", "player_display_name", "team"]]


def rank_kickers(weekly: pd.DataFrame, target_season: int, roster_teams: pd.Series | None = None) -> pd.DataFrame:
    """Recency-weighted points-per-game rank for kickers (no VORP, by design)."""
    wk = kicker_weekly_points(weekly)
    if wk.empty:
        return pd.DataFrame(columns=["player_name", "team", "weighted_ppg", "rank"])

    per_season = (
        wk.groupby(["player_id", "season"])
        .agg(
            games=("week", "nunique"),
            total=("k_points", "sum"),
            player_name=("player_display_name", "last"),
            team=("team", "last"),
        )
        .reset_index()
    )
    per_season["season_ppg"] = per_season["total"] / per_season["games"].replace(0, np.nan)

    weighted = recency_weighted_mean(per_season, ["season_ppg"], target_season)
    meta = per_season.sort_values("season").groupby("player_id").agg(
        player_name=("player_name", "last"), team=("team", "last"), games=("games", "sum")
    )
    out = weighted.merge(meta, on="player_id", how="left").rename(columns={"season_ppg": "weighted_ppg"})
    if roster_teams is not None:
        # Keep only kickers still on an NFL roster for the upcoming season.
        out["current_team"] = out["player_id"].map(roster_teams)
        out = out[out["current_team"].notna()]
        out["team"] = out["current_team"]
    out = out.dropna(subset=["weighted_ppg"]).sort_values("weighted_ppg", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    out["note"] = "Draft last / stream. Simple recency-weighted PPG; full VORP pipeline intentionally skipped."
    return out[["rank", "player_name", "team", "weighted_ppg", "games", "note"]]


def _points_allowed_score(points_allowed: float) -> float:
    for max_pa, pts in DST_SCORING["points_allowed_tiers"]:
        if points_allowed <= max_pa:
            return pts
    return DST_SCORING["points_allowed_tiers"][-1][1]


def dst_weekly_points(team_stats: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Weekly team-defense fantasy points from defensive counting stats + points allowed."""
    if team_stats.empty:
        return pd.DataFrame(columns=["team", "season", "week", "dst_points"])

    ts = team_stats.copy()
    team_col = "team" if "team" in ts.columns else None
    if team_col is None:
        return pd.DataFrame(columns=["team", "season", "week", "dst_points"])

    def num(col):
        return pd.to_numeric(ts.get(col), errors="coerce").fillna(0.0) if col in ts.columns else 0.0

    pts = (
        num("def_sacks") * DST_SCORING["sack"]
        + num("def_interceptions") * DST_SCORING["interception"]
        + num("def_fumbles_recovered" if "def_fumbles_recovered" in ts.columns else "def_fumbles") * DST_SCORING["fumble_recovery"]
        + num("def_safeties") * DST_SCORING["safety"]
        + num("def_tds") * DST_SCORING["touchdown"]
    )
    ts["dst_base"] = pts

    # Points allowed comes from the schedule: the opponent's score in that game.
    if not schedules.empty:
        s = schedules[schedules.get("game_type", "REG") == "REG"]
        home = s[["season", "week", "home_team", "away_score"]].rename(
            columns={"home_team": "team", "away_score": "points_allowed"}
        )
        away = s[["season", "week", "away_team", "home_score"]].rename(
            columns={"away_team": "team", "home_score": "points_allowed"}
        )
        pa = pd.concat([home, away], ignore_index=True)
        ts = ts.merge(pa, on=["team", "season", "week"], how="left")
        ts["pa_points"] = pd.to_numeric(ts["points_allowed"], errors="coerce").map(
            lambda v: _points_allowed_score(v) if pd.notna(v) else 0.0
        )
    else:
        ts["pa_points"] = 0.0

    ts["dst_points"] = ts["dst_base"] + ts["pa_points"]
    return ts[["team", "season", "week", "dst_points"]]


def rank_defenses(team_stats: pd.DataFrame, schedules: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """Recency-weighted points-per-game rank for team defenses (no VORP, by design)."""
    wk = dst_weekly_points(team_stats, schedules)
    if wk.empty:
        return pd.DataFrame(columns=["rank", "team", "weighted_ppg", "games", "note"])

    per_season = (
        wk.groupby(["team", "season"])
        .agg(games=("week", "nunique"), total=("dst_points", "sum"))
        .reset_index()
    )
    per_season["season_ppg"] = per_season["total"] / per_season["games"].replace(0, np.nan)

    weighted = recency_weighted_mean(per_season, ["season_ppg"], target_season, key="team")
    out = weighted.rename(columns={"season_ppg": "weighted_ppg"})
    games = per_season.groupby("team")["games"].sum()
    out["games"] = out["team"].map(games)
    out = out.dropna(subset=["weighted_ppg"]).sort_values("weighted_ppg", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    out["note"] = "Draft last / stream. Simple recency-weighted PPG; full VORP pipeline intentionally skipped."
    return out[["rank", "team", "weighted_ppg", "games", "note"]]
