"""Tier 1 data access: nflverse via nflreadpy.

This is the backbone for most of Steps 1-2. Every loader returns a pandas DataFrame
(nflreadpy hands back polars; we convert once, here, so the rest of the codebase only
deals with one dataframe library).

Season clamping is enforced in one place. nflreadpy's load_participation computes
    max_season = get_current_season(roster=True) - 1
and raises ValueError outside 2016..max_season. The roster year flips on March 15, so
during a 2026 offseason the newest completed season is 2025. Passing the target season
through unclamped would crash the run, so clamp_stat_seasons() is applied to every
historical loader rather than trusted to callers.
"""

from __future__ import annotations

import functools

import pandas as pd

from .cache import SourceLog, configure_nflreadpy, safe_load

configure_nflreadpy()

import nflreadpy as nfl  # noqa: E402 - must follow configure_nflreadpy()

# Earliest season nflverse publishes participation data for.
PARTICIPATION_MIN_SEASON = 2016


def max_stat_season() -> int:
    """Newest COMPLETED season available for historical stats.

    Mirrors nflreadpy's own internal ceiling so we clamp to exactly what it will serve.
    """
    return int(nfl.get_current_season(roster=True)) - 1


def clamp_stat_seasons(seasons) -> list[int]:
    """Restrict requested seasons to those nflverse actually has completed data for."""
    ceiling = max_stat_season()
    out = sorted({int(s) for s in seasons if PARTICIPATION_MIN_SEASON <= int(s) <= ceiling})
    return out


def _pd(df) -> pd.DataFrame:
    """Convert a polars frame to pandas, tolerating anything already pandas."""
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pd.DataFrame):
        return df
    return df.to_pandas()


# --------------------------------------------------------------------------------------
# Player stats / play-by-play
# --------------------------------------------------------------------------------------


def weekly_stats(seasons, log: SourceLog | None = None, regular_season_only: bool = True) -> pd.DataFrame:
    """Weekly player stats: the source of fantasy_points_ppr, target_share, air_yards_share.

    Feeds STEP 1 (Weighted PPG), part of STEP 2's opportunity component, and the parallel
    volatility branch's weekly logs.
    """
    seasons = clamp_stat_seasons(seasons)
    if not seasons:
        return pd.DataFrame()
    df = safe_load(
        "nflverse weekly player stats",
        f"load_player_stats(seasons={seasons})",
        lambda: nfl.load_player_stats(seasons=seasons),
        log=log,
    )
    df = _pd(df)
    if df.empty:
        return df
    if regular_season_only and "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()
    return df


def team_stats(seasons, log: SourceLog | None = None) -> pd.DataFrame:
    """Weekly team stats, including team defensive counting stats for the DST tab."""
    seasons = clamp_stat_seasons(seasons)
    if not seasons:
        return pd.DataFrame()
    df = safe_load(
        "nflverse team stats",
        f"load_team_stats(seasons={seasons})",
        lambda: nfl.load_team_stats(seasons=seasons),
        log=log,
    )
    df = _pd(df)
    if not df.empty and "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()
    return df


def play_by_play(seasons, log: SourceLog | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Play-by-play, used for red-zone touches, dropbacks, designed rushes and routes.

    Only the needed columns are kept: the full frame is ~370 columns per season and three
    seasons of it is the single biggest memory cost in the pipeline.
    """
    seasons = clamp_stat_seasons(seasons)
    if not seasons:
        return pd.DataFrame()
    df = safe_load(
        "nflverse play-by-play",
        f"load_pbp(seasons={seasons})",
        lambda: nfl.load_pbp(seasons=seasons),
        log=log,
    )
    if df is None:
        return pd.DataFrame()
    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        df = df.select(keep)
    return _pd(df)


def participation(seasons, log: SourceLog | None = None) -> pd.DataFrame:
    """Per-play personnel, used to count real routes run for YPRR.

    offense_players is a semicolon-delimited list of gsis ids for all 11 offensive players
    on the play, aligned element-wise with offense_positions. Verified populated on 100% of
    rows for 2023-2025, and every regular-season dropback in pbp has a matching row.

    Seasons are clamped, so an unreleased season degrades to an empty frame and the caller
    falls back to the pass-snap proxy rather than crashing.
    """
    seasons = clamp_stat_seasons(seasons)
    if not seasons:
        return pd.DataFrame()
    df = safe_load(
        "nflverse participation (routes)",
        f"load_participation(seasons={seasons})",
        lambda: nfl.load_participation(seasons=seasons),
        log=log,
    )
    return _pd(df)


def snap_counts(seasons, log: SourceLog | None = None) -> pd.DataFrame:
    """Snap counts / offensive snap share. Used for STEP 3a's position pool selection."""
    seasons = clamp_stat_seasons(seasons)
    if not seasons:
        return pd.DataFrame()
    df = safe_load(
        "nflverse snap counts",
        f"load_snap_counts(seasons={seasons})",
        lambda: nfl.load_snap_counts(seasons=seasons),
        log=log,
    )
    df = _pd(df)
    if not df.empty and "game_type" in df.columns:
        df = df[df["game_type"] == "REG"].copy()
    return df


def pfr_advstats(stat_type: str, seasons, log: SourceLog | None = None) -> pd.DataFrame:
    """Pro-Football-Reference advanced stats mirrored by nflverse.

    Supplies STEP 2 efficiency inputs that are not in the base stats: yards after contact
    (`yac`/`yac_att`) and broken tackles (`brk_tkl`) for rushing and receiving.
    Joins on pfr_id, not gsis_id.
    """
    seasons = clamp_stat_seasons(seasons)
    if not seasons:
        return pd.DataFrame()
    df = safe_load(
        f"nflverse PFR advanced stats ({stat_type})",
        f"load_pfr_advstats(stat_type={stat_type!r}, seasons={seasons})",
        lambda: nfl.load_pfr_advstats(seasons=seasons, stat_type=stat_type, summary_level="season"),
        log=log,
    )
    return _pd(df)


def ff_opportunity(seasons, log: SourceLog | None = None) -> pd.DataFrame:
    """Expected fantasy points / expected TDs (ffverse ffopportunity model).

    Supplies the "actual TD rate minus expected TD rate" half of the STEP 2 efficiency
    component, so we are not hand-rolling an expected-points model.
    """
    seasons = clamp_stat_seasons(seasons)
    if not seasons:
        return pd.DataFrame()
    df = safe_load(
        "ffverse ff_opportunity (expected points)",
        f"load_ff_opportunity(seasons={seasons})",
        lambda: nfl.load_ff_opportunity(seasons=seasons, stat_type="weekly", model_version="latest"),
        log=log,
    )
    return _pd(df)


# --------------------------------------------------------------------------------------
# Rosters / metadata / schedule
# --------------------------------------------------------------------------------------


def rosters(season: int, log: SourceLog | None = None) -> pd.DataFrame:
    """Current-season roster: team, position, birth_date, years_exp, pfr_id, draft_number.

    Deliberately NOT clamped -- the upcoming season's roster is published well before any
    games are played, and it is what defines the player universe we rank.
    """
    df = safe_load(
        "nflverse rosters",
        f"load_rosters(seasons=[{season}])",
        lambda: nfl.load_rosters(seasons=[season]),
        log=log,
    )
    return _pd(df)


def depth_charts(season: int, log: SourceLog | None = None) -> pd.DataFrame:
    """Depth charts for the upcoming season, reduced to the most recent snapshot.

    The raw release contains one row per player per scrape timestamp (hundreds of
    thousands of rows), so we keep only the latest `dt` per team. Feeds STEP 2's
    depth-chart-competition situational input.
    """
    df = safe_load(
        "nflverse depth charts",
        f"load_depth_charts(seasons=[{season}])",
        lambda: nfl.load_depth_charts(seasons=[season]),
        log=log,
    )
    df = _pd(df)
    if df.empty or "dt" not in df.columns:
        return df
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce", utc=True)
    latest = df.groupby("team")["dt"].transform("max")
    return df[df["dt"] == latest].copy()


def depth_charts_opening(seasons, log: SourceLog | None = None) -> pd.DataFrame:
    """Week 1 depth charts for completed seasons, normalised to player_id/position/rank.

    Separate from depth_charts() because nflverse changed this dataset's schema: 2024 and earlier
    publish a weekly row per player with a `depth_team` rank, while 2025 onward publish
    timestamped snapshots with `pos_rank`. Only the older schema is used here -- deriving a
    structural prior for "what a depth-chart slot earns" does not need the newest season, and the
    older format gives a clean opening-week view.
    """
    seasons = clamp_stat_seasons(seasons)
    if not seasons:
        return pd.DataFrame(columns=["season", "player_id", "position", "depth_chart_rank"])

    frames = []
    for season in seasons:
        df = safe_load(
            "nflverse opening depth charts",
            f"load_depth_charts(seasons=[{season}])",
            lambda s=season: nfl.load_depth_charts(seasons=[s]),
            log=log,
        )
        df = _pd(df)
        if df.empty or "depth_team" not in df.columns:
            continue  # newer snapshot schema; skip for prior derivation
        sub = df.copy()
        sub["week"] = pd.to_numeric(sub.get("week"), errors="coerce")
        if "game_type" in sub.columns:
            sub = sub[sub["game_type"] == "REG"]
        sub = sub[sub["week"] == 1]
        sub["depth_chart_rank"] = pd.to_numeric(sub["depth_team"], errors="coerce")
        sub = sub[sub["position"].isin(["QB", "RB", "WR", "TE"])]
        sub = sub.dropna(subset=["gsis_id", "depth_chart_rank"])
        # A player can appear at more than one slot; keep his best (lowest) rank.
        sub = sub.sort_values("depth_chart_rank").drop_duplicates(subset=["gsis_id"])
        frames.append(
            sub[["season", "gsis_id", "position", "depth_chart_rank"]].rename(
                columns={"gsis_id": "player_id"}
            )
        )

    if not frames:
        return pd.DataFrame(columns=["season", "player_id", "position", "depth_chart_rank"])
    return pd.concat(frames, ignore_index=True)


def schedules(seasons, log: SourceLog | None = None) -> pd.DataFrame:
    """Game schedule including Vegas spread_line / total_line, roof and coaches.

    Not clamped: the upcoming season's schedule is published with betting lines attached,
    which is where implied team totals come from (no odds API or key needed).
    """
    seasons = sorted({int(s) for s in seasons})
    df = safe_load(
        "nflverse schedules (incl. Vegas lines)",
        f"load_schedules(seasons={seasons})",
        lambda: nfl.load_schedules(seasons=seasons),
        log=log,
    )
    return _pd(df)


def injuries(seasons, log: SourceLog | None = None) -> pd.DataFrame:
    """Weekly injury reports, used to classify injury TYPE for the Injury Risk Score.

    Games missed themselves come from absence in the weekly stats; this supplies whether
    the absence was soft-tissue (recurs more often) or one-off trauma.
    """
    seasons = clamp_stat_seasons(seasons)
    if not seasons:
        return pd.DataFrame()
    df = safe_load(
        "nflverse injury reports",
        f"load_injuries(seasons={seasons})",
        lambda: nfl.load_injuries(seasons=seasons),
        log=log,
    )
    return _pd(df)


def draft_picks(log: SourceLog | None = None) -> pd.DataFrame:
    """Historical draft picks, for rookie draft capital and the rookie hit-rate lookup."""
    df = safe_load(
        "nflverse draft picks",
        "load_draft_picks()",
        lambda: nfl.load_draft_picks(),
        log=log,
    )
    return _pd(df)


def contracts(log: SourceLog | None = None) -> pd.DataFrame:
    """OverTheCap contract data mirrored by nflverse.

    Used to identify contract-year players and to build the empirical contract-year lift.
    Because this is an nflverse mirror, no scraping of OverTheCap is needed.
    """
    df = safe_load(
        "nflverse contracts (OverTheCap mirror)",
        "load_contracts()",
        lambda: nfl.load_contracts(),
        log=log,
    )
    return _pd(df)


@functools.lru_cache(maxsize=1)
def _players_cached():
    return nfl.load_players()


def players(log: SourceLog | None = None) -> pd.DataFrame:
    """Player metadata (names, positions, birth dates) across all seasons."""
    df = safe_load("nflverse players", "load_players()", _players_cached, log=log)
    return _pd(df)
