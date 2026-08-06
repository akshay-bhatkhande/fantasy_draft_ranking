"""STEP 4 -- VORP: the step that makes cross-position ranking valid.

This is the direct answer to "how do you rank players at different positions?". The pipeline
does not compare positions at all until here, and it never compares them using Composite
Z-Score, Base Projected PPG, or Final Projected Season Points -- only via VORP.

4a  Starter Count at Position = (dedicated slots x teams) + (flex slots x teams x flex share)
    DERIVED from config, never hardcoded. With this league (10 teams; 1QB/2RB/2WR/1TE plus 2
    FLEX allocated 60/30/10) that computes to QB 10, RB 32, WR 26, TE 12, so the replacement
    players are QB11, RB33, WR27 and TE13. Change the roster in config/league.py and these
    recalculate automatically.

4b  Replacement Level = the Final Projected Season Points of the player ranked immediately
    below the starter count at that position. Surfaced in a labelled column so the VORP
    arithmetic is auditable in place.

4c  VORP = player's Final Projected Season Points - his position's Replacement Level.

Because VORP subtracts each position's own baseline, every player's value becomes the same
unit: points better than a freely available replacement at their position. This is the only
number in the pipeline that is valid to compare across positions, and Overall Rank, Tier and
the main draft board are all built from it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import weights as W
from config.league import LeagueConfig

POINTS_COL = "final_projected_season_points"


def starter_counts(league: LeagueConfig) -> dict[str, int]:
    """STEP 4a: how many players at each position are draftable as starters in this league.

    FLEX slots are shared across RB/WR/TE, so they are allocated using historical league-wide
    FLEX usage rates rather than being assigned to one position.
    """
    counts: dict[str, int] = {}
    for pos in league.scored_positions:
        dedicated = league.dedicated_starters.get(pos, 0) * league.num_teams
        flex_share = league.flex_allocation.get(pos, 0.0)
        flex = league.flex_slots * league.num_teams * flex_share
        counts[pos] = int(round(dedicated + flex))
    return counts


def replacement_levels(
    df: pd.DataFrame,
    league: LeagueConfig,
    points_col: str = POINTS_COL,
) -> dict[str, float]:
    """STEP 4b: the replacement player's projected season points, per position.

    The replacement player is the one ranked immediately BELOW the starter count -- with 32 RB
    starters, that is RB33. If a position has fewer players than the starter count (only
    possible with a very thin pool), the last available player is used and the shortfall is
    implicitly visible in the starter count column.
    """
    counts = starter_counts(league)
    levels: dict[str, float] = {}
    for pos, count in counts.items():
        pool = (
            df[(df["position"] == pos) & df[points_col].notna()]
            .sort_values(points_col, ascending=False)[points_col]
            .to_numpy()
        )
        if len(pool) == 0:
            levels[pos] = float("nan")
            continue
        idx = min(count, len(pool) - 1)  # count is 0-based index of the (count+1)-th player
        levels[pos] = float(pool[idx])
    return levels


def compute_vorp(
    df: pd.DataFrame,
    league: LeagueConfig,
    points_col: str = POINTS_COL,
    vorp_col: str = "vorp",
) -> pd.DataFrame:
    """STEP 4c: VORP, plus Overall Rank, Position Rank and the auditable replacement level."""
    out = df.copy()
    counts = starter_counts(league)
    levels = replacement_levels(out, league, points_col=points_col)

    out["starter_count_at_position"] = out["position"].map(counts)
    out["replacement_level_points"] = out["position"].map(levels)
    out["replacement_player_rank"] = out["position"].map(
        {pos: count + 1 for pos, count in counts.items()}
    )
    out[vorp_col] = pd.to_numeric(out[points_col], errors="coerce") - out["replacement_level_points"]

    # Overall Rank: sort ALL players by VORP descending. This is the only cross-position sort.
    out["overall_rank"] = out[vorp_col].rank(ascending=False, method="min").astype("Int64")

    # Position Rank: by Final Projected Season Points within the position. Note this gives an
    # order identical to sorting by VORP within a position, since replacement level is a
    # constant subtracted from everyone at that position -- so it needs no separate math.
    out["position_rank"] = (
        out.groupby("position")[points_col].rank(ascending=False, method="min").astype("Int64")
    )
    return out


# --------------------------------------------------------------------------------------
# Tiering
# --------------------------------------------------------------------------------------


def detect_tiers_largest_gap(
    values: pd.Series,
    sensitivity: float | None = None,
    local_window: int | None = None,
    min_size: int | None = None,
    max_tiers: int | None = None,
    max_width: float | None = None,
    max_size: int | None = None,
) -> pd.Series:
    """Tier breaks at natural gaps in the sorted VORP list.

    Two independent triggers, because neither alone is sufficient:

    1. A GAP break, where the gap to the next player is meaningfully larger than the LOCAL
       average gap. This is the adaptive rule that makes tier sizes follow the real shape of
       the distribution rather than fixed buckets like "every 12 players". It respects
       min_size, so ordinary noise cannot fragment a tier.

    2. A FORCED break, when the running tier has grown too wide in VORP or too long in
       players. The relative rule cannot distinguish "uniform but very wide" from "uniform and
       tight": near the top of the board every gap is large, so nothing clears the
       local-average test and players pile into one enormous tier. A tier spanning 56 VORP is
       not a tier, since its best member is worth 56 more points than its worst. Forced breaks
       deliberately ignore min_size.

    max_tiers is normally None (unlimited). A finite ceiling TRUNCATES: once reached, every
    remaining player is dumped into the final tier no matter what the values do.
    """
    sensitivity = W.TIER_GAP_SENSITIVITY if sensitivity is None else sensitivity
    local_window = W.TIER_LOCAL_WINDOW if local_window is None else local_window
    min_size = W.TIER_MIN_SIZE if min_size is None else min_size
    max_tiers = W.TIER_MAX_COUNT if max_tiers is None else max_tiers
    max_width = W.TIER_MAX_WIDTH_VORP if max_width is None else max_width
    max_size = W.TIER_MAX_SIZE if max_size is None else max_size

    vals = pd.to_numeric(values, errors="coerce")
    order = vals.sort_values(ascending=False)
    order = order.dropna()
    if order.empty:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")

    arr = order.to_numpy(dtype=float)
    gaps = -np.diff(arr)  # descending order, so this is non-negative
    tiers = np.ones(len(arr), dtype=int)
    current = 1
    since_break = 1
    tier_start = 0

    for i, gap in enumerate(gaps):
        lo = max(0, i - local_window)
        hi = min(len(gaps), i + local_window + 1)
        local = gaps[lo:hi]
        local_mean = float(np.mean(local)) if len(local) else 0.0

        room_for_another_tier = max_tiers is None or current < max_tiers

        gap_break = (
            local_mean > 0
            and gap > sensitivity * local_mean
            and since_break >= min_size
            and room_for_another_tier
        )
        # Width measured as though the next player were to join the current tier.
        prospective_width = arr[tier_start] - arr[i + 1]
        forced_break = room_for_another_tier and (
            (max_width is not None and prospective_width > max_width)
            or (max_size is not None and since_break >= max_size)
        )

        if gap_break or forced_break:
            current += 1
            since_break = 1
            tier_start = i + 1
        else:
            since_break += 1
        tiers[i + 1] = current

    assigned = pd.Series(tiers, index=order.index)

    # Players with identical VORP must land in the same tier -- equal value, equal tier. Deep in
    # the board hundreds of players share an exact projection, and a tie group can straddle a
    # break, so without this the tier number depends on the arbitrary order ties came out of the
    # sort. Collapsing each tie group to its best (lowest) tier makes the column deterministic.
    # A tie group larger than max_size will exceed it, which is correct: identical values cannot
    # meaningfully be split.
    if order.duplicated().any():
        assigned = assigned.groupby(order.values).transform("min")

    return assigned.reindex(values.index).astype("Int64")


def detect_tiers_kmeans(values: pd.Series, max_clusters: int | None = None) -> pd.Series:
    """1-D k-means tiering, selecting cluster count by silhouette score.

    Alternative to largest-gap detection, selectable via TIER_METHOD in config/weights.py.
    """
    max_clusters = W.KMEANS_MAX_CLUSTERS if max_clusters is None else max_clusters
    vals = pd.to_numeric(values, errors="coerce")
    valid = vals.dropna()
    if len(valid) < 4:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        return detect_tiers_largest_gap(values)

    X = valid.to_numpy(dtype=float).reshape(-1, 1)
    best_labels, best_score, best_k = None, -1.0, 2
    for k in range(2, min(max_clusters, len(valid) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        score = silhouette_score(X, km.labels_)
        if score > best_score:
            best_labels, best_score, best_k = km.labels_, score, k
    if best_labels is None:
        return detect_tiers_largest_gap(values)

    # Relabel clusters so tier 1 is the highest-value group.
    centers = pd.Series(X.ravel()).groupby(best_labels).mean().sort_values(ascending=False)
    remap = {old: new for new, old in enumerate(centers.index, start=1)}
    tiers = pd.Series([remap[l] for l in best_labels], index=valid.index)
    return tiers.reindex(values.index).astype("Int64")


def assign_tiers(df: pd.DataFrame, value_col: str = "vorp", method: str | None = None) -> pd.Series:
    """Global VORP-based tiers using the configured detection method."""
    method = method or W.TIER_METHOD
    if method == "kmeans":
        return detect_tiers_kmeans(df[value_col])
    return detect_tiers_largest_gap(df[value_col])


def positional_tiers(df: pd.DataFrame, value_col: str = "vorp") -> pd.Series:
    """Tier breaks computed within each position, for the Tiers reference tab."""
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for pos in df["position"].dropna().unique():
        mask = df["position"] == pos
        out.loc[mask] = assign_tiers(df[mask], value_col=value_col)
    return out


def describe_starter_counts(league: LeagueConfig) -> list[str]:
    """Human-readable STEP 4a arithmetic, for the Cover tab."""
    lines = []
    counts = starter_counts(league)
    for pos in league.scored_positions:
        dedicated = league.dedicated_starters.get(pos, 0)
        share = league.flex_allocation.get(pos, 0.0)
        flex = league.flex_slots * league.num_teams * share
        lines.append(
            f"{pos}: ({dedicated} x {league.num_teams}) + ({league.flex_slots} x {league.num_teams} x {share:.2f})"
            f" = {counts[pos]} starters -> replacement player is {pos}{counts[pos] + 1}"
        )
    return lines
