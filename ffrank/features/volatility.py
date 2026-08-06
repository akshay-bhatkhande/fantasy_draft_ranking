"""Volatility / consistency -- a SEPARATE, PARALLEL calculation.

This never feeds the Composite Z-Score, Final Projected Season Points, or VORP, and it never
changes a player's rank. It exists to catch the case where two players have similar Weighted
PPG but very different weekly reliability.

Why it is kept out of the ranking: a volatile boom/bust player is a bigger risk in a locked
starting slot (QB/RB1/RB2/WR1/WR2/TE) than as a FLEX or bench dart-throw. That judgement
belongs to the drafter per roster slot, not buried inside one number.

Produces:
  * coefficient of variation (weekly stdev / weekly mean), z-scored within position, since
    WRs are inherently more volatile than RBs as a position
  * Floor (20th percentile week), Median, Ceiling (80th percentile week)
  * Consistency Score on a 0-100 scale, higher = more consistent
  * an automated same-PPG-different-volatility comparison pass across each position's pool
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import weights as W

FANTASY_COL = "fantasy_points_ppr"


def compute_weekly_distribution(
    weekly: pd.DataFrame,
    target_season: int,
    lookback_seasons: int | None = None,
    recency_weights: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Per-player weekly distribution stats, recency-weighted across the last 1-2 seasons.

    Per-season stats are computed first and then recency-weighted, rather than pooling all
    weeks together, so that last season counts for more without distorting the percentiles.
    """
    lookback_seasons = lookback_seasons or W.VOLATILITY_LOOKBACK_SEASONS
    recency_weights = recency_weights or W.VOLATILITY_RECENCY_WEIGHTS
    cols = ["player_id", "cv", "floor", "median", "ceiling", "weeks_sampled", "volatility_note"]
    if weekly.empty:
        return pd.DataFrame(columns=cols)

    df = weekly.copy()
    df[FANTASY_COL] = pd.to_numeric(df[FANTASY_COL], errors="coerce")
    df = df.dropna(subset=[FANTASY_COL])
    df["seasons_ago"] = target_season - df["season"]
    df = df[df["seasons_ago"].between(1, lookback_seasons)]
    if df.empty:
        return pd.DataFrame(columns=cols)

    per_season = (
        df.groupby(["player_id", "season", "seasons_ago"])[FANTASY_COL]
        .agg(
            weeks="count",
            mean="mean",
            std=lambda s: s.std(ddof=1),
            floor=lambda s: np.percentile(s, W.FLOOR_PERCENTILE),
            median=lambda s: np.percentile(s, W.MEDIAN_PERCENTILE),
            ceiling=lambda s: np.percentile(s, W.CEILING_PERCENTILE),
        )
        .reset_index()
    )
    per_season["cv"] = per_season["std"] / per_season["mean"].replace(0, np.nan)
    per_season["w"] = per_season["seasons_ago"].map(recency_weights).astype(float)

    rows = []
    for player_id, grp in per_season.groupby("player_id"):
        total_weeks = int(grp["weeks"].sum())
        usable = grp[grp["weeks"] >= 2]
        if usable.empty or total_weeks < W.MIN_WEEKS_FOR_VOLATILITY:
            rows.append(
                {
                    "player_id": player_id,
                    "cv": np.nan,
                    "floor": np.nan,
                    "median": np.nan,
                    "ceiling": np.nan,
                    "weeks_sampled": total_weeks,
                    "volatility_note": f"insufficient data: only {total_weeks} weeks in the last {lookback_seasons} seasons",
                }
            )
            continue
        wts = usable["w"] / usable["w"].sum()
        rows.append(
            {
                "player_id": player_id,
                "cv": float((usable["cv"] * wts).sum(skipna=True)),
                "floor": float((usable["floor"] * wts).sum()),
                "median": float((usable["median"] * wts).sum()),
                "ceiling": float((usable["ceiling"] * wts).sum()),
                "weeks_sampled": total_weeks,
                "volatility_note": "",
            }
        )
    return pd.DataFrame(rows)


def consistency_scores(dist: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """Z-score CV within position, then map to a 0-100 Consistency Score.

    Compared like to like on purpose: WRs are inherently more volatile than RBs as a
    position, so a raw CV comparison across positions would just re-measure the position.
    Higher score = more consistent, so the CV z-score is negated.
    """
    out = dist.copy()
    if out.empty:
        out["cv_z"] = []
        out["consistency_score"] = []
        return out

    out["position"] = out["player_id"].map(positions)
    out["cv_z"] = np.nan
    for pos in out["position"].dropna().unique():
        mask = out["position"] == pos
        vals = pd.to_numeric(out.loc[mask, "cv"], errors="coerce")
        sd = vals.std(ddof=0)
        if sd and not np.isnan(sd):
            out.loc[mask, "cv_z"] = (vals - vals.mean()) / sd
        else:
            out.loc[mask, "cv_z"] = 0.0

    # 50 is positional average consistency; each standard deviation is worth 15 points, so the
    # usual range lands inside 0-100 without most players pinning at the ends.
    out["consistency_score"] = (50.0 - 15.0 * out["cv_z"]).clip(0, 100).round(1)
    return out


def same_ppg_volatility_flags(
    df: pd.DataFrame,
    ppg_col: str = "weighted_ppg",
    consistency_col: str = "consistency_score",
    tolerance: float | None = None,
    gap: float | None = None,
) -> pd.Series:
    """Automated same-PPG-different-volatility comparison pass.

    For any two players at the same position within 5% of each other in Weighted PPG, if their
    Consistency Scores differ by more than 15 points, both get a Notes flag naming the
    comparison. Run automatically across each position's pool so it never has to be spotted
    by hand.
    """
    tolerance = W.SAME_PPG_TOLERANCE if tolerance is None else tolerance
    gap = W.CONSISTENCY_GAP_FLAG if gap is None else gap

    notes = pd.Series("", index=df.index, dtype=object)
    name_col = "player_name" if "player_name" in df.columns else "player_id"

    for pos in df["position"].dropna().unique():
        sub = df[
            (df["position"] == pos)
            & pd.to_numeric(df[ppg_col], errors="coerce").notna()
            & pd.to_numeric(df[consistency_col], errors="coerce").notna()
        ]
        if len(sub) < 2:
            continue
        sub = sub.sort_values(ppg_col, ascending=False)
        ppg = pd.to_numeric(sub[ppg_col], errors="coerce").to_numpy()
        cons = pd.to_numeric(sub[consistency_col], errors="coerce").to_numpy()
        names = sub[name_col].to_numpy()
        idx = sub.index.to_numpy()

        best_partner: dict[int, tuple[float, str, float]] = {}
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                if ppg[i] <= 0:
                    continue
                # Sorted descending, so once the PPG difference exceeds the tolerance every
                # later player is further away and the inner loop can stop.
                rel = abs(ppg[i] - ppg[j]) / ppg[i]
                if rel > tolerance:
                    break
                diff = cons[i] - cons[j]
                if abs(diff) <= gap:
                    continue
                for a, b, d in ((i, j, diff), (j, i, -diff)):
                    prev = best_partner.get(a)
                    if prev is None or abs(d) > abs(prev[0]):
                        best_partner[a] = (d, names[b], ppg[b])

        for pos_i, (diff, other_name, other_ppg) in best_partner.items():
            if diff > 0:
                text = (
                    f"Similar PPG to {other_name} ({other_ppg:.1f}), but notably MORE consistent "
                    f"week-to-week (+{diff:.0f} consistency) - better floor play than ceiling play"
                )
            else:
                text = (
                    f"Similar PPG to {other_name} ({other_ppg:.1f}), but notably LESS consistent "
                    f"week-to-week ({diff:.0f} consistency) - better ceiling play than floor play"
                )
            notes.at[idx[pos_i]] = text

    return notes
