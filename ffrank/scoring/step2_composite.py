"""STEP 2 -- Composite Z-Score (a position-relative talent/opportunity ranking).

    Composite Z = 0.40*PPGz + 0.25*Opportunityz + 0.15*Efficiencyz
                + 0.10*Situationalz + 0.10*ADPz

Every component is z-scored WITHIN its own position group, because raw values are not on
the same scale across positions (a RB's carry share and a WR's target share are simply not
the same quantity). The result typically spans about -3 to +3 and is meaningful only inside
that player's position group.

IMPORTANT: this number is NOT cross-position comparable. Do not sort or rank across
positions with it. It only becomes cross-position-comparable after STEP 4's VORP.

The position pool used to define each z-score distribution is the same pool STEP 3a uses for
the mean/stdev that converts the z-score back into points, so the two steps cannot disagree
about what "the RB population" means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import weights as W

COMPONENT_SOURCE_COLUMNS = {
    "ppg": "weighted_ppg",
    "opportunity": "opportunity_value",
    "efficiency": "efficiency_value",
    "situational": "situational_value",
    "adp": "adp_signal",
}


def build_position_pool(
    df: pd.DataFrame,
    pool_sizes: dict[str, int] | None = None,
    min_snap_share: dict[str, float] | None = None,
) -> pd.Series:
    """Flag the "relevant players" pool per position (STEP 3a's phrasing).

    Selection: players at or above a minimum share of team offensive snaps, then the top N
    at the position by recency-weighted snap share. Recomputed every run, so the pool tracks
    the actual league rather than a stale hardcoded list.

    Returns a boolean Series aligned to df.index.
    """
    pool_sizes = pool_sizes or W.POSITION_POOL_SIZES
    min_snap_share = min_snap_share or W.POSITION_POOL_MIN_SNAP_SHARE

    in_pool = pd.Series(False, index=df.index)
    snap = pd.to_numeric(df.get("snap_share"), errors="coerce")
    ppg = pd.to_numeric(df.get("weighted_ppg"), errors="coerce")

    for pos, size in pool_sizes.items():
        mask = df["position"] == pos
        if not mask.any():
            continue
        sub = df[mask].copy()
        sub["_snap"] = snap[mask]
        sub["_ppg"] = ppg[mask]
        threshold = min_snap_share.get(pos, 0.0)

        eligible = sub[sub["_snap"].fillna(0.0) >= threshold]
        if eligible.empty:
            # No snap data at all (e.g. a pure-rookie position group): fall back to PPG so
            # the pool is never silently empty, which would make STEP 3a undefined.
            eligible = sub[sub["_ppg"].notna()]
        ranked = eligible.sort_values(["_snap", "_ppg"], ascending=False).head(size)
        in_pool.loc[ranked.index] = True

    return in_pool


def build_adp_signal(df: pd.DataFrame) -> pd.Series:
    """Inverse ADP: lower ADP is better, so invert before z-scoring.

    Players absent from the ADP sample are placed a configurable distance past the last
    drafted pick rather than being dropped or given a fabricated precise value.

    ADP *variance* is deliberately NOT part of this signal -- it is tracked separately as a
    market-disagreement flag, per the methodology.
    """
    adp = pd.to_numeric(df.get("adp_blended", df.get("adp")), errors="coerce")
    if adp.notna().any():
        undrafted_value = adp.max() + W.ADP_UNDRAFTED_PADDING_PICKS
    else:
        undrafted_value = np.nan
    filled = adp.fillna(undrafted_value)
    return -filled


def zscore_within_position(
    values: pd.Series,
    positions: pd.Series,
    pool_mask: pd.Series,
) -> pd.DataFrame:
    """Z-score a column inside each position group, using pool statistics.

    The mean/stdev come from the position POOL, but the z-score is assigned to every player
    at that position (including non-pool depth players) so nobody is dropped from the board.
    Returns columns z, pos_mean, pos_std.
    """
    vals = pd.to_numeric(values, errors="coerce")
    z = pd.Series(np.nan, index=vals.index, dtype=float)
    mean_out = pd.Series(np.nan, index=vals.index, dtype=float)
    std_out = pd.Series(np.nan, index=vals.index, dtype=float)

    for pos in positions.dropna().unique():
        pos_mask = positions == pos
        pool_vals = vals[pos_mask & pool_mask].dropna()
        if len(pool_vals) < 2:
            pool_vals = vals[pos_mask].dropna()
        if len(pool_vals) < 2:
            continue
        mu = pool_vals.mean()
        sd = pool_vals.std(ddof=0)
        mean_out.loc[pos_mask] = mu
        std_out.loc[pos_mask] = sd
        if sd and not np.isnan(sd):
            z.loc[pos_mask] = (vals[pos_mask] - mu) / sd
        else:
            z.loc[pos_mask] = 0.0

    return pd.DataFrame({"z": z, "pos_mean": mean_out, "pos_std": std_out})


def compute_composite_z(
    df: pd.DataFrame,
    component_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """STEP 2: build every component z-score and the weighted Composite Z-Score.

    Expects df to contain player_id, position, weighted_ppg, opportunity_value,
    efficiency_value, situational_value, an ADP column, and snap_share.

    Adds z_ppg / z_opportunity / z_efficiency / z_situational / z_adp, composite_z,
    in_position_pool, and the STEP 3a inputs pos_mean_weighted_ppg / pos_std_weighted_ppg.

    A component that is missing for a player has its weight redistributed across that
    player's remaining components, so a missing input never silently reads as "average".
    """
    component_weights = component_weights or W.COMPOSITE_WEIGHTS
    out = df.copy()

    out["in_position_pool"] = build_position_pool(out)
    out["adp_signal"] = build_adp_signal(out)

    weight_series = {}
    for component, weight in component_weights.items():
        source = COMPONENT_SOURCE_COLUMNS[component]
        if source not in out.columns:
            out[f"z_{component}"] = np.nan
            weight_series[component] = weight
            continue
        stats = zscore_within_position(out[source], out["position"], out["in_position_pool"])
        out[f"z_{component}"] = stats["z"]
        weight_series[component] = weight
        if component == "ppg":
            # STEP 3a needs exactly these two numbers to convert the composite back to points.
            out["pos_mean_weighted_ppg"] = stats["pos_mean"]
            out["pos_std_weighted_ppg"] = stats["pos_std"]

    z_cols = [f"z_{c}" for c in component_weights]
    z_block = out[z_cols].apply(pd.to_numeric, errors="coerce")
    w_vec = pd.Series({f"z_{c}": w for c, w in weight_series.items()}, dtype=float)

    available = z_block.notna()
    applied_weight = available.mul(w_vec, axis=1)
    weight_total = applied_weight.sum(axis=1)
    out["composite_z"] = (
        z_block.fillna(0.0).mul(w_vec, axis=1).sum(axis=1) / weight_total.replace(0, np.nan)
    )
    out["composite_weight_coverage"] = weight_total

    # Rescale to unit variance inside each position pool. STEP 3a multiplies this by the
    # position's PPG standard deviation, which only recovers the real spread if the Composite Z
    # is itself a standard normal -- and a weighted average of five correlated z-scores is not.
    # Left unscaled it compresses every position by a different factor (0.58x to 0.83x here),
    # which silently tilts the board between positions. See COMPOSITE_Z_NORMALIZE.
    out["composite_z_raw"] = out["composite_z"]
    out["composite_z_scale"] = 1.0
    if W.COMPOSITE_Z_NORMALIZE:
        for pos in out["position"].dropna().unique():
            pos_mask = out["position"] == pos
            pool_vals = out.loc[pos_mask & out["in_position_pool"], "composite_z"].dropna()
            if len(pool_vals) < 3:
                continue
            sd = pool_vals.std(ddof=0)
            if not sd or np.isnan(sd):
                continue
            out.loc[pos_mask, "composite_z"] = out.loc[pos_mask, "composite_z"] / sd
            out.loc[pos_mask, "composite_z_scale"] = sd
        out["composite_z"] = out["composite_z"] * W.COMPOSITE_Z_SHRINKAGE

    def _missing_note(row) -> str:
        missing = [c.replace("z_", "") for c in z_cols if pd.isna(row[c])]
        if not missing:
            return ""
        return "insufficient data for " + ", ".join(missing) + " (weights redistributed)"

    out["composite_note"] = out.apply(_missing_note, axis=1)
    return out


COMPOSITE_Z_HEADER = "Composite Z-Score (position-relative only, not cross-position comparable)"
