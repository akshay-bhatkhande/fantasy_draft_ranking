"""Empirically fitted adjustment curves used by STEP 3b.

Three separate things live here, all *derived from data each run* rather than assumed:

1. Age curve. Fits age-versus-PPG per position over the last N seasons to find where decline
   actually begins, instead of inheriting stale folklore like "RB cliff at 27". The fitted
   peak age and decline slope are returned so they can be printed on the sheet -- this is the
   mechanism that lets backs who outperform the old aging curve be valued correctly.
   Rookies and second-year players are excluded entirely and use the rookie adjustment.

2. Contract-year lift. Compares contract-year seasons to each player's OWN surrounding-season
   baseline (average of the prior and following year), which controls for player quality far
   better than comparing contract-year players to the league. Published research tends to land
   near 1.03-1.07, but the number used is whatever this computes.

3. Rookie adjustment. A draft-capital-tiered baseline from the last N draft classes, expressed
   as a share of the position's mean PPG, since rookies have no prior NFL statistical record.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import weights as W


# ======================================================================================
# 1. Age curve
# ======================================================================================


@dataclass
class AgeCurve:
    """A fitted age curve for one position."""

    position: str
    peak_age: float
    decline_per_year: float
    sample_size: int
    fitted: bool
    r_squared: float = float("nan")
    fallback_reason: str = ""

    def multiplier(self, age: float | None) -> float:
        """Multiplier relative to the recalibrated positional peak.

        Ages at or below the peak get 1.00 (no bonus for being young -- youth is already
        reflected in the player's own production and opportunity). Past the peak, a gentle
        per-year discount applies, bounded by AGE_CURVE_BOUNDS.
        """
        if age is None or pd.isna(age):
            return 1.0
        if age <= self.peak_age:
            return 1.0
        years_past = age - self.peak_age
        mult = 1.0 - self.decline_per_year * years_past
        low, high = W.AGE_CURVE_BOUNDS
        return float(min(max(mult, low), high))

    def describe(self) -> str:
        if not self.fitted:
            reason = self.fallback_reason or f"insufficient sample ({self.sample_size})"
            return (
                f"{self.position}: fallback curve (peak {self.peak_age:.1f}, "
                f"{self.decline_per_year * 100:.1f}%/yr) - {reason}"
            )
        return (
            f"{self.position}: fitted peak age {self.peak_age:.1f}, decline "
            f"{self.decline_per_year * 100:.1f}%/yr past peak (n={self.sample_size}, R2={self.r_squared:.2f})"
        )


def fit_age_curves(
    season_ppg: pd.DataFrame,
    ages: pd.DataFrame,
    positions: pd.Series,
    lookback_seasons: int | None = None,
) -> dict[str, AgeCurve]:
    """Fit a quadratic age-versus-PPG curve per position and read off peak and slope.

    Fitting raw player-seasons directly does not work: the scatter is dominated by role
    heterogeneity (a 24-year-old backup and a 24-year-old every-week starter are both in
    there), which gives an R-squared near zero and an unstable vertex. So the fit is done on
    AGE-LEVEL means -- for each age, the games-weighted mean PPG across qualifying seasons --
    which is the standard way aging curves are built and yields a stable peak.

    Role is controlled for by requiring a genuine workload (games_played >= 8) before a
    season enters the curve, so the curve describes starters rather than the whole roster.

    Args:
        season_ppg: player_id, season, season_ppg, games_played.
        ages: player_id, season, age.
        positions: player_id -> position.
    """
    lookback_seasons = lookback_seasons or W.AGE_CURVE_LOOKBACK_SEASONS
    curves: dict[str, AgeCurve] = {}

    def _fallback(pos: str, n: int, reason: str) -> AgeCurve:
        fb = W.AGE_CURVE_FALLBACK.get(pos, {"peak_age": 27.0, "decline_per_year": 0.02})
        curve = AgeCurve(pos, fb["peak_age"], fb["decline_per_year"], n, False)
        curve.fallback_reason = reason
        return curve

    if season_ppg.empty or ages.empty:
        return {pos: _fallback(pos, 0, "no data") for pos in ("QB", "RB", "WR", "TE")}

    df = season_ppg.merge(ages, on=["player_id", "season"], how="inner")
    df["position"] = df["player_id"].map(positions)
    df = df.dropna(subset=["age", "season_ppg", "position"])
    max_season = df["season"].max()
    df = df[df["season"] > max_season - lookback_seasons]
    df = df[df["games_played"] >= 8]
    df["age"] = df["age"].round().astype(float)

    for pos in ("QB", "RB", "WR", "TE"):
        sub = df[(df["position"] == pos) & df["age"].between(21, 36)]
        n = len(sub)
        if n < W.AGE_CURVE_MIN_SAMPLE:
            curves[pos] = _fallback(pos, n, f"insufficient sample ({n})")
            continue

        # Games-weighted mean PPG per age, keeping only ages with enough observations to be
        # anything other than noise.
        grouped = sub.groupby("age").apply(
            lambda g: pd.Series(
                {
                    "mean_ppg": np.average(g["season_ppg"], weights=g["games_played"]),
                    "n": len(g),
                }
            ),
            include_groups=False,
        )
        grouped = grouped[grouped["n"] >= 3]
        if len(grouped) < 5:
            curves[pos] = _fallback(pos, n, f"only {len(grouped)} usable age buckets")
            continue

        x = grouped.index.to_numpy(dtype=float)
        y = grouped["mean_ppg"].to_numpy(dtype=float)
        wts = grouped["n"].to_numpy(dtype=float)
        try:
            coeffs = np.polyfit(x, y, 2, w=np.sqrt(wts))
        except (np.linalg.LinAlgError, ValueError):
            curves[pos] = _fallback(pos, n, "regression failed")
            continue

        a, b, c = coeffs
        if a >= 0:
            # Upward-opening parabola: no interior peak, so the fit cannot be read as an
            # aging curve. Fall back rather than invent a peak at the edge of the data.
            curves[pos] = _fallback(pos, n, "fit had no interior peak")
            continue

        peak_age = float(-b / (2 * a))
        if not (21.0 <= peak_age <= 34.0):
            curves[pos] = _fallback(pos, n, f"implausible fitted peak ({peak_age:.1f})")
            continue

        peak_value = float(np.polyval(coeffs, peak_age))
        if peak_value <= 0:
            curves[pos] = _fallback(pos, n, "non-positive fitted peak value")
            continue

        # Convert curvature into a fractional decline per year: evaluate three years past the
        # peak and annualise the drop relative to peak production.
        probe_value = float(np.polyval(coeffs, peak_age + 3.0))
        decline_per_year = float(max((peak_value - probe_value) / peak_value / 3.0, 0.0))

        pred = np.polyval(coeffs, x)
        ss_res = float(np.sum(wts * (y - pred) ** 2))
        ss_tot = float(np.sum(wts * (y - np.average(y, weights=wts)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

        curves[pos] = AgeCurve(pos, peak_age, decline_per_year, n, True, r2)

    return curves


def age_multiplier(
    age: float | None,
    position: str,
    years_exp: float | None,
    curves: dict[str, AgeCurve],
) -> tuple[float, str]:
    """Age-curve multiplier for one player, with the assumption used spelled out.

    Rookies and second-year players are excluded entirely (multiplier 1.00): there is not
    enough of their own career data to regress on, so the rookie adjustment handles them.
    """
    if years_exp is not None and not pd.isna(years_exp) and years_exp < W.AGE_CURVE_MIN_EXPERIENCE:
        return 1.00, "Age curve not applied (rookie/2nd-year; rookie adjustment used instead)"
    curve = curves.get(position)
    if curve is None:
        return 1.00, "Age curve unavailable for position"
    mult = curve.multiplier(age)
    if age is None or pd.isna(age):
        return 1.00, f"Age unknown; no age adjustment ({curve.describe()})"
    return mult, curve.describe()


# ======================================================================================
# 2. Contract-year lift
# ======================================================================================


@dataclass
class ContractYearLift:
    position: str
    multiplier: float
    sample_size: int
    derived: bool
    raw_estimate: float = float("nan")

    def describe(self) -> str:
        if not self.derived:
            return f"{self.position}: contract-year lift not derived (n={self.sample_size}), using {self.multiplier:.3f}"
        clipped = ""
        if not np.isnan(self.raw_estimate) and abs(self.raw_estimate - self.multiplier) > 1e-6:
            clipped = f" (raw {self.raw_estimate:.3f} clipped to configured bounds)"
        return (
            f"{self.position}: contract-year lift {self.multiplier:.3f} derived from "
            f"{self.sample_size} contract-year seasons vs own surrounding-season baseline{clipped}"
        )


def identify_contract_years(contracts: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """Player-seasons that were genuinely the final year of the then-active contract.

    A naive reading of the OverTheCap mirror ("every contract's last covered season is a
    contract year") massively over-flags, because players sign extensions before the old deal
    runs out. That produced 28k flagged seasons on this dataset, which then dragged the
    empirical lift to the clip floor.

    So each contract's effective end is truncated at the season before the player's NEXT
    contract was signed. A deal that was superseded early never had a contract year at all,
    and is excluded.

    Args:
        season: if given, only return rows for that season (used to flag the draft class year).
    """
    cols = ["gsis_id", "season", "is_contract_year"]
    if contracts.empty:
        return pd.DataFrame(columns=cols)

    df = contracts.copy()
    if not {"gsis_id", "year_signed", "years"}.issubset(df.columns):
        return pd.DataFrame(columns=cols)

    df["year_signed"] = pd.to_numeric(df["year_signed"], errors="coerce")
    df["years"] = pd.to_numeric(df["years"], errors="coerce")
    df = df.dropna(subset=["gsis_id", "year_signed", "years"])
    df = df[df["years"] > 0]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["own_final_season"] = df["year_signed"] + df["years"] - 1
    df = df.sort_values(["gsis_id", "year_signed"])
    # The next contract this player signed; the current deal cannot outlive it.
    df["next_signed"] = df.groupby("gsis_id")["year_signed"].shift(-1)
    df["effective_end"] = df["own_final_season"]
    superseded = df["next_signed"].notna() & (df["next_signed"] <= df["own_final_season"])
    df.loc[superseded, "effective_end"] = df.loc[superseded, "next_signed"] - 1

    # Only deals that actually ran to their own final season contain a real contract year.
    real = df[df["effective_end"] >= df["own_final_season"]].copy()
    out = real[["gsis_id", "own_final_season"]].rename(columns={"own_final_season": "season"})
    out["season"] = out["season"].astype(int)
    out["is_contract_year"] = True
    out = out.drop_duplicates()
    if season is not None:
        out = out[out["season"] == int(season)]
    return out


def derive_contract_year_lift(
    season_ppg: pd.DataFrame,
    contract_years: pd.DataFrame,
    positions: pd.Series,
    lookback_seasons: int | None = None,
) -> dict[str, ContractYearLift]:
    """Empirical contract-year lift per position.

    For each contract-year season, compare that season's PPG to the average of the player's
    own prior and following seasons. Using the player as his own control is what makes this
    meaningful; comparing contract-year players to the league at large would mostly measure
    which players earn second contracts.
    """
    lookback_seasons = lookback_seasons or W.CONTRACT_YEAR_LOOKBACK_SEASONS
    results: dict[str, ContractYearLift] = {}
    low, high = W.CONTRACT_YEAR_BOUNDS

    if season_ppg.empty or contract_years.empty:
        for pos, fb in W.CONTRACT_YEAR_FALLBACK.items():
            results[pos] = ContractYearLift(pos, fb, 0, False)
        return results

    ppg = season_ppg[["player_id", "season", "season_ppg", "games_played"]].copy()
    ppg = ppg[ppg["games_played"] >= 4]
    lookup = {(r.player_id, int(r.season)): r.season_ppg for r in ppg.itertuples()}

    cy = contract_years.rename(columns={"gsis_id": "player_id"})
    max_season = int(ppg["season"].max())
    cy = cy[(cy["season"] <= max_season) & (cy["season"] > max_season - lookback_seasons)]
    cy["position"] = cy["player_id"].map(positions)

    ratios: dict[str, list[float]] = {}
    for row in cy.itertuples():
        pos = row.position
        if pos not in ("QB", "RB", "WR", "TE"):
            continue
        this = lookup.get((row.player_id, int(row.season)))
        prior = lookup.get((row.player_id, int(row.season) - 1))
        following = lookup.get((row.player_id, int(row.season) + 1))
        # BOTH surrounding seasons are required, per the methodology's "average of the prior
        # year and the following year". Accepting a one-sided baseline silently enriches the
        # sample with players whose careers ended in their contract year -- they have no
        # following season -- which biased an earlier run's estimate down to the clip floor
        # for every position.
        if this is None or prior is None or following is None:
            continue
        if this <= 0 or prior <= 0 or following <= 0:
            continue
        baseline = float(np.mean([prior, following]))
        if baseline <= 0:
            continue
        ratios.setdefault(pos, []).append(this / baseline)

    for pos in ("QB", "RB", "WR", "TE"):
        vals = ratios.get(pos, [])
        fb = W.CONTRACT_YEAR_FALLBACK.get(pos, 1.00)
        if len(vals) < W.CONTRACT_YEAR_MIN_SAMPLE:
            results[pos] = ContractYearLift(pos, fb, len(vals), False)
            continue
        # Median resists the handful of extreme ratios produced by a player returning from a
        # near-lost season, which would otherwise inflate the estimate.
        raw = float(np.median(vals))
        est = float(min(max(raw, low), high))
        results[pos] = ContractYearLift(pos, est, len(vals), True, raw_estimate=raw)

    return results


# ======================================================================================
# 3. Rookie adjustment
# ======================================================================================


@dataclass
class RookieBaseline:
    """Rookie production baselines by draft-capital tier, as a share of position mean PPG."""

    position: str
    shares: dict[str, float] = field(default_factory=dict)
    samples: dict[str, int] = field(default_factory=dict)
    derived_tiers: set[str] = field(default_factory=set)

    def share_for(self, capital_tier: str) -> tuple[float, bool]:
        if capital_tier in self.shares:
            return self.shares[capital_tier], capital_tier in self.derived_tiers
        fallback = W.ROOKIE_FALLBACK_SHARE_OF_MEAN.get(self.position, {})
        return fallback.get(capital_tier, 0.5), False


def capital_tier(overall_pick: float | None) -> str:
    """Map an overall draft pick to a draft-capital tier label (UDFA if undrafted)."""
    if overall_pick is None or pd.isna(overall_pick) or overall_pick <= 0:
        return "udfa"
    pick = int(overall_pick)
    for lo, hi, label in W.ROOKIE_CAPITAL_TIERS:
        if lo <= pick <= hi:
            return label
    return "udfa"


def derive_rookie_baselines(
    season_ppg: pd.DataFrame,
    draft_picks: pd.DataFrame,
    positions: pd.Series,
    lookback_classes: int | None = None,
) -> dict[str, RookieBaseline]:
    """Historical rookie-season PPG by draft-capital tier, as a share of the position mean.

    Expressed as a share rather than an absolute PPG so it transfers cleanly into STEP 3a,
    where it is multiplied by the current season's position mean.
    """
    lookback_classes = lookback_classes or W.ROOKIE_DRAFT_CLASS_LOOKBACK
    out: dict[str, RookieBaseline] = {pos: RookieBaseline(pos) for pos in ("QB", "RB", "WR", "TE")}

    if season_ppg.empty or draft_picks.empty:
        return out

    dp = draft_picks.copy()
    id_col = "gsis_id" if "gsis_id" in dp.columns else None
    pick_col = next((c for c in ("pick", "overall") if c in dp.columns), None)
    if id_col is None or pick_col is None or "season" not in dp.columns:
        return out

    dp = dp[[id_col, "season", pick_col, "position"]].rename(
        columns={id_col: "player_id", "season": "draft_year", pick_col: "overall_pick"}
    )
    dp = dp.dropna(subset=["player_id"])

    ppg = season_ppg[["player_id", "season", "season_ppg", "games_played"]].copy()
    merged = ppg.merge(dp, on="player_id", how="inner")
    rookies = merged[merged["season"] == merged["draft_year"]].copy()
    if rookies.empty:
        return out

    max_class = int(rookies["draft_year"].max())
    rookies = rookies[rookies["draft_year"] > max_class - lookback_classes]
    rookies["pos"] = rookies["player_id"].map(positions).fillna(rookies["position"])
    rookies["tier"] = rookies["overall_pick"].map(capital_tier)

    # Position mean PPG per season is the denominator that turns rookie PPG into a
    # transferable share. It MUST be built the same way STEP 3a builds its position mean --
    # the top N at the position -- because the share is later multiplied by exactly that
    # number. An earlier version averaged every player with 8+ games, a much lower baseline,
    # which inflated every rookie share and pushed rookies into the top 20 overall.
    pos_means = {}
    ppg["pos"] = ppg["player_id"].map(positions)
    qualified = ppg[ppg["games_played"] >= 4]
    for (season, pos), grp in qualified.groupby(["season", "pos"]):
        pool_size = W.POSITION_POOL_SIZES.get(pos)
        if not pool_size:
            continue
        pool = grp.nlargest(pool_size, "season_ppg")
        pos_means[(int(season), pos)] = pool["season_ppg"].mean()

    for pos in ("QB", "RB", "WR", "TE"):
        sub = rookies[rookies["pos"] == pos]
        baseline = out[pos]
        fallback = W.ROOKIE_FALLBACK_SHARE_OF_MEAN.get(pos, {})
        for tier, grp in sub.groupby("tier"):
            shares = []
            for r in grp.itertuples():
                mean = pos_means.get((int(r.season), pos))
                if mean and mean > 0 and r.season_ppg is not None:
                    shares.append(r.season_ppg / mean)
            baseline.samples[tier] = len(shares)
            if len(shares) < W.ROOKIE_MIN_SAMPLE:
                continue
            observed = float(np.median(shares))
            # Shrink toward the prior in proportion to sample size. Draft-capital tiers have
            # only a handful of rookies per position per class, so an unshrunk median swings
            # wildly (an early run produced a 1.61x share off n=9). Shrinkage keeps a thin
            # tier from asserting more than its sample supports.
            prior = fallback.get(tier, 0.5)
            n = len(shares)
            k = W.ROOKIE_SHRINKAGE_STRENGTH
            weight = n / (n + k)
            baseline.shares[tier] = float(weight * observed + (1 - weight) * prior)
            baseline.samples[tier] = n
            if weight >= 0.5:
                baseline.derived_tiers.add(tier)
    return out


def rookie_note(position: str, tier: str, share: float, derived: bool) -> str:
    src = "derived from last 5 draft classes" if derived else "fallback prior (insufficient sample)"
    return (
        f"Rookie adjustment: {tier} draft capital at {position} historically produces "
        f"{share:.2f}x the position mean PPG ({src})"
    )
