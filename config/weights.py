"""All model weights, multiplier magnitudes and tunable constants.

Separated from league.py so the *model* can be retuned without touching league rules.
Every constant is labelled with the methodology step that consumes it.
"""

from __future__ import annotations

# ======================================================================================
# STEP 1 -- Weighted PPG recency weights
# ======================================================================================
# Keyed by "seasons ago" (1 = most recently completed season). Applied to per-game
# averages over GAMES ACTUALLY PLAYED -- never season totals / 17, which would punish an
# injury-shortened season twice (it is already handled separately in STEP 3c).
# Players with fewer than 3 seasons get these weights redistributed proportionally across
# whatever seasons exist, and are flagged "limited sample".
RECENCY_WEIGHTS: dict[int, float] = {1: 0.55, 2: 0.30, 3: 0.15}

# A season needs at least this many games played to contribute to STEP 1.
MIN_GAMES_FOR_SEASON = 1


# ======================================================================================
# STEP 2 -- Composite Z-Score component weights (must sum to 1.00)
# ======================================================================================
# Composite Z = 0.40*PPGz + 0.25*Opportunityz + 0.15*Efficiencyz
#             + 0.10*Situationalz + 0.10*ADPz
# This number is POSITION-RELATIVE ONLY. It is never valid to compare across positions.
COMPOSITE_WEIGHTS: dict[str, float] = {
    "ppg": 0.40,
    "opportunity": 0.25,
    "efficiency": 0.15,
    "situational": 0.10,
    "adp": 0.10,
}

# --- Opportunity sub-weights: folded into ONE sub-value per player, then z-scored -----
OPPORTUNITY_SUB_WEIGHTS: dict[str, dict[str, float]] = {
    "RB": {"carry_share": 0.60, "rz_carry_share": 0.25, "target_share": 0.15},
    "WR": {"target_share": 0.50, "air_yards_share": 0.30, "rz_target_share": 0.20},
    "TE": {"target_share": 0.50, "air_yards_share": 0.30, "rz_target_share": 0.20},
    # QB components equal-weighted per spec.
    "QB": {"dropback_share": 1 / 3, "designed_rush_share": 1 / 3, "rz_pass_attempt_share": 1 / 3},
}

# --- Efficiency sub-weights -----------------------------------------------------------
EFFICIENCY_SUB_WEIGHTS: dict[str, dict[str, float]] = {
    "RB": {"yprr": 1 / 3, "yards_after_contact": 1 / 3, "broken_tackle_rate": 1 / 3},
    "WR": {"yprr": 0.50, "td_rate_vs_expected": 0.50},
    "TE": {"yprr": 0.50, "td_rate_vs_expected": 0.50},
    "QB": {"ypa": 1 / 3, "td_rate_vs_expected": 1 / 3, "sack_rate_avoided": 1 / 3},
}

# Minimum sample before an efficiency rate is trusted; below this it is "insufficient
# data" and the component is dropped from that player's equal-weighted average rather
# than being filled with a noisy small-sample number.
MIN_ROUTES_FOR_YPRR = 50
MIN_CARRIES_FOR_RUSH_EFF = 40
MIN_RECEPTIONS_FOR_REC_EFF = 20
MIN_DROPBACKS_FOR_QB_EFF = 100
MIN_TARGETS_FOR_TD_RATE = 25

# Hard clip on any single efficiency sub-component's z-score before it is averaged.
# Rate stats on tiny samples explode: Brittain Brown's 2 broken tackles on 5 carries is a 0.40
# broken-tackle rate against a league mean of 0.047, and combined with an equally inflated
# yards-after-contact figure it produced a +12 sigma efficiency score that cancelled out a
# -2.3 sigma PPG and lifted a fringe back to 56th overall. The minimum-sample gates above are
# the real fix; this clip is a backstop so no single component can ever dominate the composite.
EFFICIENCY_Z_CLIP = 3.0

# --- Situational Context Score: equal-weighted average of these z-scores -------------
SITUATIONAL_COMPONENTS: tuple[str, ...] = (
    "oline_quality",            # esp. relevant for RBs
    "scheme_pace",              # plays per game + pass rate over expected
    "qb_quality",               # matters for pass-catchers
    "coordinator_tendency",     # head-coach/OC change scored by historical positional usage
    "implied_team_total",       # Vegas proxy for offensive opportunity
    "strength_of_schedule",     # full season
    "depth_chart_competition",  # crowded competition for touches => lower z
)

# Not every component applies to every position. Components not applicable are SKIPPED
# when averaging (not zero-filled, which would drag a player toward the mean spuriously).
SITUATIONAL_APPLICABILITY: dict[str, tuple[str, ...]] = {
    "oline_quality": ("QB", "RB", "WR", "TE"),
    "scheme_pace": ("QB", "RB", "WR", "TE"),
    "qb_quality": ("RB", "WR", "TE"),  # a QB is not his own "QB quality" input
    "coordinator_tendency": ("QB", "RB", "WR", "TE"),
    "implied_team_total": ("QB", "RB", "WR", "TE"),
    "strength_of_schedule": ("QB", "RB", "WR", "TE"),
    "depth_chart_competition": ("QB", "RB", "WR", "TE"),
}

# Informational only -- NEVER folded into any score or multiplier.
INFORMATIONAL_ONLY: tuple[str, ...] = ("playoff_sos", "bye_week", "dome_or_outdoor")


# ======================================================================================
# STEP 3a -- Position pool for the Mean/StdDev that converts Z back into points
# ======================================================================================
# Recomputed every run from that season's relevant player pool (spec example: "top 40 RBs
# by snap share"), so the distribution tracks the actual league, never a stale constant.
POSITION_POOL_SIZES: dict[str, int] = {"QB": 32, "RB": 40, "WR": 60, "TE": 24}
# Players below this share of team offensive snaps are excluded from the pool entirely.
POSITION_POOL_MIN_SNAP_SHARE: dict[str, float] = {"QB": 0.25, "RB": 0.10, "WR": 0.10, "TE": 0.10}


# ======================================================================================
# STEP 3b -- Risk multipliers, applied in this order:
#            Contract-Year x Age-Curve x Camp-Buzz x Team-Penalty
#            (the team penalty is a personal preference and is currently disabled --
#             see BIAS_TEAM in config/league.py)
# ======================================================================================

# --- Camp-Buzz Multiplier: -2..+2 score -> multiplier, hard-capped at +/-8% ------------
# Camp buzz is the least statistically grounded input, so the cap keeps it to roughly one
# tier of movement: it can nudge a player, never leapfrog him several tiers on its own.
CAMP_BUZZ_MULTIPLIERS: dict[int, float] = {-2: 0.92, -1: 0.96, 0: 1.00, 1: 1.04, 2: 1.08}
CAMP_BUZZ_MAX_ABS_EFFECT = 0.08
# camp_buzz.json older than this many days is reported as stale in the run summary.
CAMP_BUZZ_STALE_AFTER_DAYS = 7

# --- Contract-Year Multiplier ---------------------------------------------------------
# NOT a flat assumed boost. Derived empirically each run by comparing contract-year
# seasons to each player's own surrounding-season baseline (prior + following year).
# Published research tends to land near 1.03-1.07, but the tool computes its own number;
# the values below are used ONLY when a position has too thin a sample to trust.
CONTRACT_YEAR_LOOKBACK_SEASONS = 5
CONTRACT_YEAR_MIN_SAMPLE = 8
CONTRACT_YEAR_BOUNDS = (0.97, 1.12)  # sanity rails on the empirical estimate
CONTRACT_YEAR_FALLBACK: dict[str, float] = {"QB": 1.00, "RB": 1.00, "WR": 1.00, "TE": 1.00}

# --- Age-Curve Multiplier -------------------------------------------------------------
# Peak age and decline slope are FIT from the last N seasons of age-vs-PPG data, so backs
# who beat the old "RB cliff at 27" rule are valued correctly. The fitted peak age and
# slope actually used are logged into the workbook for auditing.
AGE_CURVE_LOOKBACK_SEASONS = 5
AGE_CURVE_MIN_SAMPLE = 25
AGE_CURVE_BOUNDS = (0.80, 1.08)
# Rookies and 2nd-year players are excluded entirely (multiplier 1.00) -- too little of
# their own career data to regress on. They use the Rookie Adjustment instead.
AGE_CURVE_MIN_EXPERIENCE = 2
# Used only when the regression cannot run for a position.
AGE_CURVE_FALLBACK: dict[str, dict[str, float]] = {
    "QB": {"peak_age": 30.0, "decline_per_year": 0.015},
    "RB": {"peak_age": 26.0, "decline_per_year": 0.035},
    "WR": {"peak_age": 27.0, "decline_per_year": 0.020},
    "TE": {"peak_age": 28.0, "decline_per_year": 0.020},
}

# --- Rookie Adjustment (separate lookup, NOT part of the age regression) --------------
# Position-specific hit-rate baselines from the last N draft classes, keyed on draft
# capital. Applied at the Base Projected PPG stage in place of a Composite Z-Score
# history that does not exist yet for a rookie.
ROOKIE_DRAFT_CLASS_LOOKBACK = 5
ROOKIE_MIN_SAMPLE = 5
# Draft-capital tiers hold only a handful of rookies per position per class, so the observed
# median is shrunk toward the prior below with weight n/(n+k). Higher k = more conservative.
ROOKIE_SHRINKAGE_STRENGTH = 10
# Draft-capital tiers by overall pick: (min_pick, max_pick, label)
ROOKIE_CAPITAL_TIERS: tuple[tuple[int, int, str], ...] = (
    (1, 15, "top-15"),
    (16, 32, "rd1-late"),
    (33, 64, "rd2"),
    (65, 105, "rd3"),
    (106, 175, "rd4-5"),
    (176, 262, "rd6-7"),
    (263, 9999, "udfa"),
)
# Fallback share-of-positional-mean PPG by capital tier, used when a tier has too few
# historical rookies at a position to compute an empirical rate.
#
# These are calibrated against the SAME denominator STEP 3a uses (the mean of the top-N pool
# at the position), and are deliberately in line with the empirically derived values so a
# thin tier falling back does not jump a rookie above better-supported tiers. An earlier set
# was calibrated against a lower baseline and pushed rookies into the top 20 overall.
ROOKIE_FALLBACK_SHARE_OF_MEAN: dict[str, dict[str, float]] = {
    "QB": {"top-15": 0.85, "rd1-late": 0.70, "rd2": 0.60, "rd3": 0.52, "rd4-5": 0.44, "rd6-7": 0.41, "udfa": 0.30},
    "RB": {"top-15": 1.02, "rd1-late": 0.96, "rd2": 0.90, "rd3": 0.55, "rd4-5": 0.35, "rd6-7": 0.25, "udfa": 0.18},
    "WR": {"top-15": 0.97, "rd1-late": 0.80, "rd2": 0.63, "rd3": 0.42, "rd4-5": 0.28, "rd6-7": 0.21, "udfa": 0.15},
    "TE": {"top-15": 0.75, "rd1-late": 0.68, "rd2": 0.63, "rd3": 0.46, "rd4-5": 0.44, "rd6-7": 0.32, "udfa": 0.15},
}


# ======================================================================================
# STEP 3c -- Expected Games Played
# ======================================================================================
# Historical Injury Risk bucket -> expected games missed. A probabilistic reliability
# discount from history. Entirely separate from known current-season absences, which are
# specific announced facts counted directly.
INJURY_RISK_EXPECTED_GAMES_MISSED: dict[str, float] = {"Low": 0.2, "Med": 0.7, "High": 1.5}

# Bucket thresholds on recency-weighted share of games missed.
INJURY_RISK_LOW_MAX = 0.10   # < 10% => Low
INJURY_RISK_MED_MAX = 0.25   # 10-25% => Med; > 25% => High

# Soft-tissue injuries recur at a statistically higher rate than one-off trauma, so games
# missed to them count for more when building the Injury Risk Score.
SOFT_TISSUE_KEYWORDS: tuple[str, ...] = (
    "hamstring", "groin", "calf", "quad", "hip flexor", "adductor", "achilles", "soft tissue",
)
TRAUMA_KEYWORDS: tuple[str, ...] = (
    "fracture", "broken", "concussion", "laceration", "dislocat", "acl", "mcl", "surgery",
)
SOFT_TISSUE_WEIGHT = 1.5
TRAUMA_WEIGHT = 1.0
UNKNOWN_INJURY_WEIGHT = 1.15

# Floor so a heavily discounted player never reaches zero or negative games.
MIN_EXPECTED_GAMES = 1.0


# ======================================================================================
# STEP 4 -- Tier detection on the VORP-sorted list
# ======================================================================================
# Natural breakpoints only -- never fixed-size buckets like "every 12 players".
TIER_METHOD = "largest_gap"  # "largest_gap" | "kmeans"
# A gap starts a new tier when it exceeds this multiple of the local average gap.
TIER_GAP_SENSITIVITY = 1.8
TIER_LOCAL_WINDOW = 8
# Minimum players between two GAP-driven breaks, so ordinary noise cannot fragment a tier.
# The width/size limits below deliberately override this -- see below.
TIER_MIN_SIZE = 3

# Hard ceiling on the number of tiers. None = unlimited, which is the sensible default:
# a fixed ceiling silently dumps everyone past the last allowed break into one bucket. With a
# 914-player board there are ~146 real breakpoints, so a ceiling of 14 discarded 128 of them
# and produced a single 798-player "tier 14" spanning 212 VORP.
TIER_MAX_COUNT: int | None = None

# Absolute limits that force a break even when no gap is locally unusual.
#
# The relative gap rule alone cannot tell "uniform but very wide" from "uniform and tight".
# Near the top of the board every gap is large, so nothing clears the local-average test and
# 25 players were landing in one tier spanning 56 VORP -- which is not a tier in any useful
# sense, since its best player was worth 56 more points than its worst. These caps mean a tier
# is always a group you could genuinely treat as interchangeable.
#
# 8 VORP across a 17-game season is about half a point per game.
TIER_MAX_WIDTH_VORP: float | None = 8.0
TIER_MAX_SIZE: int | None = 10

# Only used when TIER_METHOD == "kmeans". k-means picks its own cluster count by silhouette
# score, so this is a genuine algorithmic bound rather than a truncation.
KMEANS_MAX_CLUSTERS = 12


# ======================================================================================
# Volatility / Consistency -- parallel branch, NEVER feeds Steps 2-4 or VORP
# ======================================================================================
VOLATILITY_LOOKBACK_SEASONS = 2
# Same recency-weighting philosophy as STEP 1, renormalised over the shorter window.
VOLATILITY_RECENCY_WEIGHTS: dict[int, float] = {1: 0.65, 2: 0.35}
FLOOR_PERCENTILE = 20
MEDIAN_PERCENTILE = 50
CEILING_PERCENTILE = 80
MIN_WEEKS_FOR_VOLATILITY = 6

# Same-PPG-different-volatility automated comparison pass.
SAME_PPG_TOLERANCE = 0.05     # players within 5% of each other in Weighted PPG
CONSISTENCY_GAP_FLAG = 15     # Consistency Scores differing by more than this get flagged


# ======================================================================================
# Market / ADP
# ======================================================================================
# ADP variance is surfaced as a "market disagreement" flag and deliberately does NOT get
# folded into the Composite Z-Score.
ADP_DISAGREEMENT_STDEV_THRESHOLD = 6.0  # in pick spots
# Players absent from the ADP sample are placed this far past the last drafted pick so the
# inverse-ADP z-score stays defined without inventing a precise number for them.
ADP_UNDRAFTED_PADDING_PICKS = 12


# ======================================================================================
# STEP 5 -- Expert consensus sanity check (never feeds the math)
# ======================================================================================
ECR_DELTA_FLAG_THRESHOLD = 15  # spots of disagreement that trigger an auto-generated reason


def validate() -> None:
    """Fail fast on weight tables that do not sum to 1.0."""
    if abs(sum(COMPOSITE_WEIGHTS.values()) - 1.0) > 1e-9:
        raise ValueError(f"COMPOSITE_WEIGHTS must sum to 1.0, got {sum(COMPOSITE_WEIGHTS.values())}")
    if abs(sum(RECENCY_WEIGHTS.values()) - 1.0) > 1e-9:
        raise ValueError("RECENCY_WEIGHTS must sum to 1.0")
    if abs(sum(VOLATILITY_RECENCY_WEIGHTS.values()) - 1.0) > 1e-9:
        raise ValueError("VOLATILITY_RECENCY_WEIGHTS must sum to 1.0")
    for name, table in (
        ("OPPORTUNITY_SUB_WEIGHTS", OPPORTUNITY_SUB_WEIGHTS),
        ("EFFICIENCY_SUB_WEIGHTS", EFFICIENCY_SUB_WEIGHTS),
    ):
        for pos, sub in table.items():
            if abs(sum(sub.values()) - 1.0) > 1e-9:
                raise ValueError(f"{name}[{pos}] must sum to 1.0, got {sum(sub.values())}")


validate()
