"""League settings.

Everything in this file describes *your league*. Change values here and the whole
pipeline recalculates -- notably STEP 4a's starter counts and replacement levels are
DERIVED from the roster settings below, never hardcoded.

Roster-size note: the source spec said "16 total roster spots" but the slot list it gave
sums to 17 (1 QB + 2 RB + 2 WR + 1 TE + 2 FLEX + 1 K + 1 DEF + 7 bench). Confirmed with
the league owner that 17 is correct and the "16" was a typo. Bench count never enters the
STEP 4a starter-count math either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Season being drafted for
# --------------------------------------------------------------------------------------

TARGET_SEASON = 2026

# STEP 1 lookback: last season / two seasons ago / three seasons ago.
LOOKBACK_SEASONS: list[int] = [TARGET_SEASON - 1, TARGET_SEASON - 2, TARGET_SEASON - 3]

GAMES_IN_SEASON = 17


# --------------------------------------------------------------------------------------
# Scoring: Standard PPR, no bonus scoring
# --------------------------------------------------------------------------------------
# Deliberately no long-TD or yardage-milestone bonuses. nflverse already publishes a
# full-PPR fantasy points column that matches this exactly, which is what STEP 1 consumes;
# this table is kept for the K/DST tab and for auditability of the format.

SCORING = {
    "reception": 1.0,
    "passing_yard": 0.04,  # 1 pt / 25 yards
    "passing_td": 4.0,
    "interception": -2.0,
    "rushing_yard": 0.10,  # 1 pt / 10 yards
    "rushing_td": 6.0,
    "receiving_yard": 0.10,
    "receiving_td": 6.0,
    "fumble_lost": -2.0,
    "two_point_conversion": 2.0,
}

# Kicker scoring (used only on the minimal Kicker/DEF tab).
K_SCORING = {
    "fg_made_0_39": 3.0,
    "fg_made_40_49": 4.0,
    "fg_made_50_plus": 5.0,
    "fg_missed": -1.0,
    "pat_made": 1.0,
    "pat_missed": -1.0,
}

# Team-defense scoring (used only on the minimal Kicker/DEF tab).
DST_SCORING = {
    "sack": 1.0,
    "interception": 2.0,
    "fumble_recovery": 2.0,
    "safety": 2.0,
    "touchdown": 6.0,
    # (max points allowed, fantasy points) -- first matching tier wins.
    "points_allowed_tiers": [
        (0, 10.0),
        (6, 7.0),
        (13, 4.0),
        (20, 1.0),
        (27, 0.0),
        (34, -1.0),
        (999, -4.0),
    ],
}


# --------------------------------------------------------------------------------------
# Roster / draft format
# --------------------------------------------------------------------------------------

NUM_TEAMS = 10
DRAFT_TYPE = "snake"
LEAGUE_FORMAT = "redraft"

# Dedicated (position-locked) starter slots per team. Feeds STEP 4a.
DEDICATED_STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}

# FLEX slots per team, shared across RB/WR/TE.
FLEX_SLOTS = 2
FLEX_ELIGIBLE = ("RB", "WR", "TE")

# Historical league-wide FLEX usage rates. Feeds STEP 4a's shared-slot allocation.
#
# Read these as a share of the 20 FLEX slots in the league (2 per team x 10 teams). The spec's
# default put TE at 10%, i.e. TWO tight ends started in FLEX league-wide every week -- which
# requires two separate teams each rostering a TE good enough to start over their RB/WR depth.
# On 17-man rosters in a 10-team league that essentially does not happen outside a
# McBride-plus-Warren situation. TE is set to 2.5%: half a TE FLEX start league-wide, i.e. one
# such team roughly every other week. The freed share is split between RB and WR in their
# existing 2:1 ratio, which leaves that split unlitigated.
#
# This is not cosmetic. FLEX allocation sets the starter count, which sets which player is the
# replacement, which sets the baseline every VORP at that position is measured against. Measured
# directly, the startable-TE gap versus expert consensus moves +15.5 at 5%, +25.2 at 10%,
# +34.6 at 20% and +44.1 at 30% -- it was the single largest driver of the board's TE tilt.
#
# Rounding note: 2.5% gives TE 10 + 0.5 = 10.5 nominal starters, and starter counts round to
# whole players, so this resolves to 10 -- identical to setting TE FLEX to zero. The replacement
# is therefore TE11 rather than TE12. 2.5% sits exactly on the boundary: at or below it TE gets
# 10 starters, above it 11 (and above 7.5%, 12).
FLEX_ALLOCATION = {"RB": 0.65, "WR": 0.325, "TE": 0.025}

BENCH_SLOTS = 7

# Positions that go through the full STEP 1-4 VORP pipeline.
SCORED_POSITIONS = ("QB", "RB", "WR", "TE")

# How many players to actually put on the board. Only 170 picks happen in this league
# (10 teams x 17 spots), so a 900-row sheet is mostly noise -- hundreds of those players share
# an identical fallback projection and will never be drafted.
#
# This truncates the OUTPUT only. Every statistical input is still computed from the full player
# population first: the STEP 3a position mean/stdev, the STEP 2 z-score pools, and the STEP 4b
# replacement levels. Cutting before those were computed would move the replacement level and
# silently change every VORP on the sheet. Set to None to keep everyone.
OUTPUT_PLAYER_LIMIT: int | None = 250

# Positions handled minimally (recency-weighted PPG rank only, no VORP) -- draft last/stream.
MINIMAL_POSITIONS = ("K", "DST")


# --------------------------------------------------------------------------------------
# Personal-preference team penalty -- DISABLED
# --------------------------------------------------------------------------------------
# Currently off: no team is penalised, and the rankings are purely data-driven.
#
# The mechanism is still here because it is a general team-bias switch, not a 49ers-specific
# hack. To turn it back on, set BIAS_TEAM to a team abbreviation (e.g. "SF"). When it is
# active the penalty is applied as a flat multiplier kept deliberately separate from every
# data-driven multiplier, the workbook gains 49ers/pre-penalty columns, and the pipeline runs
# a second unbiased pass so the model's unbiased opinion stays visible. While it is None,
# that second pass is skipped and those columns are omitted entirely.
BIAS_TEAM: str | None = None
BIAS_TEAM_MULTIPLIER = 0.92


@dataclass(frozen=True)
class LeagueConfig:
    """Bundle of league settings, so functions take one object instead of many globals."""

    num_teams: int = NUM_TEAMS
    draft_type: str = DRAFT_TYPE
    league_format: str = LEAGUE_FORMAT
    dedicated_starters: dict[str, int] = field(default_factory=lambda: dict(DEDICATED_STARTERS))
    flex_slots: int = FLEX_SLOTS
    flex_eligible: tuple[str, ...] = FLEX_ELIGIBLE
    flex_allocation: dict[str, float] = field(default_factory=lambda: dict(FLEX_ALLOCATION))
    bench_slots: int = BENCH_SLOTS
    games_in_season: int = GAMES_IN_SEASON
    target_season: int = TARGET_SEASON
    lookback_seasons: tuple[int, ...] = tuple(LOOKBACK_SEASONS)
    scored_positions: tuple[str, ...] = SCORED_POSITIONS
    minimal_positions: tuple[str, ...] = MINIMAL_POSITIONS
    output_player_limit: int | None = OUTPUT_PLAYER_LIMIT
    bias_team: str | None = BIAS_TEAM
    bias_team_multiplier: float = BIAS_TEAM_MULTIPLIER

    @property
    def total_roster_spots(self) -> int:
        """17 for this league -- see the roster-size note at the top of this module."""
        return sum(self.dedicated_starters.values()) + self.flex_slots + self.bench_slots

    @property
    def total_picks(self) -> int:
        return self.num_teams * self.total_roster_spots

    def validate(self) -> None:
        total = sum(self.flex_allocation.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"FLEX_ALLOCATION must sum to 1.0, got {total}")
        for pos in self.flex_allocation:
            if pos not in self.flex_eligible:
                raise ValueError(f"FLEX_ALLOCATION has {pos}, not in FLEX_ELIGIBLE {self.flex_eligible}")


LEAGUE = LeagueConfig()
LEAGUE.validate()
