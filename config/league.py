"""League settings.

Everything in this file describes *your league*. Change values here and the whole
pipeline recalculates -- notably STEP 4a's starter counts and replacement levels are
DERIVED from the roster settings below, never hardcoded.

Authoritative roster (confirmed): 1 QB + 2 RB + 2 WR + 1 TE + 2 FLEX + 1 K + 1 DST + 8 bench
= 18 spots. Bench never enters the STEP 4a starter-count / VORP math -- it only changes
total draft length (180 picks) and how many rounds the Pick-slot strategy tabs cover.
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

# Your snake-draft slot (1-indexed). Only this Pick tab is written to the workbook.
MY_DRAFT_SLOT = 3

# Round-1 board states at YOUR pick. Each becomes its own Excel tab: the named players are
# treated as already drafted when picks 1-2 are over, and the tab answers "who do I take at
# pick 3, and how do I draft after that?" Sheet names must stay <= 31 characters (Excel limit).
ROUND1_SCENARIOS: tuple[dict, ...] = (
    {
        "id": "bijan_gibbs_gone",
        "sheet_name": "If Bijan+Gibbs Gone",
        "title": "Pick 3 — Bijan & Gibbs already drafted",
        "gone": ("Bijan Robinson", "Jahmyr Gibbs"),
        "blurb": (
            "Picks 1-2 took the two elite RBs. The board opens for WR (or a third path if the "
            "market also sniped a receiver)."
        ),
    },
    {
        "id": "gibbs_puka_gone",
        "sheet_name": "If Gibbs+Puka Gone",
        "title": "Pick 3 — Gibbs & Puka already drafted",
        "gone": ("Jahmyr Gibbs", "Puka Nacua"),
        "blurb": (
            "One elite RB and the model's WR1 are gone. Decide between Bijan and the next WR tier."
        ),
    },
    {
        "id": "puka_bijan_gone",
        "sheet_name": "If Puka+Bijan Gone",
        "title": "Pick 3 — Puka & Bijan already drafted",
        "gone": ("Puka Nacua", "Bijan Robinson"),
        "blurb": (
            "The model's top WR and top RB are gone. Gibbs vs Chase/St. Brown is the live decision."
        ),
    },
)

# Dedicated (position-locked) starter slots per team. Feeds STEP 4a.
DEDICATED_STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}

# FLEX slots per team, shared across RB/WR/TE.
FLEX_SLOTS = 2
FLEX_ELIGIBLE = ("RB", "WR", "TE")

# Historical league-wide FLEX usage rates. Feeds STEP 4a's shared-slot allocation.
#
# Read these as a share of the 20 FLEX slots in the league (2 per team x 10 teams).
#
# DERIVED, not assumed. The spec's suggested 60/30/10 is a half-PPR-ish convention and is simply
# wrong for full PPR. The correct split is the one where the MARGINAL flex starter is worth the
# same at every position -- if it is not, a rational manager swaps, so the allocation is not an
# equilibrium. Simulating that directly on realised 2023-25 PPR outcomes (lock the top 20 RB,
# 20 WR and 10 TE into dedicated slots, then hand each of the 20 flex slots to whichever position
# offers the best remaining player) gives:
#
#     season   RB    WR    TE     marginal values at the cutoff
#     2023     25%   75%   0%     RB 187 / WR 190
#     2024     25%   75%   0%     RB 187 / WR 184
#     2025     25%   70%   5%     RB 179 / WR 172 / TE 176
#
# Strikingly stable, and the marginal values do equalise, which is the check that the method is
# right. Full PPR is the reason: reception points make WR21-WR35 more valuable than RB21-RB30,
# and there are simply more useful receivers than backs.
#
# Independent confirmation: this allocation also brings the board into agreement with expert
# consensus on positional structure. Startable-player gap versus consensus went
#     RB +23.8 / WR -15.6  at 60/30/10-style 65/32.5/2.5
#     RB  +4.8 / WR  +2.4  here
# The market prices players as though flex runs ~25/73, so two independent methods agree.
#
# To re-derive after a scoring change, re-run the greedy simulation described above.
#
# Rounding note: starter counts round to whole players. TE at 2% gives 10.4 nominal starters,
# which resolves to 10 -- the same as zero TE flex -- so the TE replacement is TE11. TE needs
# more than 2.5% to buy an 11th starter.
FLEX_ALLOCATION = {"RB": 0.25, "WR": 0.73, "TE": 0.02}

BENCH_SLOTS = 8

# Positions that go through the full STEP 1-4 VORP pipeline.
SCORED_POSITIONS = ("QB", "RB", "WR", "TE")

# How many players to actually put on the board. Only 180 picks happen in this league
# (10 teams x 18 spots), so a 900-row sheet is mostly noise -- hundreds of those players share
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
# Personal-preference penalties (NOT data-driven)
# --------------------------------------------------------------------------------------
# Kept deliberately separate from every data-driven multiplier. When either is active the
# workbook shows Pre-Penalty VORP / rank so the unbiased model stays visible, and the
# pipeline runs a second unbiased pass.
#
# Team penalty: set BIAS_TEAM to a team abbreviation (e.g. "SF") to deprioritize a whole
# roster. Currently off.
BIAS_TEAM: str | None = None
BIAS_TEAM_MULTIPLIER = 0.92

# Player fades: map display name -> multiplier + reason. Matched via normalize_name.
# 0.90 ≈ an 10% PPG haircut -- enough to drop CMC a tier without inventing a season-ending
# absence. Add/remove names here; empty dict disables the feature.
PLAYER_FADES: dict[str, dict] = {
    "Christian McCaffrey": {
        "multiplier": 0.90,
        "reason": "Personal fade: chronic injury / workload history",
    },
    "Kyren Williams": {
        "multiplier": 0.82,
        "reason": "Personal fade: Corum committee trend / not a true bellcow — align closer to market (ADP ~32, ECR ~42)",
    },
}


@dataclass(frozen=True)
class LeagueConfig:
    """Bundle of league settings, so functions take one object instead of many globals."""

    num_teams: int = NUM_TEAMS
    draft_type: str = DRAFT_TYPE
    league_format: str = LEAGUE_FORMAT
    my_draft_slot: int = MY_DRAFT_SLOT
    round1_scenarios: tuple[dict, ...] = ROUND1_SCENARIOS
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
    player_fades: dict[str, dict] = field(default_factory=lambda: dict(PLAYER_FADES))

    @property
    def personal_bias_active(self) -> bool:
        """True when any personal-preference penalty (team or player fade) is configured."""
        return bool(self.bias_team) or bool(self.player_fades)

    @property
    def total_roster_spots(self) -> int:
        """18 for this league -- see the roster-size note at the top of this module."""
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
        if not (1 <= self.my_draft_slot <= self.num_teams):
            raise ValueError(
                f"MY_DRAFT_SLOT must be in 1..{self.num_teams}, got {self.my_draft_slot}"
            )
        for sc in self.round1_scenarios:
            name = sc.get("sheet_name", "")
            if not name or len(name) > 31:
                raise ValueError(
                    f"ROUND1_SCENARIOS sheet_name must be 1..31 chars (Excel limit), got {name!r}"
                )
            if not sc.get("gone"):
                raise ValueError(f"ROUND1_SCENARIOS entry {name!r} needs a non-empty 'gone' list")
        for player, spec in self.player_fades.items():
            mult = float(spec.get("multiplier", 1.0))
            if not (0.5 <= mult <= 1.0):
                raise ValueError(
                    f"PLAYER_FADES[{player!r}] multiplier must be in [0.5, 1.0], got {mult}"
                )


LEAGUE = LeagueConfig()
LEAGUE.validate()
