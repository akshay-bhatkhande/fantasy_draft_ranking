"""Workbook formats, column definitions and conditional-formatting helpers.

The column list is the contract between the pipeline and the workbook: every column is named
to match the methodology's own vocabulary, so any number on the sheet can be traced back to a
specific labelled step.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    """One workbook column: source field, display header, number format and width."""

    key: str
    header: str
    width: float = 12
    fmt: str | None = None  # logical format name resolved in build_formats()


# The Main Rankings column set. Slot tabs reuse this identical set (plus slot-specific extras)
# so no column ever means something different from one tab to the next.
MAIN_COLUMNS: tuple[Column, ...] = (
    Column("overall_rank", "Overall Rank (by VORP, cross-position)", 11, "int"),
    Column("position_rank", "Position Rank", 9, "int"),
    Column("tier", "Tier (VORP breakpoints)", 9, "int"),
    Column("player_name", "Name", 22),
    Column("position", "Position", 8, "center"),
    Column("team", "Team", 7, "center"),
    Column("bye_week", "Bye Week", 8, "center_int"),
    Column("weighted_ppg", "Weighted PPG (Step 1)", 12, "num2"),
    Column("composite_z", "Composite Z-Score (Step 2 - position-relative ONLY, not cross-position comparable)", 16, "num2"),
    Column("base_projected_ppg", "Base Projected PPG (Step 3a)", 12, "num2"),
    Column("final_projected_ppg", "Final Projected PPG (Step 3b)", 12, "num2"),
    Column("expected_games_played", "Expected Games Played (Step 3c)", 12, "num1"),
    Column("final_projected_season_points", "Final Projected Season Points (Step 3d)", 13, "num1"),
    Column("vorp", "VORP (Step 4d - draft-adjusted, cross-position rank key)", 12, "num1"),
    Column("vorp_raw", "VORP Raw (Step 4c - points minus replacement)", 12, "num1"),
    Column("vorp_draft_scale", "VORP Draft Scarcity Scale (effective vs raw)", 10, "num2"),
    Column("vorp_wait_penalty", "VORP Wait Penalty (QB/TE draft scarcity)", 10, "num1"),
    Column("replacement_level_points", "Replacement Level used (Step 4b)", 12, "num1"),
    Column("starter_count_at_position", "Starter Count at Position (Step 4a)", 10, "int"),
    Column("floor", "Floor (20th pct weekly)", 10, "num1"),
    Column("median", "Median weekly", 10, "num1"),
    Column("ceiling", "Ceiling (80th pct weekly)", 10, "num1"),
    Column("consistency_score", "Consistency Score (0-100, higher = steadier)", 12, "num1"),
    Column("same_ppg_volatility_flag", "Same-PPG-Different-Volatility Flag", 46, "wrap"),
    Column("adp_blended", "ADP (avg across sources)", 10, "num1"),
    Column("adp_stdev", "ADP source spread / variance", 11, "num1"),
    Column("market_disagreement_flag", "Market Disagreement", 16, "wrap"),
    Column("injury_risk_bucket", "Injury Risk Bucket", 10, "center"),
    Column("expected_snap_pct", "Expected Snap % (RB role estimate)", 11, "num1"),
    Column("rb_workload_multiplier", "RB Workload Multiplier (from expected snap %)", 11, "num3"),
    Column("rb_role_label", "RB Role Label", 18, "wrap"),
    Column("rb_committee_flag", "RB Committee Flag", 14, "wrap"),
    Column("rb_team_structure", "Team RB Structure (prior season)", 14, "wrap"),
    Column("hist_snap_pct", "Historical Snap % (recency-weighted)", 11, "num1"),
    Column("carry_share_pct", "Carry Share % (recency-weighted)", 11, "num1"),
    Column("rb_committee_note", "RB Committee / Snap Audit", 44, "wrap"),
    Column("expected_games_missed_from_bucket", "Expected Games Missed (from bucket)", 11, "num1"),
    Column("known_games_missed", "Known Current-Season Games Missed", 11, "num1"),
    Column("known_absence_source", "Known Absence Source", 22, "wrap"),
    Column("camp_buzz_flag", "Camp Buzz Flag", 11, "center"),
    Column("camp_buzz_score", "Camp Buzz Score (-2..+2)", 9, "center_int"),
    Column("camp_buzz_multiplier", "Camp Buzz Multiplier", 10, "num3"),
    Column("contract_year_flag", "Contract Year (Y/N)", 9, "center"),
    Column("contract_year_multiplier", "Contract Year Multiplier (derived)", 11, "num3"),
    Column("age", "Age", 7, "num1"),
    Column("age_curve_multiplier", "Age-Curve Multiplier", 10, "num3"),
    Column("age_curve_assumption", "Age-Curve / Rookie Assumption Used", 44, "wrap"),
    Column("rookie_adjustment_applied", "Rookie Adjustment (share of position mean)", 11, "num3"),
    Column("team_bias_flag", "Team Penalty Flag (Y/N)", 9, "center"),
    Column("team_bias_multiplier", "Team Penalty Multiplier", 10, "num3"),
    Column("player_fade_flag", "Player Fade Flag (Y/N)", 9, "center"),
    Column("player_fade_multiplier", "Player Fade Multiplier", 10, "num3"),
    Column("player_fade_reason", "Player Fade Reason", 36, "wrap"),
    Column("pre_penalty_vorp", "Pre-Penalty VORP", 11, "num1"),
    Column("pre_penalty_overall_rank", "Pre-Penalty Overall Rank", 11, "int"),
    Column("consensus_rank", "Consensus Rank (Step 5, sanity check only)", 11, "int"),
    Column("ecr_delta", "Delta vs My Rank", 10, "int"),
    Column("ecr_flag", "ECR Flag (>15 spots)", 18, "wrap"),
    Column("ecr_reason", "ECR Disagreement Reason (auto)", 44, "wrap"),
    Column("sos_full", "Full-Season SOS (scored)", 10, "num3"),
    Column("sos_playoffs", "Weeks 15-17 SOS (informational only)", 11, "num3"),
    Column("dome_share", "Dome Share of Home Games (informational)", 10, "num2"),
    Column("implied_total_per_game", "Vegas Implied Team Total", 10, "num1"),
    Column("notes", "Notes / Sourcing", 90, "wrap"),
)

# Extra columns shown only on the per-slot / scenario tabs.
SLOT_EXTRA_COLUMNS: tuple[Column, ...] = (
    Column("slot_adjusted_tier", "Slot-Adjusted Tier (NOT the global Tier)", 12, "int"),
    Column("likely_available_at_next_pick", "Likely Available at Your Next Pick", 16, "center"),
    Column("availability_probability", "Availability Probability", 11, "num2"),
    Column("realistic_target_pick", "Realistic Target Pick #", 11, "int"),
)

SCENARIO_EXTRA_COLUMNS: tuple[Column, ...] = (
    Column("scenario_gone", "Already Drafted in This Scenario (Y/N)", 12, "center"),
) + SLOT_EXTRA_COLUMNS

# Columns that only make sense when a personal-preference penalty is switched on.
TEAM_PENALTY_COLUMN_KEYS: frozenset[str] = frozenset(
    {"team_bias_flag", "team_bias_multiplier"}
)
PLAYER_FADE_COLUMN_KEYS: frozenset[str] = frozenset(
    {"player_fade_flag", "player_fade_multiplier", "player_fade_reason"}
)
PRE_PENALTY_COLUMN_KEYS: frozenset[str] = frozenset(
    {"pre_penalty_vorp", "pre_penalty_overall_rank"}
)


def main_columns(
    *,
    include_team_penalty: bool = False,
    include_player_fade: bool = False,
) -> tuple[Column, ...]:
    """Main Rankings columns, omitting inactive personal-preference blocks."""
    drop: set[str] = set()
    if not include_team_penalty:
        drop |= TEAM_PENALTY_COLUMN_KEYS
    if not include_player_fade:
        drop |= PLAYER_FADE_COLUMN_KEYS
    if not (include_team_penalty or include_player_fade):
        drop |= PRE_PENALTY_COLUMN_KEYS
    return tuple(c for c in MAIN_COLUMNS if c.key not in drop)


KDST_COLUMNS: tuple[Column, ...] = (
    Column("rank", "Rank", 6, "int"),
    Column("player_name", "Name", 22),
    Column("team", "Team", 7, "center"),
    Column("weighted_ppg", "Recency-Weighted PPG", 12, "num2"),
    Column("games", "Games in Sample", 10, "int"),
    Column("note", "Note", 80, "wrap"),
)


def build_formats(workbook) -> dict:
    """Create every cell format used across the workbook."""
    base = {"font_name": "Calibri", "font_size": 10}
    f = {
        "header": workbook.add_format({
            **base, "bold": True, "bg_color": "#1F3864", "font_color": "white",
            "text_wrap": True, "valign": "vcenter", "align": "center", "border": 1,
        }),
        "title": workbook.add_format({**base, "bold": True, "font_size": 16}),
        "subtitle": workbook.add_format({**base, "bold": True, "font_size": 12}),
        "text": workbook.add_format({**base, "valign": "top"}),
        "wrap": workbook.add_format({**base, "text_wrap": True, "valign": "top"}),
        "body_wrap": workbook.add_format({**base, "text_wrap": True, "valign": "top"}),
        "int": workbook.add_format({**base, "num_format": "0", "align": "center"}),
        "center_int": workbook.add_format({**base, "num_format": "0", "align": "center"}),
        "center": workbook.add_format({**base, "align": "center"}),
        "num1": workbook.add_format({**base, "num_format": "0.0", "align": "center"}),
        "num2": workbook.add_format({**base, "num_format": "0.00", "align": "center"}),
        "num3": workbook.add_format({**base, "num_format": "0.000", "align": "center"}),
        "link": workbook.add_format({**base, "font_color": "blue", "underline": 1}),
        "label": workbook.add_format({**base, "bold": True}),
        "callout": workbook.add_format({
            **base, "text_wrap": True, "valign": "top", "bg_color": "#FFF2CC", "border": 1,
        }),
        "bias": workbook.add_format({**base, "bg_color": "#F8CBAD", "align": "center"}),
        "good": workbook.add_format({**base, "bg_color": "#C6EFCE", "align": "center"}),
        "bad": workbook.add_format({**base, "bg_color": "#FFC7CE", "align": "center"}),
        "warn": workbook.add_format({**base, "bg_color": "#FFEB9C", "align": "center"}),
    }
    return f


def resolve_format(formats: dict, name: str | None):
    if name is None:
        return formats["text"]
    return formats.get(name, formats["text"])


def apply_flag_formatting(worksheet, workbook, columns, n_rows: int, first_data_row: int = 1) -> None:
    """Conditional formatting so risk / buzz / bias flags are visually scannable.

    Colour scales are used for the continuous numbers a drafter scans (VORP, consistency,
    availability) and discrete highlights for the flag columns.
    """
    if n_rows <= 0:
        return
    last_row = first_data_row + n_rows - 1
    by_key = {c.key: i for i, c in enumerate(columns)}

    def col_range(key):
        idx = by_key.get(key)
        if idx is None:
            return None
        return first_data_row, idx, last_row, idx

    # Higher VORP is better -- a 3-colour scale makes the tier structure visible at a glance.
    for key in ("vorp", "final_projected_season_points", "consistency_score", "availability_probability"):
        rng = col_range(key)
        if rng:
            worksheet.conditional_format(*rng, {
                "type": "3_color_scale",
                "min_color": "#F8696B", "mid_color": "#FFEB84", "max_color": "#63BE7B",
            })

    # Lower is better for these.
    for key in ("adp_blended", "overall_rank"):
        rng = col_range(key)
        if rng:
            worksheet.conditional_format(*rng, {
                "type": "3_color_scale",
                "min_color": "#63BE7B", "mid_color": "#FFEB84", "max_color": "#F8696B",
            })

    red = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
    green = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
    amber = workbook.add_format({"bg_color": "#FFEB9C", "font_color": "#9C5700"})
    orange = workbook.add_format({"bg_color": "#F8CBAD", "font_color": "#843C0C"})

    rng = col_range("injury_risk_bucket")
    if rng:
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"High"', "format": red})
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"Med"', "format": amber})
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"Low"', "format": green})

    rng = col_range("rb_committee_flag")
    if rng:
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"Committee likely"', "format": red})
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"Lean committee"', "format": amber})
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"Lean bellcow"', "format": green})
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"Bellcow likely"', "format": green})

    rng = col_range("camp_buzz_flag")
    if rng:
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"Riser"', "format": green})
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"Faller"', "format": red})

    # The personal-preference penalty gets its own distinct colour so it is never mistaken for
    # one of the data-driven adjustments.
    rng = col_range("team_bias_flag")
    if rng:
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"Y"', "format": orange})
    rng = col_range("player_fade_flag")
    if rng:
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"Y"', "format": orange})
    rng = col_range("scenario_gone")
    if rng:
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"Y"', "format": red})

    rng = col_range("contract_year_flag")
    if rng:
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"Y"', "format": green})

    for key in ("ecr_flag", "market_disagreement_flag", "same_ppg_volatility_flag"):
        rng = col_range(key)
        if rng:
            worksheet.conditional_format(*rng, {
                "type": "text", "criteria": "containing", "value": " ", "format": amber,
            })

    rng = col_range("likely_available_at_next_pick")
    if rng:
        worksheet.conditional_format(*rng, {"type": "text", "criteria": "containing", "value": "Likely available", "format": green})
        worksheet.conditional_format(*rng, {"type": "text", "criteria": "containing", "value": "Coin flip", "format": amber})
        worksheet.conditional_format(*rng, {"type": "text", "criteria": "containing", "value": "gone", "format": red})

    rng = col_range("known_games_missed")
    if rng:
        worksheet.conditional_format(*rng, {"type": "cell", "criteria": ">", "value": 0, "format": red})
