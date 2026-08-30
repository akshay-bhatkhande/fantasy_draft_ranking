"""Separate workbook ranked by raw VORP (points over replacement).

This is the objective "how good could this player be" board:
  * ranked by vorp_raw (no QB/TE draft-scarcity haircut)
  * built from the unbiased projection pass (no personal player fades / team bias)
  * carries opportunity, efficiency, injury, and RB role stats for inspection
"""

from __future__ import annotations

import pandas as pd

from .formatting import Column, apply_flag_formatting, build_formats
from .workbook import _prepare_frame, _tier_row_formats, _write_table


RAW_SHEET = "Raw VORP Board"

RAW_VORP_COLUMNS: tuple[Column, ...] = (
    Column("overall_rank", "Raw VORP Rank", 10, "int"),
    Column("position_rank", "Pos Rank", 8, "int"),
    Column("tier", "Tier (raw VORP breaks)", 9, "int"),
    Column("player_name", "Name", 22),
    Column("position", "Pos", 6, "center"),
    Column("team", "Team", 7, "center"),
    Column("bye_week", "Bye", 6, "center_int"),
    Column("vorp_raw", "VORP Raw (pts over replacement)", 12, "num1"),
    Column("vorp", "VORP Draft-Adjusted (comparison only)", 12, "num1"),
    Column("final_projected_season_points", "Proj Season Pts", 11, "num1"),
    Column("final_projected_ppg", "Final PPG", 10, "num2"),
    Column("base_projected_ppg", "Base PPG", 10, "num2"),
    Column("weighted_ppg", "Weighted PPG", 10, "num2"),
    Column("expected_games_played", "Exp Games", 9, "num1"),
    Column("replacement_level_points", "Replacement Pts", 11, "num1"),
    Column("floor", "Floor (20th)", 9, "num1"),
    Column("median", "Median", 9, "num1"),
    Column("ceiling", "Ceiling (80th)", 10, "num1"),
    Column("consistency_score", "Consistency", 10, "num1"),
    Column("injury_risk_bucket", "Injury Risk", 10, "center"),
    Column("expected_games_missed_from_bucket", "Exp Missed (bucket)", 10, "num1"),
    Column("known_games_missed", "Known Games Out", 10, "num1"),
    Column("known_absence_reason", "Absence Reason", 28, "wrap"),
    Column("known_absence_source", "Absence Source", 20, "wrap"),
    Column("snap_share_pct", "Hist Snap %", 10, "num1"),
    Column("expected_snap_pct", "Expected Snap % (RB)", 11, "num1"),
    Column("hist_snap_pct", "RB Hist Snap %", 10, "num1"),
    Column("carry_share_pct", "Carry Share %", 10, "num1"),
    Column("carry_share_pct_raw", "Carry Share % (opp)", 11, "num1"),
    Column("rz_carry_share_pct", "RZ Carry Share %", 11, "num1"),
    Column("target_share_pct", "Target Share %", 11, "num1"),
    Column("rz_target_share_pct", "RZ Target Share %", 11, "num1"),
    Column("air_yards_share_pct", "Air Yards Share %", 11, "num1"),
    Column("dropback_share_pct", "Dropback Share %", 11, "num1"),
    Column("designed_rush_share_pct", "Designed Rush Share %", 12, "num1"),
    Column("rz_pass_attempt_share_pct", "RZ Pass Att Share %", 12, "num1"),
    Column("rb_role_label", "RB Role", 16, "wrap"),
    Column("rb_committee_flag", "Committee?", 12, "wrap"),
    Column("rb_team_structure", "Team RB Structure", 14, "wrap"),
    Column("opportunity_detail", "Opportunity Components", 40, "wrap"),
    Column("yprr", "YPRR", 8, "num2"),
    Column("yards_after_contact", "YAC / att", 9, "num2"),
    Column("broken_tackle_rate", "Broken Tackle Rate", 11, "num3"),
    Column("td_rate_vs_expected", "TD rate vs expected", 11, "num3"),
    Column("ypa", "YPA", 8, "num2"),
    Column("sack_rate_avoided", "Sack rate avoided", 11, "num3"),
    Column("efficiency_detail", "Efficiency Components", 40, "wrap"),
    Column("total_routes", "Routes (sample)", 10, "int"),
    Column("composite_z", "Composite Z", 10, "num2"),
    Column("age", "Age", 6, "num1"),
    Column("age_curve_multiplier", "Age Mult", 9, "num3"),
    Column("contract_year_flag", "Contract Y/N", 9, "center"),
    Column("contract_year_multiplier", "Contract Mult", 10, "num3"),
    Column("camp_buzz_score", "Camp Buzz", 9, "center_int"),
    Column("camp_buzz_multiplier", "Camp Mult", 9, "num3"),
    Column("limited_sample", "Limited Sample", 10, "center"),
    Column("adp_blended", "ADP", 8, "num1"),
    Column("consensus_rank", "ECR", 8, "int"),
    Column("ecr_delta", "Delta vs Raw Rank", 10, "int"),
    Column("sos_full", "SOS", 8, "num3"),
    Column("implied_total_per_game", "Vegas Team Total", 10, "num1"),
    Column("notes", "Notes", 80, "wrap"),
)


def write_raw_vorp_workbook(result, path) -> str:
    """Write the raw-VORP spreadsheet and return the path."""
    import xlsxwriter

    df = result.raw_vorp_rankings
    if df is None or df.empty:
        raise ValueError("PipelineResult.raw_vorp_rankings is empty; cannot write raw VORP workbook")

    prepared = _prepare_frame(df)
    # Only emit columns that exist on the frame.
    columns = tuple(c for c in RAW_VORP_COLUMNS if c.key in prepared.columns or c.key in ("contract_year_flag",))

    workbook = xlsxwriter.Workbook(str(path), {"nan_inf_to_errors": True})
    formats = build_formats(workbook)
    ws = workbook.add_worksheet(RAW_SHEET)
    ws.freeze_panes(2, 4)

    title = workbook.add_format(
        {"bold": True, "font_size": 14, "font_color": "#1F4E79", "font_name": "Calibri"}
    )
    blurb = workbook.add_format(
        {"font_size": 10, "font_color": "#595959", "font_name": "Calibri", "text_wrap": True}
    )
    ws.write(0, 0, "Raw VORP Board — objective production value (no personal fades)", title)
    ws.merge_range(
        1,
        0,
        1,
        min(8, len(columns) - 1),
        (
            "Ranked by VORP Raw = projected season points − replacement. "
            "Ignores QB/TE draft-scarcity adjustments and personal player fades. "
            f"Built {result.run_timestamp}."
        ),
        blurb,
    )
    ws.set_row(1, 32)

    tier_fmts = _tier_row_formats(workbook, formats)
    n = _write_table(
        ws,
        workbook,
        formats,
        prepared,
        columns,
        start_row=3,
        autofilter=True,
        freeze=None,
        tier_key="tier",
        tier_formats=tier_fmts,
    )
    apply_flag_formatting(ws, workbook, columns, n_rows=n, first_data_row=4)
    workbook.close()
    return str(path)
