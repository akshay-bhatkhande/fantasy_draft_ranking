"""Excel workbook builder.

Sheets, in order:
    Cover            index, plain-language methodology, Last Updated, audit of assumptions used
    Main Rankings    the single source of truth; every other tab filters or annotates it
    Pick N           your draft slot only (MY_DRAFT_SLOT): strategy, availability, Slot-Adjusted Tier
    Scenario tabs    Round-1 "if X and Y are gone" boards for that same slot
    Overall Tiers    full board colour-banded by global VORP tier
    QB/RB/WR/TE Tiers  per-position boards colour-banded by position tier
    Bye Week Check   bye-week clustering risk among top-ranked players
    Kicker-DEF       minimal recency-weighted PPG ranks, no VORP pipeline
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import weights as W
from ..scoring import strategy as strat
from ..scoring.step4_vorp import describe_starter_counts
from .formatting import (
    KDST_COLUMNS,
    SCENARIO_EXTRA_COLUMNS,
    SLOT_EXTRA_COLUMNS,
    Column,
    apply_flag_formatting,
    build_formats,
    main_columns,
    resolve_format,
)

COVER_SHEET = "Cover"
MAIN_SHEET = "Main Rankings"
OVERALL_TIERS_SHEET = "Overall Tiers"
BYE_SHEET = "Bye Week Check"
KDST_SHEET = "Kicker-DEF"

# Soft fills cycled by tier number (1-indexed). Tier 1 is the strongest green.
TIER_FILL_COLORS: tuple[str, ...] = (
    "#C6EFCE",  # 1
    "#A9D08E",  # 2
    "#FFE699",  # 3
    "#FCE4D6",  # 4
    "#DDEBF7",  # 5
    "#E2D5F1",  # 6
    "#D6DCE4",  # 7
    "#FFF2CC",  # 8
    "#F8CBAD",  # 9
    "#D9D9D9",  # 10+
)


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add the display-only derived columns the workbook expects."""
    out = df.copy()
    if "contract_year" in out.columns:
        contract = out["contract_year"].fillna(False).astype(bool)
    else:
        contract = pd.Series(False, index=out.index)
    out["contract_year_flag"] = np.where(contract, "Y", "N")

    # One column answering "which assumption moved this player": the fitted age curve for
    # veterans, the rookie draft-capital baseline for rookies.
    def _assumption(row):
        if bool(row.get("is_rookie")) and row.get("rookie_note"):
            return row["rookie_note"]
        return row.get("age_curve_note") or ""

    out["age_curve_assumption"] = out.apply(_assumption, axis=1)
    if "scenario_gone" in out.columns:
        out["scenario_gone"] = np.where(out["scenario_gone"].fillna(False).astype(bool), "Y", "N")
    return out


def _tier_fill(tier_value) -> str:
    """Background colour for a tier number (cycles after the palette length)."""
    try:
        if tier_value is None or (isinstance(tier_value, float) and pd.isna(tier_value)):
            return TIER_FILL_COLORS[-1]
        t = int(tier_value)
    except (TypeError, ValueError):
        return TIER_FILL_COLORS[-1]
    if t < 1:
        return TIER_FILL_COLORS[-1]
    return TIER_FILL_COLORS[min(t, len(TIER_FILL_COLORS)) - 1]


def _tier_row_formats(workbook, base_formats: dict) -> dict[int, dict[str, object]]:
    """Cache of per-tier cell formats keyed by tier number, mirroring base numeric/text fmts."""
    cache: dict[int, dict[str, object]] = {}
    for i, color in enumerate(TIER_FILL_COLORS, start=1):
        cache[i] = {
            "text": workbook.add_format({"font_name": "Calibri", "font_size": 10, "bg_color": color, "valign": "top"}),
            "wrap": workbook.add_format({"font_name": "Calibri", "font_size": 10, "bg_color": color, "text_wrap": True, "valign": "top"}),
            "int": workbook.add_format({"font_name": "Calibri", "font_size": 10, "bg_color": color, "num_format": "0", "align": "center"}),
            "center_int": workbook.add_format({"font_name": "Calibri", "font_size": 10, "bg_color": color, "num_format": "0", "align": "center"}),
            "center": workbook.add_format({"font_name": "Calibri", "font_size": 10, "bg_color": color, "align": "center"}),
            "num1": workbook.add_format({"font_name": "Calibri", "font_size": 10, "bg_color": color, "num_format": "0.0", "align": "center"}),
            "num2": workbook.add_format({"font_name": "Calibri", "font_size": 10, "bg_color": color, "num_format": "0.00", "align": "center"}),
            "num3": workbook.add_format({"font_name": "Calibri", "font_size": 10, "bg_color": color, "num_format": "0.000", "align": "center"}),
        }
    return cache


def _write_table(
    worksheet,
    workbook,
    formats: dict,
    df: pd.DataFrame,
    columns: tuple[Column, ...],
    start_row: int = 0,
    autofilter: bool = True,
    freeze: tuple[int, int] | None = None,
    tier_key: str | None = None,
    tier_formats: dict[int, dict[str, object]] | None = None,
) -> int:
    """Write a header row plus data rows, returning the number of data rows written.

    When tier_key + tier_formats are provided, each data row is colour-banded by that tier.
    """
    for idx, col in enumerate(columns):
        worksheet.write(start_row, idx, col.header, formats["header"])
        worksheet.set_column(idx, idx, col.width)
    worksheet.set_row(start_row, 46)

    n = 0
    for r, (_, row) in enumerate(df.iterrows(), start=start_row + 1):
        row_fmts = None
        if tier_key and tier_formats is not None:
            tv = row.get(tier_key)
            try:
                t_idx = int(tv) if tv is not None and not (isinstance(tv, float) and pd.isna(tv)) else None
            except (TypeError, ValueError):
                t_idx = None
            if t_idx is not None:
                t_idx = min(max(t_idx, 1), len(TIER_FILL_COLORS))
                row_fmts = tier_formats.get(t_idx)
        for c, col in enumerate(columns):
            value = row.get(col.key)
            if row_fmts is not None:
                fmt = row_fmts.get(col.fmt or "text", row_fmts["text"])
            else:
                fmt = resolve_format(formats, col.fmt)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                worksheet.write_blank(r, c, None, fmt)
                continue
            try:
                if pd.isna(value):
                    worksheet.write_blank(r, c, None, fmt)
                    continue
            except (TypeError, ValueError):
                pass
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            elif isinstance(value, (np.bool_, bool)):
                value = "Y" if bool(value) else "N"
            worksheet.write(r, c, value, fmt)
        n += 1

    if autofilter and n:
        worksheet.autofilter(start_row, 0, start_row + n, len(columns) - 1)
    if freeze:
        worksheet.freeze_panes(*freeze)
    return n


def _write_cover(workbook, formats, result, sheet_names: list[str]) -> None:
    ws = workbook.add_worksheet(COVER_SHEET)
    ws.set_column(0, 0, 3)
    ws.set_column(1, 1, 42)
    ws.set_column(2, 2, 92)
    ws.hide_gridlines(2)

    league = result.league
    row = 1
    ws.write(row, 1, f"{league.target_season} Pre-Draft Fantasy Rankings", formats["title"])
    row += 1
    ws.write(row, 1, f"{league.num_teams}-team snake, redraft, full PPR", formats["subtitle"])
    row += 1
    roster_bits = (
        f"1 QB / 2 RB / 2 WR / 1 TE / {league.flex_slots} FLEX / 1 K / 1 DST / "
        f"{league.bench_slots} bench ({league.total_roster_spots} spots, {league.total_picks} picks)"
    )
    ws.write(row, 1, roster_bits, formats["text"])
    row += 1
    my_picks = strat.pick_sequence(league.my_draft_slot, league.num_teams, league.total_roster_spots)
    early = ", ".join(str(p) for p in my_picks[:6])
    ws.write(
        row,
        1,
        f"Your draft slot: Pick {league.my_draft_slot}  "
        f"(early picks {early}… — use the 'Pick {league.my_draft_slot}' tab and the Round-1 scenario tabs)",
        formats["subtitle"],
    )
    row += 2

    ws.write(row, 1, "Last Updated", formats["label"])
    ws.write(row, 2, result.run_timestamp, formats["text"])
    row += 1
    camp_age = (
        f"{result.camp_buzz_age_days:.1f} days old" if result.camp_buzz_age_days is not None else "no date recorded"
    )
    ws.write(row, 1, "camp_buzz.json status", formats["label"])
    ws.write(row, 2, f"{result.camp_buzz_status} ({camp_age})", formats["text"])
    row += 2

    ws.write(row, 1, "How players are ranked", formats["subtitle"])
    row += 1
    ws.merge_range(
        row, 1, row + 2, 2,
        "Players are scored within their own position (never against other positions), converted "
        "into projected fantasy points, and then ranked league-wide by how many points they are "
        "worth above a replacement-level player at their position (VORP). See the Main Rankings "
        "column headers for the full detail of every step. VORP is the only number on this "
        "workbook that is valid to compare across positions.",
        formats["callout"],
    )
    row += 4

    ws.write(row, 1, "Contents", formats["subtitle"])
    row += 1
    ws.write(row, 1, f"{len(sheet_names) + 1} tabs in total (this Cover tab plus {len(sheet_names)})", formats["text"])
    row += 1
    for name in sheet_names:
        ws.write_url(row, 1, f"internal:'{name}'!A1", formats["link"], name)
        ws.write(row, 2, _sheet_description(name, league), formats["text"])
        row += 1
    row += 1

    ws.write(row, 1, "Step 4a starter counts (derived from config, not hardcoded)", formats["subtitle"])
    row += 1
    for line in describe_starter_counts(league):
        ws.write(row, 1, line, formats["text"])
        row += 1
    row += 1

    ws.write(row, 1, "Step 4b replacement levels used this run", formats["subtitle"])
    row += 1
    for pos, level in result.replacement_levels.items():
        ws.write(row, 1, f"{pos}{result.starter_counts.get(pos, 0) + 1} replacement level", formats["label"])
        ws.write(row, 2, f"{level:.1f} projected season points", formats["text"])
        row += 1
    row += 1

    ws.write(row, 1, "Fitted assumptions used this run", formats["subtitle"])
    row += 1
    for pos, curve in result.age_curves.items():
        ws.write(row, 1, f"Age curve - {pos}", formats["label"])
        ws.write(row, 2, curve.describe(), formats["text"])
        row += 1
    for pos, lift in result.contract_lifts.items():
        ws.write(row, 1, f"Contract-year lift - {pos}", formats["label"])
        ws.write(row, 2, lift.describe(), formats["text"])
        row += 1
    for pos, baseline in result.rookie_baselines.items():
        shares = ", ".join(f"{k}={v:.2f}" for k, v in sorted(baseline.shares.items()))
        ws.write(row, 1, f"Rookie baseline - {pos}", formats["label"])
        ws.write(row, 2, shares or "no derived tiers; fallback priors used", formats["text"])
        row += 1
    row += 1

    ws.write(row, 1, "Personal-preference adjustment", formats["subtitle"])
    row += 1
    if result.bias_active:
        bits = []
        if league.bias_team:
            bits.append(
                f"Players on {league.bias_team} have Final Projected PPG multiplied by "
                f"{league.bias_team_multiplier:.2f}."
            )
        if league.player_fades:
            fade_bits = [
                f"{name} x{float(spec.get('multiplier', 1.0)):.2f} "
                f"({spec.get('reason', 'personal fade')})"
                for name, spec in league.player_fades.items()
            ]
            bits.append("Player fades: " + "; ".join(fade_bits) + ".")
        bits.append(
            "These are personal preferences, not objective analysis, and are kept separate from "
            "every data-driven multiplier. Main Rankings shows Pre-Penalty VORP and Pre-Penalty "
            "Overall Rank so the unbiased model stays visible."
        )
        ws.merge_range(row, 1, row + 2, 2, " ".join(bits), formats["callout"])
        row += 3
    else:
        ws.merge_range(
            row, 1, row + 1, 2,
            "None. No team or player is personally penalised, so every ranking on this workbook "
            "is purely data-driven. Set BIAS_TEAM or PLAYER_FADES in config/league.py to switch "
            "penalties on (and restore the pre-penalty columns).",
            formats["callout"],
        )
        row += 3

    ws.write(row, 1, "What is NOT in the ranking", formats["subtitle"])
    row += 1
    for line in (
        "Volatility / consistency (floor, median, ceiling, Consistency Score) is calculated in "
        "parallel and shown alongside, but never enters the projection chain or changes a rank.",
        "Consensus Rank (Step 5) is a sanity check only and never feeds Composite Z-Score, "
        "projected points, or VORP.",
        "Weeks 15-17 SOS, bye week and dome/outdoor are informational columns only.",
        "ADP variance is tracked as a market-disagreement flag and is not folded into the "
        "Composite Z-Score.",
    ):
        ws.write(row, 1, line, formats["wrap"])
        ws.set_row(row, 28)
        row += 1
    row += 1

    ws.write(row, 1, "Data sources consulted this run", formats["subtitle"])
    row += 1
    if result.sources:
        for rec in result.sources.records:
            ws.write(row, 1, rec.name, formats["label"])
            ws.write(row, 2, rec.as_line(), formats["text"])
            row += 1


def _sheet_description(name: str, league=None) -> str:
    if name == MAIN_SHEET:
        return "Single source of truth. Every other tab filters or annotates these exact numbers."
    if name.startswith("Pick "):
        slot = name.split(" ")[1]
        return f"Your draft slot {slot}: pick sequence, recommended strategy, availability and Slot-Adjusted Tier."
    if league is not None:
        for sc in league.round1_scenarios:
            if sc.get("sheet_name") == name:
                gone = " + ".join(sc.get("gone", ()))
                return f"Round-1 scenario at Pick {league.my_draft_slot}: if {gone} are already drafted."
    if name == OVERALL_TIERS_SHEET:
        return "Full board sorted by global VORP tier, colour-banded so each tier is scannable."
    if name.endswith(" Tiers") and name.split()[0] in {"QB", "RB", "WR", "TE"}:
        pos = name.split()[0]
        return f"{pos}-only board sorted by position tier, colour-banded."
    if name == BYE_SHEET:
        return "Bye-week clustering risk among the top-ranked players at each position."
    if name == KDST_SHEET:
        return "Kickers and team defenses. Draft last / stream; full VORP pipeline intentionally skipped."
    return ""


def _write_main(workbook, formats, df: pd.DataFrame, columns: tuple[Column, ...]) -> None:
    ws = workbook.add_worksheet(MAIN_SHEET)
    tier_fmts = _tier_row_formats(workbook, formats)
    n = _write_table(
        ws, workbook, formats, df, columns, freeze=(1, 4),
        tier_key="tier", tier_formats=tier_fmts,
    )
    apply_flag_formatting(ws, workbook, columns, n)


def _write_slot(workbook, formats, df: pd.DataFrame, slot: int, league, columns: tuple[Column, ...]) -> None:
    ws = workbook.add_worksheet(f"Pick {slot}")
    picks = strat.pick_sequence(slot, league.num_teams, league.total_roster_spots)
    board = strat.slot_board(df, slot, league)
    strategy, reasoning = strat.recommend_strategy(df, slot, picks, league)
    notes = strat.contingency_notes(df, slot, league)

    ws.set_column(0, 0, 22)
    ws.set_column(1, 1, 100)
    ws.write(0, 0, f"Draft Slot {slot}", formats["title"])
    ws.write(1, 0, "Pick sequence", formats["label"])
    ws.write(1, 1, ", ".join(str(p) for p in picks), formats["text"])
    ws.write(2, 0, "Recommended strategy", formats["label"])
    ws.write(2, 1, strategy, formats["subtitle"])
    ws.merge_range(3, 1, 5, 10, reasoning, formats["callout"])

    row = 7
    ws.write(row, 0, "Round contingencies", formats["label"])
    for note in notes:
        ws.write(row, 1, note["note"], formats["wrap"])
        ws.set_row(row, 26)
        row += 1
    row += 1

    slot_columns = columns[:3] + SLOT_EXTRA_COLUMNS + columns[3:]
    prepared = _prepare_frame(board)
    tier_fmts = _tier_row_formats(workbook, formats)
    n = _write_table(
        ws, workbook, formats, prepared, slot_columns, start_row=row, freeze=(row + 1, 0),
        tier_key="slot_adjusted_tier", tier_formats=tier_fmts,
    )
    apply_flag_formatting(ws, workbook, slot_columns, n, first_data_row=row + 1)


def _fmt_target_line(targets: list[dict]) -> str:
    parts = []
    for t in targets:
        adp = t.get("adp")
        adp_txt = f"ADP {adp:.1f}" if adp is not None and not (isinstance(adp, float) and pd.isna(adp)) else "ADP n/a"
        parts.append(f"{t['name']} ({t['position']}, VORP {t['vorp']:.0f}, {adp_txt})")
    return "; ".join(parts) if parts else "no clear targets"


def _write_scenario(
    workbook,
    formats,
    df: pd.DataFrame,
    league,
    scenario: dict,
    columns: tuple[Column, ...],
) -> None:
    sheet = scenario["sheet_name"]
    gone = tuple(scenario.get("gone") or ())
    ws = workbook.add_worksheet(sheet)
    plan = strat.scenario_pick3_plan(df, league, gone)
    picks = plan["picks"]
    board = strat.slot_board(df, league.my_draft_slot, league, gone_names=gone)

    ws.set_column(0, 0, 24)
    ws.set_column(1, 1, 110)
    ws.write(0, 0, scenario.get("title") or sheet, formats["title"])
    ws.write(1, 0, "Already drafted", formats["label"])
    ws.write(1, 1, " + ".join(gone), formats["subtitle"])
    ws.write(2, 0, "Situation", formats["label"])
    ws.merge_range(2, 1, 3, 10, scenario.get("blurb") or "", formats["callout"])

    best = plan.get("best_pick3")
    ws.write(5, 0, "Take at pick 3", formats["label"])
    if best:
        ws.write(
            5, 1,
            f"{best['name']} ({best['position']}, {best['team']}) — VORP {best['vorp']:.1f}",
            formats["subtitle"],
        )
    else:
        ws.write(5, 1, "No remaining players", formats["text"])

    ws.write(6, 0, "Recommended structure", formats["label"])
    ws.write(6, 1, plan["strategy"], formats["subtitle"])
    ws.merge_range(7, 1, 8, 10, plan["reasoning"], formats["callout"])

    row = 10
    ws.write(row, 0, "Early-pick cheat sheet", formats["label"])
    row += 1
    for pick in picks[:3]:
        targets = plan["targets_by_pick"].get(pick, [])
        ws.write(row, 0, f"Pick {pick}", formats["label"])
        ws.write(row, 1, _fmt_target_line(targets), formats["wrap"])
        ws.set_row(row, 28)
        row += 1
    row += 1

    ws.write(row, 0, "Round contingencies", formats["label"])
    for note in plan["notes"]:
        ws.write(row, 1, note["note"], formats["wrap"])
        ws.set_row(row, 26)
        row += 1
    row += 1

    scenario_columns = columns[:3] + SCENARIO_EXTRA_COLUMNS + columns[3:]
    prepared = _prepare_frame(board)
    tier_fmts = _tier_row_formats(workbook, formats)
    n = _write_table(
        ws, workbook, formats, prepared, scenario_columns, start_row=row, freeze=(row + 1, 0),
        tier_key="slot_adjusted_tier", tier_formats=tier_fmts,
    )
    apply_flag_formatting(ws, workbook, scenario_columns, n, first_data_row=row + 1)


def _tier_columns(league, *, include_team_penalty: bool, position_mode: bool) -> tuple[Column, ...]:
    cols = (
        Column("position_tier", "Position Tier", 9, "int"),
        Column("tier", "Global Tier", 9, "int"),
        Column("overall_rank", "Overall Rank", 10, "int"),
        Column("position_rank", "Position Rank", 9, "int"),
        Column("player_name", "Name", 22),
        Column("position", "Pos", 6, "center"),
        Column("team", "Team", 7, "center"),
        Column("bye_week", "Bye", 6, "center_int"),
        Column("vorp", "VORP", 10, "num1"),
        Column("final_projected_season_points", "Final Projected Season Points", 13, "num1"),
        Column("camp_buzz_score", "Camp Buzz", 9, "center_int"),
        Column("consistency_score", "Consistency", 10, "num1"),
        Column("injury_risk_bucket", "Injury Risk", 10, "center"),
    )
    if position_mode:
        # Position sheet already filters; Pos column is redundant noise.
        cols = tuple(c for c in cols if c.key != "position")
    if include_team_penalty:
        cols += (Column("team_bias_flag", "Team Penalty", 9, "center"),)
    if league.player_fades:
        cols += (Column("player_fade_flag", "Player Fade", 9, "center"),)
    return cols


def _write_overall_tiers(workbook, formats, df: pd.DataFrame, league, include_team_penalty: bool) -> None:
    ws = workbook.add_worksheet(OVERALL_TIERS_SHEET)
    tier_fmts = _tier_row_formats(workbook, formats)
    ws.write(0, 0, "Overall Tiers (by global VORP)", formats["title"])
    ws.write(
        1, 0,
        "Same global Tier as Main Rankings. Rows are colour-banded by tier so breaks are obvious "
        "at a glance. Independent of draft slot.",
        formats["wrap"],
    )
    cols = _tier_columns(league, include_team_penalty=include_team_penalty, position_mode=False)
    board = df.sort_values(["tier", "vorp"], ascending=[True, False])
    n = _write_table(
        ws, workbook, formats, board, cols,
        start_row=3, freeze=(4, 4),
        tier_key="tier", tier_formats=tier_fmts,
    )
    apply_flag_formatting(ws, workbook, cols, n, first_data_row=4)


def _write_position_tiers(
    workbook, formats, df: pd.DataFrame, league, position: str, include_team_penalty: bool
) -> None:
    ws = workbook.add_worksheet(f"{position} Tiers")
    tier_fmts = _tier_row_formats(workbook, formats)
    sub = df[df["position"] == position].sort_values(["position_tier", "vorp"], ascending=[True, False])
    level = sub["replacement_level_points"].dropna()
    starters = sub["starter_count_at_position"].dropna()
    subtitle = f"{position} tiers (position-relative)"
    if len(level) and len(starters):
        subtitle = (
            f"{position} tiers — replacement is {position}{int(starters.iloc[0]) + 1} "
            f"at {float(level.iloc[0]):.1f} projected season points"
        )
    ws.write(0, 0, subtitle, formats["title"])
    ws.write(
        1, 0,
        "Colour-banded by Position Tier. Global Tier is shown for cross-reference with Overall Tiers / Main Rankings.",
        formats["wrap"],
    )
    cols = _tier_columns(league, include_team_penalty=include_team_penalty, position_mode=True)
    n = _write_table(
        ws, workbook, formats, sub, cols,
        start_row=3, freeze=(4, 4),
        tier_key="position_tier", tier_formats=tier_fmts,
    )
    apply_flag_formatting(ws, workbook, cols, n, first_data_row=4)


def _write_bye(workbook, formats, df: pd.DataFrame, league) -> None:
    ws = workbook.add_worksheet(BYE_SHEET)
    ws.write(0, 0, "Bye Week Clustering Check", formats["title"])
    ws.write(
        1, 0,
        "Counts how many of your top-ranked players at each position share a bye week. "
        "Informational only -- bye weeks never affect any score or multiplier.",
        formats["wrap"],
    )

    top_n = {"QB": 14, "RB": 40, "WR": 40, "TE": 16}
    frames = []
    for pos in league.scored_positions:
        sub = df[df["position"] == pos].nlargest(top_n.get(pos, 30), "vorp")
        frames.append(sub.assign(_pos=pos))
    if not frames:
        return
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["bye_week"].notna()]
    if pool.empty:
        ws.write(3, 0, "No bye-week data available for the upcoming season.", formats["text"])
        return

    grid = (
        pool.pivot_table(index="bye_week", columns="_pos", values="player_name", aggfunc="count")
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    grid["Total"] = grid.sum(axis=1)

    ws.write(3, 0, "Bye Week", formats["header"])
    for c, col in enumerate(grid.columns, start=1):
        ws.write(3, c, str(col), formats["header"])
    ws.set_column(0, 0, 10)
    ws.set_column(1, len(grid.columns), 10)

    for r, (bye, row_vals) in enumerate(grid.iterrows(), start=4):
        ws.write(r, 0, int(bye), formats["center_int"])
        for c, col in enumerate(grid.columns, start=1):
            ws.write(r, c, int(row_vals[col]), formats["center_int"])

    last = 3 + len(grid)
    ws.conditional_format(4, 1, last, len(grid.columns), {
        "type": "3_color_scale",
        "min_color": "#C6EFCE", "mid_color": "#FFEB9C", "max_color": "#FFC7CE",
    })

    row = last + 2
    ws.write(row, 0, "Clustering warnings", formats["subtitle"])
    row += 1
    threshold = max(3, int(grid["Total"].mean() + grid["Total"].std()))
    flagged = grid[grid["Total"] >= threshold]
    if flagged.empty:
        ws.write(row, 0, "No unusual bye-week clustering among top-ranked players.", formats["text"])
    else:
        for bye, vals in flagged.iterrows():
            ws.write(
                row, 0,
                f"Week {int(bye)}: {int(vals['Total'])} top-ranked players are on bye "
                f"- check you are not stacking too many starters here.",
                formats["wrap"],
            )
            row += 1

    row += 1
    ws.write(row, 0, "Top players by bye week", formats["subtitle"])
    row += 1
    for bye, grp in pool.groupby("bye_week"):
        names = ", ".join(grp.nlargest(8, "vorp")["player_name"].astype(str))
        ws.write(row, 0, f"Week {int(bye)}", formats["label"])
        ws.write(row, 1, names, formats["wrap"])
        row += 1
    ws.set_column(1, 1, 100)


def _write_kdst(workbook, formats, kickers: pd.DataFrame, defenses: pd.DataFrame) -> None:
    ws = workbook.add_worksheet(KDST_SHEET)
    ws.write(0, 0, "Kickers and Team Defenses", formats["title"])
    ws.merge_range(
        1, 0, 2, 6,
        "Draft these last or stream them. The full Steps 1-4 VORP pipeline is intentionally "
        "skipped here -- a simple recency-weighted points-per-game rank is sufficient, and "
        "spending ranking effort on these positions is not worth it in a 10-team league.",
        formats["callout"],
    )

    row = 4
    ws.write(row, 0, "Kickers", formats["subtitle"])
    row += 1
    if kickers.empty:
        ws.write(row, 0, "insufficient data: no kicker scoring available", formats["text"])
        row += 2
    else:
        n = _write_table(ws, workbook, formats, kickers.head(32), KDST_COLUMNS, start_row=row, autofilter=False)
        row += n + 3

    ws.write(row, 0, "Team Defenses", formats["subtitle"])
    row += 1
    dst_columns = tuple(c for c in KDST_COLUMNS if c.key != "player_name")
    if defenses.empty:
        ws.write(row, 0, "insufficient data: no team defense scoring available", formats["text"])
    else:
        _write_table(ws, workbook, formats, defenses.head(32), dst_columns, start_row=row, autofilter=False)


def write_workbook(result, path) -> str:
    """Write the full workbook and return the path."""
    import xlsxwriter

    df = _prepare_frame(result.rankings)
    league = result.league
    columns = main_columns(
        include_team_penalty=bool(league.bias_team),
        include_player_fade=bool(league.player_fades),
    )

    slot_name = f"Pick {league.my_draft_slot}"
    scenario_names = [sc["sheet_name"] for sc in league.round1_scenarios]
    pos_tier_names = [f"{pos} Tiers" for pos in league.scored_positions]
    sheet_names = [
        MAIN_SHEET,
        slot_name,
        *scenario_names,
        OVERALL_TIERS_SHEET,
        *pos_tier_names,
        BYE_SHEET,
        KDST_SHEET,
    ]

    workbook = xlsxwriter.Workbook(str(path), {"nan_inf_to_errors": True})
    formats = build_formats(workbook)

    _write_cover(workbook, formats, result, sheet_names)
    _write_main(workbook, formats, df, columns)
    _write_slot(workbook, formats, df, league.my_draft_slot, league, columns)
    for scenario in league.round1_scenarios:
        _write_scenario(workbook, formats, df, league, scenario, columns)
    _write_overall_tiers(workbook, formats, df, league, bool(league.bias_team))
    for pos in league.scored_positions:
        _write_position_tiers(workbook, formats, df, league, pos, bool(league.bias_team))
    _write_bye(workbook, formats, df, league)
    _write_kdst(workbook, formats, result.kickers, result.defenses)

    workbook.close()
    return str(path)
