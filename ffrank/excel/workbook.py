"""Excel workbook builder.

Sheets, in order:
    Cover            index, plain-language methodology, Last Updated, audit of assumptions used
    Main Rankings    the single source of truth; every other tab filters or annotates it
    Pick 1..Pick 10  one tab per draft slot, same VORP, plus slot strategy and availability
    Tiers            global VORP tier bands by position, visual reference
    Bye Week Check   bye-week clustering risk among top-ranked players
    Kicker-DEF       minimal recency-weighted PPG ranks, no VORP pipeline

Note on the tab count: the brief said "14 total" and then listed Cover + Main Rankings +
10 slot tabs + Tiers + Bye Week Check + Kicker/DEF, which sums to 15. All the listed tabs are
built (dropping one to satisfy the count would lose requested content); the Cover tab states
the real count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import weights as W
from ..scoring import strategy as strat
from ..scoring.step4_vorp import describe_starter_counts
from .formatting import (
    KDST_COLUMNS,
    SLOT_EXTRA_COLUMNS,
    Column,
    apply_flag_formatting,
    build_formats,
    main_columns,
    resolve_format,
)

COVER_SHEET = "Cover"
MAIN_SHEET = "Main Rankings"
TIERS_SHEET = "Tiers"
BYE_SHEET = "Bye Week Check"
KDST_SHEET = "Kicker-DEF"


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
    return out


def _write_table(
    worksheet,
    workbook,
    formats: dict,
    df: pd.DataFrame,
    columns: tuple[Column, ...],
    start_row: int = 0,
    autofilter: bool = True,
    freeze: tuple[int, int] | None = None,
) -> int:
    """Write a header row plus data rows, returning the number of data rows written."""
    for idx, col in enumerate(columns):
        worksheet.write(start_row, idx, col.header, formats["header"])
        worksheet.set_column(idx, idx, col.width)
    worksheet.set_row(start_row, 46)

    n = 0
    for r, (_, row) in enumerate(df.iterrows(), start=start_row + 1):
        for c, col in enumerate(columns):
            value = row.get(col.key)
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
        ws.write(row, 2, _sheet_description(name), formats["text"])
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
        ws.merge_range(
            row, 1, row + 1, 2,
            f"Players on {league.bias_team} have Final Projected PPG multiplied by "
            f"{league.bias_team_multiplier:.2f}. This is a personal preference, not objective "
            f"analysis, and is kept separate from every data-driven multiplier. The Main Rankings tab "
            f"shows Pre-Penalty VORP and Pre-Penalty Overall Rank so the unbiased model stays visible.",
            formats["callout"],
        )
    else:
        ws.merge_range(
            row, 1, row + 1, 2,
            "None. No team is penalised, so every ranking on this workbook is purely data-driven. "
            "The team-penalty mechanism is still available -- set BIAS_TEAM in config/league.py to a "
            "team abbreviation to switch it on, which also restores the penalty and pre-penalty "
            "columns on the Main Rankings tab.",
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


def _sheet_description(name: str) -> str:
    if name == MAIN_SHEET:
        return "Single source of truth. Every other tab filters or annotates these exact numbers."
    if name.startswith("Pick "):
        slot = name.split(" ")[1]
        return f"Draft slot {slot}: pick sequence, recommended strategy, availability and Slot-Adjusted Tier."
    if name == TIERS_SHEET:
        return "Global VORP tier bands by position, colour-banded. Independent of draft slot."
    if name == BYE_SHEET:
        return "Bye-week clustering risk among the top-ranked players at each position."
    if name == KDST_SHEET:
        return "Kickers and team defenses. Draft last / stream; full VORP pipeline intentionally skipped."
    return ""


def _write_main(workbook, formats, df: pd.DataFrame, columns: tuple[Column, ...]) -> None:
    ws = workbook.add_worksheet(MAIN_SHEET)
    n = _write_table(ws, workbook, formats, df, columns, freeze=(1, 4))
    apply_flag_formatting(ws, workbook, columns, n)


def _write_slot(workbook, formats, df: pd.DataFrame, slot: int, league, columns: tuple[Column, ...]) -> None:
    ws = workbook.add_worksheet(f"Pick {slot}")
    picks = strat.pick_sequence(slot, league.num_teams, league.total_roster_spots)
    board = strat.slot_board(df, slot, league)
    strategy, reasoning = strat.recommend_strategy(df, slot, picks, league)
    notes = strat.contingency_notes(df, slot, league)

    ws.set_column(0, 0, 18)
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
    n = _write_table(ws, workbook, formats, board, slot_columns, start_row=row, freeze=(row + 1, 0))
    apply_flag_formatting(ws, workbook, slot_columns, n, first_data_row=row + 1)


def _write_tiers(workbook, formats, df: pd.DataFrame, league, include_team_penalty: bool) -> None:
    ws = workbook.add_worksheet(TIERS_SHEET)
    ws.write(0, 0, "Global Tiers by Position", formats["title"])
    ws.write(
        1, 0,
        "Same global VORP-based tiers as Main Rankings, grouped by position for visual reference. "
        "Independent of draft slot -- see the Pick tabs for Slot-Adjusted Tiers.",
        formats["wrap"],
    )

    tier_columns = (
        Column("position_tier", "Position Tier", 9, "int"),
        Column("tier", "Global Tier", 9, "int"),
        Column("overall_rank", "Overall Rank", 10, "int"),
        Column("position_rank", "Position Rank", 9, "int"),
        Column("player_name", "Name", 22),
        Column("team", "Team", 7, "center"),
        Column("bye_week", "Bye", 6, "center_int"),
        Column("vorp", "VORP", 10, "num1"),
        Column("final_projected_season_points", "Final Projected Season Points", 13, "num1"),
        Column("consistency_score", "Consistency", 10, "num1"),
        Column("injury_risk_bucket", "Injury Risk", 10, "center"),
    )
    if include_team_penalty:
        tier_columns += (Column("team_bias_flag", "Team Penalty", 9, "center"),)

    row = 3
    for pos in league.scored_positions:
        sub = df[df["position"] == pos].sort_values("vorp", ascending=False)
        if sub.empty:
            continue
        level = sub["replacement_level_points"].dropna()
        starters = sub["starter_count_at_position"].dropna()
        label = f"{pos}"
        if len(level) and len(starters):
            label = (
                f"{pos} - replacement level is {pos}{int(starters.iloc[0]) + 1} "
                f"at {float(level.iloc[0]):.1f} projected season points"
            )
        ws.write(row, 0, label, formats["subtitle"])
        row += 1
        n = _write_table(ws, workbook, formats, sub.head(60), tier_columns, start_row=row, autofilter=False)
        apply_flag_formatting(ws, workbook, tier_columns, n, first_data_row=row + 1)
        row += n + 3


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
    columns = main_columns(include_team_penalty=result.bias_active)

    slot_names = [f"Pick {s}" for s in range(1, league.num_teams + 1)]
    sheet_names = [MAIN_SHEET, *slot_names, TIERS_SHEET, BYE_SHEET, KDST_SHEET]

    workbook = xlsxwriter.Workbook(str(path), {"nan_inf_to_errors": True})
    formats = build_formats(workbook)

    _write_cover(workbook, formats, result, sheet_names)
    _write_main(workbook, formats, df, columns)
    for slot in range(1, league.num_teams + 1):
        _write_slot(workbook, formats, df, slot, league, columns)
    _write_tiers(workbook, formats, df, league, result.bias_active)
    _write_bye(workbook, formats, df, league)
    _write_kdst(workbook, formats, result.kickers, result.defenses)

    workbook.close()
    return str(path)
