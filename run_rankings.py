#!/usr/bin/env python3
"""Single entry point: refresh all data, run Steps 1-5, and write the Excel workbook.

    python run_rankings.py

This is ordinary Python calling real APIs. No AI, no chat, no API keys that cost money, and
no scheduled job -- run it whenever you want fresh numbers.

Camp-buzz and current-news inputs are NOT fetched here. They live in manual_overrides/, which
you refresh occasionally by asking Cursor's agent in chat (see README.md). This script only
reads whatever is currently in those files and runs fine when they are stale or missing.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date
from pathlib import Path

from config.league import LEAGUE
from ffrank.data.cache import OUTPUT_DIR, SourceLog
from ffrank.pipeline import build_rankings


def _fmt_camp_status(result) -> str:
    if result.camp_buzz_status == "missing":
        return "camp_buzz.json not found (all players neutral)"
    if result.camp_buzz_status == "malformed":
        return "camp_buzz.json unreadable (all players neutral)"
    if result.camp_buzz_age_days is None:
        return "camp_buzz.json present but has no last_updated date"
    return f"camp_buzz.json last updated {result.camp_buzz_age_days:.1f} days ago"


def print_summary(result, output_path: Path) -> None:
    """Short terminal summary so you can see what is current without opening the workbook."""
    d = result.diagnostics
    print()
    print("=" * 78)
    print(f"  {LEAGUE.target_season} rankings refreshed - {result.run_timestamp}")
    print("=" * 78)
    pool_note = ""
    if d.get("output_limit") and d.get("full_pool_size", 0) > d["players_ranked"]:
        pool_note = (
            f" (trimmed from {d['full_pool_size']}; replacement levels and position "
            f"distributions still computed on all {d['full_pool_size']})"
        )
    print(f"  Players ranked          : {d['players_ranked']} across {', '.join(LEAGUE.scored_positions)}{pool_note}")
    print(
        f"  League                  : {LEAGUE.num_teams}-team snake {LEAGUE.league_format}, "
        f"full PPR, {LEAGUE.total_roster_spots} roster spots "
        f"({LEAGUE.bench_slots} bench), your slot Pick {LEAGUE.my_draft_slot}"
    )
    print(f"  ADP matched             : {d['adp_players']} players (Fantasy Football Calculator, "
          f"{LEAGUE.num_teams}-team PPR)")
    print(f"  Consensus (ECR) matched : {d['ecr_matched']} players (sanity check only)")
    print(f"  Routes / YPRR source    : {d['routes_source']}")
    def _file_vs_board(on_board: int, in_file: int) -> str:
        """A player can be in an override file but rank outside the output limit."""
        if in_file == on_board:
            return f"{on_board}"
        return f"{on_board} on the board, {in_file} in the file ({in_file - on_board} ranked outside the top {d['players_ranked']})"

    print(f"  Camp buzz               : {_file_vs_board(d['camp_buzz_players'], d.get('camp_buzz_in_file', d['camp_buzz_players']))}"
          f" scored; {_fmt_camp_status(result)}")
    print(f"  Known absences          : {_file_vs_board(d['known_absences'], d.get('known_absences_in_file', d['known_absences']))}")

    print()
    print("  Replacement levels (Step 4b):")
    for pos, level in result.replacement_levels.items():
        rank = result.starter_counts.get(pos, 0) + 1
        print(f"    {pos:3} {pos}{rank:<4} = {level:7.1f} projected season points")

    print()
    print("  Fitted assumptions (Step 3b):")
    for pos, curve in result.age_curves.items():
        print(f"    {curve.describe()}")
    for pos, lift in result.contract_lifts.items():
        print(f"    {lift.describe()}")

    degraded = result.sources.degraded() if result.sources else []
    if degraded:
        print()
        print("  Degraded / unavailable sources (affected columns marked 'insufficient data'):")
        for rec in degraded:
            print(f"    - {rec.name}: {rec.message}")

    print()
    top = result.rankings.head(10)
    print("  Top 10 by VORP:")
    for r in top.itertuples():
        flag = " [team penalty]" if str(getattr(r, "team_bias_flag", "N")) == "Y" else ""
        print(
            f"    {int(r.overall_rank):>2}. {r.player_name:<24} {r.position:<3} {r.team:<4} "
            f"VORP {r.vorp:6.1f}  tier {r.tier}{flag}"
        )

    print()
    print(f"  Workbook written to: {output_path}")
    raw_path = d.get("raw_vorp_path")
    if raw_path:
        print(f"  Raw VORP workbook   : {raw_path}")
        raw = getattr(result, "raw_vorp_rankings", None)
        if raw is not None and not raw.empty:
            print()
            print("  Top 10 by Raw VORP (no fades / no draft-scarcity rank key):")
            for i, row in enumerate(raw.head(10).itertuples(), start=1):
                print(
                    f"   {i:2}. {row.player_name:22s} {row.position:2s}  "
                    f"raw {float(row.vorp_raw):6.1f}  adj {float(row.vorp):6.1f}"
                )
    print("=" * 78)
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate pre-draft fantasy football rankings.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .xlsx path (default: output/rankings_<date>.xlsx)",
    )
    parser.add_argument(
        "--no-excel", action="store_true",
        help="Run the pipeline and print the summary without writing the workbook.",
    )
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or OUTPUT_DIR / f"rankings_{date.today().isoformat()}.xlsx"

    log = SourceLog()
    try:
        result = build_rankings(LEAGUE, log=log)
    except Exception:  # noqa: BLE001
        print("Pipeline failed. Data sources consulted before the failure:", file=sys.stderr)
        for line in log.summary_lines():
            print(f"  {line}", file=sys.stderr)
        print(file=sys.stderr)
        traceback.print_exc()
        return 1

    if not args.no_excel:
        from ffrank.excel.raw_vorp import write_raw_vorp_workbook
        from ffrank.excel.workbook import write_workbook

        write_workbook(result, output_path)
        raw_path = output_path.with_name(
            output_path.name.replace("rankings_", "raw_vorp_", 1)
            if output_path.name.startswith("rankings_")
            else f"raw_vorp_{output_path.name}"
        )
        write_raw_vorp_workbook(result, raw_path)
        result.diagnostics["raw_vorp_path"] = str(raw_path)

    print_summary(result, output_path if not args.no_excel else Path("(skipped)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
