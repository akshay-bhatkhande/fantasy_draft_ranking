# Pre-Draft Fantasy Rankings (10-team, PPR, redraft)

Generates pre-draft fantasy football rankings and writes them to an Excel workbook. Built to
be re-run through the offseason as news changes.

Everything it uses is free. No paid subscriptions, and no API keys that cost money.

---

## The two commands

### 1. Refresh the numbers (plain Python, no AI, run any time)

```bash
python run_rankings.py
```

Pulls all automated data, runs the full Steps 1-4 scoring pipeline, and regenerates the
workbook. Ordinary Python calling real APIs -- zero chat involvement, zero added cost. Run it
straight from Cursor's terminal whenever you want fresh numbers.

Output lands at:

```
output/rankings_<today's date>.xlsx        e.g. output/rankings_2026-08-06.xlsx
```

Useful flags: `--output path.xlsx` to choose the file, `--no-excel` to run the pipeline and
print the summary without writing a workbook.

### 2. Refresh camp buzz (through Cursor's AI chat, every 3-7 days during camp)

Camp news needs reading and interpreting live articles, so it is done by asking Cursor's agent
in chat rather than by the script. Paste this prompt:

> Research training camp buzz from the last 1-2 weeks for the top 150 players in
> `output/rankings_<date>.xlsx` and update `manual_overrides/camp_buzz.json`. Use the -2 to +2
> scale: +2 = multiple beat writers reporting a clear role or position-battle win, +1 = positive
> but limited mentions, 0 = no notable camp news, -1/-2 = mirrored for negative news. Every
> entry must include a source and a date -- drop any player you cannot source. Also set
> `last_updated` to today. While you are there, check for newly announced injuries, holdouts or
> suspensions and record expected games missed in `manual_overrides/known_absences.csv`.

Then re-run `python run_rankings.py` to fold it in.

The script never searches the web itself. It just reads whatever is currently in those files,
and it runs fine if they are stale or missing -- camp buzz simply degrades to neutral and the
terminal summary tells you how old the file is.

This is not a background job and there is no cron. Run the script on demand; refresh camp buzz
every few days while news is moving, and once more right before the draft.

---

## Manual / agent-maintained files

These are the inputs that cannot be automated for free. Edit them by hand or ask Cursor's agent
to fill them in. The script only ever reads them.

| File | What it is | How often |
|---|---|---|
| `manual_overrides/camp_buzz.json` | -2..+2 camp-buzz scores with source + date | every 3-7 days in camp |
| `manual_overrides/known_absences.csv` | already-announced games missed for the coming season | as news breaks |
| `manual_overrides/contract_years.csv` | optional corrections to the auto-detected contract years | once per offseason |
| `manual_overrides/adp_manual.csv` | optional multi-platform ADP paste-in | once at setup |

---

## How the ranking works

The whole pipeline stays in fantasy points from start to finish. There is no abstract unitless
score that has to be converted back later.

```
Weighted PPG (Step 1)
  -> Composite Z-Score, position-relative only (Step 2)
  -> Base Projected PPG, back in real points via position mean/stdev (Step 3a)
  -> x contract-year x age-curve x camp-buzz x 49ers-penalty = Final Projected PPG (Step 3b)
  -> x Expected Games Played = Final Projected Season Points (Step 3c-3d)
  -> minus position Replacement Level = VORP (Step 4)
```

**VORP is the only number that is ever valid to compare across positions.** Composite Z-Score,
Base Projected PPG and Final Projected Season Points are all position-relative -- a QB's 300
projected points and a RB's 300 projected points are not the same thing. Subtracting each
position's own replacement baseline is what puts everyone in the same unit: points better than a
freely available replacement at that position.

Starter counts, and therefore replacement levels, are **derived from `config/league.py`**, not
hardcoded. For this league they compute to:

```
QB: (1 x 10) + 0                  = 10 starters -> replacement is QB11
RB: (2 x 10) + (2 x 10 x 0.60)    = 32 starters -> replacement is RB33
WR: (2 x 10) + (2 x 10 x 0.30)    = 26 starters -> replacement is WR27
TE: (1 x 10) + (2 x 10 x 0.10)    = 12 starters -> replacement is TE13
```

Change the roster or the FLEX allocation and these recalculate automatically.

Volatility (floor / median / ceiling / Consistency Score) is computed **in parallel** and shown
alongside every player, but never enters the chain above and never changes a rank. A boom/bust
player is a bigger risk in a locked starting slot than as a FLEX or bench flyer, and that call
is yours to make per roster slot.

Expert consensus (Step 5) is a **sanity check only**. It never feeds the math. Any player more
than 15 spots away from consensus gets an auto-generated reason naming the input responsible.

---

## Configuration

| File | What lives there |
|---|---|
| `config/league.py` | scoring, teams, roster slots, FLEX allocation, target season, 49ers penalty |
| `config/weights.py` | all component weights, multiplier magnitudes, injury buckets, tier method |

Both are plain constants with comments. Nothing in `ffrank/` hardcodes a league rule or a
weight, so you can retune without touching the scoring logic.

---

## Workbook tabs

15 sheets: **Cover**, **Main Rankings**, **Pick 1**-**Pick 10**, **Tiers**, **Bye Week Check**,
**Kicker-DEF**.

(The original brief said "14 total" and then listed these fifteen. All the listed tabs are
built, since dropping one to match the count would lose requested content.)

- **Cover** -- index with hyperlinks, plain-language methodology, Last Updated, plus an audit
  block showing the replacement levels, fitted age curves, derived contract-year lifts and
  rookie baselines actually used this run.
- **Main Rankings** -- the single source of truth. Every other tab filters or annotates these
  exact numbers.
- **Pick 1-10** -- one per draft slot: computed snake pick sequence, recommended strategy with
  reasoning tied to the VORP obtainable at *your* picks, a "likely available at your next pick"
  indicator, a clearly-labelled **Slot-Adjusted Tier** (never confused with the global Tier),
  and per-round pivot contingencies. VORP and projected points are never recomputed here.
- **Tiers** -- global VORP tier bands by position, colour-banded, independent of draft slot.
- **Bye Week Check** -- bye clustering risk among top-ranked players.
- **Kicker-DEF** -- minimal, recency-weighted PPG only. Draft last or stream.

---

## Data sources

All free, no paid tier, no keys.

**Automated (the script fetches these):**

- **nflverse** via `nflreadpy` -- weekly and season stats, play-by-play, participation (for real
  routes run), snap counts, PFR advanced stats (yards after contact, broken tackles), injury
  reports, rosters, depth charts, draft picks, schedules, contracts (an OverTheCap mirror, so no
  scraping), and expected fantasy points from the ffopportunity model.
- **Fantasy Football Calculator** -- ADP queried for this exact format (10-team PPR) with
  per-player standard deviation. Chosen over Sleeper, which has no public ADP endpoint.
- **Vegas implied team totals** -- computed from the `spread_line` and `total_line` already
  carried in the nflverse schedule release. No odds API and no key needed.
- **FantasyPros ECR** -- real expert consensus via the dynastyprocess mirror, filtered to
  redraft-overall.

**Not automated (see the two files above):** camp buzz, and current injury/holdout/suspension
news. These need live reading, which is the chat step.

Anything time-sensitive carries its source and pull date in the Notes column. Where data is
genuinely unavailable the sheet says **"insufficient data"** rather than guessing.

### Two honest caveats surfaced on the sheet

- **RB YPRR is pass-snap-based.** Routes are counted as "eligible receiver on the field for a
  dropback", which includes pass-blocking snaps. Measured targets-per-pass-snap implies roughly a
  5% overcount for WRs and TEs but 25-35% for RBs, who stay in to block. Since YPRR is z-scored
  within the position pool, a systematic proportional bias largely cancels.
- **Vegas lines are early-season only.** In early August the schedule release carries lines for
  roughly the first four weeks, so implied team totals average a handful of games. The count is
  reported so you can weigh it.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_rankings.py
```

The first run downloads several seasons of play-by-play and participation data and takes a
couple of minutes. It is cached in `.cache/` (gitignored), so later runs are fast. Delete
`.cache/` to force a cold refresh.

Run the tests with:

```bash
python -m pytest tests/ -q
```

---

## Layout

```
run_rankings.py              single entry point
config/league.py             league rules
config/weights.py            model weights and constants
ffrank/data/                 data access: nflverse, market (ADP/ECR), id crosswalk, overrides, cache
ffrank/features/             weighted_ppg (Step 1), opportunity, efficiency, situational,
                             risk, curves, volatility, kdst
ffrank/scoring/              step2_composite, step3_projection, step4_vorp, step5_ecr, strategy
ffrank/excel/                workbook, formatting
ffrank/pipeline.py           orchestration: loads once, runs Steps 1-5 in order
manual_overrides/            the hand/agent-maintained inputs
output/                      generated workbooks
tests/                       worked-example and unit tests
```

Every scoring function's docstring names the step it implements, so the code and the
methodology stay traceable to each other.
