"""Manual / agent-maintained override files.

These are the Tier 3 inputs that cannot be scraped for free and are instead maintained by
hand or by asking Cursor's agent to research them in chat. The core script only ever READS
whatever is currently in these files. It never fetches or searches anything itself, and it
must run successfully when a file is stale, malformed or missing -- in which case the
affected input degrades to neutral / "insufficient data" and the run summary says so.

Files:
  manual_overrides/camp_buzz.json      -2..+2 camp-buzz scores with source + date
  manual_overrides/known_absences.csv  already-announced games missed for the coming season
  manual_overrides/contract_years.csv  optional manual contract-year corrections
  manual_overrides/adp_manual.csv      optional multi-platform ADP paste-in (see market.py)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from config import weights as W
from .cache import MANUAL_OVERRIDES_DIR, SourceLog
from .sleeper import normalize_name


@dataclass
class CampBuzz:
    """Parsed camp-buzz file plus freshness metadata for the run summary."""

    scores: pd.DataFrame  # name_key, camp_buzz_score, camp_buzz_source, camp_buzz_date, camp_buzz_note
    last_updated: str | None
    age_days: float | None
    is_stale: bool
    status: str  # "ok" | "stale" | "missing" | "malformed"
    message: str = ""

    @property
    def available(self) -> bool:
        return not self.scores.empty


def load_camp_buzz(log: SourceLog | None = None) -> CampBuzz:
    """Read camp_buzz.json, the single place camp news enters the pipeline.

    Expected shape:
        {
          "last_updated": "2026-08-05",
          "players": [
            {"player": "Name", "score": 1, "source": "Beat writer, Outlet",
             "date": "2026-08-04", "note": "won the slot role"}
          ]
        }

    Any player whose entry lacks a source is dropped: the methodology forbids unsourced
    camp adjustments, so a missing citation must not silently move a player's projection.
    """
    path = MANUAL_OVERRIDES_DIR / "camp_buzz.json"
    empty = pd.DataFrame(
        columns=["name_key", "camp_buzz_score", "camp_buzz_source", "camp_buzz_date", "camp_buzz_note"]
    )

    if not path.exists():
        if log:
            log.add("camp_buzz.json", str(path.name), status="unavailable", message="file missing; camp buzz neutral for all players")
        return CampBuzz(empty, None, None, True, "missing", "camp_buzz.json not found; all players neutral")

    try:
        with path.open() as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        if log:
            log.add("camp_buzz.json", str(path.name), status="unavailable", message=f"unreadable: {exc}")
        return CampBuzz(empty, None, None, True, "malformed", f"camp_buzz.json unreadable ({exc}); all players neutral")

    rows = []
    dropped_unsourced = 0
    for rec in payload.get("players", []) or []:
        name = rec.get("player") or rec.get("name")
        if not name:
            continue
        source = (rec.get("source") or "").strip()
        if not source:
            dropped_unsourced += 1
            continue
        try:
            score = int(round(float(rec.get("score", 0))))
        except (TypeError, ValueError):
            continue
        score = max(-2, min(2, score))
        rows.append(
            {
                "name_key": normalize_name(name),
                "camp_buzz_score": score,
                "camp_buzz_source": source,
                "camp_buzz_date": rec.get("date") or payload.get("last_updated"),
                "camp_buzz_note": rec.get("note") or "",
            }
        )

    scores = pd.DataFrame(rows) if rows else empty
    last_updated = payload.get("last_updated")
    age_days = None
    if last_updated:
        try:
            dt = datetime.fromisoformat(str(last_updated).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        except ValueError:
            age_days = None

    is_stale = age_days is None or age_days > W.CAMP_BUZZ_STALE_AFTER_DAYS
    status = "stale" if is_stale else "ok"
    msg = ""
    if dropped_unsourced:
        msg = f"{dropped_unsourced} entries dropped for missing source (unsourced camp adjustments are not allowed)"
    if log:
        log.add(
            "camp_buzz.json",
            f"{path.name} (last_updated={last_updated})",
            status="ok" if not is_stale else "degraded",
            rows=len(scores),
            message=msg or (f"{age_days:.1f} days old" if age_days is not None else "no last_updated field"),
        )
    return CampBuzz(scores, last_updated, age_days, is_stale, status, msg)


def load_known_absences(log: SourceLog | None = None) -> pd.DataFrame:
    """Already-announced games missed for the upcoming season (injury/holdout/suspension).

    This is a known, specific, current fact and is subtracted directly in STEP 3c -- it is
    NOT the same thing as the probabilistic historical Injury Risk bucket, which is applied
    on top of it.

    Expected columns: player, games_missed, reason, source, date
    """
    path = MANUAL_OVERRIDES_DIR / "known_absences.csv"
    cols = ["name_key", "known_games_missed", "absence_reason", "absence_source", "absence_date"]
    if not path.exists():
        if log:
            log.add("known_absences.csv", str(path.name), status="unavailable", message="none on file")
        return pd.DataFrame(columns=cols)

    try:
        raw = pd.read_csv(path, comment="#")
    except Exception as exc:  # noqa: BLE001
        if log:
            log.add("known_absences.csv", str(path.name), status="unavailable", message=str(exc))
        return pd.DataFrame(columns=cols)

    if raw.empty or "player" not in raw.columns:
        if log:
            log.add("known_absences.csv", str(path.name), status="ok", rows=0)
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(
        {
            "name_key": raw["player"].map(normalize_name),
            "known_games_missed": pd.to_numeric(raw.get("games_missed"), errors="coerce").fillna(0.0),
            "absence_reason": raw.get("reason", pd.Series([""] * len(raw))).fillna(""),
            "absence_source": raw.get("source", pd.Series([""] * len(raw))).fillna(""),
            "absence_date": raw.get("date", pd.Series([""] * len(raw))).astype(str).fillna(""),
        }
    )
    out = out[out["known_games_missed"] > 0]
    if log:
        log.add("known_absences.csv", str(path.name), status="ok", rows=len(out))
    return out


def load_contract_year_overrides(log: SourceLog | None = None) -> pd.DataFrame:
    """Optional manual contract-year corrections layered on top of the OTC-derived flags.

    Expected columns: player, contract_year (Y/N), source
    """
    path = MANUAL_OVERRIDES_DIR / "contract_years.csv"
    cols = ["name_key", "contract_year_override", "contract_year_source"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        raw = pd.read_csv(path, comment="#")
    except Exception as exc:  # noqa: BLE001
        if log:
            log.add("contract_years.csv", str(path.name), status="unavailable", message=str(exc))
        return pd.DataFrame(columns=cols)
    if raw.empty or "player" not in raw.columns:
        return pd.DataFrame(columns=cols)

    flag = (
        raw.get("contract_year", pd.Series([""] * len(raw)))
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["Y", "YES", "TRUE", "1"])
    )
    out = pd.DataFrame(
        {
            "name_key": raw["player"].map(normalize_name),
            "contract_year_override": flag,
            "contract_year_source": raw.get("source", pd.Series([""] * len(raw))).fillna(""),
        }
    )
    if log:
        log.add("contract_years.csv", str(path.name), status="ok", rows=len(out))
    return out
