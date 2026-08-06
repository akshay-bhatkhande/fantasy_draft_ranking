"""Caching + source-provenance plumbing shared by every loader.

Two separate concerns live here:

1. nflreadpy's own filesystem cache, pointed at a project-local, gitignored directory so
   only the first run pays the download cost for three seasons of participation and
   play-by-play. Offseason reruns are then fast.
2. A tiny JSON disk cache for the non-nflverse HTTP sources we fetch ourselves (currently
   Fantasy Football Calculator's ADP endpoint).

It also holds the SourceLog, which every loader writes to. The workbook's Notes and
"Last Updated" cells are built from this, so anything time-sensitive carries a source and
a pull date rather than appearing as an unattributed number.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"
NFLREADPY_CACHE_DIR = CACHE_DIR / "nflreadpy"
HTTP_CACHE_DIR = CACHE_DIR / "http"
MANUAL_OVERRIDES_DIR = PROJECT_ROOT / "manual_overrides"
OUTPUT_DIR = PROJECT_ROOT / "output"


def configure_nflreadpy() -> None:
    """Point nflreadpy at the project-local filesystem cache.

    Must run before the first nflreadpy call. Cache duration is deliberately long (7 days)
    because completed-season data never changes; delete .cache/ to force a cold refresh.
    """
    NFLREADPY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NFLREADPY_CACHE", "filesystem")
    os.environ.setdefault("NFLREADPY_CACHE_DIR", str(NFLREADPY_CACHE_DIR))
    os.environ.setdefault("NFLREADPY_CACHE_DURATION", str(7 * 24 * 3600))
    os.environ.setdefault("NFLREADPY_VERBOSE", "False")
    os.environ.setdefault("NFLREADPY_TIMEOUT", "120")


# --------------------------------------------------------------------------------------
# Source provenance
# --------------------------------------------------------------------------------------


@dataclass
class SourceRecord:
    """One data source actually consulted during a run."""

    name: str
    detail: str
    pull_date: str
    status: str  # "ok" | "degraded" | "unavailable"
    rows: int | None = None
    message: str = ""

    def as_line(self) -> str:
        bits = [f"{self.name}: {self.status}"]
        if self.rows is not None:
            bits.append(f"{self.rows} rows")
        bits.append(f"pulled {self.pull_date}")
        if self.message:
            bits.append(self.message)
        return " | ".join(bits)


@dataclass
class SourceLog:
    """Collects provenance for every source touched in a run.

    The general sourcing rule is enforced here: where data is genuinely unavailable we
    record "unavailable" and downstream marks the affected columns "insufficient data",
    rather than fabricating a value.
    """

    records: list[SourceRecord] = field(default_factory=list)

    def add(
        self,
        name: str,
        detail: str,
        status: str = "ok",
        rows: int | None = None,
        message: str = "",
    ) -> SourceRecord:
        rec = SourceRecord(
            name=name,
            detail=detail,
            pull_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            status=status,
            rows=rows,
            message=message,
        )
        self.records.append(rec)
        return rec

    def get(self, name: str) -> SourceRecord | None:
        for rec in reversed(self.records):
            if rec.name == name:
                return rec
        return None

    def degraded(self) -> list[SourceRecord]:
        return [r for r in self.records if r.status != "ok"]

    def summary_lines(self) -> list[str]:
        return [r.as_line() for r in self.records]


SOURCES = SourceLog()


def safe_load(
    name: str,
    detail: str,
    loader: Callable[[], Any],
    log: SourceLog | None = None,
    required: bool = False,
):
    """Run a loader, recording provenance and degrading gracefully on failure.

    Tier 1 sources (nflverse) are effectively stable; Tier 2 sources (unofficial JSON
    endpoints, third-party pages) can change structure or disappear without notice, so
    every call routes through here. Returns None on failure unless required=True.
    """
    log = log if log is not None else SOURCES
    try:
        result = loader()
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any source can break
        log.add(name, detail, status="unavailable", message=f"{type(exc).__name__}: {exc}")
        if required:
            raise
        return None

    rows = None
    try:
        rows = int(getattr(result, "height", None) or len(result))
    except Exception:  # noqa: BLE001
        rows = None
    log.add(name, detail, status="ok", rows=rows)
    return result


# --------------------------------------------------------------------------------------
# Small JSON disk cache for our own HTTP fetches
# --------------------------------------------------------------------------------------


def http_cache_get(key: str, max_age_seconds: int) -> Any | None:
    """Return cached JSON for `key` if it exists and is fresher than max_age_seconds."""
    path = HTTP_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > max_age_seconds:
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def http_cache_set(key: str, payload: Any) -> None:
    HTTP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = HTTP_CACHE_DIR / f"{key}.json"
    try:
        with path.open("w") as fh:
            json.dump(payload, fh)
    except OSError:
        pass  # a cache write failure must never break a run


def cache_age_days(key: str) -> float | None:
    """Age in days of a cached HTTP payload, for the run summary."""
    path = HTTP_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    return (time.time() - path.stat().st_mtime) / 86400.0
