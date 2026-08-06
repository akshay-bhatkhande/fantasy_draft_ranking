"""Player-ID crosswalk, plus Sleeper as a fallback metadata source.

The pipeline joins across several ID spaces: nflverse/gsis (stats, participation), pfr_id
(PFR advanced stats), and plain names (FFC ADP, FantasyPros ECR). dynastyprocess's
db_playerids table carries gsis_id, pfr_id, fantasypros_id and sleeper_id in a single
table, so it is the primary crosswalk and Sleeper's 14MB player dump is only a fallback.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd
import requests

from .cache import SourceLog, http_cache_get, http_cache_set, safe_load
from .nflverse import _pd

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_STATE_URL = "https://api.sleeper.app/v1/state/nfl"
SLEEPER_CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600
REQUEST_TIMEOUT = 60

# Common suffixes and punctuation that differ between sources for the same human.
_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?$", flags=re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")

# Hand-maintained aliases for players whose names differ irreconcilably between sources.
NAME_ALIASES = {
    "kenny gainwell": "kenneth gainwell",
    "gabe davis": "gabriel davis",
    "hollywood brown": "marquise brown",
    "chig okonkwo": "chigoziem okonkwo",
    "cam ward": "cameron ward",
    "deebo samuel sr": "deebo samuel",
    "michael pittman jr": "michael pittman",
    "marvin harrison jr": "marvin harrison",
    "brian thomas jr": "brian thomas",
    "kenneth walker iii": "kenneth walker",
    "travis etienne jr": "travis etienne",
    "tank bigsby": "thomas bigsby",
    "jaxon smithnjigba": "jaxon smith njigba",
}


def normalize_name(name) -> str:
    """Collapse a player name to a join key that survives source-to-source formatting.

    Strips accents, punctuation, generational suffixes and casing, then applies the alias
    table. Name joins are a last resort (used for ADP and ECR, which publish no stable ids),
    so this is deliberately aggressive.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "").replace(".", " ").replace("-", " ")
    text = _NON_ALNUM_RE.sub(" ", text)
    text = " ".join(text.split())
    text = _SUFFIX_RE.sub("", text).strip()
    tokens = text.split()

    # Collapse runs of single-character tokens so initials survive punctuation differences
    # between sources: "K.C. Concepcion" and "KC Concepcion" must produce the same key, as
    # must "A.J. Brown" / "AJ Brown" and "T.J. Hockenson" / "TJ Hockenson".
    merged: list[str] = []
    for token in tokens:
        if len(token) == 1 and merged and len(merged[-1]) <= 2 and merged[-1].isalpha():
            merged[-1] += token
        else:
            merged.append(token)
    text = " ".join(merged)
    return NAME_ALIASES.get(text, text)


def load_id_crosswalk(log: SourceLog | None = None) -> pd.DataFrame:
    """Primary crosswalk from dynastyprocess via nflreadpy (gsis/pfr/fantasypros/sleeper)."""
    import nflreadpy as nfl

    df = safe_load(
        "dynastyprocess player-id crosswalk",
        "load_ff_playerids()",
        lambda: nfl.load_ff_playerids(),
        log=log,
    )
    df = _pd(df)
    if df.empty:
        return df
    keep = [
        c
        for c in ("name", "mergename", "gsis_id", "pfr_id", "sleeper_id", "fantasypros_id", "position", "team", "birthdate", "draft_year", "draft_round", "draft_pick", "draft_ovr")
        if c in df.columns
    ]
    out = df[keep].copy()
    if "name" in out.columns:
        out["name_key"] = out["name"].map(normalize_name)
    return out


def fetch_sleeper_players(log: SourceLog | None = None) -> pd.DataFrame:
    """Fallback metadata source. Only called if the primary crosswalk is unavailable.

    Note this endpoint returns ~14MB, hence the long disk cache and fallback-only status.
    """

    def _load():
        cached = http_cache_get("sleeper_players", SLEEPER_CACHE_MAX_AGE_SECONDS)
        if cached is None:
            resp = requests.get(SLEEPER_PLAYERS_URL, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            cached = resp.json()
            http_cache_set("sleeper_players", cached)
        rows = []
        for pid, rec in cached.items():
            if rec.get("position") not in ("QB", "RB", "WR", "TE", "K", "DEF"):
                continue
            rows.append(
                {
                    "sleeper_id": pid,
                    "full_name": rec.get("full_name"),
                    "position": rec.get("position"),
                    "team": rec.get("team"),
                    "gsis_id": rec.get("gsis_id"),
                    "years_exp": rec.get("years_exp"),
                    "injury_status": rec.get("injury_status"),
                    "depth_chart_order": rec.get("depth_chart_order"),
                }
            )
        return pd.DataFrame(rows)

    df = safe_load("Sleeper players (fallback)", SLEEPER_PLAYERS_URL, _load, log=log)
    df = _pd(df)
    if not df.empty:
        df["name_key"] = df["full_name"].map(normalize_name)
    return df


def sleeper_state(log: SourceLog | None = None) -> dict:
    """Sleeper's view of the current season/week, used for the run summary header."""

    def _load():
        resp = requests.get(SLEEPER_STATE_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    state = safe_load("Sleeper league state", SLEEPER_STATE_URL, _load, log=log)
    return state or {}
