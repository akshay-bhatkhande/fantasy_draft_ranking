"""Market data: ADP and expert consensus rankings.

ADP feeds the 10% market-signal component of the Composite Z-Score (STEP 2) and the
per-slot "likely available at your next pick" logic. ECR is a STEP 5 sanity check only and
never re-enters the math.

Source choice for ADP: Fantasy Football Calculator's public API, because it can be queried
for exactly this league's format (10 teams, PPR) and returns per-player `stdev`, `high` and
`low` -- which is what the market-disagreement flag needs. Sleeper has no public ADP
endpoint (its GraphQL schema has no `adp` field, and `search_rank` is a non-unique
popularity value), so it is not used for ADP. A manual multi-platform override file is
supported for anyone who wants to paste in Underdog/ESPN/Yahoo numbers by hand.
"""

from __future__ import annotations

import pandas as pd
import requests

from config import weights as W
from .cache import (
    MANUAL_OVERRIDES_DIR,
    SourceLog,
    cache_age_days,
    http_cache_get,
    http_cache_set,
    safe_load,
)
from .nflverse import _pd

FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{scoring}"
FFC_CACHE_MAX_AGE_SECONDS = 12 * 3600  # ADP does not move hour to hour
REQUEST_TIMEOUT = 30


# --------------------------------------------------------------------------------------
# ADP
# --------------------------------------------------------------------------------------


def _fetch_ffc(year: int, teams: int, scoring: str) -> dict:
    key = f"ffc_adp_{scoring}_{teams}team_{year}"
    cached = http_cache_get(key, FFC_CACHE_MAX_AGE_SECONDS)
    if cached is not None:
        return cached
    resp = requests.get(
        FFC_URL.format(scoring=scoring),
        params={"teams": teams, "year": year},
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "fantasy_draft_ranking/1.0"},
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "Success" or not payload.get("players"):
        raise ValueError(f"FFC returned no players for {year} ({payload.get('status')!r})")
    http_cache_set(key, payload)
    return payload


def fetch_adp(
    year: int,
    teams: int = 10,
    scoring: str = "ppr",
    log: SourceLog | None = None,
) -> pd.DataFrame:
    """Fetch format-matched ADP, falling back one season if the new year has no drafts yet.

    Early in an offseason the upcoming year's ADP may not exist. Rather than fabricate
    values we fall back to the prior year's final ADP and mark it degraded, so the Notes
    column can say so.
    """

    def _load():
        try:
            payload = _fetch_ffc(year, teams, scoring)
            fallback = False
        except Exception:
            payload = _fetch_ffc(year - 1, teams, scoring)
            fallback = True
        df = pd.DataFrame(payload["players"])
        df["adp_meta_teams"] = payload.get("meta", {}).get("teams")
        df["adp_meta_start"] = payload.get("meta", {}).get("start_date")
        df["adp_meta_end"] = payload.get("meta", {}).get("end_date")
        df["adp_total_drafts"] = payload.get("meta", {}).get("total_drafts")
        df["adp_is_prior_year_fallback"] = fallback
        return df

    df = safe_load(
        "Fantasy Football Calculator ADP",
        f"{scoring.upper()} / {teams}-team / {year}",
        _load,
        log=log,
    )
    df = _pd(df)
    if df.empty:
        return df

    df = df.rename(columns={"name": "adp_name", "position": "adp_position", "team": "adp_team"})
    df["adp_stdev"] = pd.to_numeric(df.get("stdev"), errors="coerce")
    df["adp"] = pd.to_numeric(df["adp"], errors="coerce")
    df["adp_high"] = pd.to_numeric(df.get("high"), errors="coerce")
    df["adp_low"] = pd.to_numeric(df.get("low"), errors="coerce")
    df["adp_spread"] = df["adp_low"] - df["adp_high"]
    df["adp_source"] = "FFC " + scoring.upper() + f" {teams}-team"
    return df


def load_manual_adp_override(log: SourceLog | None = None) -> pd.DataFrame:
    """Optional hand-maintained multi-platform ADP.

    A true blended ADP across Underdog/ESPN/Yahoo/NFC needs either paid API access or
    scraping sites whose terms discourage it, so this is the sanctioned manual path: paste
    2-3 platforms' numbers into manual_overrides/adp_manual.csv once at setup time.

    Expected columns: player (required), plus any of adp_underdog / adp_espn / adp_yahoo /
    adp_nfc / adp_other. Whatever is present is averaged with the FFC value.
    """
    path = MANUAL_OVERRIDES_DIR / "adp_manual.csv"
    if not path.exists():
        return pd.DataFrame()

    def _load():
        return pd.read_csv(path, comment="#")

    df = safe_load("manual ADP override", str(path.name), _load, log=log)
    return _pd(df)


def blend_adp(ffc: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Average FFC ADP with any manually supplied platform columns.

    The spread ACROSS SOURCES is tracked separately as a market-disagreement signal and is
    deliberately never folded into the Composite Z-Score.
    """
    if ffc.empty:
        return ffc
    out = ffc.copy()
    out["adp_sources_used"] = 1
    out["adp_blended"] = out["adp"]

    if manual.empty or "player" not in manual.columns:
        out["adp_cross_source_spread"] = pd.NA
        return out

    manual = manual.copy()
    manual["_join"] = manual["player"].astype(str).str.strip().str.lower()
    out["_join"] = out["adp_name"].astype(str).str.strip().str.lower()
    platform_cols = [c for c in manual.columns if c.startswith("adp_") and c != "adp_name"]
    if not platform_cols:
        out["adp_cross_source_spread"] = pd.NA
        return out.drop(columns=["_join"])

    merged = out.merge(manual[["_join", *platform_cols]], on="_join", how="left")
    all_cols = ["adp", *platform_cols]
    vals = merged[all_cols].apply(pd.to_numeric, errors="coerce")
    merged["adp_blended"] = vals.mean(axis=1, skipna=True)
    merged["adp_sources_used"] = vals.notna().sum(axis=1)
    merged["adp_cross_source_spread"] = vals.max(axis=1) - vals.min(axis=1)
    return merged.drop(columns=["_join"])


def adp_cache_age_days(year: int, teams: int = 10, scoring: str = "ppr") -> float | None:
    return cache_age_days(f"ffc_adp_{scoring}_{teams}team_{year}")


# --------------------------------------------------------------------------------------
# Expert consensus (STEP 5 -- sanity check only)
# --------------------------------------------------------------------------------------

# The dynastyprocess ECR file bundles many ranking sets in one table: redraft, dynasty,
# best-ball and IDP. Filtering is mandatory -- comparing our redraft board against dynasty
# consensus would produce plausible-looking but meaningless delta flags.
ECR_REDRAFT_PAGE_TYPE = "redraft-overall"
ECR_REDRAFT_TYPE = "ro"


def load_ecr_redraft(log: SourceLog | None = None) -> pd.DataFrame:
    """FantasyPros redraft ECR via nflreadpy's dynastyprocess mirror.

    Real ECR, free, no key. Despite a stale docstring in nflreadpy warning about .rds
    files, load_ff_rankings("draft") explicitly requests CSV, so there is no R dependency.
    """
    import nflreadpy as nfl

    def _load():
        return nfl.load_ff_rankings("draft")

    df = safe_load("FantasyPros ECR (dynastyprocess mirror)", "load_ff_rankings('draft')", _load, log=log)
    df = _pd(df)
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)
    if "page_type" in df.columns:
        mask &= df["page_type"] == ECR_REDRAFT_PAGE_TYPE
    if "ecr_type" in df.columns:
        mask &= df["ecr_type"] == ECR_REDRAFT_TYPE
    out = df[mask].copy()

    out["ecr"] = pd.to_numeric(out.get("ecr"), errors="coerce")
    out = out.dropna(subset=["ecr"]).sort_values("ecr")
    # ECR is an average rank with ties/decimals; give it a clean integer ordering to
    # compare rank-vs-rank against our board.
    out["consensus_rank"] = range(1, len(out) + 1)
    out["ecr_scrape_date"] = out.get("scrape_date")
    return out


def ecr_flag_threshold() -> int:
    return W.ECR_DELTA_FLAG_THRESHOLD
