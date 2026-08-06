"""Injury Risk Score and buckets, feeding Expected Games Played in STEP 3c.

This is a probabilistic reliability discount built from history. It is NOT a multiplier on
PPG, and it is NOT the same thing as a known current-season absence (that is a specific
announced fact, loaded from manual_overrides/known_absences.csv and subtracted separately).

Games missed are attributed to injury only when the absence coincides with an injury-report
designation that week. Counting every absence would punish backups who were simply inactive,
which has nothing to do with durability.

Games missed are recency-weighted 55/30/15 like STEP 1, and weighted more heavily for
recurring soft-tissue injuries (hamstring, groin, calf) than for one-off trauma, because soft
tissue issues recur at a statistically higher rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import weights as W

OUT_STATUSES = {"Out", "Doubtful", "Injured Reserve", "IR", "PUP"}


def _injury_severity_weight(text: str) -> tuple[float, str]:
    """Map an injury description to a recurrence-risk weight and a category label."""
    if not text or pd.isna(text):
        return W.UNKNOWN_INJURY_WEIGHT, "unspecified"
    low = str(text).lower()
    for kw in W.SOFT_TISSUE_KEYWORDS:
        if kw in low:
            return W.SOFT_TISSUE_WEIGHT, "soft tissue"
    for kw in W.TRAUMA_KEYWORDS:
        if kw in low:
            return W.TRAUMA_WEIGHT, "trauma"
    return W.UNKNOWN_INJURY_WEIGHT, "other"


def compute_injury_history(
    weekly: pd.DataFrame,
    injuries: pd.DataFrame,
    schedules: pd.DataFrame,
    target_season: int,
    recency_weights: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Per-player injury history, bucketed Low/Med/High.

    Returns player_id, injury_risk_score (severity-weighted, recency-weighted share of games
    missed), injury_risk_bucket, expected_games_missed_from_bucket, and a description.
    """
    recency_weights = recency_weights or W.RECENCY_WEIGHTS
    cols = [
        "player_id", "injury_risk_score", "injury_risk_bucket",
        "expected_games_missed_from_bucket", "injury_history_note",
    ]
    if weekly.empty:
        return pd.DataFrame(columns=cols)

    # Weeks each team actually played, per season, so a bye week is never counted as missed.
    if not schedules.empty:
        s = schedules[schedules.get("game_type", "REG") == "REG"]
        home = s[["season", "week", "home_team"]].rename(columns={"home_team": "team"})
        away = s[["season", "week", "away_team"]].rename(columns={"away_team": "team"})
        team_weeks = pd.concat([home, away], ignore_index=True)
        team_game_counts = team_weeks.groupby(["team", "season"]).size().rename("team_games")
    else:
        team_game_counts = pd.Series(dtype=int, name="team_games")

    played = (
        weekly.groupby(["player_id", "season"])
        .agg(games_played=("week", "nunique"), team=("team", "last"))
        .reset_index()
    )
    played = played.merge(team_game_counts, on=["team", "season"], how="left")
    played["team_games"] = played["team_games"].fillna(17)

    # Whether the player carried an injury designation at any point in the season.
    #
    # Counting only weeks explicitly marked Out badly under-attributes absences, because a
    # player placed on injured reserve DROPS OFF the weekly injury report entirely -- the
    # status vocabulary in this dataset is only {Questionable, Out, Doubtful, Note}, with no
    # IR or PUP value. Christian McCaffrey's 2024 is the clean example: he played 4 games, yet
    # appears as "Out" for exactly one week, so a week-by-week match rated him Low risk.
    #
    # So the season is treated as injury-affected if the player was designated Out/Doubtful,
    # or listed with a specific injury, or did not participate in practice at any point. All
    # of that season's absences are then attributed to injury.
    if not injuries.empty and "report_status" in injuries.columns:
        inj = injuries.copy()
        status = inj["report_status"].fillna("")
        has_named_injury = inj.get("report_primary_injury", pd.Series(index=inj.index, dtype=object)).notna()
        dnp = inj.get("practice_status", pd.Series(index=inj.index, dtype=object)).fillna("").str.contains(
            "Did Not Participate", case=False, na=False
        )
        flagged = inj[status.isin(OUT_STATUSES) | has_named_injury | dnp]
        injury_flags = (
            flagged.groupby(["gsis_id", "season"])
            .agg(
                injury_weeks_flagged=("week", "nunique"),
                primary=(
                    "report_primary_injury",
                    lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None,
                ),
            )
            .reset_index()
            .rename(columns={"gsis_id": "player_id"})
        )
    else:
        injury_flags = pd.DataFrame(columns=["player_id", "season", "injury_weeks_flagged", "primary"])

    hist = played.merge(injury_flags, on=["player_id", "season"], how="left")
    hist["injury_weeks_flagged"] = hist["injury_weeks_flagged"].fillna(0)

    hist["absences"] = (hist["team_games"] - hist["games_played"]).clip(lower=0)
    hist["injury_games_missed"] = np.where(
        hist["injury_weeks_flagged"] > 0, hist["absences"], 0.0
    )

    severity = hist["primary"].map(lambda t: _injury_severity_weight(t)[0])
    category = hist["primary"].map(lambda t: _injury_severity_weight(t)[1])
    hist["severity_weight"] = severity
    hist["injury_category"] = category
    hist["weighted_missed"] = hist["injury_games_missed"] * hist["severity_weight"]

    hist["seasons_ago"] = target_season - hist["season"]
    hist = hist[hist["seasons_ago"].isin(recency_weights.keys())].copy()
    if hist.empty:
        return pd.DataFrame(columns=cols)
    hist["w"] = hist["seasons_ago"].map(recency_weights).astype(float)

    # Recency-weighted share of games missed: weighted missed games over weighted team games.
    wsum = hist.groupby("player_id")["w"].transform("sum")
    hist["applied_w"] = hist["w"] / wsum
    agg = hist.groupby("player_id").apply(
        lambda g: pd.Series(
            {
                "injury_risk_score": float(
                    (g["weighted_missed"] * g["applied_w"]).sum()
                    / max((g["team_games"] * g["applied_w"]).sum(), 1e-9)
                ),
                "raw_missed": float(g["injury_games_missed"].sum()),
                "categories": ", ".join(sorted({c for c in g["injury_category"] if c and c != "unspecified"})),
                "seasons": int(g["season"].nunique()),
            }
        ),
        include_groups=False,
    ).reset_index()

    def _bucket(score: float) -> str:
        if score < W.INJURY_RISK_LOW_MAX:
            return "Low"
        if score <= W.INJURY_RISK_MED_MAX:
            return "Med"
        return "High"

    agg["injury_risk_bucket"] = agg["injury_risk_score"].map(_bucket)
    agg["expected_games_missed_from_bucket"] = agg["injury_risk_bucket"].map(
        W.INJURY_RISK_EXPECTED_GAMES_MISSED
    )

    def _note(row) -> str:
        base = (
            f"Injury risk {row['injury_risk_bucket']}: {row['injury_risk_score'] * 100:.1f}% "
            f"severity-weighted games missed over {int(row['seasons'])} season(s) "
            f"({row['raw_missed']:.0f} games)"
        )
        if row["categories"]:
            base += f"; injuries: {row['categories']}"
            if "soft tissue" in row["categories"]:
                base += " (soft tissue weighted higher for recurrence risk)"
        return base

    agg["injury_history_note"] = agg.apply(_note, axis=1)
    return agg[cols]


def expected_games_played(
    known_games_missed: float,
    bucket_games_missed: float,
    games_in_season: int,
    min_games: float | None = None,
) -> float:
    """STEP 3c: a literal games count, not a multiplier on PPG.

        Expected Games Played = 17 - known current-season games missed
                                   - probabilistic games from the injury risk bucket

    Missed games work this way in reality, which is exactly why this is kept separate from
    the PPG multipliers in STEP 3b.
    """
    min_games = W.MIN_EXPECTED_GAMES if min_games is None else min_games
    known = 0.0 if known_games_missed is None or pd.isna(known_games_missed) else float(known_games_missed)
    prob = 0.0 if bucket_games_missed is None or pd.isna(bucket_games_missed) else float(bucket_games_missed)
    return float(max(games_in_season - known - prob, min_games))
