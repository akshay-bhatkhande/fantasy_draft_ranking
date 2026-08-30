"""STEP 5 -- Expert consensus rankings as a SANITY CHECK ONLY.

ECR is pulled separately and used purely as a comparison column. It never feeds the Composite
Z-Score, Final Projected Season Points, or VORP -- it is a downstream check, full stop.

Any player whose Overall Rank differs from ECR by more than the configured threshold (default
15 spots) is flagged with a one-line auto-generated reason pointing at whichever input actually
drove the difference, so a disagreement is explainable rather than mysterious.
"""

from __future__ import annotations

import pandas as pd

from config import weights as W

# Component columns inspected when explaining a disagreement, with human-readable labels.
DRIVER_LABELS = {
    "z_opportunity": "opportunity share",
    "z_efficiency": "efficiency",
    "z_situational": "situational context",
    "z_ppg": "recent per-game production",
    "z_adp": "market signal (ADP)",
}


def attach_consensus(df: pd.DataFrame, ecr: pd.DataFrame) -> pd.DataFrame:
    """Join ECR onto the board by normalised name and compute the delta versus our rank."""
    out = df.copy()
    out["consensus_rank"] = pd.NA
    out["consensus_source"] = ""
    out["ecr_delta"] = pd.NA
    out["ecr_flag"] = ""
    out["ecr_reason"] = ""

    if ecr.empty or "name_key" not in ecr.columns:
        out["consensus_source"] = "insufficient data: ECR unavailable"
        return out

    lookup = ecr.drop_duplicates(subset=["name_key"]).set_index("name_key")
    out["consensus_rank"] = out["name_key"].map(lookup["consensus_rank"]).astype("Int64")
    scrape = lookup["ecr_scrape_date"].dropna()
    scrape_date = scrape.iloc[0] if len(scrape) else "unknown date"
    out.loc[out["consensus_rank"].notna(), "consensus_source"] = (
        f"FantasyPros redraft ECR via dynastyprocess, scraped {scrape_date} (sanity check only)"
    )

    my_rank = pd.to_numeric(out["overall_rank"], errors="coerce")
    their_rank = pd.to_numeric(out["consensus_rank"], errors="coerce")
    out["ecr_delta"] = (their_rank - my_rank).astype("Int64")
    return out


def flag_threshold_for_rank(rank, base: int | None = None) -> float:
    """How many spots of disagreement count as meaningful at a given depth.

    Scales with rank because VORP density does. Near the top, 15 spots separates tiers; at rank
    200 the same 15 spots separates two players who are both far below replacement.
    """
    base = W.ECR_DELTA_FLAG_THRESHOLD if base is None else base
    if rank is None or pd.isna(rank):
        return float(base)
    return float(min(base + W.ECR_DELTA_FLAG_PER_RANK * float(rank), W.ECR_DELTA_FLAG_MAX))


def explain_disagreement(row, threshold: int | None = None) -> tuple[str, str]:
    """Build the flag and one-line reason for a player who diverges from consensus.

    The reason names the component z-score that is furthest from average, since that is the
    input actually responsible for the disagreement.
    """
    delta = row.get("ecr_delta")
    if delta is None or pd.isna(delta):
        return "", ""

    # Below replacement there is no decision to inform, so a disagreement is not worth surfacing.
    if W.ECR_FLAG_REQUIRE_ABOVE_REPLACEMENT:
        vorp = row.get("vorp")
        if vorp is not None and not pd.isna(vorp) and float(vorp) <= 0:
            return "", ""

    effective = (
        flag_threshold_for_rank(row.get("overall_rank")) if threshold is None else float(threshold)
    )
    if abs(delta) <= effective:
        return "", ""

    higher_than_consensus = delta > 0  # consensus ranks him later => we like him more

    drivers = []
    for col, label in DRIVER_LABELS.items():
        val = row.get(col)
        if val is None or pd.isna(val):
            continue
        # A component only counts as a driver if it points the same way as the disagreement.
        if higher_than_consensus and val > 0.5:
            drivers.append((abs(float(val)), label))
        elif not higher_than_consensus and val < -0.5:
            drivers.append((abs(float(val)), label))
    drivers.sort(reverse=True)

    # Some common causes are not component z-scores, so they are checked explicitly. These are
    # DIRECTIONAL: a games-played discount and the personal-preference penalty can only push a
    # player DOWN, so they must never be offered as evidence for ranking someone higher.
    downward = []
    exp_games = row.get("expected_games_played")
    if exp_games is not None and not pd.isna(exp_games) and float(exp_games) < 16.0:
        downward.append(f"Expected Games Played discount ({float(exp_games):.1f} of 17)")
    if str(row.get("team_bias_flag", "N")) == "Y":
        downward.append("the personal-preference team penalty")
    if str(row.get("player_fade_flag", "N")) == "Y":
        downward.append("a personal player fade")

    upward = []
    camp = row.get("camp_buzz_score")
    if camp is not None and not pd.isna(camp) and float(camp) != 0:
        if float(camp) > 0:
            upward.append(f"positive camp buzz ({float(camp):+.0f})")
        else:
            downward.append(f"negative camp buzz ({float(camp):+.0f})")

    if higher_than_consensus:
        causes = [d[1] for d in drivers[:2]] + upward[:1]
        if causes:
            reason = f"Ranked higher than consensus due to elevated {', '.join(causes)}"
        else:
            reason = "Ranked higher than consensus on a combination of model inputs"
        return f"Higher than consensus by {int(abs(delta))}", reason

    causes = [d[1] for d in drivers[:2]] + downward[:2]
    if causes:
        reason = f"Ranked lower than consensus due to {', '.join(causes)}"
    else:
        reason = "Ranked lower than consensus on a combination of model inputs"
    return f"Lower than consensus by {int(abs(delta))}", reason


def run_step5(df: pd.DataFrame, ecr: pd.DataFrame) -> pd.DataFrame:
    """Attach ECR, compute deltas and generate flags/reasons for large disagreements."""
    out = attach_consensus(df, ecr)
    results = out.apply(explain_disagreement, axis=1)
    out["ecr_flag"] = [r[0] for r in results]
    out["ecr_reason"] = [r[1] for r in results]
    return out
