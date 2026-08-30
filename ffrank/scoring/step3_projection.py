"""STEP 3 -- Convert the Composite Z-Score back into real points, then apply risk.

3a  Base Projected PPG = Position Mean Weighted PPG + (Composite Z x Position StdDev)
    Rookies bypass this: they have no Composite Z-Score history worth converting, so their
    Base Projected PPG comes from the draft-capital rookie baseline instead.

3b  Final Projected PPG = Base Projected PPG
                          x Contract-Year x Age-Curve x Camp-Buzz
                          x Team-Penalty x Player-Fade x RB-Workload
    Applied in exactly that order. Team and player fades are personal-preference adjustments
    and resolve to 1.00 when disabled / unmatched. RB-Workload uses expected offensive snap %
    (committee / bellcow estimate) and is 1.00 for non-RBs.

3c  Expected Games Played = 17 - known current-season games missed - probabilistic games
    from the injury-risk bucket. A literal games count, never a multiplier on PPG.

3d  Final Projected Season Points = Final Projected PPG x Expected Games Played

The whole step stays in fantasy points, so every intermediate number is auditable in a real
unit rather than an abstract score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import weights as W
from config.league import LeagueConfig
from ..data.sleeper import normalize_name
from ..features import curves as C
from ..features.risk import expected_games_played
from ..features.rb_role import rb_workload_multiplier


def base_projected_ppg(
    df: pd.DataFrame,
    rookie_baselines: dict[str, C.RookieBaseline],
) -> pd.DataFrame:
    """STEP 3a: turn the position-relative z-score back into a real points-per-game number.

    Requires pos_mean_weighted_ppg and pos_std_weighted_ppg (produced in STEP 2 from the same
    position pool), plus composite_z.

    Two groups bypass the z-score conversion and use the draft-capital baseline instead:

    * Rookies, who have no NFL record to convert.
    * Anyone else with no usable production history. This matters more than it sounds: a quarter
      of the board has no Weighted PPG at all, and only a seventh of those are rookies. The rest
      are mostly second-year players who barely played. Without this branch their Composite
      Z-Score is assembled from situational context and ADP alone, those two roughly cancel, and
      the player inherits something close to the positional MEAN -- so a camp-body back with no
      carries and no ADP was landing near 12 PPG and inside the top 100 overall, ahead of real
      starters. Draft capital is a far better prior for a player with no record than "average
      NFL starter" is.
    """
    out = df.copy()
    mean = pd.to_numeric(out["pos_mean_weighted_ppg"], errors="coerce")
    std = pd.to_numeric(out["pos_std_weighted_ppg"], errors="coerce")
    z = pd.to_numeric(out["composite_z"], errors="coerce")

    out["base_projected_ppg"] = mean + z * std
    out["base_ppg_method"] = "STEP 3a: position mean + (Composite Z x position stdev)"
    out["rookie_adjustment_applied"] = np.nan
    out["rookie_note"] = ""

    is_rookie = out["is_rookie"].fillna(False).astype(bool)
    # "No usable history" = STEP 1 produced nothing for him.
    no_history = pd.to_numeric(out["weighted_ppg"], errors="coerce").isna()
    use_draft_capital = is_rookie | no_history

    for idx in out.index[use_draft_capital]:
        pos = out.at[idx, "position"]
        baseline = rookie_baselines.get(pos)
        tier = out.at[idx, "draft_capital_tier"]
        pos_mean = mean.at[idx]
        if baseline is None or pd.isna(pos_mean):
            continue
        share, derived = baseline.share_for(tier)
        out.at[idx, "base_projected_ppg"] = pos_mean * share
        out.at[idx, "rookie_adjustment_applied"] = share
        if bool(is_rookie.at[idx]):
            out.at[idx, "base_ppg_method"] = "STEP 3a (rookie): position mean x draft-capital baseline"
            out.at[idx, "rookie_note"] = C.rookie_note(pos, tier, share, derived)
        else:
            out.at[idx, "base_ppg_method"] = (
                "STEP 3a (no production history): position mean x draft-capital baseline"
            )
            out.at[idx, "rookie_note"] = (
                f"No usable production history in the lookback window, so the Composite Z-Score "
                f"is not meaningful; projected from {tier} draft capital instead "
                f"({share:.2f}x the {pos} pool mean)"
            )

    # A projection below zero is not meaningful in a PPR format.
    out["base_projected_ppg"] = out["base_projected_ppg"].clip(lower=0.0)
    return out


def camp_buzz_multiplier(score, limited_sample: bool = False) -> float:
    """Camp-Buzz Multiplier from a -2..+2 score, hard-capped at +/-8%.

    The cap exists because camp buzz is the least statistically grounded input: it should
    nudge a player at most about one tier, never leapfrog him several tiers on its own.

    Limited-sample players (one season / no usable history) keep only a fraction of the
    deviation from 1.0 — same philosophy as the RB workload limited-sample shrink.
    """
    if score is None or pd.isna(score):
        return 1.00
    s = int(max(-2, min(2, round(float(score)))))
    mult = float(W.CAMP_BUZZ_MULTIPLIERS.get(s, 1.00))
    mult = float(min(max(mult, 1 - W.CAMP_BUZZ_MAX_ABS_EFFECT), 1 + W.CAMP_BUZZ_MAX_ABS_EFFECT))
    if limited_sample and mult != 1.0:
        shrink = float(W.CAMP_BUZZ_LIMITED_SAMPLE_SHRINK)
        mult = 1.0 + (mult - 1.0) * shrink
    return float(mult)


def apply_multipliers(
    df: pd.DataFrame,
    age_curves: dict[str, C.AgeCurve],
    contract_lifts: dict[str, C.ContractYearLift],
    league: LeagueConfig,
    apply_team_bias: bool = True,
) -> pd.DataFrame:
    """STEP 3b: apply the four multipliers, in order, to Base Projected PPG.

    Args:
        apply_team_bias: when False the personal-preference team penalty is forced to 1.00.
            The pipeline runs this twice -- once with the penalty and once without -- so the
            workbook can show the pre-penalty rank alongside the biased one.
    """
    out = df.copy()

    # --- Contract-Year multiplier (empirically derived per position) --------------------
    def _contract(row) -> float:
        if not bool(row.get("contract_year")):
            return 1.00
        lift = contract_lifts.get(row["position"])
        return float(lift.multiplier) if lift else 1.00

    out["contract_year_multiplier"] = out.apply(_contract, axis=1)
    out["contract_year_note"] = out.apply(
        lambda r: (contract_lifts[r["position"]].describe() if bool(r.get("contract_year")) and r["position"] in contract_lifts else ""),
        axis=1,
    )

    # --- Age-Curve multiplier (fitted, with the assumption used recorded) ---------------
    age_results = out.apply(
        lambda r: C.age_multiplier(r.get("age"), r.get("position"), r.get("years_exp"), age_curves),
        axis=1,
    )
    out["age_curve_multiplier"] = [a[0] for a in age_results]
    out["age_curve_note"] = [a[1] for a in age_results]

    # --- Camp-Buzz multiplier ----------------------------------------------------------
    limited = (
        out["limited_sample"].fillna(False).astype(bool)
        if "limited_sample" in out.columns
        else pd.Series(False, index=out.index)
    )
    out["camp_buzz_multiplier"] = [
        camp_buzz_multiplier(score, limited_sample=bool(lim))
        for score, lim in zip(out["camp_buzz_score"], limited)
    ]

    # --- Personal-preference team penalty ----------------------------------------------
    if apply_team_bias and league.bias_team:
        is_biased = out["team"] == league.bias_team
        out["team_bias_multiplier"] = np.where(is_biased, league.bias_team_multiplier, 1.00)
        out["team_bias_flag"] = np.where(is_biased, "Y", "N")
    else:
        out["team_bias_multiplier"] = 1.00
        out["team_bias_flag"] = "N"

    # --- Personal-preference player fades ----------------------------------------------
    fade_mult = pd.Series(1.00, index=out.index, dtype=float)
    fade_flag = pd.Series("N", index=out.index, dtype=object)
    fade_reason = pd.Series("", index=out.index, dtype=object)
    if apply_team_bias and league.player_fades:
        by_key = {
            normalize_name(name): spec
            for name, spec in league.player_fades.items()
        }
        keys = out["name_key"] if "name_key" in out.columns else out["player_name"].map(normalize_name)
        for idx, key in keys.items():
            spec = by_key.get(key)
            if not spec:
                continue
            fade_mult.at[idx] = float(spec.get("multiplier", 1.0))
            fade_flag.at[idx] = "Y"
            fade_reason.at[idx] = str(spec.get("reason", "Personal fade"))
    out["player_fade_multiplier"] = fade_mult
    out["player_fade_flag"] = fade_flag
    out["player_fade_reason"] = fade_reason

    # --- RB workload from expected snap % (committee / bellcow) -----------------------
    camp_m = pd.to_numeric(out["camp_buzz_multiplier"], errors="coerce").fillna(1.0)
    snaps = (
        out["expected_snap_pct"]
        if "expected_snap_pct" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    out["rb_workload_multiplier"] = [
        rb_workload_multiplier(snap, pos, limited_sample=bool(lim), camp_buzz_multiplier=float(camp))
        for snap, pos, lim, camp in zip(snaps, out["position"], limited, camp_m)
    ]

    out["final_projected_ppg"] = (
        pd.to_numeric(out["base_projected_ppg"], errors="coerce")
        * out["contract_year_multiplier"]
        * out["age_curve_multiplier"]
        * out["camp_buzz_multiplier"]
        * out["team_bias_multiplier"]
        * out["player_fade_multiplier"]
        * out["rb_workload_multiplier"]
    )
    return out


def apply_expected_games(df: pd.DataFrame, league: LeagueConfig) -> pd.DataFrame:
    """STEP 3c and 3d: games count, then Final Projected Season Points."""
    out = df.copy()
    out["expected_games_played"] = [
        expected_games_played(
            row.get("known_games_missed"),
            row.get("expected_games_missed_from_bucket"),
            league.games_in_season,
        )
        for _, row in out.iterrows()
    ]
    out["final_projected_season_points"] = (
        pd.to_numeric(out["final_projected_ppg"], errors="coerce") * out["expected_games_played"]
    )
    return out


def run_step3(
    df: pd.DataFrame,
    age_curves: dict[str, C.AgeCurve],
    contract_lifts: dict[str, C.ContractYearLift],
    rookie_baselines: dict[str, C.RookieBaseline],
    league: LeagueConfig,
    apply_team_bias: bool = True,
) -> pd.DataFrame:
    """Run STEP 3a through 3d in order."""
    out = base_projected_ppg(df, rookie_baselines)
    out = apply_multipliers(out, age_curves, contract_lifts, league, apply_team_bias=apply_team_bias)
    out = apply_expected_games(out, league)
    return out


def camp_flag(
    camp_score,
    my_rank,
    adp,
    num_teams: int,
) -> str:
    """Camp riser / faller flag.

    A player is a "camp riser" specifically when camp buzz is POSITIVE *and* his current ADP
    is later than the rank the rest of the pipeline implies -- that is, the market has not
    caught up yet. Any positive report on its own is not the interesting case, so it is not
    flagged as one. The faller condition is the mirror image.
    """
    if camp_score is None or pd.isna(camp_score) or int(camp_score) == 0:
        return "Neutral"
    if my_rank is None or pd.isna(my_rank) or adp is None or pd.isna(adp):
        return "Neutral"
    score = float(camp_score)
    # ADP is in picks; convert our overall rank to a comparable pick number.
    implied_pick = float(my_rank)
    market_pick = float(adp)
    if score > 0 and market_pick > implied_pick:
        return "Riser"
    if score < 0 and market_pick < implied_pick:
        return "Faller"
    return "Neutral"
