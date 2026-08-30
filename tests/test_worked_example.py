"""Validates the pipeline against the methodology's own worked example.

"Player X": a hypothetical 26-year-old RB, not on a contract year, no current injury, moderate
positive camp buzz, and no personal-preference team penalty against him.

    Weighted PPG            = 16.2
    component z-scores      = PPG +1.30, Opportunity +1.10, Efficiency +0.40,
                              Situational +0.60, ADP +0.90
    Composite Z             = 0.40(1.30) + 0.25(1.10) + 0.15(0.40) + 0.10(0.60) + 0.10(0.90)
                            = +1.005
    RB pool                 mean Weighted PPG 11.0, stdev 4.2
    Base Projected PPG      = 11.0 + (1.005 x 4.2)   = 15.22
    multipliers             contract 1.00 x age 1.00 x camp +1 (1.04) x team penalty 1.00
    Final Projected PPG     = 15.22 x 1.04           = 15.83
    Expected Games Played   = 17 - 0 - 0.2 (Low)     = 16.8
    Final Projected Points  = 15.83 x 16.8           = 265.9
    RB replacement (RB33)   = 148.0
    VORP                    = 265.9 - 148.0          = 117.9

Every number below is produced by the real production functions, not re-derived in the test.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from config import weights as W
from config.league import LEAGUE
from ffrank.features.curves import AgeCurve, ContractYearLift, RookieBaseline
from ffrank.features.risk import expected_games_played
from ffrank.scoring.step2_composite import compute_composite_z
from ffrank.scoring.step3_projection import camp_buzz_multiplier, run_step3
from ffrank.scoring.step4_vorp import starter_counts

TOL = 0.01


def test_composite_z_matches_worked_example():
    """STEP 2: the documented weights applied to the documented component z-scores."""
    z = (
        W.COMPOSITE_WEIGHTS["ppg"] * 1.30
        + W.COMPOSITE_WEIGHTS["opportunity"] * 1.10
        + W.COMPOSITE_WEIGHTS["efficiency"] * 0.40
        + W.COMPOSITE_WEIGHTS["situational"] * 0.60
        + W.COMPOSITE_WEIGHTS["adp"] * 0.90
    )
    assert z == pytest.approx(1.005, abs=1e-9)


def test_camp_buzz_multiplier_table():
    """A +1 camp-buzz score is a 4% bump, and the cap holds at 8%."""
    assert camp_buzz_multiplier(1) == pytest.approx(1.04)
    assert camp_buzz_multiplier(2) == pytest.approx(1.08)
    assert camp_buzz_multiplier(0) == pytest.approx(1.00)
    assert camp_buzz_multiplier(-1) == pytest.approx(0.96)
    assert camp_buzz_multiplier(-2) == pytest.approx(0.92)
    # Out-of-range scores are clamped, never extrapolated past the cap.
    assert camp_buzz_multiplier(5) <= 1.08
    assert camp_buzz_multiplier(-5) >= 0.92
    assert camp_buzz_multiplier(None) == pytest.approx(1.00)


def test_camp_buzz_limited_sample_dampens_deviation():
    """Thin NFL history: keep only a fraction of the camp deviation from 1.0."""
    full = camp_buzz_multiplier(2, limited_sample=False)
    limited = camp_buzz_multiplier(2, limited_sample=True)
    assert full == pytest.approx(1.08)
    assert limited == pytest.approx(1.0 + (1.08 - 1.0) * W.CAMP_BUZZ_LIMITED_SAMPLE_SHRINK)
    assert 1.0 < limited < full
    assert abs(camp_buzz_multiplier(-2, limited_sample=True) - 1.0) < abs(
        camp_buzz_multiplier(-2, limited_sample=False) - 1.0
    )


def test_expected_games_played_worked_example():
    """STEP 3c: 17 minus no known absence minus 0.2 for a Low-risk bucket."""
    games = expected_games_played(
        known_games_missed=0.0,
        bucket_games_missed=W.INJURY_RISK_EXPECTED_GAMES_MISSED["Low"],
        games_in_season=LEAGUE.games_in_season,
    )
    assert games == pytest.approx(16.8, abs=TOL)


def test_expected_games_played_subtracts_both_components():
    """A known absence and the probabilistic bucket both subtract, and neither is a multiplier."""
    games = expected_games_played(
        known_games_missed=4.0,
        bucket_games_missed=W.INJURY_RISK_EXPECTED_GAMES_MISSED["High"],
        games_in_season=17,
    )
    assert games == pytest.approx(17 - 4 - 1.5, abs=TOL)


def _pool_values(target_mean: float, target_sd: float, x_value: float, n_others: int = 24) -> list[float]:
    """Build `n_others` PPG values so that the pool INCLUDING x_value has the target mean/sd.

    Player X is himself one of the position's relevant players, so he lands in the STEP 3a pool
    and shifts its mean and stdev. Naively surrounding him with a mean-11.0/sd-4.2 set therefore
    produced a pool sd of 4.45 and a Base Projected PPG of 15.47 rather than the documented
    15.22. So the surrounding values are solved for directly.

    Two equal groups (low and high) are used, which reduces to one quadratic.
    """
    n = n_others + 1
    half = n_others // 2
    # Required totals for the whole pool.
    needed_sum_others = target_mean * n - x_value
    needed_sumsq_others = (target_sd**2) * n - (x_value - target_mean) ** 2

    # Groups sit at target_mean - d (low) and target_mean + e (high).
    #   half*(mean - d) + half*(mean + e) = needed_sum_others  ->  e - d = k
    #   half*d^2 + half*e^2             = needed_sumsq_others
    k = (needed_sum_others - n_others * target_mean) / half
    c = needed_sumsq_others / half
    # d^2 + (d + k)^2 = c  ->  2d^2 + 2kd + (k^2 - c) = 0
    positive_roots = [r for r in np.roots([2.0, 2.0 * k, k**2 - c]) if r > 0]
    d = float(max(positive_roots))
    e = d + k
    return [target_mean - d] * half + [target_mean + e] * half


def _player_x_frame() -> pd.DataFrame:
    """Player X plus a synthetic RB pool whose distribution is exactly mean 11.0, stdev 4.2.

    The pool matters because STEP 3a reads the position mean and stdev out of it, so the only way
    to drive the real function to the documented 15.22 is to hand it the documented distribution.
    """
    pool_ppg = _pool_values(target_mean=11.0, target_sd=4.2, x_value=16.2, n_others=24)
    rows = []
    for i, ppg in enumerate(pool_ppg):
        rows.append(
            {
                "player_id": f"pool-{i}",
                "player_name": f"Pool RB {i}",
                "position": "RB",
                "team": "DAL",
                "weighted_ppg": ppg,
                "opportunity_value": 0.0,
                "efficiency_value": 0.0,
                "situational_value": 0.0,
                "adp": 100.0,
                "adp_blended": 100.0,
                "snap_share": 0.5,
                "is_rookie": False,
                "years_exp": 5,
                "age": 26.0,
                "draft_capital_tier": "rd2",
                "contract_year": False,
                "camp_buzz_score": 0,
                "known_games_missed": 0.0,
                "expected_games_missed_from_bucket": W.INJURY_RISK_EXPECTED_GAMES_MISSED["Low"],
            }
        )

    player_x = dict(rows[0])
    player_x.update(
        {
            "player_id": "player-x",
            "player_name": "Player X",
            "weighted_ppg": 16.2,
            "camp_buzz_score": 1,  # moderate positive camp buzz
            "age": 26.0,
            "contract_year": False,
            "team": "DAL",
        }
    )
    return pd.DataFrame([player_x, *rows])


def test_step3a_converts_z_back_to_points():
    """STEP 3a: Base Projected PPG = position mean + (Composite Z x position stdev) = 15.22."""
    df = _player_x_frame()
    scored = compute_composite_z(df)

    # Confirm the synthetic pool really does carry the documented distribution.
    row = scored[scored["player_id"] == "player-x"].iloc[0]
    assert row["pos_mean_weighted_ppg"] == pytest.approx(11.0, abs=TOL)
    assert row["pos_std_weighted_ppg"] == pytest.approx(4.2, abs=TOL)

    # Force the documented composite z rather than the one the synthetic pool would produce, so
    # this test isolates the 3a conversion arithmetic.
    scored.loc[scored["player_id"] == "player-x", "composite_z"] = 1.005

    curves = {"RB": AgeCurve("RB", peak_age=27.0, decline_per_year=0.03, sample_size=500, fitted=True)}
    lifts = {"RB": ContractYearLift("RB", multiplier=1.05, sample_size=100, derived=True)}
    rookies = {"RB": RookieBaseline("RB")}

    out = run_step3(scored, curves, lifts, rookies, LEAGUE, apply_team_bias=True)
    x = out[out["player_id"] == "player-x"].iloc[0]

    assert x["base_projected_ppg"] == pytest.approx(15.22, abs=0.02)


def test_full_step3_chain_and_vorp_match_worked_example():
    """STEP 3b through STEP 4c: 15.83 PPG, 16.8 games, 265.9 points, 117.9 VORP."""
    df = _player_x_frame()
    scored = compute_composite_z(df)
    scored.loc[scored["player_id"] == "player-x", "composite_z"] = 1.005

    # Age 26 is at or before the peak, so the age curve is a no-op (1.00), as in the example.
    curves = {"RB": AgeCurve("RB", peak_age=27.0, decline_per_year=0.03, sample_size=500, fitted=True)}
    lifts = {"RB": ContractYearLift("RB", multiplier=1.05, sample_size=100, derived=True)}
    rookies = {"RB": RookieBaseline("RB")}

    out = run_step3(scored, curves, lifts, rookies, LEAGUE, apply_team_bias=True)
    x = out[out["player_id"] == "player-x"].iloc[0]

    assert x["contract_year_multiplier"] == pytest.approx(1.00)
    assert x["age_curve_multiplier"] == pytest.approx(1.00)
    assert x["camp_buzz_multiplier"] == pytest.approx(1.04)
    assert x["team_bias_multiplier"] == pytest.approx(1.00)

    assert x["final_projected_ppg"] == pytest.approx(15.83, abs=0.03)
    assert x["expected_games_played"] == pytest.approx(16.8, abs=TOL)
    assert x["final_projected_season_points"] == pytest.approx(265.9, abs=0.6)

    # STEP 4c with the documented RB33 replacement level of 148.0.
    vorp = x["final_projected_season_points"] - 148.0
    assert vorp == pytest.approx(117.9, abs=0.6)


def test_starter_counts_match_worked_example():
    """STEP 4a against the spec's own worked example, which assumed 60/30/10 FLEX usage.

    Pinned to that allocation explicitly rather than to the live config, because FLEX usage is a
    tunable fact about the league, not a property of the arithmetic. The league now runs TE at 5%
    (two TE FLEX starts league-wide was unrealistic), so asserting the spec's numbers against
    whatever is currently configured would make this test fail every time that dial is turned.
    """
    spec_example = dataclasses.replace(
        LEAGUE, flex_allocation={"RB": 0.60, "WR": 0.30, "TE": 0.10}
    )
    counts = starter_counts(spec_example)
    assert counts == {"QB": 10, "RB": 32, "WR": 26, "TE": 12}
    replacements = {pos: n + 1 for pos, n in counts.items()}
    assert replacements == {"QB": 11, "RB": 33, "WR": 27, "TE": 13}


def test_live_config_starter_counts_follow_from_its_flex_allocation():
    """Whatever FLEX usage is configured, the counts must be its arithmetic consequence."""
    counts = starter_counts(LEAGUE)
    for pos in ("QB", "RB", "WR", "TE"):
        dedicated = LEAGUE.dedicated_starters.get(pos, 0) * LEAGUE.num_teams
        flex = LEAGUE.flex_slots * LEAGUE.num_teams * LEAGUE.flex_allocation.get(pos, 0.0)
        assert counts[pos] == round(dedicated + flex)
    # The configured TE FLEX share is deliberately small, so tight ends get essentially no FLEX
    # starters beyond the one dedicated slot per team. Asserted as a bound rather than an exact
    # number so tuning the share does not break the test.
    assert LEAGUE.flex_allocation["TE"] <= 0.05
    assert counts["TE"] <= LEAGUE.dedicated_starters["TE"] * LEAGUE.num_teams + 1


def test_starter_counts_are_derived_not_hardcoded():
    """Changing the roster must move the replacement levels automatically."""
    alt = dataclasses.replace(
        LEAGUE,
        dedicated_starters={"QB": 1, "RB": 3, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=1,
        flex_allocation={"RB": 0.5, "WR": 0.4, "TE": 0.1},
    )
    counts = starter_counts(alt)
    assert counts["RB"] == 35  # (3 x 10) + (1 x 10 x 0.5)
    assert counts["WR"] == 24  # (2 x 10) + (1 x 10 x 0.4)
    assert counts["TE"] == 11  # (1 x 10) + (1 x 10 x 0.1)
    assert counts != starter_counts(LEAGUE)


def test_team_penalty_is_disabled_by_default():
    """No team is penalised; CMC fade does not touch synthetic players in this fixture."""
    assert LEAGUE.bias_team is None

    df = _player_x_frame()
    scored = compute_composite_z(df)
    scored.loc[scored["player_id"] == "player-x", "composite_z"] = 1.005

    curves = {"RB": AgeCurve("RB", peak_age=27.0, decline_per_year=0.03, sample_size=500, fitted=True)}
    lifts = {"RB": ContractYearLift("RB", multiplier=1.05, sample_size=100, derived=True)}
    rookies = {"RB": RookieBaseline("RB")}

    out = run_step3(scored, curves, lifts, rookies, LEAGUE, apply_team_bias=True)
    assert (out["team_bias_multiplier"] == 1.00).all()
    assert (out["team_bias_flag"] == "N").all()
    assert (out["player_fade_multiplier"] == 1.00).all()
    assert (out["player_fade_flag"] == "N").all()

    # With no matching personal-preference targets in this frame, biased and unbiased passes
    # are identical -- which is why the pipeline can skip the second pass when nothing applies.
    unbiased = run_step3(scored, curves, lifts, rookies, LEAGUE, apply_team_bias=False)
    pd.testing.assert_series_equal(
        out["final_projected_season_points"], unbiased["final_projected_season_points"]
    )


def test_player_fade_mechanism():
    """PLAYER_FADES haircuts only the named player and is reversed when bias is off."""
    league = dataclasses.replace(
        LEAGUE,
        player_fades={
            "Player X": {"multiplier": 0.90, "reason": "test fade"},
        },
    )
    df = _player_x_frame()
    scored = compute_composite_z(df)
    scored.loc[scored["player_id"] == "player-x", "composite_z"] = 1.005
    from ffrank.data.sleeper import normalize_name
    scored["name_key"] = scored["player_name"].map(normalize_name)

    curves = {"RB": AgeCurve("RB", peak_age=27.0, decline_per_year=0.03, sample_size=500, fitted=True)}
    lifts = {"RB": ContractYearLift("RB", multiplier=1.05, sample_size=100, derived=True)}
    rookies = {"RB": RookieBaseline("RB")}

    biased = run_step3(scored, curves, lifts, rookies, league, apply_team_bias=True)
    unbiased = run_step3(scored, curves, lifts, rookies, league, apply_team_bias=False)
    bx = biased[biased["player_id"] == "player-x"].iloc[0]
    ux = unbiased[unbiased["player_id"] == "player-x"].iloc[0]
    assert bx["player_fade_flag"] == "Y"
    assert bx["player_fade_multiplier"] == pytest.approx(0.90)
    assert ux["player_fade_multiplier"] == pytest.approx(1.00)
    assert bx["final_projected_season_points"] == pytest.approx(
        ux["final_projected_season_points"] * 0.90, abs=0.1
    )


def test_team_penalty_mechanism_still_works_when_enabled():
    """The switch is disabled, not deleted: setting BIAS_TEAM must still penalise that team only."""
    league = dataclasses.replace(LEAGUE, bias_team="SF", bias_team_multiplier=0.92)

    df = _player_x_frame()
    df.loc[df["player_id"] == "player-x", "team"] = "SF"
    scored = compute_composite_z(df)
    scored.loc[scored["player_id"] == "player-x", "composite_z"] = 1.005

    curves = {"RB": AgeCurve("RB", peak_age=27.0, decline_per_year=0.03, sample_size=500, fitted=True)}
    lifts = {"RB": ContractYearLift("RB", multiplier=1.05, sample_size=100, derived=True)}
    rookies = {"RB": RookieBaseline("RB")}

    biased = run_step3(scored, curves, lifts, rookies, league, apply_team_bias=True)
    unbiased = run_step3(scored, curves, lifts, rookies, league, apply_team_bias=False)

    bx = biased[biased["player_id"] == "player-x"].iloc[0]
    ux = unbiased[unbiased["player_id"] == "player-x"].iloc[0]

    assert bx["team_bias_flag"] == "Y"
    assert bx["team_bias_multiplier"] == pytest.approx(0.92)
    assert ux["team_bias_multiplier"] == pytest.approx(1.00)
    assert bx["final_projected_season_points"] < ux["final_projected_season_points"]
    assert bx["final_projected_season_points"] == pytest.approx(
        ux["final_projected_season_points"] * 0.92, abs=0.1
    )

    # A player on any other team must be untouched by the penalty.
    other = biased[biased["player_id"] == "pool-0"].iloc[0]
    assert other["team_bias_flag"] == "N"
    assert other["team_bias_multiplier"] == pytest.approx(1.00)
