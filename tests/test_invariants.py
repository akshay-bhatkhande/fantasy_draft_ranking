"""Structural guarantees of the methodology that must never silently regress.

These are the rules that are easy to break with an innocent-looking refactor: VORP being the
only cross-position number, volatility staying out of the ranking, ECR staying out of the math,
tiers not degenerating into fixed-size buckets, and the participation season clamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import weights as W
from config.league import LEAGUE
from ffrank.data.nflverse import clamp_stat_seasons, max_stat_season
from ffrank.data.sleeper import normalize_name
from ffrank.features.opportunity import blend_role_expectation, derive_role_opportunity_priors
from ffrank.features.volatility import consistency_scores, same_ppg_volatility_flags
from ffrank.scoring.step4_vorp import (
    compute_vorp,
    detect_tiers_largest_gap,
    replacement_levels,
    starter_counts,
)
from ffrank.scoring.step5_ecr import explain_disagreement


def _board() -> pd.DataFrame:
    """A synthetic multi-position board with realistic point scales.

    QBs deliberately score far more raw points than RBs, which is exactly why raw projected
    points are not cross-position comparable and VORP is required.
    """
    rows = []
    rng = np.random.default_rng(7)
    scales = {"QB": 320.0, "RB": 240.0, "WR": 250.0, "TE": 190.0}
    counts = {"QB": 30, "RB": 60, "WR": 70, "TE": 30}
    for pos, n in counts.items():
        for i in range(n):
            rows.append(
                {
                    "player_id": f"{pos}-{i}",
                    "player_name": f"{pos} Player {i}",
                    "position": pos,
                    "team": "DAL",
                    "final_projected_season_points": scales[pos] - i * 4.0 + rng.normal(0, 1),
                    "weighted_ppg": 18.0 - i * 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_vorp_is_the_only_cross_position_comparison():
    """Raw projected points rank QBs above everyone; VORP corrects that."""
    df = compute_vorp(_board(), LEAGUE)

    top_by_points = df.nlargest(10, "final_projected_season_points")["position"].unique().tolist()
    assert top_by_points == ["QB"], "sanity: raw points should be QB-dominated in this fixture"

    # After subtracting each position's own replacement baseline, the top of the board is mixed.
    top_by_vorp = df.nlargest(20, "vorp")["position"].nunique()
    assert top_by_vorp > 1, "VORP must make the cross-position comparison meaningful"


def test_position_rank_matches_vorp_order_within_position():
    """Position Rank needs no separate math: replacement level is a constant per position."""
    df = compute_vorp(_board(), LEAGUE)
    for pos in df["position"].unique():
        sub = df[df["position"] == pos]
        by_points = sub.sort_values("final_projected_season_points", ascending=False)["player_id"].tolist()
        by_vorp = sub.sort_values("vorp", ascending=False)["player_id"].tolist()
        assert by_points == by_vorp


def test_replacement_level_is_the_player_below_the_starter_count():
    """RB33's points are the RB replacement level when there are 32 RB starters."""
    df = compute_vorp(_board(), LEAGUE)
    counts = starter_counts(LEAGUE)
    levels = replacement_levels(df, LEAGUE)
    for pos, count in counts.items():
        pool = (
            df[df["position"] == pos]
            .sort_values("final_projected_season_points", ascending=False)["final_projected_season_points"]
            .to_numpy()
        )
        # count is a 0-based index here, so it addresses the (count+1)-th player.
        assert levels[pos] == pytest.approx(float(pool[count]))

    # And the replacement player himself has VORP of exactly zero.
    rb = df[df["position"] == "RB"].sort_values("final_projected_season_points", ascending=False)
    assert rb.iloc[counts["RB"]]["vorp"] == pytest.approx(0.0, abs=1e-9)


def test_output_limit_must_not_change_replacement_levels_or_vorp():
    """Trimming the board is cosmetic: it must never move a replacement level.

    Replacement level is the (starter_count + 1)-th player at a position, so if the board were
    trimmed BEFORE STEP 4 the replacement player could be cut and every VORP at that position
    would shift. The pipeline therefore trims after STEP 4; this pins that ordering down.
    """
    board = _board()
    full = compute_vorp(board, LEAGUE)
    levels_full = replacement_levels(board, LEAGUE)

    # Simulate trimming to a draftable board, then recompute.
    trimmed = full.nlargest(60, "vorp")
    levels_trimmed = replacement_levels(trimmed, LEAGUE)

    # The trimmed pool no longer contains the replacement players, so recomputing on it gives
    # different (wrong) answers -- which is exactly why order of operations matters.
    assert levels_trimmed != levels_full

    # And the values carried on the trimmed rows are still the ones from the full population.
    for pos in LEAGUE.scored_positions:
        rows = trimmed[trimmed["position"] == pos]
        if rows.empty:
            continue
        assert rows["replacement_level_points"].nunique() == 1
        assert rows["replacement_level_points"].iloc[0] == pytest.approx(levels_full[pos])


def test_trimming_keeps_a_prefix_of_every_positions_ordering():
    """Taking the top N by VORP must never skip over a better player at a position.

    Asserted as a prefix property rather than as integer contiguity: tied projections share a
    rank under method='min' and skip the next integer, so a real board legitimately reads
    ...72, 73, 73, 75... That is a tie, not an omission.
    """
    full = compute_vorp(_board(), LEAGUE).sort_values("vorp", ascending=False).reset_index(drop=True)
    trimmed = full.head(80)
    retained = set(trimmed["player_id"])

    assert trimmed["overall_rank"].min() == 1

    for pos in LEAGUE.scored_positions:
        kept = trimmed[trimmed["position"] == pos]
        if kept.empty:
            continue
        worst_kept_rank = kept["position_rank"].max()
        at_or_above = full[(full["position"] == pos) & (full["position_rank"] <= worst_kept_rank)]
        missing = set(at_or_above["player_id"]) - retained
        assert not missing, f"{pos}: trimming skipped better-ranked players {sorted(missing)}"


def test_tiers_are_not_fixed_size_buckets():
    """Tier sizes must follow real gaps in the distribution, not a constant width."""
    # Three clearly separated clusters with a big gap between each.
    values = pd.Series([100, 99, 98, 97] + [70, 69, 68, 67, 66] + [30, 29, 28])
    tiers = detect_tiers_largest_gap(values, sensitivity=1.5, local_window=4, min_size=2, max_tiers=10)
    assert tiers.nunique() >= 2, "clustered data must produce more than one tier"
    sizes = tiers.value_counts().tolist()
    assert len(set(sizes)) > 1 or tiers.nunique() > 2, "tiers should not all be identical width"


def test_tier_one_contains_the_best_players():
    df = compute_vorp(_board(), LEAGUE).sort_values("vorp", ascending=False).reset_index(drop=True)
    tiers = detect_tiers_largest_gap(df["vorp"])
    assert tiers.iloc[0] == 1
    assert tiers.dropna().is_monotonic_increasing


def test_tiers_keep_forming_deep_in_the_board():
    """No truncation bucket.

    A finite tier ceiling used to dump everyone past the last allowed break into one tier -- on
    the real board that was 798 players in a single 212-VORP "tier". Tiers must keep forming all
    the way down.
    """
    values = pd.Series(np.linspace(120, -220, 900))
    tiers = detect_tiers_largest_gap(values)

    # A large tier is only acceptable when its members are genuinely interchangeable, i.e. it
    # spans almost no value. Truncation instead produces a tier that is both large AND wide.
    # (Tie-collapse can merge two near-identical values into one large tier, which is fine.)
    sizes = tiers.value_counts()
    for tier_id, size in sizes.items():
        span = values[tiers == tier_id]
        width = span.max() - span.min()
        assert width <= W.TIER_MAX_WIDTH_VORP + 1e-9, (
            f"tier {tier_id} spans {width:.1f} VORP across {size} players - looks like a truncation bucket"
        )
    # The bottom of the board must not share a tier with the middle.
    assert tiers.iloc[-1] != tiers.iloc[len(tiers) // 2]
    assert tiers.nunique() > 20


def test_no_tier_spans_an_implausible_value_range():
    """A tier is a group you can treat as interchangeable, so its span is bounded.

    The relative gap rule alone cannot split a region where every gap is large, which produced
    a 25-player tier spanning 56 VORP near the top of the board.
    """
    # Uniform, widely spaced values: no gap is locally unusual, so only the width/size caps
    # can break this up.
    values = pd.Series(np.arange(100, 0, -2.0))
    tiers = detect_tiers_largest_gap(values)
    assert tiers.nunique() > 1, "a uniformly spaced run must still be split into tiers"
    for tier_id in tiers.unique():
        span = values[tiers == tier_id]
        assert span.max() - span.min() <= W.TIER_MAX_WIDTH_VORP + 1e-9
        assert len(span) <= W.TIER_MAX_SIZE


def test_tied_values_always_share_a_tier():
    """Equal VORP must mean equal tier, regardless of how ties fell out of the sort."""
    values = pd.Series([50.0, 40.0, 30.0] + [7.5] * 40 + [1.0, 0.5])
    tiers = detect_tiers_largest_gap(values)
    tied = tiers[values == 7.5]
    assert tied.nunique() == 1, "players with identical VORP were split across tiers"


def test_tiering_is_deterministic_regardless_of_input_order():
    """Row order must not influence the tier column."""
    rng = np.random.default_rng(11)
    values = pd.Series(np.sort(rng.normal(0, 30, 400))[::-1])
    first = detect_tiers_largest_gap(values)
    shuffled = values.sample(frac=1, random_state=5)
    second = detect_tiers_largest_gap(shuffled).reindex(values.index)
    pd.testing.assert_series_equal(first, second)


def test_volatility_never_changes_rank():
    """Consistency is displayed alongside the board and must not feed VORP or the ranking."""
    df = compute_vorp(_board(), LEAGUE)
    baseline_order = df.sort_values("vorp", ascending=False)["player_id"].tolist()

    # Attach wildly varying consistency data.
    rng = np.random.default_rng(3)
    df["consistency_score"] = rng.uniform(0, 100, len(df))
    df["cv"] = rng.uniform(0.2, 1.5, len(df))

    after = compute_vorp(df, LEAGUE)
    assert after.sort_values("vorp", ascending=False)["player_id"].tolist() == baseline_order
    assert "consistency_score" not in ("vorp", "final_projected_season_points")


def test_consistency_score_is_position_relative_and_bounded():
    """WRs are inherently more volatile than RBs, so CV is compared within position."""
    dist = pd.DataFrame(
        {
            "player_id": ["a", "b", "c", "d", "e", "f"],
            "cv": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
        }
    )
    positions = pd.Series({"a": "RB", "b": "RB", "c": "RB", "d": "WR", "e": "WR", "f": "WR"})
    out = consistency_scores(dist, positions)
    assert out["consistency_score"].between(0, 100).all()
    # The least volatile player inside each position group scores above his group's median.
    rb = out[out["position"] == "RB"]
    wr = out[out["position"] == "WR"]
    assert rb.loc[rb["cv"].idxmin(), "consistency_score"] > rb["consistency_score"].median()
    assert wr.loc[wr["cv"].idxmin(), "consistency_score"] > wr["consistency_score"].median()


def test_same_ppg_volatility_flag_fires_only_on_similar_ppg():
    df = pd.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "player_name": ["Steady Sam", "Boom Bust Bob", "Far Away Fred"],
            "position": ["RB", "RB", "RB"],
            # a and b are within 5%; c is far off.
            "weighted_ppg": [15.0, 14.6, 5.0],
            "consistency_score": [80.0, 30.0, 30.0],
        }
    )
    flags = same_ppg_volatility_flags(df)
    assert "less consistent" in flags.iloc[1].lower() or "more consistent" in flags.iloc[1].lower()
    assert flags.iloc[0] != ""
    assert flags.iloc[2] == "", "a player with dissimilar PPG must not be flagged"


def test_ecr_never_feeds_the_math():
    """Step 5 only produces a flag and a reason -- no numeric column it touches feeds VORP."""
    row = {
        "ecr_delta": 40,
        "z_opportunity": 1.4,
        "z_efficiency": 0.2,
        "z_situational": 0.1,
        "z_ppg": 0.9,
        "z_adp": -0.2,
        "expected_games_played": 16.8,
        "team_bias_flag": "N",
        "camp_buzz_score": 0,
    }
    flag, reason = explain_disagreement(row)
    assert "Higher than consensus" in flag
    assert "opportunity" in reason

    # Within the threshold, nothing is flagged at all.
    quiet, quiet_reason = explain_disagreement({**row, "ecr_delta": 5})
    assert quiet == "" and quiet_reason == ""


def test_ecr_reason_points_at_games_played_when_that_is_the_driver():
    row = {
        "ecr_delta": -30,
        "z_opportunity": -0.1,
        "z_efficiency": 0.0,
        "z_situational": 0.0,
        "z_ppg": 0.0,
        "z_adp": 0.0,
        "expected_games_played": 13.0,
        "team_bias_flag": "N",
        "camp_buzz_score": 0,
    }
    flag, reason = explain_disagreement(row)
    assert "Lower than consensus" in flag
    assert "Expected Games Played" in reason


def test_participation_seasons_are_clamped_to_completed_seasons():
    """nflreadpy raises outside 2016..max_season, so the data layer must clamp, not pass through."""
    ceiling = max_stat_season()
    assert clamp_stat_seasons([ceiling + 5, ceiling + 1, ceiling]) == [ceiling]
    assert clamp_stat_seasons([2015, 2016]) == [2016]
    clamped = clamp_stat_seasons(LEAGUE.lookback_seasons)
    assert clamped and max(clamped) <= ceiling
    assert LEAGUE.target_season > ceiling, "the target season is not yet complete, by definition"


def _blend_inputs(hist, rank, snap, pos="RB"):
    idx = range(len(hist))
    return (
        pd.Series(hist, index=idx, dtype=float),
        pd.Series([pos] * len(hist), index=idx),
        pd.Series(rank, index=idx, dtype=float),
        pd.Series(snap, index=idx, dtype=float),
    )


def test_role_blend_lifts_a_promoted_player_with_thin_evidence():
    """A back promoted to RB1 whose history is backup-sized should move toward the RB1 prior."""
    priors = {("RB", 1): 0.49, ("RB", 2): 0.20}
    hist, pos, rank, snap = _blend_inputs([0.20], [1], [0.21])
    blended, w_hist, note = blend_role_expectation(hist, pos, rank, snap, priors)
    assert blended.iloc[0] > hist.iloc[0], "a promoted player must be lifted toward the RB1 prior"
    assert blended.iloc[0] < priors[("RB", 1)], "but not all the way; his own record still counts"
    assert 0.0 < w_hist.iloc[0] < 1.0
    assert "depth chart" in note.iloc[0]


def test_role_blend_barely_moves_an_established_starter():
    """A full-workload starter is trusted on his own record."""
    priors = {("RB", 1): 0.49}
    hist, pos, rank, snap = _blend_inputs([0.57], [1], [0.80])
    blended, w_hist, _ = blend_role_expectation(hist, pos, rank, snap, priors)
    assert w_hist.iloc[0] > 0.65, "high snap share must keep most of the weight on history"
    assert abs(blended.iloc[0] - hist.iloc[0]) < 0.06


def test_role_blend_never_drops_a_player_below_his_own_record():
    """Under the default up_only direction, a measured record acts as a floor.

    A rank-average prior is worse information about a specific player than his own measured
    share, so the blend must not drag a productive backup down to the average for his slot.
    """
    assert W.ROLE_BLEND_DIRECTION == "up_only"
    priors = {("RB", 2): 0.20}
    hist, pos, rank, snap = _blend_inputs([0.33], [2], [0.41])
    blended, _, _ = blend_role_expectation(hist, pos, rank, snap, priors)
    assert blended.iloc[0] == pytest.approx(hist.iloc[0])


def test_role_blend_uses_prior_when_there_is_no_history_at_all():
    priors = {("RB", 1): 0.49}
    hist, pos, rank, snap = _blend_inputs([np.nan], [1], [np.nan])
    blended, w_hist, _ = blend_role_expectation(hist, pos, rank, snap, priors)
    assert blended.iloc[0] == pytest.approx(0.49)
    assert w_hist.iloc[0] == pytest.approx(0.0)


def test_role_blend_is_a_no_op_without_a_prior():
    """No prior for the slot means nothing to blend toward, so the value passes through."""
    hist, pos, rank, snap = _blend_inputs([0.30], [7], [0.50])
    blended, w_hist, note = blend_role_expectation(hist, pos, rank, snap, {})
    assert blended.iloc[0] == pytest.approx(0.30)
    assert w_hist.iloc[0] == pytest.approx(1.0)
    assert note.iloc[0] == ""


def test_role_priors_are_monotonic_down_the_depth_chart():
    """A lower depth slot can never be expected to out-earn a higher one.

    Enforced because small samples invert: a 16-player QB3 bucket came out above QB2.
    """
    depth = pd.DataFrame({
        "season": [2024] * 8,
        "player_id": [f"p{i}" for i in range(8)],
        "position": ["RB"] * 8,
        "depth_chart_rank": [1, 1, 1, 1, 2, 2, 2, 2],
    })
    # Empty pbp -> no priors at all, which must not raise.
    assert derive_role_opportunity_priors(depth, pd.DataFrame(), pd.DataFrame()) == {}


def test_composite_z_is_unit_variance_so_step3a_recovers_the_real_spread():
    """STEP 3a multiplies Composite Z by the position stdev, which needs Composite Z ~ N(0,1).

    A weighted average of five correlated z-scores has std well below 1 (0.77-0.85 measured
    here), so without rescaling every position's projections are compressed by a different
    factor and the board is silently tilted between positions.
    """
    from ffrank.scoring.step2_composite import compute_composite_z

    rng = np.random.default_rng(4)
    n = 80
    df = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "position": ["RB"] * n,
        "weighted_ppg": rng.normal(12, 4, n),
        "opportunity_value": rng.normal(0.3, 0.1, n),
        "efficiency_value": rng.normal(0, 1, n),
        "situational_value": rng.normal(0, 1, n),
        "adp_blended": rng.uniform(1, 200, n),
        "snap_share": rng.uniform(0.2, 0.9, n),
    })
    out = compute_composite_z(df)
    pool = out[out["in_position_pool"]]
    assert pool["composite_z"].std(ddof=0) == pytest.approx(1.0, abs=0.02)
    # The raw, unscaled composite is retained for auditing and must be visibly smaller.
    assert pool["composite_z_raw"].std(ddof=0) < 0.95
    assert (pool["composite_z_scale"] > 0).all()


def test_consensus_list_excludes_kickers_and_defenses():
    """Our board has no K/DST, so the consensus rank must be re-numbered over skill positions.

    Left in, 31 kickers and defenses sit inside the consensus top 250 and pad every skill
    player's rank, manufacturing disagreement that is only a difference in list contents.
    """
    from ffrank.data.market import ECR_COMPARABLE_POSITIONS
    assert set(ECR_COMPARABLE_POSITIONS) == {"QB", "RB", "WR", "TE"}


def test_ecr_flag_threshold_widens_with_depth():
    """15 spots is a tier at the top and noise at rank 200, so the threshold cannot be flat."""
    from ffrank.scoring.step5_ecr import flag_threshold_for_rank

    top = flag_threshold_for_rank(5)
    mid = flag_threshold_for_rank(100)
    deep = flag_threshold_for_rank(240)
    assert top < mid < deep
    assert top >= W.ECR_DELTA_FLAG_THRESHOLD
    assert deep <= W.ECR_DELTA_FLAG_MAX


def test_ecr_flag_suppressed_below_replacement():
    """A disagreement about two unstartable players carries no decision value."""
    from ffrank.scoring.step5_ecr import explain_disagreement

    below = {"ecr_delta": 90, "overall_rank": 200, "vorp": -40.0,
             "z_opportunity": 1.5, "expected_games_played": 16.8, "team_bias_flag": "N", "camp_buzz_score": 0}
    assert explain_disagreement(below) == ("", "")
    above = {**below, "vorp": 25.0}
    flag, reason = explain_disagreement(above)
    assert flag and reason


def test_composite_weights_sum_to_one():
    assert sum(W.COMPOSITE_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(W.RECENCY_WEIGHTS.values()) == pytest.approx(1.0)
    for pos, sub in W.OPPORTUNITY_SUB_WEIGHTS.items():
        assert sum(sub.values()) == pytest.approx(1.0), pos
    for pos, sub in W.EFFICIENCY_SUB_WEIGHTS.items():
        assert sum(sub.values()) == pytest.approx(1.0), pos


def test_name_normalization_unifies_source_spellings():
    """ADP and ECR join by name, so formatting differences must collapse to one key."""
    pairs = [
        ("K.C. Concepcion", "KC Concepcion"),
        ("A.J. Brown", "AJ Brown"),
        ("T.J. Hockenson", "TJ Hockenson"),
        ("Marvin Harrison Jr.", "Marvin Harrison"),
        ("Ja'Marr Chase", "JaMarr Chase"),
        ("Amon-Ra St. Brown", "Amon Ra St Brown"),
        ("Kenny Gainwell", "Kenneth Gainwell"),
    ]
    for a, b in pairs:
        assert normalize_name(a) == normalize_name(b), f"{a!r} != {b!r}"
    # Genuinely different players must not collide.
    assert normalize_name("Josh Allen") != normalize_name("Keenan Allen")
