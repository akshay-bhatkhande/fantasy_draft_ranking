"""RB committee / expected snap % helpers."""

import pandas as pd

from ffrank.features.rb_role import (
    parse_camp_snap_pct,
    player_snap_share_by_season,
    player_end_of_season_snap_share,
    rb_workload_multiplier,
    team_backfield_snap_trend,
    team_rb_snap_structure,
    _role_label,
    _committee_flag,
)
from config import weights as W


def test_parse_camp_snap_range():
    assert parse_camp_snap_pct("Kaboly projected roughly 65-70% of snaps.") == 0.675


def test_parse_camp_snap_single():
    assert parse_camp_snap_pct("Expected to play about 55% of snaps this year") == 0.55


def test_parse_camp_snap_missing():
    assert parse_camp_snap_pct("Clear RB1 with no snap projection") is None


def test_role_labels():
    assert _role_label(0.72, 1, "Bellcow") == "Bellcow"
    assert _role_label(0.48, 1, "Committee") == "Committee share"
    assert _role_label(0.28, 2, "Committee") == "Committee RB2"


def test_committee_flag_from_structure():
    assert _committee_flag("Committee", 0.50, 1, 3, "") == "Committee likely"
    assert _committee_flag("Bellcow", 0.70, 1, 2, "") == "Bellcow likely"
    # Crowded depth chart must not override a true lead back.
    assert _committee_flag("Bellcow", 0.64, 1, 6, "") == "Bellcow likely"
    assert _committee_flag("Bellcow", 0.56, 1, 6, "") == "Lean bellcow"


def test_player_snap_share_ignores_cameo_games():
    snaps = pd.DataFrame(
        {
            "pfr_player_id": ["X"] * 4,
            "season": [2025] * 4,
            "week": [1, 2, 3, 4],
            "offense_pct": [0.12, 0.70, 0.68, 0.21],
            "offense_snaps": [8, 50, 49, 11],  # first/last below 15-snap floor
            "position": ["RB"] * 4,
            "team": ["NYG"] * 4,
        }
    )
    out = player_snap_share_by_season(snaps, min_snaps=15)
    assert len(out) == 1
    assert abs(out.iloc[0]["offense_pct"] - 0.69) < 1e-9
    assert int(out.iloc[0]["games_used"]) == 2


def test_rb_workload_multiplier_scales_around_par():
    assert rb_workload_multiplier(60.0, "RB") == 1.0
    assert rb_workload_multiplier(72.0, "RB") == W.RB_WORKLOAD_MULT_BOUNDS[1]
    assert rb_workload_multiplier(40.0, "RB") == W.RB_WORKLOAD_MULT_BOUNDS[0]
    assert rb_workload_multiplier(70.0, "WR") == 1.0
    assert rb_workload_multiplier(None, "RB") == 1.0


def test_rb_workload_softens_limited_sample_and_camp_stack():
    # Full-sample bellcow at the cap.
    full = rb_workload_multiplier(72.0, "RB", limited_sample=False, camp_buzz_multiplier=1.0)
    assert full == W.RB_WORKLOAD_MULT_BOUNDS[1]
    # Limited sample keeps only a fraction of the lift.
    limited = rb_workload_multiplier(72.0, "RB", limited_sample=True, camp_buzz_multiplier=1.0)
    assert 1.0 < limited < full
    # Camp already lifting PPG → further shrink the workload lift (Skattebo case).
    stacked = rb_workload_multiplier(72.0, "RB", limited_sample=True, camp_buzz_multiplier=1.08)
    assert 1.0 <= stacked < limited


def test_team_structure_uses_weekly_concurrent_not_season_averages():
    # Player A featured weeks 1-2; player B featured weeks 3-4. Season means look like a
    # committee; weekly concurrent leaders do not.
    rows = []
    for week, a_pct, b_pct in [(1, 0.70, 0.20), (2, 0.68, 0.22), (3, 0.18, 0.65), (4, 0.20, 0.70)]:
        rows.append(
            {
                "season": 2025,
                "week": week,
                "team": "NYG",
                "position": "RB",
                "pfr_player_id": "A",
                "offense_pct": a_pct,
                "offense_snaps": 40,
            }
        )
        rows.append(
            {
                "season": 2025,
                "week": week,
                "team": "NYG",
                "position": "RB",
                "pfr_player_id": "B",
                "offense_pct": b_pct,
                "offense_snaps": 40,
            }
        )
    snaps = pd.DataFrame(rows)
    out = team_rb_snap_structure(snaps, prior_season=2025)
    assert len(out) == 1
    assert out.iloc[0]["rb_team_structure"] == "Bellcow"
    assert out.iloc[0]["team_rb1_snap"] >= 0.60
    assert out.iloc[0]["team_rb_gap"] >= 0.20


def test_team_backfield_snap_trend_flags_late_committee():
    rows = []
    # Early: RB1 ~77%, RB2 ~20%; late: RB1 ~60%, RB2 ~35% (Kyren/Corum shape).
    for week, rb1, rb2 in [
        (1, 0.82, 0.17),
        (2, 0.76, 0.20),
        (3, 0.74, 0.22),
        (4, 0.75, 0.21),
        (5, 0.78, 0.19),
        (6, 0.77, 0.20),
        (13, 0.67, 0.33),
        (14, 0.51, 0.31),
        (15, 0.54, 0.46),
        (16, 0.71, 0.29),
        (17, 0.71, 0.25),
        (18, 0.63, 0.33),
    ]:
        rows.append(
            {
                "season": 2025,
                "week": week,
                "team": "LA",
                "position": "RB",
                "pfr_player_id": "A" if rb1 >= rb2 else "B",
                "offense_pct": rb1,
                "offense_snaps": 40,
            }
        )
        rows.append(
            {
                "season": 2025,
                "week": week,
                "team": "LA",
                "position": "RB",
                "pfr_player_id": "B" if rb1 >= rb2 else "A",
                "offense_pct": rb2,
                "offense_snaps": 20,
            }
        )
    snaps = pd.DataFrame(rows)
    out = team_backfield_snap_trend(snaps, prior_season=2025)
    assert len(out) == 1
    assert bool(out.iloc[0]["rb_trend_committee"])


def test_compute_rb_roles_prices_late_snap_erosion():
    from ffrank.features.rb_role import compute_rb_roles

    rows = []
    for week, kyren, corum in [
        (1, 0.82, 0.17),
        (2, 0.76, 0.20),
        (3, 0.74, 0.22),
        (4, 0.75, 0.21),
        (5, 0.78, 0.19),
        (6, 0.77, 0.20),
        (13, 0.67, 0.33),
        (14, 0.51, 0.31),
        (15, 0.54, 0.46),
        (16, 0.71, 0.29),
        (17, 0.71, 0.25),
        (18, 0.63, 0.33),
    ]:
        rows.append(
            {
                "season": 2025,
                "week": week,
                "team": "LA",
                "position": "RB",
                "player": "Kyren Williams",
                "pfr_player_id": "WillKy02",
                "offense_pct": kyren,
                "offense_snaps": 45,
            }
        )
        rows.append(
            {
                "season": 2025,
                "week": week,
                "team": "LA",
                "position": "RB",
                "player": "Blake Corum",
                "pfr_player_id": "CoruBl00",
                "offense_pct": corum,
                "offense_snaps": 20,
            }
        )
    snaps = pd.DataFrame(rows)
    df = pd.DataFrame(
        [
            {
                "player_id": "kyren",
                "pfr_id": "WillKy02",
                "position": "RB",
                "team": "LA",
                "depth_chart_rank": 1,
                "depth_chart_competitors": 3,
                "snap_share": 0.759,
                "carry_share": 0.643,
                "camp_buzz_score": 1,
                "camp_buzz_note": "",
            }
        ]
    )
    out = compute_rb_roles(df, snaps, target_season=2026)
    kyren_row = out.iloc[0]
    assert float(kyren_row["expected_snap_pct"]) < 72.0
    assert kyren_row["rb_role_label"] != "Bellcow"
    assert "trending committee" in str(kyren_row["rb_committee_note"])
