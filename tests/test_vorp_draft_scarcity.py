"""Draft-scarcity VORP ceilings/penalties: wait on QB/TE unless elite value falls."""

import pandas as pd

from config.league import LEAGUE
from ffrank.scoring.step4_vorp import (
    apply_draft_scarcity_vorp,
    compute_vorp,
    draft_vorp_ceiling,
    draft_wait_penalty,
)


def test_qb_and_te_have_scarcity_rb_wr_do_not():
    assert draft_wait_penalty("QB", 2) > 0.0
    assert draft_vorp_ceiling("TE", 1) is not None
    assert draft_wait_penalty("WR", 1) == 0.0
    assert draft_vorp_ceiling("RB", 1) is None


def test_mid_qb_penalized_more_than_elite_qb():
    assert draft_wait_penalty("QB", 5) > draft_wait_penalty("QB", 1)


def test_penalty_preserves_within_position_order():
    raw = pd.Series([80.0, 40.0, 10.0, -5.0])
    pos = pd.Series(["QB", "QB", "QB", "QB"])
    rank = pd.Series([1, 2, 5, 12])
    adj, _ = apply_draft_scarcity_vorp(raw, pos, rank)
    assert adj.is_monotonic_decreasing


def test_te_ceiling_preserves_elite_order():
    raw = pd.Series([90.0, 70.0, 25.0])
    pos = pd.Series(["TE", "TE", "TE"])
    rank = pd.Series([1, 2, 3])
    adj, _ = apply_draft_scarcity_vorp(raw, pos, rank)
    assert adj.is_monotonic_decreasing
    # Both elites sit at/under the round-3 ceiling band.
    assert adj.iloc[0] <= 64.0 + 1e-9
    assert adj.iloc[1] <= 64.0 + 1e-9


def test_compute_vorp_exposes_raw_and_pushes_mid_qb_down():
    rows = []
    for i, pts in enumerate([400, 360, 340, 330] + [300 - j for j in range(20)]):
        rows.append(
            {
                "player_id": f"QB-{i}",
                "player_name": f"QB {i}",
                "position": "QB",
                "team": "BUF",
                "final_projected_season_points": float(pts),
            }
        )
    for i in range(40):
        rows.append(
            {
                "player_id": f"WR-{i}",
                "player_name": f"WR {i}",
                "position": "WR",
                "team": "DAL",
                "final_projected_season_points": 280.0 - i * 3.0,
            }
        )
    for i in range(40):
        rows.append(
            {
                "player_id": f"RB-{i}",
                "player_name": f"RB {i}",
                "position": "RB",
                "team": "DET",
                "final_projected_season_points": 270.0 - i * 3.0,
            }
        )
    for i in range(20):
        rows.append(
            {
                "player_id": f"TE-{i}",
                "player_name": f"TE {i}",
                "position": "TE",
                "team": "ARI",
                "final_projected_season_points": 250.0 - i * 4.0,
            }
        )
    df = compute_vorp(pd.DataFrame(rows), LEAGUE)
    assert "vorp_raw" in df.columns
    assert "vorp_wait_penalty" in df.columns
    qb2 = df[(df["position"] == "QB") & (df["position_rank"] == 2)].iloc[0]
    wr5 = df[(df["position"] == "WR") & (df["position_rank"] == 5)].iloc[0]
    assert qb2["vorp"] < wr5["vorp"]
    assert qb2["vorp"] < qb2["vorp_raw"]
