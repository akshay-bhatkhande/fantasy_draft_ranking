"""Per-draft-slot strategy engine.

The Pick tab shows the SAME VORP and Final Projected Season Points as Main Rankings -- these
are never recomputed per slot. What changes is:

  * the slot's actual snake pick sequence
  * a recommended strategy with reasoning tied to which tiers are realistically available
  * a "likely available at your next pick" indicator derived from ADP versus that sequence
  * a Slot-Adjusted Tier, labelled distinctly so it is never confused with the global Tier
  * per-round "if this tier is gone, pivot to X" contingency notes for the first rounds
  * Round-1 scenario boards that force named players off the board (already drafted)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.league import LeagueConfig
from ..data.sleeper import normalize_name

CONTINGENCY_ROUNDS = 5


def pick_sequence(slot: int, num_teams: int, rounds: int) -> list[int]:
    """Overall pick numbers for a snake-draft slot.

    In a 10-team snake, slot 1 picks 1, 20, 21, 40, 41...; slot 10 picks 10, 11, 30, 31...
    Odd rounds run 1..N, even rounds reverse.
    """
    picks = []
    for rnd in range(1, rounds + 1):
        if rnd % 2 == 1:
            pick = (rnd - 1) * num_teams + slot
        else:
            pick = (rnd - 1) * num_teams + (num_teams - slot + 1)
        picks.append(pick)
    return picks


def availability_probability(adp: float, adp_stdev: float, pick: int) -> float:
    """Rough probability a player is still on the board at a given pick.

    Treats ADP as the centre of a normal distribution of draft positions with the observed
    per-player standard deviation, and asks how often the player goes AFTER our pick. Uses a
    logistic approximation to the normal CDF to avoid a scipy dependency in the hot path.
    """
    if adp is None or pd.isna(adp):
        return 1.0  # undrafted in the sample: effectively always there
    sd = adp_stdev if adp_stdev and not pd.isna(adp_stdev) and adp_stdev > 0 else 6.0
    z = (float(adp) - float(pick)) / float(sd)
    return float(1.0 / (1.0 + np.exp(-1.702 * z)))


def availability_label(prob: float) -> str:
    if prob >= 0.75:
        return "Likely available"
    if prob >= 0.45:
        return "Coin flip"
    if prob >= 0.2:
        return "Probably gone"
    return "Almost certainly gone"


def positional_scarcity(df: pd.DataFrame, picks: list[int], league: LeagueConfig) -> pd.DataFrame:
    """What each position is actually worth to this slot at each of its early picks.

    The decision-relevant quantity is the VORP you can realistically OBTAIN, not a headcount of
    survivors. Counting availability is close to useless early: at pick 1 essentially every
    player is still on the board, so a raw count made TE look like the deepest position at every
    slot simply because no TE goes in the first few picks.

    So this reports, per position per pick:
      best_available_vorp  -- the highest VORP among players at least 50% likely to be there,
                              i.e. the best player you can actually expect to get
      expected_vorp_pool   -- probability-weighted VORP across the position's top 24, a measure
                              of how much total value remains at the position
      expected_top12_available -- retained as a raw survivor count, for context only
    """
    rows = []
    for pos in league.scored_positions:
        sub = df[(df["position"] == pos) & df["vorp"].notna()].nlargest(40, "vorp")
        if sub.empty:
            continue
        top12 = sub.nlargest(12, "vorp")
        top24 = sub.nlargest(24, "vorp")
        for round_idx, pick in enumerate(picks[:CONTINGENCY_ROUNDS], start=1):
            reachable = [
                r for r in sub.itertuples()
                if availability_probability(r.adp_blended, r.adp_stdev, pick) >= 0.5
            ]
            best = max((float(r.vorp) for r in reachable), default=float("nan"))
            expected_pool = sum(
                availability_probability(r.adp_blended, r.adp_stdev, pick) * max(float(r.vorp), 0.0)
                for r in top24.itertuples()
            )
            rows.append(
                {
                    "position": pos,
                    "round": round_idx,
                    "pick": pick,
                    "best_available_vorp": best,
                    "expected_vorp_pool": expected_pool,
                    "expected_top12_available": sum(
                        availability_probability(r.adp_blended, r.adp_stdev, pick) for r in top12.itertuples()
                    ),
                }
            )
    return pd.DataFrame(rows)


def recommend_strategy(
    df: pd.DataFrame,
    slot: int,
    picks: list[int],
    league: LeagueConfig,
) -> tuple[str, str]:
    """Pick a strategy for this slot and explain it in 2-3 sentences.

    The logic is driven by what the ADP-based availability model says will actually be on the
    board at this slot's first two picks, not by a fixed slot-to-strategy table.
    """
    scarcity = positional_scarcity(df, picks, league)
    if scarcity.empty:
        return (
            "Best-Player-Available",
            "ADP data was unavailable, so no slot-specific availability model could be built. "
            "Default to taking the highest VORP on the board at each pick.",
        )

    def best_at(round_idx: int, pos: str) -> float:
        row = scarcity[(scarcity["round"] == round_idx) & (scarcity["position"] == pos)]
        if row.empty:
            return float("nan")
        val = row["best_available_vorp"].iloc[0]
        return float(val) if pd.notna(val) else float("nan")

    first_pick = picks[0]
    second_pick = picks[1] if len(picks) > 1 else picks[0]
    turn_gap = second_pick - first_pick

    rb1, wr1 = best_at(1, "RB"), best_at(1, "WR")
    rb2, wr2 = best_at(2, "RB"), best_at(2, "WR")
    te1, te2 = best_at(1, "TE"), best_at(2, "TE")

    def _edge(a: float, b: float) -> float:
        if np.isnan(a) or np.isnan(b):
            return 0.0
        return a - b

    edge1 = _edge(rb1, wr1)  # positive => best obtainable RB beats best obtainable WR
    edge2 = _edge(rb2, wr2)
    MEANINGFUL = 5.0  # VORP points; smaller gaps are noise given ADP uncertainty

    # The comparison is "which position gives me more VORP at MY picks", evaluated at this
    # slot's first two turns -- which is what actually distinguishes the slots.
    if edge1 > MEANINGFUL and edge2 > 0:
        strategy = "Robust-RB"
        reason = (
            f"From slot {slot} you pick at {first_pick} and {second_pick}. The best RB you can realistically get is "
            f"worth {rb1:.0f} VORP at your first pick and still {rb2:.0f} at your second, versus {wr1:.0f} and "
            f"{wr2:.0f} for WR. RB is the better buy at both turns, so take two before the position thins -- with "
            f"2 RB slots plus 2 FLEX skewing 60% RB, that is the largest structural edge on the board."
        )
    elif edge1 > MEANINGFUL:
        strategy = "Hero-RB"
        reason = (
            f"At slot {slot} (picks {first_pick} and {second_pick}) the best available RB is worth {rb1:.0f} VORP at "
            f"your first pick, ahead of WR at {wr1:.0f}, but by your second pick WR leads ({wr2:.0f} versus "
            f"{rb2:.0f}). Anchor with one elite RB, then pivot into the deeper WR pool instead of reaching for a "
            f"second-tier back."
        )
    elif edge1 < -MEANINGFUL and edge2 <= 0:
        strategy = "Zero-RB"
        reason = (
            f"Slot {slot} picks at {first_pick} and {second_pick}, and WR is the better buy at both: {wr1:.0f} versus "
            f"{rb1:.0f} VORP at your first pick and {wr2:.0f} versus {rb2:.0f} at your second. The elite RBs are gone "
            f"before you are on the clock, so load up on receivers and attack RB later where the market is softer."
        )
    else:
        strategy = "Best-Player-Available"
        reason = (
            f"From slot {slot} (picks {first_pick}, {second_pick}) neither position shows a decisive edge -- best "
            f"available RB {rb1:.0f} VORP versus best WR {wr1:.0f} at your first pick, inside the margin of ADP "
            f"noise. Take the highest VORP on the board and let the draft dictate your structure."
        )

    if turn_gap <= 3:
        reason += (
            f" Your turn wraps quickly ({first_pick} then {second_pick}), so treat those two picks as one package "
            f"and take the two positions you would least like to miss."
        )
    # Only mention TE when an elite one is genuinely reachable AND competitive with RB/WR.
    if not np.isnan(te1) and te1 >= max(rb1 if not np.isnan(rb1) else 0, wr1 if not np.isnan(wr1) else 0) - MEANINGFUL:
        reason += (
            f" With only {int(league.dedicated_starters.get('TE', 1) * league.num_teams + league.flex_slots * league.num_teams * league.flex_allocation.get('TE', 0))} "
            f"TE starters in this league, the top TE ({te1:.0f} VORP) is a real positional edge if he lasts to you."
        )
    return strategy, reason


def slot_board(
    df: pd.DataFrame,
    slot: int,
    league: LeagueConfig,
    rounds: int | None = None,
    gone_names: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Build the per-slot board: same VORP, plus availability and a Slot-Adjusted Tier.

    Deliberately does NOT recompute VORP or Final Projected Season Points -- it re-sorts and
    annotates the shared numbers from Main Rankings.

    Args:
        gone_names: players already drafted before your first pick (Round-1 scenario). They are
            flagged and treated as unavailable for recommendations / slot tiers.
    """
    rounds = rounds or league.total_roster_spots
    picks = pick_sequence(slot, league.num_teams, rounds)
    out = df.copy()

    gone_keys = {normalize_name(n) for n in (gone_names or ())}
    if "name_key" in out.columns:
        out["scenario_gone"] = out["name_key"].isin(gone_keys)
    else:
        out["scenario_gone"] = out["player_name"].map(normalize_name).isin(gone_keys)

    out["your_next_pick"] = picks[0]
    # For each player, the earliest of this slot's picks at which he is a realistic target.
    probs_first = []
    for r in out.itertuples():
        if bool(getattr(r, "scenario_gone", False)):
            probs_first.append(0.0)
        else:
            probs_first.append(availability_probability(r.adp_blended, r.adp_stdev, picks[0]))
    out["prob_available_at_first_pick"] = probs_first

    next_pick_for_player = []
    prob_at_next = []
    labels = []
    for r in out.itertuples():
        if bool(getattr(r, "scenario_gone", False)):
            next_pick_for_player.append(pd.NA)
            prob_at_next.append(0.0)
            labels.append("Already drafted (scenario)")
            continue
        chosen, chosen_prob = picks[-1], 0.0
        for pick in picks:
            p = availability_probability(r.adp_blended, r.adp_stdev, pick)
            if p >= 0.5:
                chosen, chosen_prob = pick, p
                break
            chosen, chosen_prob = pick, p
        next_pick_for_player.append(chosen)
        prob_at_next.append(chosen_prob)
        labels.append(availability_label(chosen_prob))
    out["realistic_target_pick"] = next_pick_for_player
    out["likely_available_at_next_pick"] = labels
    out["availability_probability"] = np.round(prob_at_next, 2)

    # Slot-Adjusted Tier: tiers recomputed over only the players realistically reachable from
    # this slot, so tier 1 on a slot tab means "the best group you can actually get".
    reachable = out[(out["prob_available_at_first_pick"] >= 0.15) & (~out["scenario_gone"])]
    from .step4_vorp import assign_tiers

    slot_tier = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if not reachable.empty:
        slot_tier.loc[reachable.index] = assign_tiers(reachable, value_col="vorp")
    out["slot_adjusted_tier"] = slot_tier

    return out.sort_values(["scenario_gone", "vorp"], ascending=[True, False])


def contingency_notes(
    df: pd.DataFrame,
    slot: int,
    league: LeagueConfig,
    rounds: int | None = None,
    gone_names: tuple[str, ...] | list[str] | None = None,
) -> list[dict]:
    """Per-round "if this tier is gone, pivot to X" notes for the first few rounds."""
    rounds = rounds or league.total_roster_spots
    picks = pick_sequence(slot, league.num_teams, rounds)
    gone_keys = {normalize_name(n) for n in (gone_names or ())}
    live = df.copy()
    if gone_keys:
        if "name_key" in live.columns:
            live = live[~live["name_key"].isin(gone_keys)]
        else:
            live = live[~live["player_name"].map(normalize_name).isin(gone_keys)]
    scarcity = positional_scarcity(live, picks, league)
    notes = []

    for round_idx, pick in enumerate(picks[:CONTINGENCY_ROUNDS], start=1):
        row = scarcity[scarcity["round"] == round_idx]
        if row.empty:
            continue
        # Rank positions by the VORP actually obtainable at this pick, not by survivor count.
        by_pos = row.set_index("position")["best_available_vorp"].dropna().to_dict()
        if not by_pos:
            continue
        ranked = sorted(by_pos.items(), key=lambda kv: kv[1], reverse=True)
        primary, primary_vorp = ranked[0]
        backup, backup_vorp = ranked[1] if len(ranked) > 1 else ("BPA", float("nan"))

        target_pool = live[(live["position"] == primary) & live["vorp"].notna()].nlargest(24, "vorp")
        likely = [
            r for r in target_pool.itertuples()
            if availability_probability(r.adp_blended, r.adp_stdev, pick) >= 0.5
        ]
        names = ", ".join(str(getattr(r, "player_name", "")) for r in likely[:3]) or "next tier down"

        backup_pool = live[(live["position"] == backup) & live["vorp"].notna()].nlargest(24, "vorp")
        backup_names = ", ".join(
            str(getattr(r, "player_name", ""))
            for r in backup_pool.itertuples()
            if availability_probability(r.adp_blended, r.adp_stdev, pick) >= 0.5
        ).split(", ")[:2]

        notes.append(
            {
                "round": round_idx,
                "pick": pick,
                "primary_target": primary,
                "best_available_vorp": round(float(primary_vorp), 1),
                "note": (
                    f"Round {round_idx} (pick {pick}): best value is {primary} at about {primary_vorp:.0f} VORP "
                    f"(targets: {names}). If that tier is gone, pivot to {backup}"
                    + (f" at about {backup_vorp:.0f} VORP" if pd.notna(backup_vorp) else "")
                    + (f" ({', '.join(n for n in backup_names if n)})" if any(backup_names) else "")
                    + "."
                ),
            }
        )
    return notes


def scenario_pick3_plan(
    df: pd.DataFrame,
    league: LeagueConfig,
    gone_names: tuple[str, ...] | list[str],
    top_n: int = 5,
) -> dict:
    """What to do at pick 3 (and the early follow-ups) given named players already drafted."""
    slot = league.my_draft_slot
    picks = pick_sequence(slot, league.num_teams, league.total_roster_spots)
    gone_keys = {normalize_name(n) for n in gone_names}
    live = df.copy()
    if "name_key" in live.columns:
        mask = ~live["name_key"].isin(gone_keys)
    else:
        mask = ~live["player_name"].map(normalize_name).isin(gone_keys)
    live = live.loc[mask].sort_values("vorp", ascending=False)

    strategy, reasoning = recommend_strategy(live, slot, picks, league)
    notes = contingency_notes(df, slot, league, gone_names=gone_names)

    def _top_at(pick: int, n: int = top_n) -> list[dict]:
        rows = []
        for r in live.itertuples():
            if availability_probability(r.adp_blended, r.adp_stdev, pick) < 0.35 and pick == picks[0]:
                # At pick 3 almost everyone elite is "available" by ADP; still rank by VORP.
                pass
            rows.append(
                {
                    "name": str(r.player_name),
                    "position": str(r.position),
                    "team": str(getattr(r, "team", "") or ""),
                    "vorp": float(r.vorp) if pd.notna(r.vorp) else float("nan"),
                    "adp": float(r.adp_blended) if pd.notna(getattr(r, "adp_blended", np.nan)) else float("nan"),
                    "prob": availability_probability(r.adp_blended, r.adp_stdev, pick),
                }
            )
            if len(rows) >= n:
                break
        # For later picks, prefer players still likely available.
        if pick != picks[0]:
            ranked = sorted(
                (
                    {
                        "name": str(r.player_name),
                        "position": str(r.position),
                        "team": str(getattr(r, "team", "") or ""),
                        "vorp": float(r.vorp) if pd.notna(r.vorp) else float("nan"),
                        "adp": float(r.adp_blended) if pd.notna(getattr(r, "adp_blended", np.nan)) else float("nan"),
                        "prob": availability_probability(r.adp_blended, r.adp_stdev, pick),
                    }
                    for r in live.itertuples()
                ),
                key=lambda d: (d["prob"] < 0.45, -d["vorp"] if not np.isnan(d["vorp"]) else 0),
            )
            rows = [d for d in ranked if d["prob"] >= 0.35][:n] or ranked[:n]
        return rows

    best = live.iloc[0] if not live.empty else None
    pick3_targets = _top_at(picks[0], top_n)
    return {
        "picks": picks,
        "strategy": strategy,
        "reasoning": reasoning,
        "notes": notes,
        "best_pick3": None
        if best is None
        else {
            "name": str(best["player_name"]),
            "position": str(best["position"]),
            "vorp": float(best["vorp"]),
            "team": str(best.get("team", "") or ""),
        },
        "targets_by_pick": {
            picks[0]: pick3_targets,
            picks[1]: _top_at(picks[1], top_n) if len(picks) > 1 else [],
            picks[2]: _top_at(picks[2], top_n) if len(picks) > 2 else [],
        },
        "gone": list(gone_names),
    }
