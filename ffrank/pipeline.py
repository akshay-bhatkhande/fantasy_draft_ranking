"""Pipeline orchestration: loads every source once, then runs Steps 1-5 in order.

The full chain, start to finish:

    Weighted PPG (STEP 1)
      -> Composite Z-Score, position-relative only (STEP 2)
      -> Base Projected PPG, converted back to real points via position mean/stdev (STEP 3a)
      -> x risk multipliers = Final Projected PPG (STEP 3b)
      -> x Expected Games Played = Final Projected Season Points (STEP 3c-3d)
      -> minus position Replacement Level = VORP (STEP 4)
      -> VORP is the ONLY number ever used to rank or compare players across positions

Volatility/consistency is computed in parallel and displayed alongside, but never enters this
chain and never changes a rank. ECR (STEP 5) is attached at the end as a sanity check only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import weights as W
from config.league import LEAGUE, LeagueConfig
from .data import market, nflverse as nv, overrides, sleeper
from .data.cache import SourceLog
from .features import curves as C
from .features import kdst, risk, volatility
from .features.efficiency import compute_efficiency_score
from .features import opportunity as opportunity_mod
from .features.opportunity import PBP_COLUMNS, compute_opportunity_score
from .features.rb_role import compute_rb_roles
from .features.situational import compute_situational
from .features.weighted_ppg import compute_weighted_ppg, limited_sample_note, season_ppg_table
from .scoring import step5_ecr
from .scoring.step2_composite import compute_composite_z
from .scoring.step3_projection import camp_flag, run_step3
from .scoring.step4_vorp import assign_tiers, compute_vorp, positional_tiers, starter_counts

# Extra play-by-play columns the situational module needs on top of the opportunity set.
SITUATIONAL_PBP_COLUMNS = ["epa", "qb_epa", "pass_oe", "defteam", "xpass"]


@dataclass
class PipelineResult:
    """Everything the Excel writer needs."""

    rankings: pd.DataFrame
    kickers: pd.DataFrame
    defenses: pd.DataFrame
    age_curves: dict = field(default_factory=dict)
    contract_lifts: dict = field(default_factory=dict)
    rookie_baselines: dict = field(default_factory=dict)
    replacement_levels: dict = field(default_factory=dict)
    starter_counts: dict = field(default_factory=dict)
    sources: SourceLog | None = None
    camp_buzz_status: str = ""
    camp_buzz_age_days: float | None = None
    run_timestamp: str = ""
    league: LeagueConfig = LEAGUE
    diagnostics: dict = field(default_factory=dict)
    # True when a personal-preference team penalty was applied. Drives whether the workbook
    # shows the penalty / pre-penalty columns at all.
    bias_active: bool = False
    # Objective board ranked by raw VORP (no personal fades / draft-scarcity rank key).
    raw_vorp_rankings: pd.DataFrame | None = None


def _compute_ages(rosters: pd.DataFrame, season: int) -> pd.DataFrame:
    """Age as of 1 September of the given season, from roster birth dates."""
    if rosters.empty or "birth_date" not in rosters.columns:
        return pd.DataFrame(columns=["player_id", "season", "age"])
    r = rosters.copy()
    r["birth_date"] = pd.to_datetime(r["birth_date"], errors="coerce")
    ref = pd.to_datetime(r["season"].astype(str) + "-09-01", errors="coerce")
    r["age"] = (ref - r["birth_date"]).dt.days / 365.25
    out = r.dropna(subset=["gsis_id", "age"])[["gsis_id", "season", "age"]]
    return out.rename(columns={"gsis_id": "player_id"}).drop_duplicates(subset=["player_id", "season"])


def build_rankings(league: LeagueConfig = LEAGUE, log: SourceLog | None = None) -> PipelineResult:
    """Load all data and run the whole pipeline."""
    log = log if log is not None else SourceLog()
    target = league.target_season
    lookback = list(league.lookback_seasons)
    # The curve-fitting window must be WIDER than the nominal lookbacks. The contract-year
    # analysis compares a season to both its prior and following season, so the first and last
    # seasons of any window are unusable as contract years. Without the extra buffer the sample
    # collapses and the estimate gets pushed into the configured clip bounds.
    curve_span = max(W.AGE_CURVE_LOOKBACK_SEASONS, W.CONTRACT_YEAR_LOOKBACK_SEASONS) + 2
    curve_seasons = list(range(target - curve_span, target))

    # ---------------------------------------------------------------- load (Tier 1 and 2)
    weekly = nv.weekly_stats(lookback, log=log)
    weekly_curve = nv.weekly_stats(curve_seasons, log=log)
    pbp = nv.play_by_play(lookback, log=log, columns=PBP_COLUMNS + SITUATIONAL_PBP_COLUMNS)
    pbp_recent = pbp[pbp["season"] == max(nv.clamp_stat_seasons(lookback))] if not pbp.empty else pbp
    participation = nv.participation(lookback, log=log)
    snaps = nv.snap_counts(lookback, log=log)
    adv_rush = nv.pfr_advstats("rush", lookback, log=log)
    adv_rec = nv.pfr_advstats("rec", lookback, log=log)
    ff_opp = nv.ff_opportunity(lookback, log=log)
    injuries = nv.injuries(lookback, log=log)
    rosters_target = nv.rosters(target, log=log)
    rosters_hist = pd.concat(
        [nv.rosters(s, log=log) for s in curve_seasons], ignore_index=True
    ) if curve_seasons else pd.DataFrame()
    depth = nv.depth_charts(target, log=log)
    schedules_target = nv.schedules([target], log=log)
    schedules_hist = nv.schedules(nv.clamp_stat_seasons(curve_seasons), log=log)
    draft_picks = nv.draft_picks(log=log)
    contracts = nv.contracts(log=log)
    team_stats = nv.team_stats(lookback, log=log)

    crosswalk = sleeper.load_id_crosswalk(log=log)
    adp_raw = market.fetch_adp(target, teams=league.num_teams, scoring="ppr", log=log)
    adp_manual = market.load_manual_adp_override(log=log)
    adp = market.blend_adp(adp_raw, adp_manual)
    ecr = market.load_ecr_redraft(log=log)
    camp = overrides.load_camp_buzz(log=log)
    absences = overrides.load_known_absences(log=log)
    contract_overrides = overrides.load_contract_year_overrides(log=log)

    # ---------------------------------------------------------------- player universe
    # The upcoming season's roster defines who we rank -- not last year's stat leaders.
    universe = rosters_target[rosters_target["position"].isin(league.scored_positions)].copy()
    universe = universe.dropna(subset=["gsis_id"])
    universe = universe.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
    universe = universe.drop_duplicates(subset=["player_id"])
    keep = [
        c for c in ("player_id", "player_name", "position", "team", "years_exp", "pfr_id",
                    "sleeper_id", "birth_date", "draft_number", "rookie_year", "entry_year")
        if c in universe.columns
    ]
    universe = universe[keep]
    universe["name_key"] = universe["player_name"].map(sleeper.normalize_name)

    positions = universe.set_index("player_id")["position"]
    pfr_ids = universe.dropna(subset=["pfr_id"]).set_index("player_id")["pfr_id"]

    # ---------------------------------------------------------------- STEP 1
    step1 = compute_weighted_ppg(weekly, target)
    df = universe.merge(
        step1.drop(columns=[c for c in ("position", "player_name") if c in step1.columns]),
        on="player_id",
        how="left",
    )
    # Players on the upcoming roster with no STEP 1 history at all (rookies, or veterans with
    # no games in the window) are limited-sample by definition.
    df["limited_sample"] = df["limited_sample"].astype("object").where(df["weighted_ppg"].notna(), True)
    df["limited_sample"] = df["limited_sample"].fillna(True).astype(bool)

    # Snap share drives the STEP 3a position pool and RB role estimates.
    # Average ONLY games the player actually played with meaningful snaps — inactive weeks
    # are already absent from snap counts, but ramp-up / injury-exit cameos must not dilute
    # a featured role into a false committee average.
    if not snaps.empty:
        from .features.rb_role import player_snap_share_by_season
        from .features.weighted_ppg import recency_weighted_mean

        per_season_snap = player_snap_share_by_season(snaps).rename(
            columns={"offense_pct": "offense_pct"}
        )
        if not per_season_snap.empty:
            snap_weighted = recency_weighted_mean(
                per_season_snap[["player_id", "season", "offense_pct"]],
                ["offense_pct"],
                target,
            ).rename(columns={"player_id": "pfr_id", "offense_pct": "snap_share"})
            df = df.merge(snap_weighted, on="pfr_id", how="left")
    if "snap_share" not in df.columns:
        df["snap_share"] = np.nan

    # ---------------------------------------------------------------- STEP 2 inputs
    opportunity = compute_opportunity_score(pbp, weekly, positions, target)
    opp_cols = [
        c for c in (
            "player_id", "opportunity_value", "opportunity_detail",
            "carry_share", "rz_carry_share", "target_share", "air_yards_share",
            "rz_target_share", "dropback_share", "designed_rush_share", "rz_pass_attempt_share",
        ) if c in opportunity.columns
    ]
    df = df.merge(opportunity[opp_cols], on="player_id", how="left")
    df = df.rename(columns={"opportunity_value": "opportunity_value_historical"})

    efficiency = compute_efficiency_score(
        weekly, pbp, participation, snaps, adv_rush, adv_rec, ff_opp, positions, pfr_ids, target
    )
    eff_cols = [
        c for c in (
            "player_id", "efficiency_value", "efficiency_detail", "routes_source", "total_routes",
            "yprr", "yards_after_contact", "broken_tackle_rate", "td_rate_vs_expected",
            "ypa", "sack_rate_avoided",
        ) if c in efficiency.columns
    ]
    df = df.merge(efficiency[eff_cols], on="player_id", how="left")

    sit = compute_situational(
        df[["player_id", "position", "team"]].copy(),
        pbp_recent, adv_rush, weekly, schedules_target, schedules_hist, depth, target,
    )
    sit_cols = [
        c for c in (
            "player_id", "situational_value", "situational_detail", "situational_components_used",
            "bye_week", "sos_full", "sos_playoffs", "dome_share", "implied_total_per_game",
            "depth_chart_rank", "depth_chart_competitors", "coach_change_note",
        ) if c in sit.columns
    ]
    df = df.merge(sit[sit_cols], on="player_id", how="left")

    # Blend the player's CURRENT depth-chart role into his historical opportunity share, so a
    # role change is priced rather than ignored. Needs depth_chart_rank (from the target-season
    # depth chart, merged just above) and snap_share as the evidence weight.
    role_priors = opportunity_mod.derive_role_opportunity_priors(
        nv.depth_charts_opening(W.ROLE_PRIOR_SEASONS, log=log), pbp, weekly
    )
    depth_rank = (
        df["depth_chart_rank"] if "depth_chart_rank" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    blended, w_hist, role_note = opportunity_mod.blend_role_expectation(
        df["opportunity_value_historical"],
        df["position"],
        depth_rank,
        df["snap_share"],
        role_priors,
    )
    df["opportunity_value"] = blended
    df["opportunity_history_weight"] = w_hist
    df["role_blend_note"] = role_note

    # ADP joins by normalised name: FFC publishes no stable player id.
    if not adp.empty:
        adp_join = adp.copy()
        adp_join["name_key"] = adp_join["adp_name"].map(sleeper.normalize_name)
        adp_cols = [
            c for c in ("name_key", "adp", "adp_blended", "adp_stdev", "adp_high", "adp_low",
                        "adp_spread", "adp_cross_source_spread", "adp_sources_used", "adp_source",
                        "adp_is_prior_year_fallback", "times_drafted")
            if c in adp_join.columns
        ]
        df = df.merge(adp_join[adp_cols].drop_duplicates(subset=["name_key"]), on="name_key", how="left")
    for col in ("adp", "adp_blended", "adp_stdev"):
        if col not in df.columns:
            df[col] = np.nan

    # ---------------------------------------------------------------- STEP 2 composite
    df = compute_composite_z(df)

    # ---------------------------------------------------------------- curves for STEP 3b
    season_ppg = season_ppg_table(weekly_curve)
    ages_hist = _compute_ages(rosters_hist, target)
    ages_target = _compute_ages(rosters_target, target)
    positions_hist = (
        rosters_hist.dropna(subset=["gsis_id"]).groupby("gsis_id")["position"].last()
        if not rosters_hist.empty else positions
    )

    age_curves = C.fit_age_curves(season_ppg, ages_hist, positions_hist)
    contract_year_seasons = C.identify_contract_years(contracts)
    contract_lifts = C.derive_contract_year_lift(season_ppg, contract_year_seasons, positions_hist)
    rookie_baselines = C.derive_rookie_baselines(season_ppg, draft_picks, positions_hist)

    df = df.merge(ages_target[["player_id", "age"]], on="player_id", how="left")

    # Rookie identification and draft capital.
    df["years_exp"] = pd.to_numeric(df.get("years_exp"), errors="coerce")
    df["is_rookie"] = df["years_exp"].fillna(0) <= 0
    # The draft table can carry more than one row per gsis_id, so de-duplicate before using it
    # as a lookup index (keep the earliest draft record for the player).
    draft_lookup = (
        draft_picks.dropna(subset=["gsis_id"])
        .sort_values("season")
        .drop_duplicates(subset=["gsis_id"], keep="first")
        .set_index("gsis_id")
        if not draft_picks.empty and "gsis_id" in draft_picks.columns else pd.DataFrame()
    )
    pick_col = next((c for c in ("pick", "overall") if not draft_lookup.empty and c in draft_lookup.columns), None)
    if pick_col:
        df["overall_pick"] = df["player_id"].map(draft_lookup[pick_col])
    else:
        df["overall_pick"] = np.nan
    if "draft_number" in df.columns:
        df["overall_pick"] = df["overall_pick"].fillna(pd.to_numeric(df["draft_number"], errors="coerce"))
    df["draft_capital_tier"] = df["overall_pick"].map(C.capital_tier)

    # Contract-year flag from the OTC mirror, then any manual override on top.
    cy_target = contract_year_seasons[contract_year_seasons["season"] == target]
    df["contract_year"] = df["player_id"].isin(set(cy_target["gsis_id"]))
    df["contract_year_source"] = np.where(df["contract_year"], "nflverse OverTheCap mirror", "")
    if not contract_overrides.empty:
        ov = contract_overrides.drop_duplicates(subset=["name_key"]).set_index("name_key")
        mapped = df["name_key"].map(ov["contract_year_override"])
        src = df["name_key"].map(ov["contract_year_source"])
        df["contract_year"] = np.where(mapped.notna(), mapped.fillna(False).astype(bool), df["contract_year"])
        df["contract_year_source"] = np.where(mapped.notna(), "manual override: " + src.fillna(""), df["contract_year_source"])

    # Camp buzz (the only place camp news enters).
    if camp.available:
        cb = camp.scores.drop_duplicates(subset=["name_key"]).set_index("name_key")
        df["camp_buzz_score"] = df["name_key"].map(cb["camp_buzz_score"]).fillna(0)
        df["camp_buzz_source"] = df["name_key"].map(cb["camp_buzz_source"]).fillna("")
        df["camp_buzz_date"] = df["name_key"].map(cb["camp_buzz_date"]).fillna("")
        df["camp_buzz_note"] = df["name_key"].map(cb["camp_buzz_note"]).fillna("")
    else:
        df["camp_buzz_score"] = 0
        df["camp_buzz_source"] = ""
        df["camp_buzz_date"] = ""
        df["camp_buzz_note"] = f"camp_buzz.json {camp.status}; treated as neutral"

    # RB committee / expected snap % (informational only — does not feed VORP).
    df = compute_rb_roles(df, snaps, target)

    # Injury risk and known absences -> STEP 3c.
    injury_hist = risk.compute_injury_history(weekly, injuries, schedules_hist, target)
    df = df.merge(injury_hist, on="player_id", how="left")
    df["injury_risk_bucket"] = df["injury_risk_bucket"].fillna("Low")
    df["expected_games_missed_from_bucket"] = df["expected_games_missed_from_bucket"].fillna(
        W.INJURY_RISK_EXPECTED_GAMES_MISSED["Low"]
    )
    df["injury_history_note"] = df["injury_history_note"].fillna(
        "No injury-attributed absences found in the lookback window; treated as Low risk"
    )
    if not absences.empty:
        ab = absences.drop_duplicates(subset=["name_key"]).set_index("name_key")
        df["known_games_missed"] = df["name_key"].map(ab["known_games_missed"]).fillna(0.0)
        df["known_absence_reason"] = df["name_key"].map(ab["absence_reason"]).fillna("")
        df["known_absence_source"] = df["name_key"].map(ab["absence_source"]).fillna("")
        df["known_absence_date"] = df["name_key"].map(ab["absence_date"]).fillna("")
    else:
        df["known_games_missed"] = 0.0
        df["known_absence_reason"] = ""
        df["known_absence_source"] = ""
        df["known_absence_date"] = ""

    # ---------------------------------------------------------------- STEP 3 and STEP 4
    # A second, unbiased pass only exists to expose what the model would say without the
    # personal-preference penalties (team and/or player fades). With none configured the two
    # passes are identical by construction, so it is skipped rather than computed and thrown away.
    bias_active = league.personal_bias_active

    scored = run_step3(df, age_curves, contract_lifts, rookie_baselines, league, apply_team_bias=True)
    scored = compute_vorp(scored, league)

    if bias_active:
        unbiased = run_step3(
            df, age_curves, contract_lifts, rookie_baselines, league, apply_team_bias=False
        )
        unbiased = compute_vorp(unbiased, league)
        scored["pre_penalty_vorp"] = unbiased["vorp"]
        scored["pre_penalty_overall_rank"] = unbiased["overall_rank"]
        scored["pre_penalty_final_projected_season_points"] = unbiased["final_projected_season_points"]

    ranked = scored.sort_values("vorp", ascending=False).reset_index(drop=True)

    # Trim to the draftable board. This happens AFTER STEP 4 so replacement levels, position
    # ranks and the STEP 3a distributions were all derived from the full population -- trimming
    # first would move the replacement player and change every VORP. It happens BEFORE tiering
    # so tiers, volatility comparisons and consensus flags describe the players you can actually
    # draft, rather than being diluted by hundreds of identical fallback projections.
    #
    # Overall Rank and Position Rank stay correct without renumbering: taking the top N by VORP
    # keeps a prefix of each position's ordering, since VORP within a position differs from
    # projected points only by that position's constant replacement level.
    full_pool_size = len(ranked)
    limit = league.output_player_limit
    if limit is not None and full_pool_size > limit:
        ranked = ranked.head(limit).copy()

    ranked["tier"] = assign_tiers(ranked, value_col="vorp")
    ranked["position_tier"] = positional_tiers(ranked, value_col="vorp")

    # ---------------------------------------------------------------- volatility (parallel)
    dist = volatility.compute_weekly_distribution(weekly, target)
    dist = volatility.consistency_scores(dist, positions)
    vol_cols = [c for c in ("player_id", "cv", "cv_z", "floor", "median", "ceiling", "consistency_score", "weeks_sampled", "volatility_note") if c in dist.columns]
    ranked = ranked.merge(dist[vol_cols], on="player_id", how="left")
    ranked["same_ppg_volatility_flag"] = volatility.same_ppg_volatility_flags(ranked)

    # ---------------------------------------------------------------- camp riser/faller
    ranked["camp_buzz_flag"] = [
        camp_flag(r.camp_buzz_score, r.overall_rank, r.adp_blended, league.num_teams)
        for r in ranked.itertuples()
    ]

    # ---------------------------------------------------------------- STEP 5 (sanity check)
    if not ecr.empty:
        ecr = ecr.copy()
        ecr["name_key"] = ecr["player"].map(sleeper.normalize_name)
    ranked = step5_ecr.run_step5(ranked, ecr)

    # ---------------------------------------------------------------- market disagreement
    ranked["market_disagreement_flag"] = np.where(
        pd.to_numeric(ranked["adp_stdev"], errors="coerce") >= W.ADP_DISAGREEMENT_STDEV_THRESHOLD,
        "High ADP variance",
        "",
    )

    ranked["notes"] = _build_notes(ranked, camp)

    # ---------------------------------------------------------------- Raw-VORP board (separate workbook)
    # Objective projection value only: no personal fades, ranked by points-over-replacement
    # (vorp_raw), not the draft-scarcity-adjusted vorp used on Main Rankings.
    raw_base = unbiased if bias_active else scored
    raw_ranked = raw_base.sort_values("vorp_raw", ascending=False).reset_index(drop=True)
    if limit is not None and len(raw_ranked) > limit:
        raw_ranked = raw_ranked.head(limit).copy()
    else:
        raw_ranked = raw_ranked.copy()
    raw_ranked["overall_rank"] = np.arange(1, len(raw_ranked) + 1)
    raw_ranked["position_rank"] = (
        raw_ranked.groupby("position")["vorp_raw"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    raw_ranked["tier"] = assign_tiers(raw_ranked, value_col="vorp_raw")
    raw_ranked["position_tier"] = positional_tiers(raw_ranked, value_col="vorp_raw")
    raw_ranked = raw_ranked.merge(dist[vol_cols], on="player_id", how="left")
    raw_ranked["same_ppg_volatility_flag"] = volatility.same_ppg_volatility_flags(raw_ranked)
    raw_ranked["camp_buzz_flag"] = [
        camp_flag(r.camp_buzz_score, r.overall_rank, r.adp_blended, league.num_teams)
        for r in raw_ranked.itertuples()
    ]
    raw_ranked = step5_ecr.run_step5(raw_ranked, ecr if not ecr.empty else ecr)
    raw_ranked["market_disagreement_flag"] = np.where(
        pd.to_numeric(raw_ranked["adp_stdev"], errors="coerce") >= W.ADP_DISAGREEMENT_STDEV_THRESHOLD,
        "High ADP variance",
        "",
    )
    # Percent display helpers for the raw workbook (leave shares intact too).
    for share_col, pct_col in (
        ("snap_share", "snap_share_pct"),
        ("carry_share", "carry_share_pct_raw"),
        ("rz_carry_share", "rz_carry_share_pct"),
        ("target_share", "target_share_pct"),
        ("air_yards_share", "air_yards_share_pct"),
        ("rz_target_share", "rz_target_share_pct"),
        ("dropback_share", "dropback_share_pct"),
        ("designed_rush_share", "designed_rush_share_pct"),
        ("rz_pass_attempt_share", "rz_pass_attempt_share_pct"),
    ):
        if share_col in raw_ranked.columns:
            raw_ranked[pct_col] = pd.to_numeric(raw_ranked[share_col], errors="coerce") * 100.0
    raw_ranked["notes"] = _build_notes(raw_ranked, camp)

    # ---------------------------------------------------------------- K / DST (minimal)
    roster_teams = rosters_target.dropna(subset=["gsis_id"]).set_index("gsis_id")["team"]
    kickers = kdst.rank_kickers(weekly, target, roster_teams=roster_teams)
    defenses = kdst.rank_defenses(team_stats, schedules_hist, target)

    from .scoring.step4_vorp import replacement_levels as _levels

    return PipelineResult(
        rankings=ranked,
        raw_vorp_rankings=raw_ranked,
        kickers=kickers,
        defenses=defenses,
        age_curves=age_curves,
        contract_lifts=contract_lifts,
        rookie_baselines=rookie_baselines,
        replacement_levels=_levels(ranked, league),
        starter_counts=starter_counts(league),
        sources=log,
        camp_buzz_status=camp.status,
        camp_buzz_age_days=camp.age_days,
        run_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        league=league,
        bias_active=bias_active,
        diagnostics={
            "players_ranked": len(ranked),
            "full_pool_size": full_pool_size,
            "output_limit": league.output_player_limit,
            "adp_players": int(pd.to_numeric(ranked["adp_blended"], errors="coerce").notna().sum()),
            "ecr_matched": int(ranked["consensus_rank"].notna().sum()),
            "routes_source": (ranked["routes_source"].dropna().iloc[0] if "routes_source" in ranked.columns and ranked["routes_source"].notna().any() else "unavailable"),
            # Counted both in the override files and on the trimmed board, because a player can
            # legitimately be in a file yet rank outside the output limit (a season-ending
            # absence drops him to the bottom). Reporting only the board count reads as though
            # the file had not been loaded.
            "known_absences": int((pd.to_numeric(ranked["known_games_missed"], errors="coerce").fillna(0) > 0).sum()),
            "known_absences_in_file": int(len(absences)),
            "camp_buzz_players": int((pd.to_numeric(ranked["camp_buzz_score"], errors="coerce").fillna(0) != 0).sum()),
            "camp_buzz_in_file": int(len(camp.scores)),
        },
    )


def _build_notes(df: pd.DataFrame, camp) -> pd.Series:
    """Assemble the plain-text Notes/Sourcing column.

    Anything news-driven carries its source and date; anything non-obvious carries the reason.
    """
    def _txt(value) -> str:
        """Text fields arrive as NaN when a left join found no match; treat those as empty."""
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    notes = []
    for row in df.itertuples():
        bits = []
        limited = limited_sample_note({"limited_sample": getattr(row, "limited_sample", False),
                                       "seasons_used": getattr(row, "seasons_used", None)})
        if limited:
            bits.append(limited)
        if getattr(row, "is_rookie", False):
            bits.append(_txt(getattr(row, "rookie_note", "")))
        bits.append(_txt(getattr(row, "composite_note", "")))
        bits.append(_txt(getattr(row, "rb_committee_note", "")))
        bits.append(_txt(getattr(row, "injury_history_note", "")))
        known = getattr(row, "known_games_missed", 0) or 0
        if known and float(known) > 0:
            src = getattr(row, "known_absence_source", "") or "unsourced"
            date = getattr(row, "known_absence_date", "") or "no date"
            bits.append(
                f"Known current-season absence: {float(known):.0f} games "
                f"({getattr(row, 'known_absence_reason', '')}; source: {src}, {date})"
            )
        camp_score = getattr(row, "camp_buzz_score", 0) or 0
        if not pd.isna(camp_score) and float(camp_score) != 0:
            lim = bool(getattr(row, "limited_sample", False))
            damp = " (limited-sample dampened)" if lim else ""
            bits.append(
                f"Camp buzz {float(camp_score):+.0f} -> x{getattr(row, 'camp_buzz_multiplier', 1.0):.2f}"
                f"{damp} "
                f"(source: {_txt(getattr(row, 'camp_buzz_source', ''))}, {_txt(getattr(row, 'camp_buzz_date', ''))})"
            )
        if getattr(row, "contract_year", False):
            bits.append(_txt(getattr(row, "contract_year_note", "")) or "Contract year")
        bits.append(_txt(getattr(row, "age_curve_note", "")))
        if str(getattr(row, "team_bias_flag", "N")) == "Y":
            bits.append(
                f"Personal-preference team penalty applied (x{getattr(row, 'team_bias_multiplier', 1.0):.2f}, "
                "not objective analysis); see pre-penalty VORP and rank columns"
            )
        if str(getattr(row, "player_fade_flag", "N")) == "Y":
            reason = _txt(getattr(row, "player_fade_reason", "")) or "Personal fade"
            bits.append(
                f"{reason} (x{getattr(row, 'player_fade_multiplier', 1.0):.2f}, not objective analysis); "
                "see pre-penalty VORP and rank columns"
            )
        bits.append(_txt(getattr(row, "coach_change_note", "")))
        bits.append(_txt(getattr(row, "same_ppg_volatility_flag", "")))
        bits.append(_txt(getattr(row, "volatility_note", "")))
        bits.append(_txt(getattr(row, "ecr_reason", "")))
        if _txt(getattr(row, "market_disagreement_flag", "")):
            stdev = getattr(row, "adp_stdev", float("nan"))
            stdev_txt = "unknown" if pd.isna(stdev) else f"{float(stdev):.1f}"
            bits.append(
                f"Market disagreement: ADP stdev {stdev_txt} picks "
                f"(tracked separately; not folded into Composite Z-Score)"
            )
        routes_src = _txt(getattr(row, "routes_source", ""))
        if routes_src == "pass-snap proxy":
            bits.append("YPRR from pass-snap proxy (participation unavailable) - insufficient data for true routes")
        notes.append(" | ".join(b for b in bits if b))
    return pd.Series(notes, index=df.index)
