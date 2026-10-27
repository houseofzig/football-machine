"""
Populates weekly_projections_2026 and team_weekly_envelope_2026 (see
schema_weekly_ros_tables.sql) for a given season/week - the shared baseline
tables the site's admin-edit UI reads/writes, same shape as
projections_2026_resolved.

Touchdown fields (proj_rush_td_rate, proj_rec_td_rate, proj_td_rate) are
back-calculated as an equivalent rate (implied TDs / volume) rather than
stored directly, since the schema keeps the site's existing "rate x volume"
editable shape even though the underlying model computes TDs top-down from
Vegas-implied team totals now, not as a simple rate (see common/projection.py
module docstring). Editing one of these fields on the site reverts that
player to old-style rate x volume math going forward, same as today.

Player identity uses sleeper_id, not player_id (gsis_id/NFL ID) - the NFL ID
is NULL for ~45% of players in projections_2026_resolved (deeper bench
players added later via sync_sleeper_depth_charts.py never got a gsis_id
looked up), which risked duplicate rows on re-run (Postgres treats NULL !=
NULL in unique constraints). Sleeper's player database has no such gap.
The sleeper_id comes from the projections_2026_resolved row itself (column
added and backfilled by ID 2026-09-15). Only a row without one falls back
to the gsis crosswalk (common/db.py:build_sleeper_id_crosswalk - ff_playerids
is the canonical gsis_id<->sleeper_id<->espn_id<->pfr_id mapping table,
don't rebuild it). No name matching.
"""
import sys

from common import matchup
from common.db import build_fumble_rate_lookup, build_sleeper_id_crosswalk, get_client, get_fumble_rate, select_all, upsert_batched
from common.projection import compute_team_envelope, project_player_week
from common.ros_model import starting_qb_ids
from common.sources import get_inactive_weekly_gsis_ids
from common.team_shares import normalize_team_shares
from run_mock_projection import POSITIONS, build_opponent_map, get_player_pool
from collections import defaultdict


def _normalize_name(name):
    return (name or "").strip().lower()


def apply_week2_special_blend(supabase, season):
    """One-time override, week 2 only (user request, 2026-09-16): the normal
    model's "career" tier (preseason model output) is replaced with each
    player's already-computed ros_projections_2026 row, and the last-2-
    seasons tier is fixed at a flat 2/7 of the non-current-season weight -
    instead of the model's own confidence-scaled _last2_weight(). Combined
    with weight_current already being 30% at 1 game played this season (see
    CURRENT_SEASON_WEIGHT_BY_GAME in common/matchup.py), the nested blend
    _blend(_blend(career, last2, last2_weight), current, weight_current)
    algebraically works out to EXACTLY 30% current-season / 20% last-2-
    seasons / 50% ROS - confirmed by hand: last2_weight=2/7 makes
    prior_combined = 5/7*ROS + 2/7*last2, then *0.7 (=1-0.30) + current*0.30
    = 0.5*ROS + 0.2*last2 + 0.3*current.

    Does NOT touch compute_player_baseline/compute_player_share_baseline
    (the permanent model, shared with every other week and with ROS) - it
    monkeypatches the lower-level functions those look up their "career"
    input and last2 weight through (get_preseason_baseline,
    get_preseason_share_baseline, _last2_weight), which their own module
    globals resolve at call time regardless of how they were imported
    elsewhere - so this is invisible to every other caller. Only ever
    called by this script, only when week==2 - see run()'s call site.
    """
    ros_rows = select_all(supabase, "ros_projections_2026", "*", filters=[lambda q: q.eq("season", season)])
    by_id, by_name = {}, {}
    for r in ros_rows:
        if r.get("player_id"):
            by_id[r["player_id"]] = r
        if r.get("player_name"):
            by_name[_normalize_name(r["player_name"])] = r
    print(f"[week2 special blend] {len(ros_rows)} ROS rows loaded ({len(by_id)} by player_id, {len(by_name)} by name)")

    def _find(player_id, player_name):
        return by_id.get(player_id) or by_name.get(_normalize_name(player_name))

    # ros_projections_2026 doesn't cover every player in the pool (742 rows vs
    # ~889 in the pool) - a player missing from it must fall back to the REAL
    # preseason baseline and the model's normal confidence-scaled last2
    # weight, completely unaffected by this special blend, not silently lose
    # their career-tier data entirely (confirmed real bug: first version of
    # this returned None on a miss, which skipped 179 players instead of the
    # normal ~26). Since get_preseason_X is always called immediately before
    # _last2_weight within the SAME compute_player_baseline/
    # compute_player_share_baseline call (single-threaded, one player at a
    # time - see project_player_week), this flag set-then-read is safe.
    original_get_preseason_baseline = matchup.get_preseason_baseline
    original_get_preseason_share_baseline = matchup.get_preseason_share_baseline
    original_last2_weight = matchup._last2_weight
    had_ros = {"value": False}

    def patched_get_preseason_baseline(supabase, player_id, player_name, position):
        row = _find(player_id, player_name)
        if row:
            had_ros["value"] = True
            return matchup.get_ros_baseline_from_row(row, position)
        had_ros["value"] = False
        return original_get_preseason_baseline(supabase, player_id, player_name, position)

    def patched_get_preseason_share_baseline(supabase, player_id, player_name):
        row = _find(player_id, player_name)
        if row:
            had_ros["value"] = True
            return matchup.get_ros_share_baseline_from_row(row)
        had_ros["value"] = False
        return original_get_preseason_share_baseline(supabase, player_id, player_name)

    def patched_last2_weight(games_count):
        if had_ros["value"]:
            return 2 / 7
        return original_last2_weight(games_count)

    matchup.get_preseason_baseline = patched_get_preseason_baseline
    matchup.get_preseason_share_baseline = patched_get_preseason_share_baseline
    matchup._last2_weight = patched_last2_weight


def build_player_row(season, week, p, team, opponent, proj, sleeper_id, fum_lookup):
    stat_line = proj["stat_line"]
    eff = proj["efficiencies"]
    shares = proj["shares"]

    def adj(stat):
        return (eff.get(stat) or {}).get("adjusted")

    row = {
        "season": season, "week": week,
        "sleeper_id": sleeper_id, "player_id": p["player_id"], "player_name": p["player_name"],
        "team": team, "position": p["position"], "slot_key": p.get("slot_key"),
        "proj_carry_share": shares.get("carry_share"),
        "proj_target_share": shares.get("target_share"),
        "proj_rush_ypc": adj("rush_ypc"),
        "proj_catch_rate": adj("catch_rate"),
        "proj_rec_ypr": adj("rec_ypr"),
        "proj_fum_lost_per_touch": get_fumble_rate(p["player_id"], p["player_name"], fum_lookup),
    }

    carries = stat_line.get("carries") or 0
    rush_tds = stat_line.get("rush_tds") or 0
    row["proj_rush_td_rate"] = round(rush_tds / carries, 4) if carries else 0

    if p["position"] == "QB":
        attempts = stat_line.get("attempts") or 0
        pass_tds = stat_line.get("pass_tds") or 0
        row["proj_ypa"] = adj("pass_ypa")
        row["proj_comp_pct"] = adj("comp_pct")
        row["proj_int_rate"] = adj("pass_int_rate")
        row["proj_td_rate"] = round(pass_tds / attempts, 4) if attempts else 0
    else:
        receptions = stat_line.get("receptions") or 0
        rec_tds = stat_line.get("rec_tds") or 0
        row["proj_rec_td_rate"] = round(rec_tds / receptions, 4) if receptions else 0

    return row


def run(season, week):
    supabase = get_client()
    # Scoped to season 2026's week 2 specifically, not "week 2" forever - the
    # user's own framing was "week 2 only, for now" (2026-09-16), not a
    # permanent rule for whatever a future season's week 2 happens to be.
    if season == 2026 and week == 2:
        apply_week2_special_blend(supabase, season)
    opponent_map = build_opponent_map(supabase, season, week)
    pool = get_player_pool(supabase)
    crosswalk = build_sleeper_id_crosswalk(supabase)
    fum_lookup = build_fumble_rate_lookup(supabase)
    # Live snapshot of who's not playing AT ALL this week (IR/Out/PUP/Sus/NA/
    # COV/DNR - see INACTIVE_WEEKLY_STATUSES) - user, 2026-09-29: "if a player
    # is listed on IR then their share inputs should be 0... this means they
    # are not playing this week." No try/except here on purpose (user: "if it
    # fails then rerun") - a failed fetch should stop this run, not silently
    # produce a week of projections missing real inactives.
    inactive_gsis_ids = get_inactive_weekly_gsis_ids(supabase)
    print(f"Player pool: {len(pool)} players. Teams with a week {week} opponent: {len(opponent_map)}. Inactive this week (IR/Out/etc): {len(inactive_gsis_ids)}")
    # Starting QB per team (user, 2026-10-27: same rule as ROS - manual QB overrides, then
    # Sleeper's depth chart by sleeper_id, then the stored slot _0). The stored slot alone
    # gave Rudolph/Tua the team's passing TDs and left Rodgers/Penix/McKee at 0.
    starter_ids = starting_qb_ids(pool)

    by_team = defaultdict(list)
    for p in pool:
        by_team[p["team"]].append(p)

    envelope_rows = []
    player_rows = []
    skipped = 0
    no_sleeper_id = []

    for i, (team, team_players) in enumerate(by_team.items()):
        opponent = opponent_map.get(team)
        if not opponent:
            skipped += len(team_players)
            continue

        envelope = compute_team_envelope(supabase, team, opponent, season, week)
        envelope_rows.append({
            "season": season, "week": week, "team": team,
            "total_plays": envelope["total_plays"],
            "run_pct": envelope["run_pct"],
            "assumed_points": envelope["assumed_points"],
        })

        normalized_shares = normalize_team_shares(supabase, team, team_players, season, week, exclude_ids=inactive_gsis_ids, starter_ids=starter_ids)
        for p in team_players:
            # The sleeper_id stored on the projections_2026_resolved row is the
            # player's identity (backfilled by ID, 2026-09-15). A row without
            # one falls back to the gsis crosswalk only - never a name match,
            # which put 2 players under someone else's ID in the Week 2 dry run.
            sleeper_id = p.get("sleeper_id") or crosswalk["by_gsis"].get(p["player_id"])
            if not sleeper_id:
                # Genuine gap - no stored sleeper_id and no gsis match. Skip
                # rather than write a row with no reliable identity key.
                no_sleeper_id.append(p["player_name"])
                continue
            share_override = normalized_shares.get(p["player_name"])
            # IR/Out/etc. wins over the normal team-share split - a player not
            # on the field this week gets none of the team's volume regardless
            # of whatever their role would otherwise imply (see
            # get_inactive_weekly_gsis_ids's own docstring for why this stays
            # scoped to the weekly pipeline only, never the ROS one).
            if p["player_id"] in inactive_gsis_ids:
                share_override = {"carry_share": 0, "target_share": 0, "pass_attempt_share": 0}
            proj = project_player_week(
                supabase, p["player_id"], p["position"], team, opponent, season, week,
                player_name=p["player_name"], share_override=share_override,
            )
            if proj.get("source") in ("no_history", "no_team_history"):
                skipped += 1
                continue
            player_rows.append(build_player_row(season, week, p, team, opponent, proj, sleeper_id, fum_lookup))

        if (i + 1) % 8 == 0:
            print(f"  processed {i+1}/{len(by_team)} teams")

    print(f"\nReady to write: {len(envelope_rows)} team rows, {len(player_rows)} player rows. Skipped {skipped}.")
    if no_sleeper_id:
        print(f"No sleeper_id found for {len(no_sleeper_id)} players (excluded): {no_sleeper_id[:10]}{'...' if len(no_sleeper_id) > 10 else ''}")
    return envelope_rows, player_rows


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    dry_run = "--write" not in sys.argv

    envelope_rows, player_rows = run(season, week)

    if dry_run:
        print("\nDRY RUN - nothing written. Pass --write to actually upsert into Supabase.")
        print("\nSample team row:", envelope_rows[0] if envelope_rows else None)
        print("Sample player row:", player_rows[0] if player_rows else None)
    else:
        supabase = get_client()
        upsert_batched(supabase, "team_weekly_envelope_2026", envelope_rows, on_conflict="season,week,team")
        upsert_batched(supabase, "weekly_projections_2026", player_rows, on_conflict="season,week,sleeper_id")
        print("Done.")
