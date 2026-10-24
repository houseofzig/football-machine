"""
2026-10-22 REDESIGN (user spec): the player and team math below now lives in
common/ros_model.py - 60% this season (ramped by games) + the 10 valid games before
this season + 25% last-saved ROS; team plays/run split/yards from this season only.
SAVED_ROS_WEIGHT below is kept for reference only. See that module's docstring.

Populates ros_projections_2026 and team_ros_envelope_2026 (see
schema_weekly_ros_tables.sql) - the rest-of-season counterpart to
write_weekly_tables.py.

ROS uses the same player rate-stat baseline (compute_player_baseline /
compute_player_share_baseline - the career < last-2-seasons < current-season
blend) and team attempt-pool baseline (compute_team_attempts_baseline) as
weekly, but with NO opponent-specific adjustment layered on top - no
position-efficiency delta, no team-volume delta, no Vegas-implied TD split.
That machinery only makes sense against one specific opponent in one specific
week; ROS spans the rest of the season against many different opponents, so
those effects should roughly average out anyway. This makes ROS numbers a
plain per-game rate (points-per-game), matching how the existing preseason
model already works - not a summed season total.

Before a player has any real qualifying game this season, their ROS row is a
straight, unmodified copy of the preseason model's own numbers (per user:
"straight copy until games start", not a blend with 2024/2025 history - even
though compute_player_baseline's normal blend would pull that in for any
returning player). Once a player has at least one qualifying game this
season, this switches to the normal compute_player_baseline/share blend,
which (via the fast-ramp) leans on that real current-season data almost
immediately anyway.

Player identity uses sleeper_id (see common/db.py:build_sleeper_id_crosswalk),
same reasoning as write_weekly_tables.py.
"""
import sys

from common.db import build_fumble_rate_lookup, build_sleeper_id_crosswalk, get_client, get_fumble_rate, select_all, upsert_batched
from common.matchup import (
    compute_player_baseline,
    compute_team_attempts_baseline,
    get_preseason_baseline,
    get_preseason_share_baseline,
)
from common.sources import ROS_LONG_TERM_INACTIVE_STATUSES, get_inactive_weekly_gsis_ids
from common.ros_model import (
    compute_ros_player_baseline, compute_ros_player_shares, compute_ros_team_envelope, games_remaining_by_team,
    starting_qb_ids,
)
from common.team_shares import normalize_team_shares
from run_mock_projection import get_player_pool
from collections import defaultdict


# Share of the new ROS that is the player's LAST-SAVED ROS baseline (user,
# 2026-10-13: "make the current ROS baseline 25% of the model"). The other 75%
# is the model as before (last-2-seasons / current-season blend, 30/40/60/80/100
# ramp) but with the saved ROS no longer a tier INSIDE it - it is folded in once,
# here, at this weight, whatever the games played (so it still counts at 5+
# games, where it used to drop out). ROS write only: the weekly pipeline keeps
# the old behavior (compute_player_baseline's default saved_ros_weight=None).
# A player with no qualifying game this season keeps their saved ROS untouched
# (user's call), instead of the old straight preseason copy.
SAVED_ROS_WEIGHT = 0.25


def build_player_row(season, as_of_week, p, team, baseline, shares, sleeper_id, fum_lookup):
    row = {
        "season": season, "as_of_week": as_of_week,
        "sleeper_id": sleeper_id, "player_id": p["player_id"], "player_name": p["player_name"],
        "team": team, "position": p["position"], "slot_key": p.get("slot_key"),
        "proj_carry_share": shares.get("carry_share"),
        "proj_target_share": shares.get("target_share"),
        "proj_rush_ypc": baseline.get("rush_ypc"),
        "proj_rush_td_rate": baseline.get("rush_td_rate"),
        "proj_catch_rate": baseline.get("catch_rate"),
        "proj_rec_ypr": baseline.get("rec_ypr"),
        "proj_rec_td_rate": baseline.get("rec_td_rate"),
        "proj_fum_lost_per_touch": get_fumble_rate(p["player_id"], p["player_name"], fum_lookup),
    }
    if p["position"] == "QB":
        row["proj_ypa"] = baseline.get("pass_ypa")
        row["proj_comp_pct"] = baseline.get("comp_pct")
        row["proj_td_rate"] = baseline.get("pass_td_rate")
        row["proj_int_rate"] = baseline.get("pass_int_rate")
    return row


def run(season, as_of_week):
    supabase = get_client()
    pool = get_player_pool(supabase)
    crosswalk = build_sleeper_id_crosswalk(supabase)
    fum_lookup = build_fumble_rate_lookup(supabase)
    # IR / PUP / suspended players are left out of their team's share scaling (user,
    # 2026-10-22) so their share goes to the players who are available. Their own
    # rows stay normal ROS projections (they are not zeroed - they come back).
    long_term_out_ids = get_inactive_weekly_gsis_ids(supabase, statuses=ROS_LONG_TERM_INACTIVE_STATUSES)
    print(f"Player pool: {len(pool)} players. Computing ROS baselines as of week {as_of_week}... ({len(long_term_out_ids)} IR/PUP/Sus players left out of team share scaling)")

    games_left = games_remaining_by_team(supabase, season, as_of_week)
    starter_ids = starting_qb_ids(pool)  # manual QB overrides first, then Sleeper depth chart (by ID)
    by_team = defaultdict(list)
    for p in pool:
        by_team[p["team"]].append(p)

    envelope_rows = []
    player_rows = []
    skipped = 0
    kept_saved = []  # zero qualifying 2026 games + a saved ROS row: left exactly as saved
    no_sleeper_id = []

    for i, (team, team_players) in enumerate(by_team.items()):
        # Team volume: this season only, per game (ros_model.compute_ros_team_envelope).
        env = compute_ros_team_envelope(supabase, team, season, as_of_week)
        if env is not None and env["total_plays"]:
            envelope_rows.append({
                "season": season, "as_of_week": as_of_week, "team": team,
                "total_plays": round(env["total_plays"], 2),
                "run_pct": round(env["run_pct"], 4) if env["run_pct"] is not None else None,
                # not columns of team_ros_envelope_2026 (dropped by the upsert's column filter) - used by the
                # Team Adjustment hand-off and the dry-run report
                "ypa": round(env["ypa"], 3) if env["ypa"] is not None else None,
                "ypc": round(env["ypc"], 3) if env["ypc"] is not None else None,
                "games_played": env["games_played"],
                "games_remaining": games_left.get(team),
            })

        normalized_shares = normalize_team_shares(supabase, team, team_players, season, as_of_week, exclude_ids=long_term_out_ids, share_fn=compute_ros_player_shares, starter_ids=starter_ids, lock_top=(2, 1))
        for p in team_players:
            # By stored ID only, never a name match (same fix write_weekly_tables.py
            # got 2026-09-15: the name fallback excluded ~157 players that had a
            # stored sleeper_id and wrote 2 under someone else's ID).
            sleeper_id = p.get("sleeper_id") or crosswalk["by_gsis"].get(p["player_id"])
            if not sleeper_id:
                no_sleeper_id.append(p["player_name"])
                continue
            baseline = compute_ros_player_baseline(supabase, p["player_id"], p["position"], season, as_of_week, p["player_name"])
            if baseline.get("source") == "no_history":
                skipped += 1
                continue
            if baseline.get("source") == "ros_kept":
                kept_saved.append(p["player_name"])
                continue

            if baseline.get("games_used", 0) == 0:
                # No real 2026 game yet - straight copy of preseason, not the
                # normal blend (which would pull in real 2024/2025 stats).
                preseason_eff = get_preseason_baseline(supabase, p["player_id"], p["player_name"], p["position"])
                if preseason_eff is None:
                    skipped += 1
                    continue
                baseline = {"source": "preseason_fallback", "games_used": 0, **preseason_eff}
                shares = get_preseason_share_baseline(supabase, p["player_id"], p["player_name"]) or {}
            else:
                shares = normalized_shares.get(p["player_name"], {})

            player_rows.append(build_player_row(season, as_of_week, p, team, baseline, shares, sleeper_id, fum_lookup))

        if (i + 1) % 8 == 0:
            print(f"  processed {i+1}/{len(by_team)} teams")

    # PINNED values (user, 2026-10-22): proj_rec_ypr / proj_rec_td_rate flagged *_pinned in the saved ROS row are
    # deliberate admin overrides - keep the saved number instead of the model's (same key set on every row,
    # which a bulk upsert needs, so the saved value is written back rather than the key dropped).
    saved_pins = {str(r["sleeper_id"]): r for r in select_all(
        supabase, "ros_projections_2026", "sleeper_id,proj_rec_ypr,proj_rec_td_rate,proj_rec_ypr_pinned,proj_rec_td_pinned",
        filters=[lambda q: q.eq("season", season)]) if r.get("sleeper_id")}
    pin_ypr = pin_td = 0
    for row in player_rows:
        sv = saved_pins.get(str(row["sleeper_id"]))
        if sv and sv.get("proj_rec_ypr_pinned") and sv.get("proj_rec_ypr") is not None:
            row["proj_rec_ypr"] = sv["proj_rec_ypr"]; pin_ypr += 1
        if sv and sv.get("proj_rec_td_pinned") and sv.get("proj_rec_td_rate") is not None:
            row["proj_rec_td_rate"] = sv["proj_rec_td_rate"]; pin_td += 1
    print(f"Pinned values kept as saved: yds/catch {pin_ypr}, TD rate {pin_td}")

    print(f"\nReady to write: {len(envelope_rows)} team rows, {len(player_rows)} player rows. Skipped {skipped}.")
    print(f"Kept exactly as saved (no qualifying 2026 game yet, saved ROS row exists - not rewritten): {len(kept_saved)}")
    if no_sleeper_id:
        print(f"No sleeper_id found for {len(no_sleeper_id)} players (excluded): {no_sleeper_id[:10]}{'...' if len(no_sleeper_id) > 10 else ''}")
    return envelope_rows, player_rows


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    as_of_week = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    dry_run = "--write" not in sys.argv

    envelope_rows, player_rows = run(season, as_of_week)

    if dry_run:
        print("\nDRY RUN - nothing written. Pass --write to actually upsert into Supabase.")
        print("\nSample team row:", envelope_rows[0] if envelope_rows else None)
        print("Sample player row:", player_rows[0] if player_rows else None)
    else:
        supabase = get_client()
        upsert_batched(supabase, "team_ros_envelope_2026", envelope_rows, on_conflict="season,team")
        upsert_batched(supabase, "ros_projections_2026", player_rows, on_conflict="season,sleeper_id")
        print("Done.")
