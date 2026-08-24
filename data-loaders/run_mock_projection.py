"""
Mock weekly projection run: projects every QB/RB/WR/TE in the current
preseason player pool (projections_2026_resolved - the same table the live
dashboard reads) for a given season/week, using the full pipeline (baseline +
matchup deltas + Vegas scaling), with the preseason model as the fallback for
anyone without enough real game history yet (i.e. everyone, for week 1).

Does not write anywhere - prints results and saves a local JSON file only.
"""
import json
import sys
import time

from collections import defaultdict

from common.db import get_client, select_all
from common.projection import project_player_week
from common.team_shares import normalize_team_shares

POSITIONS = ("QB", "RB", "WR", "TE")


def build_opponent_map(supabase, season, week):
    """team -> opponent_team for this week, from the schedules table."""
    rows = select_all(
        supabase, "schedules", "home_team,away_team",
        filters=[
            lambda q: q.eq("season", season),
            lambda q: q.eq("week", week),
            lambda q: q.eq("game_type", "REG"),
        ],
    )
    m = {}
    for r in rows:
        m[r["home_team"]] = r["away_team"]
        m[r["away_team"]] = r["home_team"]
    return m


def get_player_pool(supabase):
    """
    Only one QB plays per team per week - unlike RB/WR/TE, which genuinely
    split touches across multiple rostered players, so those stay unfiltered.
    slot_key (e.g. 'NYG_qb_0') encodes depth chart rank, 0 = starter; without
    this filter every backup QB with any real 2024/2025 attempts got treated
    as a starter too (confirmed bug: Jameis Winston, Anthony Richardson,
    Daniel Jones, Riley Leonard all projected as QB1 on their own teams).
    """
    rows = select_all(
        supabase, "projections_2026_resolved", "player_id,player_name,team,position,slot_key",
        filters=[lambda q: q.eq("projection_year", 2026), lambda q: q.in_("position", list(POSITIONS))],
    )
    rows = [r for r in rows if r["position"] != "QB" or (r.get("slot_key") or "").endswith("_0")]
    seen = set()
    pool = []
    for r in rows:
        key = (r["player_name"], r["team"], r["position"])
        if key in seen:
            continue
        seen.add(key)
        pool.append(r)
    return pool


def run(season, week):
    supabase = get_client()
    opponent_map = build_opponent_map(supabase, season, week)
    pool = get_player_pool(supabase)
    print(f"Player pool: {len(pool)} players. Teams with a week {week} opponent: {len(opponent_map)}")

    by_team = defaultdict(list)
    for p in pool:
        by_team[p["team"]].append(p)

    results = {pos: [] for pos in POSITIONS}
    skipped = []
    t0 = time.time()
    processed = 0

    for team, team_players in by_team.items():
        opponent = opponent_map.get(team)
        if not opponent:
            for p in team_players:
                skipped.append((p["player_name"], team, "no scheduled opponent"))
            continue

        normalized_shares = normalize_team_shares(supabase, team, team_players, season, week)

        for p in team_players:
            share_override = normalized_shares.get(p["player_name"])
            proj = project_player_week(
                supabase, p["player_id"], p["position"], team, opponent, season, week,
                player_name=p["player_name"], share_override=share_override,
            )
            processed += 1
            if proj.get("source") in ("no_history", "no_team_history"):
                skipped.append((p["player_name"], team, proj.get("note")))
                continue
            results[p["position"]].append({
                "name": p["player_name"], "team": team, "opponent": opponent,
                "source": proj["source"], "projected_points": proj["projected_points"],
                "stat_line": proj["stat_line"],
                "efficiencies": proj["efficiencies"],
                "shares": proj["shares"],
            })
            if processed % 100 == 0:
                print(f"  processed {processed}/{len(pool)} ({time.time()-t0:.0f}s elapsed)")

    for pos in POSITIONS:
        results[pos].sort(key=lambda r: -r["projected_points"])

    print(f"\nDone in {time.time()-t0:.1f}s. Skipped {len(skipped)} players.")
    for pos in POSITIONS:
        print(f"\nTop 10 {pos}:")
        for r in results[pos][:10]:
            print(f"  {r['name']:22} {r['team']:3} vs {r['opponent']:3}  {r['projected_points']:>6.1f} pts  [{r['source']}]")

    return results, skipped


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    results, skipped = run(season, week)
    out_path = f"mock_projection_{season}_wk{week}.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "skipped": skipped}, f, indent=2)
    print(f"\nSaved to {out_path}")
