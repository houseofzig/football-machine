"""
Backtest harness: runs the weekly projection engine against a past season,
week by week, using only data that would have been available at the time
(project_player_week already respects season/week cutoffs - no lookahead).

"Top players" for a season are picked by actual season-end PPR points (a
simple, defensible proxy for "who mattered" - not the model's own opinion).
"""
from collections import defaultdict

from .db import select_all
from .projection import project_player_week


def get_top_players(supabase, season, position, top_n):
    filters = [
        lambda q: q.eq("season", season),
        lambda q: q.eq("season_type", "REG"),
        lambda q: q.eq("position", position),
    ]
    rows = select_all(supabase, "player_stats_weekly", "player_id,player_name,team,fantasy_points_ppr", filters=filters)
    totals = defaultdict(lambda: {"name": None, "team": None, "total": 0.0})
    for r in rows:
        t = totals[r["player_id"]]
        t["name"] = r["player_name"]
        t["team"] = r["team"]
        t["total"] += r.get("fantasy_points_ppr") or 0
    ranked = sorted(totals.items(), key=lambda kv: kv[1]["total"], reverse=True)
    return [{"player_id": pid, **info} for pid, info in ranked[:top_n]]


def _player_game_rows(supabase, player_id, season):
    filters = [
        lambda q: q.eq("player_id", player_id),
        lambda q: q.eq("season", season),
        lambda q: q.eq("season_type", "REG"),
    ]
    rows = select_all(supabase, "player_stats_weekly", "*", filters=filters)
    rows.sort(key=lambda r: r["week"])
    return rows


def backtest_player_season(supabase, player_id, position, season, start_week=1):
    game_rows = _player_game_rows(supabase, player_id, season)
    weekly = []
    for row in game_rows:
        week = row["week"]
        if week < start_week:
            continue
        team = row["team"]
        opponent = row["opponent_team"]
        actual_pts = row.get("fantasy_points_ppr") or 0
        proj = project_player_week(supabase, player_id, position, team, opponent, season, week)
        weekly.append({
            "week": week,
            "team": team,
            "opponent": opponent,
            "actual_points": round(actual_pts, 2),
            "projected_points": proj.get("projected_points"),
            "source": proj.get("source"),
        })
    return weekly


def backtest_season(supabase, season, top_n=None, start_week=1):
    """
    Runs the backtest for top 24 RB/WR and top 12 QB/TE (or a custom top_n
    dict like {"RB": 24, "WR": 24, "QB": 12, "TE": 12}).
    Returns {position: [{player_id, name, actual_season_total, projected_season_total, weekly: [...]}]}
    """
    top_n = top_n or {"RB": 24, "WR": 24, "QB": 12, "TE": 12}
    results = {}
    for position, n in top_n.items():
        print(f"Backtesting {position} (top {n})...")
        top_players = get_top_players(supabase, season, position, n)
        position_results = []
        for p in top_players:
            weekly = backtest_player_season(supabase, p["player_id"], position, season, start_week)
            actual_total = sum(w["actual_points"] for w in weekly)
            projected_total = sum(w["projected_points"] for w in weekly if w["projected_points"] is not None)
            position_results.append({
                "player_id": p["player_id"],
                "name": p["name"],
                "team": p["team"],
                "actual_season_total": round(actual_total, 1),
                "projected_season_total": round(projected_total, 1),
                "weeks_projected": len([w for w in weekly if w["projected_points"] is not None]),
                "weekly": weekly,
            })
            print(f"  {p['name']}: actual={round(actual_total,1)} projected={round(projected_total,1)}")
        results[position] = position_results
    return results
