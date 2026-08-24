"""
Phase 2: Vegas-implied scoring layer.

spread_line/total_line come from the schedules table (already loaded via
nflreadpy - Sleeper has no betting-odds data at all, so this stays on
nflreadpy regardless of the player-stats source priority).

nflverse convention (verified against real 2025 completed games): spread_line
is the number of points the HOME team is favored by (negative = home is the
underdog). So:
  implied_home_total = (total_line + spread_line) / 2
  implied_away_total = (total_line - spread_line) / 2

This week's implied team total is decomposed into an implied touchdown count
(see compute_implied_team_tds) - not applied as a blanket multiplier on a
player's final points, which would treat touchdown-driven scoring and
yardage-driven scoring identically and double-count volume (already adjusted
via compute_team_volume_delta in common/matchup.py).
"""
from functools import lru_cache

from .db import select_all
from .matchup import RECENT_WINDOW, QUALIFYING_GAME_THRESHOLD, _blend, _current_season_weight


def implied_team_total(schedule_row, team):
    total = schedule_row.get("total_line")
    spread = schedule_row.get("spread_line")
    if total is None or spread is None:
        return None
    if schedule_row["home_team"] == team:
        return (total + spread) / 2
    elif schedule_row["away_team"] == team:
        return (total - spread) / 2
    return None


@lru_cache(maxsize=None)
def get_schedule_row(supabase, team, season, week):
    rows = select_all(
        supabase, "schedules", "*",
        filters=[
            lambda q: q.eq("season", season),
            lambda q: q.eq("week", week),
            lambda q: q.eq("game_type", "REG"),
            lambda q: q.or_(f"home_team.eq.{team},away_team.eq.{team}"),
        ],
    )
    return rows[0] if rows else None


# Points per touchdown drive: 6 for the TD itself + ~0.94 for the PAT (NFL PAT
# make rate is consistently ~94%) + a small allowance for 2-point tries.
# Field-goal points (3 each) are the remainder of a team's actual scoring -
# never allocated to skill players, since kickers aren't projected here.
POINTS_PER_TD = 6.94


def _team_actual_scoring_game_log(supabase, team, before_season, before_week, seasons_back=2):
    """
    Real (not implied) points scored and offensive TDs scored per game, from
    schedules' actual final scores and player_stats_weekly's rushing_tds +
    passing_tds (receiving_tds is the same event as passing_tds at the team
    level - crediting both would double it). Used to learn each team's own
    historical split of "how much of our scoring is touchdowns vs field goals."
    """
    min_season = before_season - seasons_back
    sched_rows = select_all(
        supabase, "schedules", "season,week,home_team,away_team,home_score,away_score",
        filters=[
            lambda q: q.or_(f"home_team.eq.{team},away_team.eq.{team}"),
            lambda q: q.gte("season", min_season),
            lambda q: q.eq("game_type", "REG"),
        ],
    )
    sched_rows = [r for r in sched_rows if (r["season"], r["week"]) < (before_season, before_week)]

    stat_rows = select_all(
        supabase, "player_stats_weekly", "season,week,rushing_tds,passing_tds",
        filters=[
            lambda q: q.eq("team", team),
            lambda q: q.eq("season_type", "REG"),
            lambda q: q.gte("season", min_season),
        ],
    )
    tds_by_game = {}
    for r in stat_rows:
        key = (r["season"], r["week"])
        tds_by_game[key] = tds_by_game.get(key, 0) + (r.get("rushing_tds") or 0) + (r.get("passing_tds") or 0)

    games = []
    for r in sched_rows:
        key = (r["season"], r["week"])
        if r.get("home_score") is None or r.get("away_score") is None:
            continue
        points = r["home_score"] if r["home_team"] == team else r["away_score"]
        games.append({"season": r["season"], "week": r["week"], "points": points, "tds": tds_by_game.get(key, 0)})
    games.sort(key=lambda g: (g["season"], g["week"]))
    return games


@lru_cache(maxsize=None)
def compute_team_td_composition(supabase, team, season, week):
    """
    What fraction of this team's actual points historically come from
    touchdowns (vs field goals). Same fast-ramp philosophy as everywhere
    else: this season's own form (full-to-date + last-4-games-rolling)
    dominates almost immediately, prior seasons are only a minor input once
    real current-season games exist.
    """
    games = _team_actual_scoring_game_log(supabase, team, season, week)
    games = [g for g in games if g["points"] > 0]
    if not games:
        return {"td_points_share": 0.75, "rush_td_share": 0.4}  # league-ish defaults if no history at all

    current_season_games = [g for g in games if g["season"] == season]
    prior_games = [g for g in games if g["season"] != season]
    recent = current_season_games[-RECENT_WINDOW:]
    weight_current = _current_season_weight(len(current_season_games))
    within_season_weight_recent = min(1.0, len(current_season_games) / (QUALIFYING_GAME_THRESHOLD * 2)) if current_season_games else 0.0

    def td_points_share(gs):
        total_points = sum(g["points"] for g in gs)
        total_td_points = sum(g["tds"] for g in gs) * POINTS_PER_TD
        return min(1.0, total_td_points / total_points) if total_points else None

    prior_share = td_points_share(prior_games)
    season_to_date_share = td_points_share(current_season_games)
    recent_share = td_points_share(recent)
    current_share = _blend(season_to_date_share, recent_share, within_season_weight_recent)
    td_points_share_final = _blend(prior_share, current_share, weight_current)

    return {
        "td_points_share": td_points_share_final if td_points_share_final is not None else 0.75,
        "games_used": len(games),
    }


def compute_team_rush_td_share(supabase, team, season, week):
    """
    Team's share of offensive TDs that are rushing (vs passing/receiving).
    Same fast-ramp philosophy - previously this had NO recency weighting at
    all, a rush/pass TD split from 2 seasons ago counted exactly as much as
    last week's.
    """
    filters = [
        lambda q: q.eq("team", team),
        lambda q: q.eq("season_type", "REG"),
        lambda q: q.gte("season", season - 2),
    ]
    rows = select_all(supabase, "player_stats_weekly", "season,week,rushing_tds,passing_tds", filters=filters)
    rows = [r for r in rows if (r["season"], r["week"]) < (season, week)]
    if not rows:
        return 0.4  # league-ish default

    by_game = {}
    for r in rows:
        key = (r["season"], r["week"])
        g = by_game.setdefault(key, {"season": r["season"], "week": r["week"], "rush": 0, "pass": 0})
        g["rush"] += r.get("rushing_tds") or 0
        g["pass"] += r.get("passing_tds") or 0
    games = sorted(by_game.values(), key=lambda g: (g["season"], g["week"]))

    current_season_games = [g for g in games if g["season"] == season]
    prior_games = [g for g in games if g["season"] != season]
    recent = current_season_games[-RECENT_WINDOW:]
    weight_current = _current_season_weight(len(current_season_games))
    within_season_weight_recent = min(1.0, len(current_season_games) / (QUALIFYING_GAME_THRESHOLD * 2)) if current_season_games else 0.0

    def rush_share(gs):
        total_rush = sum(g["rush"] for g in gs)
        total_pass = sum(g["pass"] for g in gs)
        total = total_rush + total_pass
        return total_rush / total if total else None

    prior_rate = rush_share(prior_games)
    season_to_date_rate = rush_share(current_season_games)
    recent_rate = rush_share(recent)
    current_rate = _blend(season_to_date_rate, recent_rate, within_season_weight_recent)
    final_rate = _blend(prior_rate, current_rate, weight_current)
    return final_rate if final_rate is not None else 0.4


@lru_cache(maxsize=None)
def compute_implied_team_tds(supabase, team, opponent_team, season, week):
    """
    Decomposes this week's Vegas-implied team total into implied rushing and
    passing touchdown counts - the top-down replacement for the old bottom-up
    "player's own TD rate x opponent delta" approach, which had zero
    connection to how many points Vegas actually expects this team to score.

    Also folds in the opponent's TD-rate matchup delta (already computed and
    dampened by compute_position_efficiency_delta) at the team-pool level,
    once, rather than per-player - avoids re-introducing the missing-
    normalization problem that caused the QB share bug.
    """
    from .matchup import compute_position_efficiency_delta

    this_week_row = get_schedule_row(supabase, team, season, week)
    implied_total = implied_team_total(this_week_row, team) if this_week_row else None
    if implied_total is None:
        return {"rush_tds": 0.0, "pass_tds": 0.0}

    composition = compute_team_td_composition(supabase, team, season, week)
    rush_td_share = compute_team_rush_td_share(supabase, team, season, week)

    implied_td_points = implied_total * composition["td_points_share"]
    implied_total_tds = implied_td_points / POINTS_PER_TD

    implied_rush_tds = implied_total_tds * rush_td_share
    implied_pass_tds = implied_total_tds * (1 - rush_td_share)

    # Opponent's position-level TD-rate delta, applied once at the team-pool
    # level (already dampened/clipped - see common/matchup.py).
    rb_delta = compute_position_efficiency_delta(supabase, opponent_team, "RB", season, week)
    wr_delta = compute_position_efficiency_delta(supabase, opponent_team, "WR", season, week)
    implied_rush_tds *= rb_delta.get("rush_td_rate", 1.0)
    implied_pass_tds *= wr_delta.get("rec_td_rate", 1.0)

    return {"rush_tds": implied_rush_tds, "pass_tds": implied_pass_tds}
