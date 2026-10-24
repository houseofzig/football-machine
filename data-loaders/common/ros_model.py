"""
ROS model (user spec, 2026-10-22) - used ONLY by write_ros_tables.py; the weekly
model (matchup.compute_player_baseline / compute_player_share_baseline) is untouched.

PLAYERS - final = 60% x ramp  this season          (ramp 30/40/60/80/100% for 1/2/3/4/5+ counted games)
                + the remainder of the other 75%   the 10 valid games BEFORE this season (no overlap)
                + 25%                              the last-saved ROS row (same stat)
  "valid game" = qualifying (QB 10+ attempts, others 2+ carries+targets) and NOT a partial game
  (under half his median snap share, see matchup.partial_games). Within this season the
  last-4-games recency blend is the same as before (weight = games / 8).
  - no counted 2026 game and a saved ROS row  -> the saved ROS row is kept untouched ("ros_kept")
  - fewer than 10 prior valid games           -> use what exists; none -> that weight goes to this season
  - a missing component (no data / no saved value) is dropped and the others re-weighted
  Shares are scaled per team by common/team_shares ("strict lock": the top 2 target-getters and top 1 carry-getter keep
their model share, the rest fill what is left of 100%; IR/PUP/Sus players left out).
Shares are measured against the team the player was ON in each game (g["team"]), not
  his current team.

TEAMS - this season only, last-4 recency blend (weight = games / 8): pass attempts, rush
attempts, passing yards and rushing yards per game -> total plays, run %, yards per
attempt, yards per carry. No prior seasons, no saved values. Games remaining per team come
from the schedule (bye-aware).
"""
import re
from collections import defaultdict
from functools import lru_cache

import requests

from . import matchup as m
from .db import select_all

ROS_SAVED_WEIGHT = 0.25
ROS_CURRENT_MAX = 0.60
ROS_PRIOR_GAMES = 10
TEAM_SHARE_MIN_SEASON = 2020


def _ramp(n):
    if n <= 0:
        return 0.0
    return 1.0 if n >= 5 else m.CURRENT_SEASON_WEIGHT_BY_GAME[n]


def _combine(parts):
    """Weighted mean of (weight, value) pairs, skipping missing values and re-weighting the rest."""
    parts = [(w, v) for w, v in parts if v is not None and w > 0]
    if not parts:
        return None
    total = sum(w for w, _ in parts)
    return sum(w * v for w, v in parts) / total


def ros_weights(n_this, has_prior):
    """(this-season weight, prior-10-games weight) out of the 75% that is not the saved ROS."""
    avail = 1.0 - ROS_SAVED_WEIGHT
    if not has_prior:
        return (avail if n_this > 0 else 0.0), 0.0
    cur = ROS_CURRENT_MAX * _ramp(n_this)
    return cur, avail - cur


@lru_cache(maxsize=None)
def ros_games(supabase, player_id, position, season, week):
    valid = [g for g in m.fetch_full_games(supabase, player_id, season, week) if m.is_qualifying_game(g, position)]
    this = tuple(g for g in valid if g["season"] == season)
    prior = tuple([g for g in valid if g["season"] < season][-ROS_PRIOR_GAMES:])
    return this, prior


@lru_cache(maxsize=None)
def compute_ros_player_baseline(supabase, player_id, position, season, week, player_name=None):
    this, prior = ros_games(supabase, player_id, position, season, week)
    saved = m.get_ros_baseline(supabase, player_id, player_name, position, season)
    result = {"games_used": len(this)}
    if not this:
        if saved:
            result["source"] = "ros_kept"
            result.update(saved)
            return result
        if not prior:
            result["source"] = "no_history"
            return result
    cur_w, prior_w = ros_weights(len(this), bool(prior))
    recent_w = min(1.0, len(this) / (m.QUALIFYING_GAME_THRESHOLD * 2)) if this else 0.0
    recent = this[-m.RECENT_WINDOW:]
    result["source"] = "blend"
    for stat_name, (num_f, den_f) in m.POSITION_RATE_STATS[position].items():
        cur = m._blend(m.weighted_rate(this, num_f, den_f), m.weighted_rate(recent, num_f, den_f), recent_w) if this else None
        pr = m.weighted_rate(prior, num_f, den_f) if prior else None
        result[stat_name] = _combine([(cur_w, cur), (prior_w, pr), (ROS_SAVED_WEIGHT, (saved or {}).get(stat_name))])
    return result


def _team_week_totals(supabase, team):
    return m.team_totals_by_week(supabase, team, TEAM_SHARE_MIN_SEASON)


def _share(supabase, games, team, num_field, den_field):
    """Sum of the player's stat over the games / sum of THAT game's team total (the team he was on)."""
    if not games:
        return None
    num = sum((g.get(num_field) or 0) for g in games)
    den = sum(_team_week_totals(supabase, g.get("team") or team).get((g["season"], g["week"]), {}).get(den_field, 0) for g in games)
    return (num / den) if den else None


@lru_cache(maxsize=None)
def compute_ros_player_shares(supabase, player_id, position, team, season, week, player_name=None, is_starter=True):
    this, prior = ros_games(supabase, player_id, position, season, week)
    saved = m.get_ros_share_baseline(supabase, player_id, player_name, season)

    def finalize(result):
        # same QB rule as the weekly model: the starter takes the dropbacks, a backup gets 0 pass/carry share
        if position == "QB":
            result["pass_attempt_share"] = 1.0 if is_starter else 0.0
            if not is_starter:
                result["carry_share"] = 0.0
        return result

    if not this:
        if saved:
            return finalize({"source": "ros_kept", "carry_share": saved.get("carry_share"),
                             "target_share": saved.get("target_share"), "pass_attempt_share": saved.get("pass_attempt_share")})
        if not prior:
            return finalize({"source": "no_history", "carry_share": None, "target_share": None, "pass_attempt_share": None})
    cur_w, prior_w = ros_weights(len(this), bool(prior))
    recent_w = min(1.0, len(this) / (m.QUALIFYING_GAME_THRESHOLD * 2)) if this else 0.0
    recent = this[-m.RECENT_WINDOW:]
    out = {"source": "blend"}
    for key, num_f, den_f in (("carry_share", "carries", "rush_attempts"),
                              ("target_share", "targets", "targets"),
                              ("pass_attempt_share", "attempts", "pass_attempts")):
        cur = m._blend(_share(supabase, this, team, num_f, den_f), _share(supabase, recent, team, num_f, den_f), recent_w) if this else None
        pr = _share(supabase, prior, team, num_f, den_f) if prior else None
        out[key] = _combine([(cur_w, cur), (prior_w, pr), (ROS_SAVED_WEIGHT, (saved or {}).get(key))])
    return finalize(out)


@lru_cache(maxsize=None)
def games_remaining_by_team(supabase, season, from_week):
    """{team: regular-season games still to play, from `from_week` on} - bye-aware, from the schedule."""
    rows = select_all(supabase, "schedules", "week,home_team,away_team,game_type",
                      filters=[lambda q: q.eq("season", season), lambda q: q.gte("week", from_week), lambda q: q.eq("game_type", "REG")])
    out = defaultdict(int)
    for r in rows:
        out[r["home_team"]] += 1
        out[r["away_team"]] += 1
    return dict(out)


@lru_cache(maxsize=None)
def compute_ros_team_envelope(supabase, team, season, week):
    """This season only. Returns None when the team has no 2026 game yet."""
    rows = select_all(
        supabase, "player_stats_weekly", "season,week,attempts,carries,passing_yards,rushing_yards",
        filters=[lambda q: q.eq("team", team), lambda q: q.eq("season", season), lambda q: q.eq("season_type", "REG")],
    )
    by_week = {}
    for r in rows:
        if r["week"] >= week:
            continue
        t = by_week.setdefault(r["week"], {"pass_att": 0, "rush_att": 0, "pass_yds": 0, "rush_yds": 0})
        t["pass_att"] += r.get("attempts") or 0
        t["rush_att"] += r.get("carries") or 0
        t["pass_yds"] += r.get("passing_yards") or 0
        t["rush_yds"] += r.get("rushing_yards") or 0
    weeks = sorted(by_week)
    if not weeks:
        return None
    recent = weeks[-m.RECENT_WINDOW:]
    recent_w = min(1.0, len(weeks) / (m.QUALIFYING_GAME_THRESHOLD * 2))
    avg = lambda ws, k: sum(by_week[w][k] for w in ws) / len(ws)
    ratio = lambda ws, a, b: (sum(by_week[w][a] for w in ws) / sum(by_week[w][b] for w in ws)) if sum(by_week[w][b] for w in ws) else None
    pass_att = m._blend(avg(weeks, "pass_att"), avg(recent, "pass_att"), recent_w)
    rush_att = m._blend(avg(weeks, "rush_att"), avg(recent, "rush_att"), recent_w)
    total = pass_att + rush_att
    return {
        "games_played": len(weeks),
        "pass_attempts": pass_att, "rush_attempts": rush_att,
        "total_plays": total, "run_pct": (rush_att / total) if total else None,
        "ypa": m._blend(ratio(weeks, "pass_yds", "pass_att"), ratio(recent, "pass_yds", "pass_att"), recent_w),
        "ypc": m._blend(ratio(weeks, "rush_yds", "rush_att"), ratio(recent, "rush_yds", "rush_att"), recent_w),
    }


QB_OVERRIDES_SOURCE = "/Users/houseofzig/Working Site/football_machine.html"
_SLEEPER_TEAM_FIX = {"LA": "LAR", "JAC": "JAX", "WSH": "WAS"}


def load_qb_overrides(path=QB_OVERRIDES_SOURCE):
    """The user's manual QB depth overrides - the SAME table football_machine.html uses
    ({"TEAM:name lowercased": depth number}), read from that file so the two can't disagree."""
    html = open(path, encoding="utf-8").read()
    start = html.index("const QB_OVERRIDES = {")
    block = html[start:html.index("};", start)]
    overrides = {k: float(v) for k, v in re.findall(r"'([A-Z]+:[^']+)':\s*([0-9]+(?:\.[0-9]+)?)", block)}
    if not overrides:
        raise RuntimeError("QB_OVERRIDES could not be read from " + path)
    return overrides


def starting_qb_ids(pool):
    """{gsis player_id} of each team's ROS starting QB (user, 2026-10-22): the manual QB overrides
    first, then Sleeper's depth chart matched by sleeper_id (only while Sleeper has him on that
    team), then the stored slot_key _0. Lowest depth number wins; ties go to the stored slot."""
    overrides = load_qb_overrides()
    sleeper = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=60).json()
    by_team = defaultdict(list)
    for p in pool:
        if p["position"] == "QB":
            by_team[p["team"]].append(p)
    starters = set()
    for team, qbs in by_team.items():
        def depth(p):
            key = f"{team}:{p['player_name'].lower()}"
            if key in overrides:
                return overrides[key]
            sp = sleeper.get(str(p.get("sleeper_id")))
            if sp and _SLEEPER_TEAM_FIX.get(sp.get("team"), sp.get("team")) == team and sp.get("depth_chart_order") is not None:
                return float(sp["depth_chart_order"])
            return 99.0
        best = min(qbs, key=lambda p: (depth(p), 0 if (p.get("slot_key") or "").endswith("_0") else 1, p.get("slot_key") or ""))
        starters.add(best["player_id"])
    return starters
