"""
Phase 1: baseline + matchup calculations for the weekly/ROS projection engine.

Everything here is read-only computation against player_stats_weekly - no new
tables, no writes. Phase 3 (the weekly projection engine, not built yet) is
what will actually call these functions and store the resulting projections.

Design (agreed with the user):
  - "Efficiency" reuses the same rate-stat definitions Football Machine's
    existing preseason model already uses (yards/carry, yards/reception,
    yards/attempt, TD rate, catch rate, comp%, int%) - not a new invented unit.
  - Team weekly attempt pool (pass attempts, rush attempts) is a straight sum
    of player_stats_weekly rows for that team/week - not snap-derived.
  - Baseline blend is a three-tier weighted mix, always: career (the
    preseason model's own output) < last 2 real seasons < current season.
    Current season dominates fast, not gradually - weight is looked up by
    games played this season from CURRENT_SEASON_WEIGHT_BY_GAME (30% at 1
    game, up to 1.0/fully dominant by 5 games, prior tiers drop out entirely)
    - see _current_season_weight. Whatever weight isn't current-season is split LAST2_VS_CAREER_WEIGHT
    toward last-2-seasons and the rest toward career/preseason - last 2
    seasons always outweighs career, never the other way around. If a
    player has zero real games anywhere (true rookie with no preseason row
    either), there's nothing to compute a baseline from at all.
  - "Qualifying complete game" (to filter out cameo/injury-shortened games):
      QB: >=10 pass attempts. RB/WR/TE: >=2 combined carries+targets.
  - Matchup deltas (opponent effect vs league average, blended the same way -
    recent weeks + full season) come in two flavors:
      1) Position-efficiency delta - how a defense affects a position's rate
         stats (e.g. RBs get better/worse yards-per-carry against them).
      2) Team-volume delta - how a defense affects the whole offense's shape
         (run funnel / pass funnel, and total play volume) vs league average.
"""
from functools import lru_cache

from .db import select_all

RECENT_WINDOW = 4
QUALIFYING_GAME_THRESHOLD = 4
# Current-season weight by games played this season (updated 2026-09-16, user
# request - replaces the old floor+linear-ramp formula since these numbers
# don't fit a straight line: +10 from game 1->2, then +20 each game after).
# 0 games isn't listed - _current_season_weight returns 0 for that case
# (nothing this season to weight yet). 5+ games -> 1.0, same as before: once
# current season is fully dominant it just stays there for the rest of the
# season, prior tiers drop out entirely.
CURRENT_SEASON_WEIGHT_BY_GAME = {1: 0.30, 2: 0.40, 3: 0.60, 4: 0.80}
LAST2_VS_CAREER_WEIGHT = 0.7  # of the non-current-season weight, how much goes to last-2-seasons vs career/preseason (at full confidence)
LAST2_TRUST_GAMES = 8  # real games in the last 2 seasons before that tier is trusted at its full LAST2_VS_CAREER_WEIGHT


def _current_season_weight(games_this_season):
    """
    Fast ramp: this season's own data should dominate by a large margin
    almost immediately, not gradually earn trust over half a season. Zero
    games this season -> 0 (nothing to weight yet). See
    CURRENT_SEASON_WEIGHT_BY_GAME for games 1-4; by 5 games -> 100%, prior
    (last-2-seasons + career, or last season for opponent/team effects)
    drops out entirely.
    """
    if games_this_season <= 0:
        return 0.0
    if games_this_season >= 5:
        return 1.0
    return CURRENT_SEASON_WEIGHT_BY_GAME[games_this_season]


def _last2_weight(games_count):
    """
    LAST2_VS_CAREER_WEIGHT scaled down when the last-2-seasons sample itself
    is thin - a flat 70% weight on "last 2 seasons" regardless of whether
    that's 1 game or 20 caused fringe players with a single emergency-duty
    game to jump right back to a wildly inflated share (confirmed on real
    data: Ben Sims 19.4%, Nikko Remigio 24.8%, re-testing after adding the
    fixed weight). A thin last-2-seasons sample should lean toward career
    instead, same shrinkage idea used for opponent deltas.
    """
    return LAST2_VS_CAREER_WEIGHT * min(1.0, games_count / LAST2_TRUST_GAMES)

# projections_2026_resolved column -> our rate/share stat name. Reused as-is
# (read-only) - never touch the preseason model's own math, per project rule.
PRESEASON_RATE_MAP = {
    "proj_ypa": "pass_ypa",
    "proj_td_rate": "pass_td_rate",
    "proj_int_rate": "pass_int_rate",
    "proj_rush_ypc": "rush_ypc",
    "proj_rush_td_rate": "rush_td_rate",
    "proj_catch_rate": "catch_rate",
    "proj_rec_ypr": "rec_ypr",
    "proj_rec_td_rate": "rec_td_rate",
}
PRESEASON_SHARE_MAP = {
    "proj_carry_share": "carry_share",
    "proj_target_share": "target_share",
}

# stat_name -> (numerator column, denominator column) in player_stats_weekly
POSITION_RATE_STATS = {
    "QB": {
        "comp_pct": ("completions", "attempts"),
        "pass_ypa": ("passing_yards", "attempts"),
        "pass_td_rate": ("passing_tds", "attempts"),
        "pass_int_rate": ("passing_interceptions", "attempts"),
        "rush_ypc": ("rushing_yards", "carries"),
        "rush_td_rate": ("rushing_tds", "carries"),
    },
    "RB": {
        "rush_ypc": ("rushing_yards", "carries"),
        "rush_td_rate": ("rushing_tds", "carries"),
        "catch_rate": ("receptions", "targets"),
        "rec_ypr": ("receiving_yards", "receptions"),
        "rec_td_rate": ("receiving_tds", "receptions"),
    },
    "WR": {
        "catch_rate": ("receptions", "targets"),
        "rec_ypr": ("receiving_yards", "receptions"),
        "rec_td_rate": ("receiving_tds", "receptions"),
        "rush_ypc": ("rushing_yards", "carries"),
        "rush_td_rate": ("rushing_tds", "carries"),
    },
    "TE": {
        "catch_rate": ("receptions", "targets"),
        "rec_ypr": ("receiving_yards", "receptions"),
        "rec_td_rate": ("receiving_tds", "receptions"),
    },
}


def is_qualifying_game(row, position):
    if position == "QB":
        return (row.get("attempts") or 0) >= 10
    return (row.get("carries") or 0) + (row.get("targets") or 0) >= 2


def weighted_rate(games, numerator_field, denominator_field):
    """Sum-of-numerator / sum-of-denominator across games (volume-weighted, not average-of-averages)."""
    num = sum((g.get(numerator_field) or 0) for g in games)
    den = sum((g.get(denominator_field) or 0) for g in games)
    if den == 0:
        return None
    return num / den


def _week1_prior(preseason_val, blended_prior, week):
    """
    Week 1 special case: by definition there's no real current-season data
    yet, so instead of blending real last-2-seasons performance into the
    "prior" tier (career_rate/career_cs above is actually already the
    preseason projection value - see compute_player_baseline/
    compute_player_share_baseline), just use the preseason projection
    as-is. Falls back to the normal blended prior if this player/stat has
    no preseason value at all (no site preseason row - rare fringe case).
    Every other week is untouched (current_rate/current_cs always resolve
    to None at week 1 anyway, since qualifying_this_season is empty, so
    this only ever changes what week-1 rows compute).
    """
    if week == 1 and preseason_val is not None:
        return preseason_val
    return blended_prior


def _blend(a, b, weight_b):
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a * (1 - weight_b) + b * weight_b


@lru_cache(maxsize=None)
def fetch_player_games(supabase, player_id, before_season, before_week):
    """All of a player's REG-season games strictly before before_season/before_week."""
    filters = [
        lambda q: q.eq("player_id", player_id),
        lambda q: q.eq("season_type", "REG"),
    ]
    rows = select_all(supabase, "player_stats_weekly", "*", filters=filters)
    rows = [r for r in rows if (r["season"], r["week"]) < (before_season, before_week)]
    rows.sort(key=lambda r: (r["season"], r["week"]))
    return rows


def _match_preseason_row(supabase, player_id, player_name):
    """
    projections_2026_resolved is the live preseason model's own output - read
    only, never recomputed here. ~45% of rows (mostly rookies/deep bench added
    via sync_sleeper_depth_charts.py) have no gsis_id crosswalk, so name is a
    required fallback match, not an edge case.
    """
    rows = select_all(
        supabase, "projections_2026_resolved", "*",
        filters=[lambda q: q.eq("projection_year", 2026)],
    )
    if player_id:
        for r in rows:
            if r.get("player_id") == player_id:
                return r
    if player_name:
        target = player_name.strip().lower()
        for r in rows:
            if (r.get("player_name") or "").strip().lower() == target:
                return r
    return None


def get_preseason_baseline(supabase, player_id, player_name, position):
    """Preseason rate-stat baseline for a player with no real game history yet (rookies, etc)."""
    row = _match_preseason_row(supabase, player_id, player_name)
    if not row:
        return None
    out = {}
    for col, stat_name in PRESEASON_RATE_MAP.items():
        if stat_name in POSITION_RATE_STATS.get(position, {}):
            out[stat_name] = row.get(col)
    return out


def get_preseason_share_baseline(supabase, player_id, player_name):
    row = _match_preseason_row(supabase, player_id, player_name)
    if not row:
        return None
    return {stat_name: row.get(col) for col, stat_name in PRESEASON_SHARE_MAP.items()}


# ── One-time week-2 special blend (user request, 2026-09-16) ────────────────
# ros_projections_2026 uses the exact same proj_* column names as
# projections_2026_resolved (both written by the same pipeline shape), so the
# existing PRESEASON_RATE_MAP/PRESEASON_SHARE_MAP translate a ROS row exactly
# as they do a preseason row - no new mapping needed. These two take an
# ALREADY-FETCHED row (the caller fetches ros_projections_2026 once for every
# player, not per-player) rather than looking one up themselves, unlike
# get_preseason_baseline/get_preseason_share_baseline above.
def get_ros_baseline_from_row(row, position):
    if not row:
        return None
    return {stat_name: row.get(col) for col, stat_name in PRESEASON_RATE_MAP.items()
            if stat_name in POSITION_RATE_STATS.get(position, {})}


def get_ros_share_baseline_from_row(row):
    if not row:
        return None
    return {stat_name: row.get(col) for col, stat_name in PRESEASON_SHARE_MAP.items()}


@lru_cache(maxsize=None)
def compute_player_baseline(supabase, player_id, position, season, week, player_name=None):
    """
    Returns {"source": ..., "games_used": N, <stat_name>: rate, ...} for this
    player heading into `season`/`week`. Three-tier weighted blend (career <
    last 2 seasons < current season - see module docstring). `source` is
    "no_history" (nothing anywhere to compute from), "preseason_fallback"
    (zero real games, pure preseason model output), or "blend" (some mix of
    the three tiers, per-stat weights vary with how much real data exists).
    """
    all_games = fetch_player_games(supabase, player_id, season, week)
    this_season_games = [g for g in all_games if g["season"] == season]
    qualifying_this_season = [g for g in this_season_games if is_qualifying_game(g, position)]
    prior_seasons_games = [
        g for g in all_games
        if g["season"] in (season - 1, season - 2) and is_qualifying_game(g, position)
    ]

    stats = POSITION_RATE_STATS.get(position, {})
    result = {"games_used": len(qualifying_this_season)}

    if not qualifying_this_season and not prior_seasons_games:
        preseason = get_preseason_baseline(supabase, player_id, player_name, position)
        if not preseason:
            result["source"] = "no_history"
            return result
        result["source"] = "preseason_fallback"
        result.update(preseason)
        return result

    preseason = get_preseason_baseline(supabase, player_id, player_name, position)
    weight_current = _current_season_weight(len(qualifying_this_season))
    last2_weight = _last2_weight(len(prior_seasons_games))
    recent = qualifying_this_season[-RECENT_WINDOW:]
    # in-season recency: as this season's own sample grows, lean harder on its last 4 games
    within_season_weight_recent = min(1.0, len(qualifying_this_season) / (QUALIFYING_GAME_THRESHOLD * 2)) if qualifying_this_season else 0.0

    result["source"] = "blend"
    for stat_name, (num_f, den_f) in stats.items():
        career_rate = (preseason or {}).get(stat_name)
        last2_rate = weighted_rate(prior_seasons_games, num_f, den_f)
        season_to_date_rate = weighted_rate(qualifying_this_season, num_f, den_f)
        recent_rate = weighted_rate(recent, num_f, den_f)
        current_rate = _blend(season_to_date_rate, recent_rate, within_season_weight_recent)

        prior_combined = _week1_prior(career_rate, _blend(career_rate, last2_rate, last2_weight), week)
        result[stat_name] = _blend(prior_combined, current_rate, weight_current)
    return result


@lru_cache(maxsize=None)
def team_totals_by_week(supabase, team, min_season):
    """{(season, week): {"pass_attempts", "rush_attempts", "targets"}} for one team, from min_season onward."""
    filters = [
        lambda q: q.eq("team", team),
        lambda q: q.eq("season_type", "REG"),
        lambda q: q.gte("season", min_season),
    ]
    rows = select_all(supabase, "player_stats_weekly", "season,week,attempts,carries,targets", filters=filters)
    totals = {}
    for r in rows:
        key = (r["season"], r["week"])
        t = totals.setdefault(key, {"pass_attempts": 0, "rush_attempts": 0, "targets": 0})
        t["pass_attempts"] += r.get("attempts") or 0
        t["rush_attempts"] += r.get("carries") or 0
        t["targets"] += r.get("targets") or 0
    return totals


@lru_cache(maxsize=None)
def compute_team_attempts_baseline(supabase, team, season, week):
    """
    Team pass/rush attempt pool heading into season/week - no preseason
    team-attempts source exists, so this is a 2-tier version of the same
    philosophy as compute_player_baseline: last 2 real seasons < current
    season, with current season dominating quickly (weight ramps to 1.0 by
    5 games played this season - see _current_season_weight /
    CURRENT_SEASON_WEIGHT_BY_GAME).
    """
    team_totals = team_totals_by_week(supabase, team, season - 2)
    all_games = sorted(
        [{"season": s, "week": w, **v} for (s, w), v in team_totals.items() if (s, w) < (season, week)],
        key=lambda g: (g["season"], g["week"]),
    )
    this_season_games = [g for g in all_games if g["season"] == season]
    prior_seasons_games = [g for g in all_games if g["season"] in (season - 1, season - 2)]

    if not this_season_games and not prior_seasons_games:
        return {"source": "no_history", "pass_attempts": None, "rush_attempts": None}

    def avg(games, field):
        vals = [g[field] for g in games]
        return sum(vals) / len(vals) if vals else None

    weight_current = _current_season_weight(len(this_season_games))
    recent = this_season_games[-RECENT_WINDOW:]
    within_season_weight_recent = min(1.0, len(this_season_games) / (QUALIFYING_GAME_THRESHOLD * 2)) if this_season_games else 0.0

    current_pass = _blend(avg(this_season_games, "pass_attempts"), avg(recent, "pass_attempts"), within_season_weight_recent)
    current_rush = _blend(avg(this_season_games, "rush_attempts"), avg(recent, "rush_attempts"), within_season_weight_recent)
    prior_pass = avg(prior_seasons_games, "pass_attempts")
    prior_rush = avg(prior_seasons_games, "rush_attempts")

    pass_baseline = _blend(prior_pass, current_pass, weight_current)
    rush_baseline = _blend(prior_rush, current_rush, weight_current)
    return {"source": "blend", "pass_attempts": pass_baseline, "rush_attempts": rush_baseline}


@lru_cache(maxsize=None)
def compute_player_share_baseline(supabase, player_id, position, team, season, week, player_name=None, is_starter=True):
    """
    Blended (tiered) carry share and target share for a player, using the same
    volume-weighted approach as compute_player_baseline: sum(player's stat
    over games) / sum(team's stat over the same games), not average-of-ratios.
    Falls back to the preseason model's own carry_share/target_share when
    there's no real prior-season history (rookies, etc) - same pattern as
    compute_player_baseline. The preseason table has no QB pass-attempt-share
    equivalent, so a rookie projected as a starter defaults to 1.0 there
    (assume they take ~all the team's dropbacks) rather than 0.

    is_starter: whether this player holds the depth-chart starter slot at
    their position (slot_key ending "_0" - see run_mock_projection.
    get_player_pool). Only matters for QB - see finalize() below. Defaults
    to True so any caller not yet passing it keeps the old single-QB-in-pool
    behavior; callers that project a full depth chart (get_player_pool no
    longer filters to starters only) must pass this explicitly.
    """
    all_games = fetch_player_games(supabase, player_id, season, week)
    this_season_games = [g for g in all_games if g["season"] == season]
    qualifying_this_season = [g for g in this_season_games if is_qualifying_game(g, position)]

    min_season_needed = season - 2
    team_totals = team_totals_by_week(supabase, team, min_season_needed)

    def shares_for(games):
        carry_num = sum((g.get("carries") or 0) for g in games)
        carry_den = sum(team_totals.get((g["season"], g["week"]), {}).get("rush_attempts", 0) for g in games)
        target_num = sum((g.get("targets") or 0) for g in games)
        target_den = sum(team_totals.get((g["season"], g["week"]), {}).get("targets", 0) for g in games)
        att_num = sum((g.get("attempts") or 0) for g in games)
        att_den = sum(team_totals.get((g["season"], g["week"]), {}).get("pass_attempts", 0) for g in games)
        carry_share = carry_num / carry_den if carry_den else None
        target_share = target_num / target_den if target_den else None
        pass_attempt_share = att_num / att_den if att_den else None
        return carry_share, target_share, pass_attempt_share

    def finalize(result):
        # The depth-chart starter is, by definition, the one taking the
        # team's dropbacks that week - real historical share computed from a
        # thin sample (a QB1 who missed games, a spot-starter game) is not a
        # better estimate than just "they get the team's attempts." A
        # backup QB gets 0 by default instead - not on the field, but still
        # a real row that exists and can be edited/admin-overridden (e.g. to
        # model an injury takeover), rather than being excluded from the
        # pipeline entirely. Getting this right is why get_player_pool no
        # longer filters backups out silently - see its docstring for the
        # original bug (every QB with any history treated as a 1.0-share
        # starter) this is fixing without reintroducing.
        if position == "QB":
            result["pass_attempt_share"] = 1.0 if is_starter else 0.0
        return result

    prior_seasons_games = [
        g for g in all_games
        if g["season"] in (season - 1, season - 2) and is_qualifying_game(g, position)
    ]

    if not qualifying_this_season and not prior_seasons_games:
        preseason = get_preseason_share_baseline(supabase, player_id, player_name)
        if not preseason:
            return finalize({"source": "no_history", "carry_share": None, "target_share": None, "pass_attempt_share": None})
        return finalize({
            "source": "preseason_fallback",
            "carry_share": preseason.get("carry_share"),
            "target_share": preseason.get("target_share"),
            "pass_attempt_share": preseason.get("pass_attempt_share"),
        })

    # Three-tier weighted blend, same philosophy as compute_player_baseline:
    # career (preseason model) < last 2 real seasons < current season, with
    # current season dominating quickly as real games accumulate. Previously
    # this collapsed "last 2 seasons" and "current season" into one bucket
    # once ANY real games existed - a single qualifying game (as low as 2
    # touches/targets) produced shares as extreme as 27-35% for deep-bench
    # players with one emergency-duty game (confirmed on real data: Ben
    # Sims, Nikko Remigio). Keeping the tiers separate and properly weighted
    # fixes that at the root instead of just shrinking the symptom.
    preseason = get_preseason_share_baseline(supabase, player_id, player_name)
    weight_current = _current_season_weight(len(qualifying_this_season))
    last2_weight = _last2_weight(len(prior_seasons_games))
    recent = qualifying_this_season[-RECENT_WINDOW:]
    within_season_weight_recent = min(1.0, len(qualifying_this_season) / (QUALIFYING_GAME_THRESHOLD * 2)) if qualifying_this_season else 0.0

    career_cs, career_ts, career_as = (
        (preseason or {}).get("carry_share"), (preseason or {}).get("target_share"), (preseason or {}).get("pass_attempt_share"),
    )
    last2_cs, last2_ts, last2_as = shares_for(prior_seasons_games)
    season_cs, season_ts, season_as = shares_for(qualifying_this_season)
    recent_cs, recent_ts, recent_as = shares_for(recent)
    current_cs = _blend(season_cs, recent_cs, within_season_weight_recent)
    current_ts = _blend(season_ts, recent_ts, within_season_weight_recent)
    current_as = _blend(season_as, recent_as, within_season_weight_recent)

    prior_cs = _week1_prior(career_cs, _blend(career_cs, last2_cs, last2_weight), week)
    prior_ts = _week1_prior(career_ts, _blend(career_ts, last2_ts, last2_weight), week)
    prior_as = _week1_prior(career_as, _blend(career_as, last2_as, last2_weight), week)

    return finalize({
        "source": "blend",
        "carry_share": _blend(prior_cs, current_cs, weight_current),
        "target_share": _blend(prior_ts, current_ts, weight_current),
        "pass_attempt_share": _blend(prior_as, current_as, weight_current),
    })


def compute_team_weekly_attempts(supabase, team, season, week):
    """Team's total pass attempts and rush attempts for one week - straight sum, no snap derivation."""
    filters = [
        lambda q: q.eq("team", team),
        lambda q: q.eq("season", season),
        lambda q: q.eq("week", week),
        lambda q: q.eq("season_type", "REG"),
    ]
    rows = select_all(supabase, "player_stats_weekly", "attempts,carries", filters=filters)
    return {
        "pass_attempts": sum((r.get("attempts") or 0) for r in rows),
        "rush_attempts": sum((r.get("carries") or 0) for r in rows),
    }


_SUM_FIELDS = [
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
]


def _aggregate_by_game(rows, group_key_fn):
    """
    Multiple player rows can belong to the same real game (e.g. every RB who
    faced a defense that week). Groups rows by group_key_fn(row) and sums the
    stat fields, so "recent N" and "games faced" mean actual games, not raw
    player-row counts (a real bug found in testing: with ~5 RB rows per game,
    "last 4 rows" was grabbing one game's worth of data, not four games).
    """
    games = {}
    for r in rows:
        key = group_key_fn(r)
        g = games.setdefault(key, {"season": r["season"], "week": r["week"]})
        for f in _SUM_FIELDS:
            g[f] = g.get(f, 0) + (r.get(f) or 0)
    ordered = sorted(games.values(), key=lambda g: (g["season"], g["week"]))
    return ordered


@lru_cache(maxsize=None)
def _fetch_opponent_games(supabase, opponent_team, position, before_season, before_week, seasons_back=1):
    """One aggregated row per real game this defense has faced this position group in."""
    min_season = before_season - seasons_back
    filters = [
        lambda q: q.eq("opponent_team", opponent_team),
        lambda q: q.eq("position", position),
        lambda q: q.eq("season_type", "REG"),
        lambda q: q.gte("season", min_season),
    ]
    rows = select_all(supabase, "player_stats_weekly", "*", filters=filters)
    rows = [
        r for r in rows
        if (min_season, 1) <= (r["season"], r["week"]) < (before_season, before_week)
    ]
    return _aggregate_by_game(rows, lambda r: (r["season"], r["week"]))


@lru_cache(maxsize=None)
def _league_average_games(supabase, position, before_season, before_week, seasons_back=1):
    """One aggregated row per team-game (every team's weekly total vs that position), for league averaging."""
    min_season = before_season - seasons_back
    filters = [
        lambda q: q.eq("position", position),
        lambda q: q.eq("season_type", "REG"),
        lambda q: q.gte("season", min_season),
    ]
    rows = select_all(supabase, "player_stats_weekly", "*", filters=filters)
    rows = [
        r for r in rows
        if (min_season, 1) <= (r["season"], r["week"]) < (before_season, before_week)
    ]
    return _aggregate_by_game(rows, lambda r: (r["season"], r["week"], r["opponent_team"]))


# TD/INT rates are rare-event stats over a thin per-defense sample (a defense
# only faces ~9-17 games a season) - raw ratios on these swung as wide as
# 0.0x-4.04x in testing on real data, purely from small-sample noise, not a
# real signal. Yardage/percentage stats have a much larger per-game sample
# (every touch/target counts, not just scores) and are comparatively stable.
# Clip band is the hard safety rail; shrinkage below softens it further based
# on how many games actually informed the number.
VOLATILE_STATS = {"rush_td_rate", "rec_td_rate", "pass_td_rate", "pass_int_rate"}
CLIP_BANDS = {"volatile": (0.9, 1.1), "stable": (0.9, 1.1)}
SHRINKAGE_GAMES = 12  # games faced before a delta is trusted at its full raw value


def _dampen_delta(raw_ratio, sample_size):
    if raw_ratio is None:
        return 1.0
    weight = min(1.0, sample_size / SHRINKAGE_GAMES)
    shrunk = 1.0 + (raw_ratio - 1.0) * weight
    lo, hi = CLIP_BANDS["stable"]
    return min(hi, max(lo, shrunk))


def _dampen_volatile_delta(raw_ratio, sample_size):
    if raw_ratio is None:
        return 1.0
    weight = min(1.0, sample_size / SHRINKAGE_GAMES)
    shrunk = 1.0 + (raw_ratio - 1.0) * weight
    lo, hi = CLIP_BANDS["volatile"]
    return min(hi, max(lo, shrunk))


@lru_cache(maxsize=None)
def compute_position_efficiency_delta(supabase, opponent_team, position, season, week):
    """
    For each rate stat, how this defense's allowed rate compares to league
    average, expressed as a ratio (1.0 = league average). Same fast-ramp
    philosophy as player baselines: this season's own form (full-to-date +
    last-4-games-rolling) dominates almost immediately, last season is only
    a minor prior once real current-season games exist. Dampened by total
    sample size and clipped to a safety band (tighter for TD/INT-rate stats,
    which are rare events and far noisier than yardage/percentage stats).
    """
    all_games = _fetch_opponent_games(supabase, opponent_team, position, season, week)
    current_season_games = [g for g in all_games if g["season"] == season]
    prior_season_games = [g for g in all_games if g["season"] != season]
    recent = current_season_games[-RECENT_WINDOW:]
    league_games = _league_average_games(supabase, position, season, week)

    stats = POSITION_RATE_STATS.get(position, {})
    result = {}
    weight_current = _current_season_weight(len(current_season_games))
    within_season_weight_recent = min(1.0, len(current_season_games) / (QUALIFYING_GAME_THRESHOLD * 2)) if current_season_games else 0.0
    sample_size = len(all_games)

    for stat_name, (num_f, den_f) in stats.items():
        prior_rate = weighted_rate(prior_season_games, num_f, den_f)
        season_to_date_rate = weighted_rate(current_season_games, num_f, den_f)
        recent_rate = weighted_rate(recent, num_f, den_f)
        current_rate = _blend(season_to_date_rate, recent_rate, within_season_weight_recent)
        team_rate = _blend(prior_rate, current_rate, weight_current)

        league_rate = weighted_rate(league_games, num_f, den_f)
        if team_rate is None or league_rate is None or league_rate == 0:
            result[stat_name] = 1.0
            continue
        raw_ratio = team_rate / league_rate
        if stat_name in VOLATILE_STATS:
            result[stat_name] = _dampen_volatile_delta(raw_ratio, sample_size)
        else:
            result[stat_name] = _dampen_delta(raw_ratio, sample_size)
    return result


def _avg_ratio(gs):
    totals = [g["pass_attempts"] + g["rush_attempts"] for g in gs if (g["pass_attempts"] + g["rush_attempts"]) > 0]
    rush_shares = [
        g["rush_attempts"] / (g["pass_attempts"] + g["rush_attempts"])
        for g in gs if (g["pass_attempts"] + g["rush_attempts"]) > 0
    ]
    if not totals:
        return None, None
    return sum(totals) / len(totals), sum(rush_shares) / len(rush_shares)


@lru_cache(maxsize=None)
def _league_volume_average(supabase, season, week, seasons_back=1):
    """
    League-wide average total plays / rush share for a season/week window -
    identical for every opponent_team, so cached independent of team
    (previously recomputed per-team, and without a server-side season filter
    at all - was fetching the ENTIRE historical table every call, ~60s/call).
    """
    min_season = season - seasons_back
    filters = [
        lambda q: q.eq("season_type", "REG"),
        lambda q: q.gte("season", min_season),
    ]
    all_rows = select_all(supabase, "player_stats_weekly", "season,week,team,attempts,carries", filters=filters)
    all_rows = [r for r in all_rows if (min_season, 1) <= (r["season"], r["week"]) < (season, week)]
    league_by_game = {}
    for r in all_rows:
        key = (r["season"], r["week"], r["team"])
        g = league_by_game.setdefault(key, {"pass_attempts": 0, "rush_attempts": 0})
        g["pass_attempts"] += r.get("attempts") or 0
        g["rush_attempts"] += r.get("carries") or 0
    return _avg_ratio(list(league_by_game.values()))


@lru_cache(maxsize=None)
def compute_team_volume_delta(supabase, opponent_team, season, week):
    """
    Run-funnel/pass-funnel + total-play-volume delta: how this defense affects
    the shape of opposing offenses' attempt pool, vs league average, as ratios.
    Same fast-ramp philosophy as everywhere else: this season's own form
    (full-to-date + last-4-games-rolling) dominates almost immediately, last
    season is only a minor prior once real current-season games exist.
    """
    min_season = season - 1
    filters = [
        lambda q: q.eq("opponent_team", opponent_team),
        lambda q: q.eq("season_type", "REG"),
        lambda q: q.gte("season", min_season),
    ]
    rows = select_all(supabase, "player_stats_weekly", "season,week,team,attempts,carries", filters=filters)
    rows = [r for r in rows if (min_season, 1) <= (r["season"], r["week"]) < (season, week)]

    by_game = {}
    for r in rows:
        key = (r["season"], r["week"], r["team"])
        g = by_game.setdefault(key, {"season": r["season"], "week": r["week"], "pass_attempts": 0, "rush_attempts": 0})
        g["pass_attempts"] += r.get("attempts") or 0
        g["rush_attempts"] += r.get("carries") or 0
    games = sorted(by_game.values(), key=lambda g: (g["season"], g["week"]))
    current_season_games = [g for g in games if g["season"] == season]
    prior_season_games = [g for g in games if g["season"] != season]
    recent_games = current_season_games[-RECENT_WINDOW:]

    prior_total, prior_rush_share = _avg_ratio(prior_season_games)
    season_to_date_total, season_to_date_rush_share = _avg_ratio(current_season_games)
    recent_total, recent_rush_share = _avg_ratio(recent_games)
    weight_current = _current_season_weight(len(current_season_games))
    within_season_weight_recent = min(1.0, len(current_season_games) / (QUALIFYING_GAME_THRESHOLD * 2)) if current_season_games else 0.0

    current_total = _blend(season_to_date_total, recent_total, within_season_weight_recent)
    current_rush_share = _blend(season_to_date_rush_share, recent_rush_share, within_season_weight_recent)
    team_total_plays = _blend(prior_total, current_total, weight_current)
    team_rush_share = _blend(prior_rush_share, current_rush_share, weight_current)

    league_total_plays, league_rush_share = _league_volume_average(supabase, season, week)
    sample_size = len(games)

    raw_total_plays_ratio = (team_total_plays / league_total_plays) if team_total_plays and league_total_plays else None
    raw_rush_share_ratio = (team_rush_share / league_rush_share) if team_rush_share and league_rush_share else None

    return {
        "total_plays_ratio": _dampen_delta(raw_total_plays_ratio, sample_size),
        "rush_share_ratio": _dampen_delta(raw_rush_share_ratio, sample_size),
    }
