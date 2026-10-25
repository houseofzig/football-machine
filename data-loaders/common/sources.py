"""
Source-abstraction layer for weekly player stats.

Primary: Sleeper's undocumented bulk endpoint
(api.sleeper.app/v1/stats/nfl/regular/<season>/<week>) - a single call returns
every player's full box-score line for the week (yards, TDs, receptions,
targets, carries, snaps, pre-computed fantasy points). Sleeper is a live,
always-on API (not a nightly-refreshed batch data project), and the site
already keys headshots off Sleeper's player ID elsewhere, so this is the
simplest, most consistent source to build around. Sleeper uses its own player
IDs, so results are crosswalked to gsis_id via the ff_playerids table already
loaded in Supabase.

Fallback 1: nflreadpy (nflverse) - used only if Sleeper is down or returns
nothing for a given week. Also the source for schedules/Vegas lines
(spread_line/total_line), which Sleeper has no equivalent for at all - that
part of the pipeline (load_schedules.py) still runs through nflreadpy
regardless of this fallback order.

Fallback 2: ESPN's unofficial scoreboard + per-game boxscore endpoints - last
resort, only if both of the above fail. One scoreboard call lists the week's
games, then one summary call per game returns box-score stat lines per athlete
(espn_id), crosswalked the same way. Only returns detailed box scores for
recent games (roughly the last year) - not usable for older historical weeks.

All three normalize down to the same column subset used by player_stats_weekly
(see data-loaders/common/db.py). Sleeper and ESPN can't fill nflreadpy's
advanced/derived metrics (epa, racr, wopr, etc.) - only core box-score counts -
which is fine since the weekly/ROS model only needs rate stats (target share,
carry share, snap share) derivable from raw counts, not those advanced metrics.
"""
import requests

from .db import clean_value, fix_team_abbrs, select_all

SLEEPER_TO_NORMALIZED = {
    "pass_cmp": "completions",
    "pass_att": "attempts",
    "pass_yd": "passing_yards",
    "pass_td": "passing_tds",
    "pass_int": "passing_interceptions",
    "rush_att": "carries",
    "rush_yd": "rushing_yards",
    "rush_td": "rushing_tds",
    "rec": "receptions",
    "rec_tgt": "targets",
    "rec_yd": "receiving_yards",
    "rec_td": "receiving_tds",
    "pts_std": "fantasy_points",
    "pts_ppr": "fantasy_points_ppr",
}

ESPN_STAT_INDEX = {
    "passing": {
        "passingYards": "passing_yards",
        "passingTouchdowns": "passing_tds",
        "interceptions": "passing_interceptions",
    },
    "rushing": {
        "rushingAttempts": "carries",
        "rushingYards": "rushing_yards",
        "rushingTouchdowns": "rushing_tds",
    },
    "receiving": {
        "receptions": "receptions",
        "receivingYards": "receiving_yards",
        "receivingTouchdowns": "receiving_tds",
        "receivingTargets": "targets",
    },
}


def get_current_week():
    """
    Returns (season, week) - always the current REGULAR season week, never
    preseason or playoffs. Sleeper's state endpoint reports a raw week number
    that resets across season phases (season_type: 'pre' weeks 1-4, 'regular'
    weeks 1-18, 'post' for playoffs) - using that number directly without
    checking season_type would misread e.g. preseason week 3 as regular
    season week 3 (confirmed happening: 2026-08-24, Sleeper reported
    season_type='pre', week=3, while the real regular season hadn't started).
    Before the season starts, this returns week 1 (nothing meaningful to
    compute yet, but a safe/valid week number for callers). Once the season
    is over (season_type='post'), this clamps to week 18 rather than
    reporting a playoff week number - the projection pipeline has no concept
    of playoff weeks.
    """
    r = requests.get("https://api.sleeper.app/v1/state/nfl", timeout=15)
    r.raise_for_status()
    d = r.json()
    season = int(d["league_season"])
    season_type = d.get("season_type")
    if season_type == "pre":
        return season, 1
    if season_type == "post":
        return season, 18
    return season, int(d["display_week"])


def fetch_nflreadpy(season, week=None):
    """Returns a list of normalized dict rows, or [] on failure."""
    try:
        import nflreadpy as nfl
        stats = nfl.load_player_stats(seasons=[season], summary_level="week").to_pandas()
        stats = stats[stats["position"].isin(["QB", "RB", "WR", "TE"])]
        if week is not None:
            stats = stats[stats["week"] == week]
        if stats.empty:
            return []
        stats = fix_team_abbrs(stats, "team", "opponent_team")
        stats = stats.replace([float("inf"), float("-inf")], None)
        stats = stats.where(stats.notnull(), None)
        rows = stats.to_dict(orient="records")
        return [{k: clean_value(v) for k, v in row.items()} for row in rows]
    except Exception as e:
        print(f"  [nflreadpy] fetch failed: {e}")
        return []


def _sleeper_crosswalk(supabase):
    """sleeper_id -> {gsis_id, name, position} from the already-loaded ff_playerids table."""
    data = select_all(
        supabase, "ff_playerids", "gsis_id,sleeper_id,name,position",
        filters=[lambda q: q.not_.is_("sleeper_id", "null")], order_by="gsis_id",
    )
    out = {}
    for row in data:
        sid = row.get("sleeper_id")
        if sid is None:
            continue
        out[str(int(float(sid)))] = row
    return out


# Sleeper injury_status values meaning "not on the field this week at all" -
# confirmed via a live pull of /v1/players/nfl, 2026-09-29: these are the
# distinct non-"Questionable" statuses in real use that week (IR 241, Out 224,
# PUP 38, NA 96, Sus 11, COV 2, DNR 2). User, 2026-09-29: "if a player is
# listed on IR then their share inputs should be 0... clear out those players
# too" (confirming PUP/Sus/NA belong in the same bucket) - COV/DNR are
# grouped in by the same reasoning (also clearly not playing), though rare
# enough they weren't individually confirmed live.
#
# "Out" deliberately removed (2026-09-30, user correction after the first
# real run): unlike IR/PUP/Sus/NA, "Out" is tied to a SPECIFIC game, and
# there's no confirmed guarantee Sleeper has already rolled it forward to
# the upcoming week's game by the time this pipeline runs (Tuesday morning) -
# it could still be reflecting last week's now-finished game. Rather than
# assume the field means "out for the game we're about to project," it's
# excluded until that timing is independently confirmed. "Questionable"/
# "Doubtful" remain deliberately excluded too - those players are still
# genuinely expected to potentially play, so their normal projected share
# should stand.
INACTIVE_WEEKLY_STATUSES = {"IR", "PUP", "Sus", "NA", "COV", "DNR"}


# ROS team-share scaling (2026-10-22, user: "IR players you can leave out"): only
# the multi-week statuses. Sleeper's "Out" is not in INACTIVE_WEEKLY_STATUSES at
# all, and NA/COV/DNR are not included here either.
ROS_LONG_TERM_INACTIVE_STATUSES = {"IR", "PUP", "Sus"}


def get_inactive_weekly_gsis_ids(supabase, statuses=None):
    """
    {gsis_id} for every player Sleeper currently lists with a status in
    `statuses` (default INACTIVE_WEEKLY_STATUSES) - i.e., not expected to play
    AT ALL this week. The ROS writer passes ROS_LONG_TERM_INACTIVE_STATUSES.

    Weekly-only by design: this is a snapshot of right now, not a durable
    fact about the player (someone on IR this week can easily be back in a
    month) - callers must apply it only inside the weekly pipeline
    (write_weekly_tables.py), via project_player_week's share_override, never
    inside compute_player_baseline/compute_player_share_baseline themselves,
    since write_ros_tables.py also calls those and must not have a player's
    whole rest-of-season outlook zeroed just because they're out this one week.

    No fallback on failure, matching get_current_week()'s own style - user,
    2026-09-29: "if it fails then rerun" - a broken/incomplete fetch here
    should stop the whole pipeline run rather than silently produce a week of
    projections missing this week's real inactives.
    """
    crosswalk = _sleeper_crosswalk(supabase)
    r = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=30)
    r.raise_for_status()
    players = r.json()
    out = set()
    for sleeper_id, p in players.items():
        if p.get("injury_status") not in (statuses if statuses is not None else INACTIVE_WEEKLY_STATUSES):
            continue
        row = crosswalk.get(sleeper_id)
        if row and row.get("gsis_id"):
            out.add(row["gsis_id"])
    return out


def fetch_sleeper(season, week, supabase):
    """Returns a list of normalized dict rows, or [] on failure."""
    try:
        url = f"https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            return []
        crosswalk = _sleeper_crosswalk(supabase)
        rows = []
        for sleeper_id, stat_line in data.items():
            if not stat_line or sleeper_id.startswith("TEAM_"):
                continue
            player = crosswalk.get(sleeper_id)
            if not player or not player.get("gsis_id"):
                continue
            if player.get("position") not in ("QB", "RB", "WR", "TE"):
                continue
            row = {
                "player_id": player["gsis_id"],
                "player_name": player.get("name"),
                "position": player.get("position"),
                "season": season,
                "week": week,
                "season_type": "REG",
            }
            for sleeper_key, norm_key in SLEEPER_TO_NORMALIZED.items():
                if sleeper_key in stat_line:
                    row[norm_key] = clean_value(stat_line[sleeper_key])
            rows.append(row)
        return rows
    except Exception as e:
        print(f"  [sleeper] fetch failed: {e}")
        return []


def _espn_crosswalk(supabase):
    data = select_all(
        supabase, "ff_playerids", "gsis_id,espn_id,name,position",
        filters=[lambda q: q.not_.is_("espn_id", "null")], order_by="gsis_id",
    )
    out = {}
    for row in data:
        eid = row.get("espn_id")
        if eid is None:
            continue
        out[str(int(float(eid)))] = row
    return out


def fetch_espn(season, week, supabase):
    """Returns a list of normalized dict rows, or [] on failure."""
    try:
        scoreboard_url = (
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            f"?week={week}&seasontype=2&year={season}"
        )
        r = requests.get(scoreboard_url, timeout=20)
        r.raise_for_status()
        events = r.json().get("events", [])
        if not events:
            return []

        crosswalk = _espn_crosswalk(supabase)
        rows_by_player = {}

        for ev in events:
            event_id = ev["id"]
            summary_url = (
                "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
                f"?event={event_id}"
            )
            sr = requests.get(summary_url, timeout=20)
            sr.raise_for_status()
            box = sr.json().get("boxscore", {})
            for team_block in box.get("players", []):
                team_abbr = team_block.get("team", {}).get("abbreviation")
                for stat_category in team_block.get("statistics", []):
                    cat_name = stat_category.get("name")
                    field_map = ESPN_STAT_INDEX.get(cat_name)
                    if not field_map:
                        continue
                    keys = stat_category.get("keys", [])
                    for athlete_entry in stat_category.get("athletes", []):
                        athlete = athlete_entry.get("athlete", {})
                        espn_id = str(athlete.get("id"))
                        player = crosswalk.get(espn_id)
                        if not player or not player.get("gsis_id"):
                            continue
                        if player.get("position") not in ("QB", "RB", "WR", "TE"):
                            continue
                        gsis_id = player["gsis_id"]
                        row = rows_by_player.setdefault(gsis_id, {
                            "player_id": gsis_id,
                            "player_name": player.get("name"),
                            "position": player.get("position"),
                            "season": season,
                            "week": week,
                            "season_type": "REG",
                            "team": team_abbr,
                        })
                        values = athlete_entry.get("stats", [])
                        for key, val in zip(keys, values):
                            norm_key = field_map.get(key)
                            if not norm_key:
                                continue
                            try:
                                row[norm_key] = clean_value(float(val))
                            except (TypeError, ValueError):
                                pass
        return list(rows_by_player.values())
    except Exception as e:
        print(f"  [espn] fetch failed: {e}")
        return []


def get_weekly_player_stats(season, week, supabase, prefer=("nflreadpy", "sleeper", "espn")):
    """
    Tries each source in order until one returns rows. Returns (source_name, rows).
    """
    fetchers = {
        "nflreadpy": lambda: fetch_nflreadpy(season, week),
        "sleeper": lambda: fetch_sleeper(season, week, supabase),
        "espn": lambda: fetch_espn(season, week, supabase),
    }
    for source in prefer:
        print(f"  trying source: {source}")
        rows = fetchers[source]()
        if rows:
            print(f"  source '{source}' returned {len(rows)} rows")
            return source, rows
        print(f"  source '{source}' returned no rows, falling back")
    return None, []
