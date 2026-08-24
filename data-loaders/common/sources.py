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

from .db import clean_value, select_all

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
    """Returns (season, week) per Sleeper's live NFL state endpoint."""
    r = requests.get("https://api.sleeper.app/v1/state/nfl", timeout=15)
    r.raise_for_status()
    d = r.json()
    return int(d["league_season"]), int(d["display_week"])


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


def get_weekly_player_stats(season, week, supabase, prefer=("sleeper", "nflreadpy", "espn")):
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
