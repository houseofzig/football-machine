"""
Loads weekly player stats (QB/RB/WR/TE) into Supabase.

Modes:
  python load_player_stats_weekly.py                  Incremental: current season, all weeks played
                                                        so far, via nflreadpy -> Sleeper -> ESPN fallback.
  python load_player_stats_weekly.py --season 2026 --week 3
                                                        Pulls one specific week (also runs the fallback
                                                        chain). Useful for the automation job's weekly run
                                                        and for manual backfilling a single missed week.
  python load_player_stats_weekly.py --backfill        Full historical reload, 1999 through current season,
                                                        via nflreadpy only (fallback sources don't carry
                                                        historical bulk data).
  python load_player_stats_weekly.py --source sleeper  Forces a specific source instead of the fallback
                                                        chain (for testing one source in isolation).
"""
import argparse

import nflreadpy as nfl

from common.db import CURRENT_SEASON, clean_rows, get_client, upsert_batched
from common.sources import get_current_week, get_weekly_player_stats

ON_CONFLICT = "player_id,season,week"


def backfill(supabase):
    print("Pulling 1999-current historical weekly player stats via nflreadpy...")
    stats = nfl.load_player_stats(
        seasons=list(range(1999, CURRENT_SEASON + 1)), summary_level="week"
    ).to_pandas()
    stats = stats[stats["position"].isin(["QB", "RB", "WR", "TE"])]
    stats = stats.replace([float("inf"), float("-inf")], None)
    stats = stats.where(stats.notnull(), None)
    rows = clean_rows(stats.to_dict(orient="records"))
    upsert_batched(supabase, "player_stats_weekly", rows, on_conflict=ON_CONFLICT, label="backfill")


def load_week(supabase, season, week, prefer=None):
    kwargs = {"prefer": (prefer,)} if prefer else {}
    source, rows = get_weekly_player_stats(season, week, supabase, **kwargs)
    if not rows:
        print(f"No data available for {season} week {week} from any source.")
        return
    upsert_batched(
        supabase, "player_stats_weekly", rows, on_conflict=ON_CONFLICT,
        label=f"{season}-wk{week}-{source}",
    )


def load_current_season_incremental(supabase):
    season, current_week = get_current_week()
    print(f"Incremental pull: {season}, weeks 1-{current_week}...")
    for week in range(1, current_week + 1):
        load_week(supabase, season, week)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--source", choices=["nflreadpy", "sleeper", "espn"], default=None)
    args = parser.parse_args()

    supabase = get_client()

    if args.backfill:
        backfill(supabase)
    elif args.season and args.week:
        load_week(supabase, args.season, args.week, prefer=args.source)
    else:
        load_current_season_incremental(supabase)

    print("Done! player_stats_weekly table is up to date.")
