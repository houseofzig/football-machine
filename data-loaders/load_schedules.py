"""
Loads NFL schedules (including spread_line/total_line Vegas data) into Supabase.

Modes:
  python load_schedules.py                 Incremental: pulls current season only (2026), upserts.
  python load_schedules.py --season 2024   Pulls one specific season only.
  python load_schedules.py --backfill      Full historical reload, 1999 through current season.
  python load_schedules.py --vegas-only    Refreshes only the upcoming week's spread_line/total_line
                                            (meant to run on its own, more frequent cadence, since
                                            Vegas doesn't post lines until ~6-10 days before kickoff).
"""
import argparse

import nflreadpy as nfl

from common.db import CURRENT_SEASON, df_to_clean_rows, get_client, upsert_batched
from common.sources import get_current_week


def load_seasons(supabase, seasons, label=None):
    print(f"Pulling schedules for seasons: {seasons}...")
    schedules = nfl.load_schedules(seasons=seasons).to_pandas()
    schedules["gameday"] = schedules["gameday"].astype(str)
    rows = df_to_clean_rows(schedules)
    upsert_batched(supabase, "schedules", rows, on_conflict="game_id", label=label)


def load_vegas_only(supabase):
    season, week = get_current_week()
    print(f"Refreshing Vegas lines for {season} week {week}...")
    schedules = nfl.load_schedules(seasons=[season]).to_pandas()
    schedules = schedules[schedules["week"] == week]
    if schedules.empty:
        print(f"No schedule rows found for {season} week {week}, nothing to refresh.")
        return
    schedules["gameday"] = schedules["gameday"].astype(str)
    keep_cols = ["game_id", "season", "week", "spread_line", "total_line"]
    schedules = schedules[[c for c in keep_cols if c in schedules.columns]]
    rows = df_to_clean_rows(schedules)
    upsert_batched(supabase, "schedules", rows, on_conflict="game_id", label="vegas-only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--vegas-only", action="store_true")
    args = parser.parse_args()

    supabase = get_client()

    if args.vegas_only:
        load_vegas_only(supabase)
    elif args.backfill:
        load_seasons(supabase, list(range(1999, CURRENT_SEASON + 1)), label="backfill")
    elif args.season:
        load_seasons(supabase, [args.season], label=str(args.season))
    else:
        load_seasons(supabase, [CURRENT_SEASON], label="incremental")

    print("Done! schedules table is up to date.")
