"""
Loads team-level season stats into Supabase.

Modes:
  python load_team_stats.py                Incremental: current season only (2026), upserts.
  python load_team_stats.py --season 2024   Pulls one specific season only.
  python load_team_stats.py --backfill      Full historical reload, 1999 through current season.
"""
import argparse

import nflreadpy as nfl

from common.db import CURRENT_SEASON, df_to_clean_rows, get_client, upsert_batched

ON_CONFLICT = "team,season"


def load_seasons(supabase, seasons, label=None):
    print(f"Pulling team stats for seasons: {seasons}...")
    team_stats = nfl.load_team_stats(seasons=seasons, summary_level="reg").to_pandas()
    team_stats = team_stats.drop_duplicates(subset=["team", "season"])
    rows = df_to_clean_rows(team_stats)
    upsert_batched(supabase, "team_stats", rows, on_conflict=ON_CONFLICT, label=label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()

    supabase = get_client()

    if args.backfill:
        load_seasons(supabase, list(range(1999, CURRENT_SEASON + 1)), label="backfill")
    elif args.season:
        load_seasons(supabase, [args.season], label=str(args.season))
    else:
        load_seasons(supabase, [CURRENT_SEASON], label="incremental")

    print("Done! team_stats table is up to date.")
