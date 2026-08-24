"""
Loads snap counts into Supabase.

Modes:
  python load_snap_counts.py                Incremental: current season only (2026), upserts.
  python load_snap_counts.py --season 2024  Pulls one specific season only.
  python load_snap_counts.py --backfill     Full historical reload, 2012 through current season.
"""
import argparse

import nflreadpy as nfl

from common.db import CURRENT_SEASON, df_to_clean_rows, get_client, upsert_batched

ON_CONFLICT = "pfr_player_id,season,week"


def load_seasons(supabase, seasons, label=None):
    print(f"Pulling snap counts for seasons: {seasons}...")
    snaps = nfl.load_snap_counts(seasons=seasons).to_pandas()
    rows = df_to_clean_rows(snaps)
    upsert_batched(supabase, "snap_counts", rows, on_conflict=ON_CONFLICT, label=label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()

    supabase = get_client()

    if args.backfill:
        load_seasons(supabase, list(range(2012, CURRENT_SEASON + 1)), label="backfill")
    elif args.season:
        load_seasons(supabase, [args.season], label=str(args.season))
    else:
        load_seasons(supabase, [CURRENT_SEASON], label="incremental")

    print("Done! snap_counts table is up to date.")
