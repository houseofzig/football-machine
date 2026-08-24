"""
Orchestrates the recurring in-season data pull. Meant to be run by the GitHub
Actions schedule (see .github/workflows/weekly_pull.yml) on two cadences:

  Tuesday morning  - the day after Monday Night Football closes out the week.
                      nflreadpy updates nightly during the season, so this pull
                      should have real numbers, but NFL stat corrections still
                      land through Wednesday.
  Thursday          - reconciliation pass. Re-runs the exact same pull; nflverse's
                      own docs call Thursday's data "the cleanest we have" since
                      corrections are done landing by then. Upserts are safe to
                      repeat, so this is just "run it again."

Pulls (in order): schedules (full-season upsert + a Vegas-lines refresh for the
upcoming week), player stats for the last two weeks (covers the just-finished
week even if Sleeper's "display_week" hasn't flipped yet), team stats, and
snap counts - current season only, since these feed the weekly/ROS engine
which only cares about the current season plus this season's game log.

Does NOT compute weekly/ROS projections itself - that's Phase 3/4 of the
build (see project_weekly_ros_rankings_plan.md). This script's only job is
keeping the raw Supabase tables (schedules, player_stats_weekly, team_stats,
snap_counts) fresh.
"""
import argparse
import subprocess
import sys

from common.sources import get_current_week


def run(cmd, pass_label):
    print(f"\n=== [{pass_label}] running: {' '.join(cmd)} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"=== [{pass_label}] FAILED: {' '.join(cmd)} (exit {result.returncode}) ===")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-label", choices=["tuesday", "thursday", "manual"], default="manual")
    args = parser.parse_args()

    season, display_week = get_current_week()
    weeks_to_pull = sorted(set(w for w in (display_week - 1, display_week) if w >= 1))
    print(f"[{args.pass_label}] season={season}, display_week={display_week}, "
          f"pulling weeks={weeks_to_pull}")

    failures = []

    if not run([sys.executable, "load_schedules.py"], args.pass_label):
        failures.append("load_schedules.py")
    if not run([sys.executable, "load_schedules.py", "--vegas-only"], args.pass_label):
        failures.append("load_schedules.py --vegas-only")

    for week in weeks_to_pull:
        cmd = [sys.executable, "load_player_stats_weekly.py", "--season", str(season), "--week", str(week)]
        if not run(cmd, args.pass_label):
            failures.append(f"load_player_stats_weekly.py week {week}")

    if not run([sys.executable, "load_team_stats.py"], args.pass_label):
        failures.append("load_team_stats.py")
    if not run([sys.executable, "load_snap_counts.py"], args.pass_label):
        failures.append("load_snap_counts.py")

    if failures:
        print(f"\n[{args.pass_label}] Completed with failures: {failures}")
        sys.exit(1)

    print(f"\n[{args.pass_label}] All pulls completed successfully.")


if __name__ == "__main__":
    main()
