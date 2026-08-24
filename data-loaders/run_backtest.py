import json
import sys
import time

from common.backtest import backtest_season
from common.db import get_client

if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    start_week = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out_path = sys.argv[3] if len(sys.argv) > 3 else f"backtest_{season}.json"

    sb = get_client()
    t0 = time.time()
    results = backtest_season(sb, season, start_week=start_week)
    elapsed = time.time() - t0
    print(f"\nTotal backtest time: {elapsed:.1f}s")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")
