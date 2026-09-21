import math
import os
import re
import time

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

CURRENT_SEASON = 2026
BATCH_SIZE = 500


def execute_with_retry(query, retries=4, base_delay=1.0):
    """
    High request volume (e.g. the backtest harness) occasionally hits
    transient local connection errors (httpx.ReadError / 'Can't assign
    requested address' - ephemeral port pressure from rapid sequential
    requests), not a real API/data problem. Retries with backoff before
    giving up.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return query.execute()
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc


def get_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def clean_value(v):
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        # Postgres rejects "14.0" for an int column (PostgREST sends JSON
        # numbers as literal text); a whole-number float is safe to send as
        # int either way, since int inserts cleanly into numeric/float columns too.
        if v.is_integer():
            return int(v)
    return v


def clean_rows(rows):
    return [{k: clean_value(v) for k, v in row.items()} for row in rows]


def df_to_clean_rows(df):
    """Accepts a pandas DataFrame, returns upsert-ready dicts."""
    df = df.replace([float("inf"), float("-inf")], None)
    df = df.where(df.notnull(), None)
    return clean_rows(df.to_dict(orient="records"))


_table_columns_cache = {}


def get_table_columns(supabase, table):
    """
    nflreadpy adds/renames columns across releases; the live Supabase table
    schema doesn't move in lockstep, so upserting every DataFrame column can
    fail with 'column not found'. Introspect the table's real columns (via one
    existing row) and cache per-process, so loaders only ever send columns the
    table actually has.

    Returns None (not an error) when the table has no rows yet to sample from
    (e.g. a brand-new table on its first-ever write) - restrict_to_known_columns
    treats that as "can't verify, so don't filter" rather than blocking the
    write. Once the first row lands, introspection works normally after that.
    """
    if table in _table_columns_cache:
        return _table_columns_cache[table]
    resp = execute_with_retry(supabase.table(table).select("*").limit(1))
    if not resp.data:
        return None
    columns = set(resp.data[0].keys())
    _table_columns_cache[table] = columns
    return columns


def restrict_to_known_columns(supabase, table, rows):
    if not rows:
        return rows
    columns = get_table_columns(supabase, table)
    if columns is None:
        return rows
    dropped = set()
    filtered = []
    for row in rows:
        filtered.append({k: v for k, v in row.items() if k in columns})
        dropped |= (row.keys() - columns)
    if dropped:
        print(f"  {table}: dropping columns not present in live schema: {sorted(dropped)}")
    return filtered


def select_all(supabase, table, columns, filters=None, order_by="id"):
    """
    PostgREST caps a single select at ~1000 rows by default, which silently
    truncates results on any table bigger than that (bit us once already on
    ff_playerids, ~8000 rows). Pages through with .range() until a page comes
    back short of the page size.

    order_by must be a column that's unique per row (a real PK, not just "a
    column that's usually there") - without a stable sort, Postgres doesn't
    guarantee consistent row order across separate paginated requests, and
    .range() pagination can silently skip or duplicate rows between pages
    (confirmed happening in testing: identical calls returned 286 vs 3965
    rows for the same query). Most tables here use the default 'id' PK;
    ff_playerids has no 'id' column, so callers pass order_by='gsis_id'.
    """
    page_size = 1000
    out = []
    start = 0
    while True:
        query = supabase.table(table).select(columns)
        if filters:
            for f in filters:
                query = f(query)
        resp = execute_with_retry(query.order(order_by).range(start, start + page_size - 1))
        out.extend(resp.data)
        if len(resp.data) < page_size:
            break
        start += page_size
    return out


_sleeper_id_crosswalk_cache = None

# Same suffix list/stripping pattern as projections/sync_sleeper_depth_charts.py
# - reused for consistency, not reinvented. Fixes real mismatches confirmed on
# live data: ff_playerids has "Odell Beckham", projections_2026_resolved has
# "Odell Beckham Jr" - exact-match alone missed him and a handful of others.
_NAME_SUFFIXES = ["jr.", "jr", "sr.", "sr", "ii", "iii", "iv", "v"]

# Punctuation also breaks exact matching (confirmed: ff_playerids has "Tre
# Harris", projections_2026_resolved has "Tre' Harris"). Stripping apostrophes/
# hyphens/periods recovers a few more real mismatches, though most of the
# remaining gap after this is NOT a name-formatting problem at all - it's
# players missing from ff_playerids entirely, or present there with no
# sleeper_id populated (both are upstream data gaps, not fixable by matching
# better on our end - confirmed on real examples: CJ Daniels, Jerand Bradley).


def _normalize_name(name):
    name = re.sub(r"[.'\-]", "", name)
    parts = name.lower().strip().split()
    if parts and parts[-1] in _NAME_SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts)


def get_sleeper_id(gsis_id, player_name, crosswalk):
    """
    Looks up a player's sleeper_id, by gsis_id first (reliable when present)
    then by suffix-normalized name (fallback - matches the ~45% of
    projections_2026_resolved rows with no gsis_id crosswalk). Returns None
    if neither matches - Sleeper's player database is comprehensive, so this
    should be rare; a None here means a genuine crosswalk data gap (the
    player isn't in ff_playerids at all under any name - confirmed happening
    for some very new/deep players), not a Sleeper coverage gap.
    """
    if gsis_id and gsis_id in crosswalk["by_gsis"]:
        return crosswalk["by_gsis"][gsis_id]
    if player_name and _normalize_name(player_name) in crosswalk["by_name"]:
        return crosswalk["by_name"][_normalize_name(player_name)]
    return None


def build_sleeper_id_crosswalk(supabase):
    """
    ff_playerids is the canonical ID-mapping table (gsis_id <-> sleeper_id
    <-> espn_id <-> pfr_id, etc. - built by load_ff_playerids.py from
    nflverse's ff_playerids dataset). Look here first for any future
    "how do I match a player across sources" need - don't rebuild this.
    Cached per-process since it's ~8000 rows and doesn't change mid-run.
    """
    global _sleeper_id_crosswalk_cache
    if _sleeper_id_crosswalk_cache is not None:
        return _sleeper_id_crosswalk_cache

    rows = select_all(
        supabase, "ff_playerids", "gsis_id,sleeper_id,name",
        filters=[lambda q: q.not_.is_("sleeper_id", "null")],
        order_by="gsis_id",
    )
    by_gsis, by_name = {}, {}
    for r in rows:
        sid = r.get("sleeper_id")
        if sid is None:
            continue
        sid = str(int(float(sid)))
        if r.get("gsis_id"):
            by_gsis[r["gsis_id"]] = sid
        if r.get("name"):
            by_name[_normalize_name(r["name"])] = sid
    _sleeper_id_crosswalk_cache = {"by_gsis": by_gsis, "by_name": by_name}
    return _sleeper_id_crosswalk_cache


def build_fumble_rate_lookup(supabase):
    """
    proj_fum_lost_per_touch isn't part of the 3-tier blend model at all (not
    in common/matchup.py's POSITION_RATE_STATS) - there's no "blended" value
    to compute for weekly_projections_2026/ros_projections_2026. Straight
    passthrough from the preseason model's own number is the most honest
    value available right now, for every player (see
    schema_add_fumble_rate.sql for why this column exists on those two
    tables at all - it was silently missing, which corrupted QB point totals
    entirely and understated RB/WR/TE fumble penalties by falling back to a
    generic rate instead of each player's own - see
    docs/weekly_ros_pipeline.md). Fetched once per script run, not per
    player, to avoid ~800 repeat table scans.
    """
    rows = select_all(
        supabase, "projections_2026_resolved", "player_id,player_name,proj_fum_lost_per_touch",
        filters=[lambda q: q.eq("projection_year", 2026)],
    )
    by_id, by_name = {}, {}
    for r in rows:
        val = r.get("proj_fum_lost_per_touch")
        if val is None:
            continue
        if r.get("player_id"):
            by_id[r["player_id"]] = val
        if r.get("player_name"):
            by_name[r["player_name"].strip().lower()] = val
    return {"by_id": by_id, "by_name": by_name}


def get_fumble_rate(player_id, player_name, lookup):
    if player_id and player_id in lookup["by_id"]:
        return lookup["by_id"][player_id]
    if player_name and player_name.strip().lower() in lookup["by_name"]:
        return lookup["by_name"][player_name.strip().lower()]
    return None


def upsert_batched(supabase, table, rows, on_conflict, label=None):
    tag = f"[{label}] " if label else ""
    rows = restrict_to_known_columns(supabase, table, rows)
    total = len(rows)
    print(f"{tag}{table}: {total} rows to upsert")
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        execute_with_retry(supabase.table(table).upsert(batch, on_conflict=on_conflict))
        print(f"{tag}{table}: upserted {min(i + BATCH_SIZE, total)}/{total}")
    print(f"{tag}{table}: done")
