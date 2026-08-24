import math
import os
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
    """
    if table in _table_columns_cache:
        return _table_columns_cache[table]
    resp = execute_with_retry(supabase.table(table).select("*").limit(1))
    if not resp.data:
        raise RuntimeError(
            f"Can't introspect columns for '{table}': table has no rows to sample. "
            f"Run a --backfill once first, or add columns manually."
        )
    columns = set(resp.data[0].keys())
    _table_columns_cache[table] = columns
    return columns


def restrict_to_known_columns(supabase, table, rows):
    if not rows:
        return rows
    columns = get_table_columns(supabase, table)
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
