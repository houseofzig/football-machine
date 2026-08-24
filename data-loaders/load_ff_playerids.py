import nflreadpy as nfl

from common.db import clean_rows, get_client, upsert_batched

supabase = get_client()

print("Loading ff_playerids...")
df = nfl.load_ff_playerids().to_pandas()

df = df.where(df.notnull(), None)
df = df.replace([float("inf"), float("-inf")], None)
df = df.dropna(subset=["gsis_id"])
df = df.drop_duplicates(subset=["gsis_id"])

print(f"Rows after filter: {len(df)}")

rows = clean_rows(df.to_dict(orient="records"))
upsert_batched(supabase, "ff_playerids", rows, on_conflict="gsis_id")

print("Done!")
