# Raw Data

This folder is intentionally **not committed to git** (see `.gitignore`). It is
fully regenerable from the NYC Open Data Socrata API and should never be the
source of truth — the dbt-managed DuckDB warehouse (`pipeline/warehouse/`,
also gitignored) is built from these files but is itself disposable too.

## How to regenerate

From the project root, with the virtual environment active:

```powershell
python pipeline\ingest\fetch_311_data.py --start-date 2023-06-01 --end-date <today>
python pipeline\ingest\load_to_duckdb.py
```

This will take several minutes - the script pages through the Socrata API
50,000 records at a time with rate-limit-friendly delays, covering 3 years
across the 15 highest-volume complaint types (see `TARGET_COMPLAINT_TYPES`
in `fetch_311_data.py`).

## Layout

Raw files are partitioned by year and month, mirroring how the data is
typically queried downstream:

data/raw/

year=2023/month=06/311_requests_202306_<timestamp>.parquet

year=2023/month=07/311_requests_202307_<timestamp>.parquet



Each file is named with a Unix timestamp suffix to avoid collisions if a
partition is re-fetched. **Only one file should exist per partition** at any
time — if a partition shows multiple files (e.g. from a stale test run),
delete the older one before reloading into DuckDB, since the loader globs
and concatenates *all* parquet files under `data/raw/**`.

## Incremental updates

Once an initial backfill exists, new data can be appended without
re-pulling the full history:

```powershell
python pipeline\ingest\fetch_311_data.py --incremental
python pipeline\ingest\load_to_duckdb.py
```

`--incremental` checks the max `created_date` already present on disk and
only fetches records newer than that.

## Known data quality notes

- `closed_date` arrives from the API with mixed null/string values and is
  **not** typed as a timestamp in the raw layer — this is handled explicitly
  in the dbt staging model (`stg_311_requests.sql`), not here.
- A small number of true duplicate `unique_key` values exist across the full
  3-year pull (~4 out of 6.2M rows) — deduplicated in staging, not in raw.
- `incident_zip` has a small percentage of nulls (~0.3%) for non-geocoded
  requests — preserved as-is in raw, filtered for plausibility in staging.