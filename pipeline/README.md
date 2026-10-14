# Pipeline (Component A)

Ingests NYC 311 Service Request data from the Socrata Open Data API and
transforms it through a staging → intermediate → marts dbt architecture
into analysis-ready tables backed by DuckDB.

## Architecture

```
Socrata API
    │  fetch_311_data.py (pagination, rate limiting, incremental loading)
    ▼
data/raw/  (partitioned parquet, year=YYYY/month=MM)
    │  load_to_duckdb.py
    ▼
raw_311_requests  (DuckDB source table)
    │  dbt: staging layer
    ▼
stg_311_requests  (typed, deduplicated, cleaned -- view)
    │  dbt: intermediate layer
    ▼
int_daily_volume_by_category  (daily aggregation -- view)
    │  dbt: marts layer
    ▼
fct_daily_demand + dim_complaint_category  (consumption-ready -- tables)
```

**Why staging/intermediate/marts:** this is a standard dbt layering
convention. Staging does light cleaning with no business logic (one
input table in, one output table out). Intermediate aggregates.
Marts are the stable, documented interface that forecasting, the
dashboard, and the LLM assistant all consume -- nothing downstream
ever queries staging or intermediate directly. Staging/intermediate
materialize as views (cheap, always fresh); marts materialize as
tables (worth the storage cost for query speed on the consumption
layer).

## How to run

```bash
python pipeline/ingest/fetch_311_data.py --start-date 2023-06-01 --end-date <today>
python pipeline/ingest/load_to_duckdb.py
cd pipeline/dbt_project
dbt deps
dbt run
dbt test
```

`data/raw/` is gitignored, so a fresh clone starts without it. To pick up
new records without re-pulling three years of history from Socrata,
rebuild it from the committed warehouse first, then run an incremental
fetch:

```bash
python pipeline/ingest/bootstrap_raw_from_warehouse.py
python pipeline/ingest/fetch_311_data.py --incremental
python pipeline/ingest/load_to_duckdb.py
```

## Data quality findings

Several real data quality issues were found and explicitly handled
during this build, rather than papered over:

1. **`closed_date` mistyping.** Arrives from the raw API as a mixed
   null/string column rather than a proper timestamp. Explicitly
   cast with `try_cast` in staging; null-count verified to match
   the raw layer exactly post-cast (2,192 nulls in the 1-week pilot
   sample), confirming no valid dates were silently corrupted.

2. **Same-timestamp batch closures.** ~70,000 records (out of 6.2M)
   have `closed_date` identical to `created_date` -- concentrated in
   DOT (Street Condition) and DEP (Water System), consistent with
   bulk/batch-closure processes rather than genuinely instant
   resolution. These are preserved in `request_count` (real
   requests) but excluded from `avg_resolution_hours` (tracked
   separately via `same_day_batch_closed_count`) to avoid skewing
   the average misleadingly low.

3. **"Sanitation Condition" category verification.** Included in
   initial category scoping from a reference list, but found via
   direct API verification (querying the live Socrata endpoint
   independent of this pipeline's own filters) to have zero records
   within the 2023-06 to 2026-06 analysis window -- all 62,768
   instances of this label system-wide predate the window. This
   appears to be a retired/legacy label, likely superseded by
   "Dirty Condition" / "Dirty Conditions" (which carry similar
   semantics and meaningful volume in the live feed). Proceeded
   with 14 verified categories rather than substituting an
   unvalidated replacement category without its own multi-year
   history pulled and checked.

4. **Duplicate `unique_key` values.** A small number (4 out of
   6.2M rows) of true duplicate request IDs exist in the raw pull,
   likely from incremental re-fetch edge cases. Deduplicated in
   staging via `row_number()` partitioned on `unique_key`, keeping
   the latest `created_date`.

## Orchestration

Ingestion and dbt runs used to be manual/on-demand only. There are now
two ways this runs on a schedule instead:

**GitHub Actions** (`.github/workflows/monthly-pipeline.yml`) -- actually
runs unattended, for free, on GitHub's own runners. Monthly cron (1st of
the month, 6am UTC) plus a manual trigger. Each run: rebuilds `data/raw/`
from the committed warehouse (it's gitignored, so nothing persists
between runs on a stateless runner -- `bootstrap_raw_from_warehouse.py`
regenerates it from `raw_311_requests`, which always matches reality
regardless of GitHub cache state), fetches new 311 records
(`--incremental`), rebuilds the warehouse, runs dbt, checks drift and
retrains whatever needs it, and commits the result back.

**Dagster** (`pipeline/orchestration/definitions.py`) -- the same job,
runnable and inspectable locally:

```bash
dagster dev -f pipeline/orchestration/definitions.py
```

`dagster dev` runs both the webserver and the daemon that fires its
schedule, but only for as long as that process stays open. There's no
hosted Dagster deployment behind this project, so it's the
interactive/local way to run and watch the same pipeline -- the GitHub
Actions workflow is what's actually unattended.

See `forecasting/README.md` for what the drift check/retrain step
actually does (MLflow tracking, model registry, drift detection).

## Known limitations

- Incremental load state is inferred by scanning existing parquet
  files for the max `created_date`, rather than tracked in a
  separate metadata table. Fine at this data volume; wouldn't scale
  to a much larger raw layer.