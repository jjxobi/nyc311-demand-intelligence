# Pipeline

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

## Known limitations

- Ingestion and dbt runs are currently manual/on-demand, not
  scheduled. A GitHub Actions cron workflow could be used in the future to automate this.
- Incremental load state is inferred by scanning existing parquet
  files for the max `created_date`, rather than tracked in a
  separate metadata table. Fine at this data volume; wouldn't scale
  to a much larger raw layer.