# NYC 311 Demand Intelligence

**Live demo**: [nyc311-demand-intelligence-portfolioproject.streamlit.app](https://nyc311-demand-intelligence-portfolioproject.streamlit.app/)

An end-to-end data platform for forecasting NYC 311 service request demand and translating forecasts into operational staffing guidance. Built on 3 years of live government data across 14 complaint categories and 6.2M service requests.

---

## What it does

- **Ingests** NYC 311 data from the Socrata Open Data API — paginated, rate-limit-aware, with incremental loading support
- **Transforms** raw data through a production-style dbt pipeline (staging → intermediate → marts) into a DuckDB warehouse
- **Forecasts** daily complaint volume by category using a hybrid Prophet + LightGBM model (27.6% mean MAPE)
- **Recommends** staffing levels by translating forecasts into FTE guidance with configurable capacity assumptions
- **Answers** plain-English questions about the data via a Gemini-powered NL-to-SQL assistant with safety validation
- **Tracks and versions** every model with MLflow — experiment tracking, a model registry with staging/production
  promotion, drift-based monitoring, and an automated retraining trigger, run unattended on a monthly
  GitHub Actions schedule (and locally runnable as a Dagster job)

---

## Architecture

```
Socrata Open Data API
        │
        ▼
fetch_311_data.py          Paginated ingestion, incremental loading, parquet output
        │
        ▼
data/raw/                  Partitioned parquet (year=YYYY/month=MM)
        │
        ▼
load_to_duckdb.py          Hive-partitioned glob load into DuckDB
        │
        ▼
dbt: staging               Type casting, deduplication, data quality fixes
dbt: intermediate          Daily aggregation by category and borough
dbt: marts                 fct_daily_demand + dim_complaint_category
        │
        ├── forecasting/   SARIMA baseline · Prophet · LightGBM · train.py
        │       │
        │       ├── MLflow        Experiment tracking + model registry (staging/production)
        │       ├── monitor.py    Drift check: trailing MAPE vs. training-time baseline
        │       └── retrain_trigger.py   Retrains only what drifted or went stale
        │
        ├── dashboard/     Streamlit app (Forecast Explorer, Staffing, Ask the Data, Model Monitoring)
        │
        ├── assistant/     Gemini NL-to-SQL with read-only safety validation
        │
        └── Orchestration   GitHub Actions (monthly, unattended) · Dagster (local, same job)
```

---

## Model performance (90-day held-out test)

| Category | Model | MAPE |
|---|---|---|
| Illegal Parking | Prophet | 7.1% |
| Blocked Driveway | LightGBM | 8.5% |
| PAINT/PLASTER | LightGBM | 10.5% |
| Noise - Residential | LightGBM | 12.3% |
| PLUMBING | LightGBM | 17.4% |
| Noise - Commercial | LightGBM | 20.1% |
| Air Quality | LightGBM | 24.0% |
| Water System | LightGBM | 27.3% |
| Sewer | LightGBM | 27.9% |
| Street Condition | LightGBM | 35.3% |
| Noise - Street/Sidewalk | Prophet | 39.9% |
| Traffic Signal Condition | LightGBM | 48.2% |
| Damaged Tree | LightGBM | 48.7% |
| HEAT/HOT WATER | Prophet | 59.2% |
| **Hybrid mean** | | **27.6%** |

Prophet used for HEAT/HOT WATER, Illegal Parking, and Noise - Street/Sidewalk (strong yearly/weekly seasonality). LightGBM for remaining 11 categories (lag features outperform Fourier decomposition on recent-history-driven demand). SARIMA citywide baseline: 15.87% MAPE (aggregate only, not directly comparable).

---

## Stack

| Layer | Tools |
|---|---|
| Ingestion | Python · Socrata API · pyarrow |
| Warehouse | DuckDB · dbt-core · dbt-duckdb |
| Orchestration | GitHub Actions (monthly, unattended) · Dagster (same job, local/interactive) |
| Forecasting | statsmodels (SARIMA) · Prophet · LightGBM · scikit-learn |
| MLOps | MLflow — experiment tracking, model registry, staging/production promotion, drift monitoring |
| Dashboard | Streamlit · Plotly |
| LLM Assistant | Gemini API · google-genai |
| Deployment | Streamlit Community Cloud · Git LFS |

All tools free and open-source. No paid infrastructure.

---

## Local setup

```bash
git clone https://github.com/your-username/nyc311-demand-intelligence
cd nyc311-demand-intelligence
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Regenerate the warehouse from scratch

```bash
python pipeline/ingest/fetch_311_data.py --start-date 2023-06-01 --end-date 2026-06-19
python pipeline/ingest/load_to_duckdb.py
cd pipeline/dbt_project
dbt deps
dbt run
dbt test
```

### Retrain models

```bash
python forecasting/src/train.py
```

Every run is tracked in MLflow (experiment tracking, model registry,
staging/production promotion) — see `forecasting/README.md` for the full
MLOps layer: drift monitoring and the automated retraining trigger. That
trigger runs unattended on a monthly GitHub Actions schedule
(`.github/workflows/monthly-pipeline.yml`) and is also runnable/inspectable
locally as a Dagster job.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db    # inspect runs/registry
python forecasting/src/monitor.py                    # check drift against recent actuals
python forecasting/src/retrain_trigger.py --dry-run  # see what would be retrained and why
dagster dev -f pipeline/orchestration/definitions.py # run the same pipeline locally
```

### Run the dashboard locally

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```

Then:
```bash
streamlit run dashboard/app.py
```

---

## Data quality findings

Four real data quality issues found and explicitly handled during the build:

1. **`closed_date` mistyping** — arrives as VARCHAR from the API; explicitly cast in dbt staging with null-count verification post-cast.
2. **Same-timestamp batch closures** — ~70K records where `closed_date = created_date`, concentrated in DOT and DEP agencies. Preserved in `request_count` but excluded from `avg_resolution_hours` to avoid skewing the average.
3. **"Sanitation Condition" category** — included in initial scoping but confirmed via direct API query to have zero records in the 2023-2026 window. Documented and excluded rather than silently swapped.
4. **Duplicate `unique_key` values** — 4 out of 6.2M rows; deduplicated in staging via `row_number()` partitioned on `unique_key`.

---

## Project structure

```
nyc311-demand-intelligence/
├── .github/workflows/     monthly-pipeline.yml — unattended ingest → dbt → drift check/retrain
├── pipeline/              Component A — ingestion, dbt
│   └── orchestration/     definitions.py — same pipeline as a local/interactive Dagster job
├── forecasting/           Component B — EDA, SARIMA, Prophet, LightGBM, train.py
│   ├── src/
│   │   ├── train.py           Trains + logs to MLflow + registers + auto-promotes
│   │   ├── monitor.py         Drift check: trailing MAPE vs. training-time baseline
│   │   ├── retrain_trigger.py Decides what needs retraining, retrains it
│   │   └── mlflow_utils.py    Shared tracking/registry config
│   └── monitoring/        drift_log.json, retrain_log.json (append-only decision history)
├── dashboard/             Component C — Streamlit app
│   ├── app.py
│   └── pages/
│       ├── 1_forecast_explorer.py
│       ├── 2_staffing_recommendation.py
│       ├── 3_ask_the_data.py
│       └── 4_model_monitoring.py  Registry state, run history, drift + retrain logs
├── mlflow.db / mlartifacts/   MLflow tracking store + model artifacts (committed, see below)
├── data/                  Raw parquet files (gitignored, regenerable)
└── docs/                  Architecture diagram, demo assets
```

---

*Built by Jesse O'Brien as a project demonstrating end-to-end data engineering, time series forecasting, and applied ML deployment.*