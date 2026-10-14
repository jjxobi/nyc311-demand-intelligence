# Forecasting (Component B)

Three models compared on a 90-day held-out test window against 3 years of
daily complaint volume data across 14 categories.

## Models

**SARIMA** (`02_baseline_sarima.ipynb`)
Classical seasonal ARIMA baseline on citywide aggregate volume. Establishes
that the series has learnable temporal structure before reaching for ML.
Result: 15.87% MAPE citywide. Key limitation: a single 7-day seasonal term
cannot capture the yearly HEAT/HOT WATER on/off cycle simultaneously —
the structural reason Prophet was tested next.

**Prophet** (`03_prophet_model.ipynb`)
Category-level models with weekly + yearly seasonality and US holiday
regressors. One model per category, independently fitted.
Result: 40.8% mean MAPE across 14 categories. Wins on Illegal Parking (7.1%)
and HEAT/HOT WATER (59.2% vs LightGBM's 69.6%) where smooth seasonality
gives Fourier terms a structural advantage.

**LightGBM** (`04_lightgbm_model.ipynb`)
Gradient boosting with engineered lag features (1/7/14/28-day), rolling
statistics (7/28-day mean and std), and calendar features (day of week,
month, is_weekend, is_holiday). One model per category, trained with early
stopping on a 30-day validation set carved chronologically from the training
window.
Result: 28.4% mean MAPE. Wins on 11 of 14 categories.

**Hybrid recommendation** (`05_model_comparison.ipynb`)
Prophet for HEAT/HOT WATER, Illegal Parking, and Noise - Street/Sidewalk.
LightGBM for the remaining 11. Mean MAPE: 27.6%.

## Key findings

- No single model wins universally — category demand structure drives model
  selection, not a blanket preference.
- Damaged Tree is not reliably forecastable with either model (best MAPE:
  48.7%). Storm events are not predictable from historical patterns alone.
- Air Quality MAPE (24-28%) overstates model error — absolute MAE of 5-7
  requests/day is negligible operationally.
- LightGBM's lag_1 feature dominates for HEAT/HOT WATER, which explains
  its poor performance there: when a cold snap hits after days of near-zero
  volume, yesterday's value is a bad predictor of today's spike.

## How to retrain

```bash
python forecasting/src/train.py
python forecasting/src/train.py --test-days 90 --output-dir forecasting/models
```

Models are serialized to `forecasting/models/` as `.pkl` files.
Metrics summary written to `forecasting/models/metrics_summary.json`.

## Train/test split

All models use the same 90-day chronological split — train on the past,
test on a held-out future window. No random shuffling. Time series must
respect chronological order or test metrics are meaningless.

---

## MLOps layer

Training used to just dump a `.pkl` per category into `forecasting/models/`
with no record of what changed between runs or why. That's fine for a
one-off project but it's not how you'd run this if NYC 311 data actually
kept arriving daily. The pieces below are what closes that gap: every
training run is tracked, every model is versioned, and there's an actual
decision process for when to retrain rather than "someone remembers to
run train.py again."

### Experiment tracking + model registry (MLflow)

`train.py` logs every category's training run to MLflow — params,
metrics (MAE/RMSE/MAPE), and the serialized model itself — and registers
it as a new version of `nyc311-<category>` in the MLflow Model Registry.
A new version is automatically promoted to the `production` alias if its
test MAPE is no worse than the current production version's; otherwise
it's left at the `staging` alias for manual review instead of silently
replacing a model that's actually working.

The tracking store is `mlflow.db` (SQLite) with artifacts in
`mlartifacts/` — both committed to the repo, same reasoning as the
committed DuckDB warehouse: the monitoring dashboard needs to read real
registry state after a fresh clone or a Streamlit Cloud redeploy, and
there's no hosted tracking server behind this project. A real production
setup would point `MLFLOW_TRACKING_URI` at a hosted server backed by
Postgres + S3 so the registry survives independently of any one
deploy — see `forecasting/src/mlflow_utils.py` for the one place that
assumption lives, and what to change it to.

```bash
python forecasting/src/train.py                          # retrain + log + register everything
python forecasting/src/train.py --categories "Sewer,Air Quality"
mlflow ui --backend-store-uri sqlite:///mlflow.db          # inspect runs/registry in a browser
```

### Drift monitoring

`forecasting/src/monitor.py` backtests each category's current production
model against the most recent trailing window of actual, already-closed
days, and compares that trailing MAPE to the MAPE the model recorded at
training time. A category is flagged for drift only when the trailing
MAPE is *both* more than 1.35x the baseline *and* more than 15 points
worse — requiring both avoids flagging a low-MAE category like Air
Quality (MAE ~5 requests/day) over noise that looks dramatic in
percentage terms but is operationally meaningless.

```bash
python forecasting/src/monitor.py --window-days 30
```

**Caveat, stated plainly:** this project's warehouse is a static,
bounded historical snapshot (Jun 2023–Jun 2026), not a live feed, so
there isn't currently any genuinely new reality for a model to have
missed — the trailing window is just the tail of the same data it was
trained on, so it reads as healthy by construction. The drift check is
the *mechanism* that would matter the moment ingestion is actually
running on a schedule (see Orchestration below) and new complaint volume
starts arriving that the model never saw — a storm spiking Damaged Tree
reports, a cold snap hitting HEAT/HOT WATER, a policy change shifting
how Illegal Parking gets reported, or a category getting retired the way
"Sanitation Condition" already was in this data (see
`pipeline/README.md`). It's demonstrated end-to-end here against real
numbers; it isn't a claim that it's caught real drift yet, because there
hasn't been a live feed for drift to occur against.

### Automated retraining trigger

`forecasting/src/retrain_trigger.py` is the decision layer: it runs the
drift check, decides which categories actually need retraining, and
retrains only those (not a wasted full retrain of all 14 every time).
A category gets retrained if either is true:

1. `monitor.py` flagged it for drift, or
2. its production model hasn't been retrained in 30 days (catches decay
   too slow to trip the drift threshold on any single check).

```bash
python forecasting/src/retrain_trigger.py              # check + retrain whatever needs it
python forecasting/src/retrain_trigger.py --dry-run     # report candidates without training
python forecasting/src/retrain_trigger.py --force       # retrain everything regardless of drift
```

Every run — including "nothing needed retraining" — is appended to
`forecasting/monitoring/retrain_log.json`, and every drift check is
appended to `forecasting/monitoring/drift_log.json`. The dashboard's
Model Monitoring page reads both directly.

### Orchestration

Two paths run the same ingest → dbt build → drift check/retrain
sequence, replacing the "run it by hand" workflow this project started
with:

- `.github/workflows/monthly-pipeline.yml` -- actually unattended, for
  free, on a monthly GitHub Actions cron (plus manual trigger). See
  `pipeline/README.md` for how it handles `data/raw/` not being
  committed.
- `pipeline/orchestration/definitions.py` -- the same pipeline as a
  Dagster job, for running and inspecting it locally:

  ```bash
  dagster dev -f pipeline/orchestration/definitions.py
  ```

  `dagster dev` runs the daemon that fires its schedule, but only while
  that process stays open -- there's no hosted Dagster deployment here.