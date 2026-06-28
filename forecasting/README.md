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