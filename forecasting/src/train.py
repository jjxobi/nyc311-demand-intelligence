"""
train.py

CLI-runnable training script for the NYC 311 demand forecasting models.
Trains one LightGBM model per complaint category (plus Prophet for
HEAT/HOT WATER and Illegal Parking per the hybrid recommendation),
evaluates on a held-out test window, and serializes trained models
to forecasting/models/.

Usage:
    python forecasting/src/train.py
    python forecasting/src/train.py --test-days 90 --output-dir forecasting/models
"""

import argparse
import json
import pickle
from pathlib import Path

import duckdb
import holidays as hols
import lightgbm as lgb
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE_PATH = PROJECT_ROOT / "pipeline" / "warehouse" / "nyc311.duckdb"

PROPHET_CATEGORIES = ["HEAT/HOT WATER", "Illegal Parking", "Noise - Street/Sidewalk"]
# All other categories use LightGBM per the hybrid model comparison recommendation.

FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_28", "roll_std_7", "roll_std_28",
    "day_of_week", "month", "year", "is_weekend", "is_holiday",
]

US_HOLIDAYS = set(hols.US(years=range(2023, 2028)).keys())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    df = con.execute("""
        SELECT request_date as ds, complaint_type, SUM(request_count) as y
        FROM fct_daily_demand
        GROUP BY request_date, complaint_type
        ORDER BY request_date, complaint_type
    """).fetchdf()
    con.close()

    # Drop trailing partial day
    cutoff = df["ds"].max()
    df = df[df["ds"] < cutoff].copy()
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(cat_df: pd.DataFrame, category_name: str) -> pd.DataFrame:
    d = cat_df.copy().sort_values("ds").reset_index(drop=True)
    d["complaint_type"] = category_name

    d["lag_1"]  = d["y"].shift(1)
    d["lag_7"]  = d["y"].shift(7)
    d["lag_14"] = d["y"].shift(14)
    d["lag_28"] = d["y"].shift(28)

    d["roll_mean_7"]  = d["y"].shift(1).rolling(7).mean()
    d["roll_mean_28"] = d["y"].shift(1).rolling(28).mean()
    d["roll_std_7"]   = d["y"].shift(1).rolling(7).std()
    d["roll_std_28"]  = d["y"].shift(1).rolling(28).std()

    d["day_of_week"] = d["ds"].dt.dayofweek
    d["month"]       = d["ds"].dt.month
    d["year"]        = d["ds"].dt.year
    d["is_weekend"]  = (d["ds"].dt.dayofweek >= 5).astype(int)
    d["is_holiday"]  = d["ds"].apply(lambda x: int(x.date() in US_HOLIDAYS))

    return d.dropna().reset_index(drop=True)


def build_holiday_df() -> pd.DataFrame:
    us_hols = hols.US(years=range(2023, 2028))
    return pd.DataFrame([
        {"ds": pd.Timestamp(date), "holiday": name}
        for date, name in us_hols.items()
    ]).sort_values("ds").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lightgbm(cat_df: pd.DataFrame, test_days: int, val_days: int = 30):
    train_full = cat_df.iloc[:-test_days]
    test_df    = cat_df.iloc[-test_days:]
    train_df   = train_full.iloc[:-val_days]
    val_df     = train_full.iloc[-val_days:]

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        train_df[FEATURES], train_df["y"],
        eval_set=[(val_df[FEATURES], val_df["y"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    preds = np.maximum(model.predict(test_df[FEATURES]), 0)
    return model, test_df, preds


def train_prophet(cat_df: pd.DataFrame, test_days: int, holiday_df: pd.DataFrame):
    train = cat_df[["ds", "y"]].iloc[:-test_days].copy()
    test  = cat_df[["ds", "y"]].iloc[-test_days:].copy()

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holiday_df,
        seasonality_mode="multiplicative",
        interval_width=0.95,
    )
    m.fit(train)

    future   = m.make_future_dataframe(periods=test_days)
    forecast = m.predict(future)
    preds    = np.maximum(
        forecast[forecast["ds"].isin(test["ds"])]["yhat"].values, 0
    )
    return m, test, preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(actuals: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "mae":  round(mean_absolute_error(actuals, preds), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(actuals, preds))), 2),
        "mape": round(mean_absolute_percentage_error(actuals, preds) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(test_days: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data from DuckDB warehouse...")
    df = load_data()
    categories = sorted(df["complaint_type"].unique())
    holiday_df = build_holiday_df()

    print(f"Training on {len(categories)} categories, test window: {test_days} days\n")

    all_metrics = {}

    for cat in categories:
        cat_df = df[df["complaint_type"] == cat].reset_index(drop=True)

        if cat in PROPHET_CATEGORIES:
            model, test_df, preds = train_prophet(cat_df, test_days, holiday_df)
            model_type = "prophet"
            model_path = output_dir / f"prophet_{cat.replace('/', '_').replace(' ', '_')}.pkl"
        else:
            featured = engineer_features(cat_df, cat)
            model, test_df, preds = train_lightgbm(featured, test_days)
            model_type = "lightgbm"
            model_path = output_dir / f"lgbm_{cat.replace('/', '_').replace(' ', '_')}.pkl"

        metrics = evaluate(test_df["y"].values, preds)
        all_metrics[cat] = {"model_type": model_type, **metrics}

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"[{model_type:<10}] {cat:<35} MAE={metrics['mae']:7.1f}  MAPE={metrics['mape']:.1f}%")

    # Save metrics summary
    metrics_path = output_dir / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nMetrics saved to {metrics_path}")
    print(f"Models saved to {output_dir}")

    mean_mape = np.mean([v["mape"] for v in all_metrics.values()])
    print(f"\nMean MAPE across all categories: {mean_mape:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NYC 311 demand forecasting models.")
    parser.add_argument("--test-days",   type=int,  default=90,
                        help="Number of days to hold out for evaluation (default: 90)")
    parser.add_argument("--output-dir",  type=str,
                        default=str(PROJECT_ROOT / "forecasting" / "models"),
                        help="Directory to save trained models and metrics")
    args = parser.parse_args()

    main(
        test_days=args.test_days,
        output_dir=Path(args.output_dir),
    )