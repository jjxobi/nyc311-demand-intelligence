"""
train.py

CLI-runnable training script for the NYC 311 demand forecasting models.
Trains one LightGBM model per complaint category (plus Prophet for
HEAT/HOT WATER and Illegal Parking per the hybrid recommendation),
evaluates on a held-out test window, and serializes trained models
to forecasting/models/.

Every training run is also logged to MLflow (params, metrics, the model
artifact) and registered as a new version of nyc311-<category> in the
MLflow Model Registry. A new version is auto-promoted to production if
it beats the CURRENT production model backtested on that same test
window (not the production model's own historical number -- different
time periods aren't a fair comparison); otherwise it's left in staging.

Usage:
    python forecasting/src/train.py
    python forecasting/src/train.py --test-days 90 --output-dir forecasting/models
    python forecasting/src/train.py --categories "Sewer,Air Quality"
"""

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import duckdb
import holidays as hols
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from mlflow_utils import (
    get_production_version,
    load_production_model,
    log_and_register_model,
    promote_to_production,
    set_staging,
    setup_mlflow,
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

LGBM_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)


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

    model = lgb.LGBMRegressor(**LGBM_PARAMS, verbose=-1)
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
# Per-category training + MLflow logging/registration
# ---------------------------------------------------------------------------

def _production_mape_on_window(cat: str, model_type: str, test_df: pd.DataFrame,
                                output_dir: Path) -> float | None:
    """Backtests the CURRENT production model on the same held-out window
    the new candidate was just evaluated on. Comparing to the production
    model's own historical training-time MAPE would be comparing two
    different time periods -- unfair, and the reason a category whose
    real-world data has genuinely gotten harder to predict could stay
    stuck in staging forever even though the new model is actually the
    better one for current conditions. None if there's no usable
    production model to compare against (promote by default)."""
    model, _ = load_production_model(cat, model_type == "prophet", output_dir)
    if model is None:
        return None
    try:
        if model_type == "prophet":
            preds = np.maximum(model.predict(test_df[["ds"]])["yhat"].values, 0)
        else:
            preds = np.maximum(model.predict(test_df[FEATURES]), 0)
        return evaluate(test_df["y"].values, preds)["mape"]
    except Exception:
        return None


def train_one_category(cat: str, df: pd.DataFrame, holiday_df: pd.DataFrame,
                        test_days: int, output_dir: Path, auto_promote: bool = True) -> dict:
    cat_df = df[df["complaint_type"] == cat].reset_index(drop=True)

    if cat in PROPHET_CATEGORIES:
        model, test_df, preds = train_prophet(cat_df, test_days, holiday_df)
        model_type = "prophet"
        params = {"test_days": test_days, "seasonality_mode": "multiplicative"}
        model_path = output_dir / f"prophet_{cat.replace('/', '_').replace(' ', '_')}.pkl"
    else:
        featured = engineer_features(cat_df, cat)
        model, test_df, preds = train_lightgbm(featured, test_days)
        model_type = "lightgbm"
        params = {"test_days": test_days, **LGBM_PARAMS}
        model_path = output_dir / f"lgbm_{cat.replace('/', '_').replace(' ', '_')}.pkl"

    metrics = evaluate(test_df["y"].values, preds)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with mlflow.start_run(run_name=cat, nested=True) as run:
        version = log_and_register_model(cat, model_type, model, params, metrics)

        prod = get_production_version(cat)
        promoted, promotion_reason = False, "auto-promote disabled"
        if auto_promote:
            if prod is None:
                promote_to_production(cat, version.version)
                promoted, promotion_reason = True, "no prior production version"
            else:
                prod_mape = _production_mape_on_window(cat, model_type, test_df, output_dir)
                if prod_mape is None:
                    promote_to_production(cat, version.version)
                    promoted = True
                    promotion_reason = "existing production model couldn't be backtested on this window"
                elif metrics["mape"] <= prod_mape:
                    promote_to_production(cat, version.version)
                    promoted = True
                    promotion_reason = (
                        f"mape {metrics['mape']} <= production's {prod_mape:.2f} "
                        f"on the same test window"
                    )
                else:
                    set_staging(cat, version.version)
                    promoted = False
                    promotion_reason = (
                        f"mape {metrics['mape']} worse than production's {prod_mape:.2f} "
                        f"on the same test window, held in staging"
                    )
        mlflow.set_tags({"promoted": str(promoted), "promotion_reason": promotion_reason})

        result = {
            "model_type": model_type, **metrics,
            "mlflow_run_id": run.info.run_id,
            "mlflow_version": version.version,
            "promoted": promoted,
            "promotion_reason": promotion_reason,
        }

    print(
        f"[{model_type:<10}] {cat:<35} MAE={metrics['mae']:7.1f}  "
        f"MAPE={metrics['mape']:.1f}%  "
        f"{'-> production' if promoted else '-> staging'} (v{version.version})"
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(test_days: int, output_dir: Path, categories: list | None = None,
         trigger_reason: str = "manual", auto_promote: bool = True) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_mlflow()

    print("Loading data from DuckDB warehouse...")
    df = load_data()
    all_categories = sorted(df["complaint_type"].unique())

    if categories:
        unknown = set(categories) - set(all_categories)
        if unknown:
            raise ValueError(f"Unknown categories requested: {sorted(unknown)}")
        run_categories = categories
    else:
        run_categories = all_categories

    holiday_df = build_holiday_df()
    print(f"Training {len(run_categories)} of {len(all_categories)} categories, "
          f"test window: {test_days} days, trigger: {trigger_reason}\n")

    new_metrics = {}
    run_name = f"train-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    with mlflow.start_run(run_name=run_name, tags={
        "trigger_reason": trigger_reason,
        "categories": ",".join(run_categories),
    }):
        for cat in run_categories:
            new_metrics[cat] = train_one_category(
                cat, df, holiday_df, test_days, output_dir, auto_promote,
            )

    # Merge into the existing metrics summary rather than clobbering
    # categories that weren't part of this run (relevant for partial
    # retrains triggered by drift on a single category).
    metrics_path = output_dir / "metrics_summary.json"
    existing = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)
    existing.update(new_metrics)
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nMetrics saved to {metrics_path}")
    print(f"Models saved to {output_dir}")

    mean_mape = np.mean([v["mape"] for v in new_metrics.values()])
    print(f"\nMean MAPE across trained categories: {mean_mape:.1f}%")

    return new_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NYC 311 demand forecasting models.")
    parser.add_argument("--test-days",   type=int,  default=90,
                        help="Number of days to hold out for evaluation (default: 90)")
    parser.add_argument("--output-dir",  type=str,
                        default=str(PROJECT_ROOT / "forecasting" / "models"),
                        help="Directory to save trained models and metrics")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated list of categories to retrain. "
                             "Default: all categories.")
    parser.add_argument("--trigger-reason", type=str, default="manual",
                        help="Tag recorded on the MLflow run (e.g. 'manual', "
                             "'auto-drift', 'auto-stale'). Default: manual.")
    parser.add_argument("--no-auto-promote", action="store_true",
                        help="Register the new model version in Staging only, "
                             "skip the automatic Production promotion check.")
    args = parser.parse_args()

    main(
        test_days=args.test_days,
        output_dir=Path(args.output_dir),
        categories=[c.strip() for c in args.categories.split(",")] if args.categories else None,
        trigger_reason=args.trigger_reason,
        auto_promote=not args.no_auto_promote,
    )
