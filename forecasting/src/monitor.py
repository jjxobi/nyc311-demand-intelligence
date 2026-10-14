"""
monitor.py

Drift check. Backtests each category's production model against the most
recent trailing window of actual days and compares against its training-
time MAPE. retrain_trigger.py uses this to decide what needs retraining;
the monitoring dashboard page reads the log this writes.

Usage:
    python forecasting/src/monitor.py
    python forecasting/src/monitor.py --window-days 30
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import numpy as np
import pandas as pd

from mlflow_utils import (
    DRIFT_ABSOLUTE_DELTA_POINTS,
    DRIFT_RELATIVE_MULTIPLIER,
    get_production_version,
    get_run_metrics,
    load_production_model,
    setup_mlflow,
)
from train import (
    FEATURES,
    PROPHET_CATEGORIES,
    engineer_features,
    evaluate,
    load_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "forecasting" / "models"
MONITORING_DIR = PROJECT_ROOT / "forecasting" / "monitoring"
DRIFT_LOG_PATH = MONITORING_DIR / "drift_log.json"

MAX_LOG_ENTRIES = 500


def _production_baseline_mape(category: str, version: str | None) -> float | None:
    """The model's OWN training-time test MAPE -- used for drift detection
    (has this same model's error grown over time). A different, fair
    question from the one train.py asks when deciding whether to promote
    a new model (is the new one better than the old one, evaluated on the
    SAME window) -- see train.py's promotion comparison for that one."""
    if version is not None:
        prod = get_production_version(category)
        if prod is not None:
            return get_run_metrics(prod.run_id).get("mape")

    metrics_path = MODELS_DIR / "metrics_summary.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f).get(category, {}).get("mape")
    return None


def _load_production_model(category: str):
    model, version = load_production_model(category, category in PROPHET_CATEGORIES, MODELS_DIR)
    if model is None:
        return None, None, None
    baseline_mape = _production_baseline_mape(category, version)
    return model, baseline_mape, version


def _backtest_prophet(model, cat_df: pd.DataFrame, window_days: int) -> dict:
    window = cat_df.tail(window_days)
    forecast = model.predict(window[["ds"]])
    preds = np.maximum(forecast["yhat"].values, 0)
    return evaluate(window["y"].values, preds)


def _backtest_lightgbm(model, cat_df: pd.DataFrame, category: str, window_days: int) -> dict:
    featured = engineer_features(cat_df, category)
    window = featured.tail(window_days)
    preds = np.maximum(model.predict(window[FEATURES]), 0)
    return evaluate(window["y"].values, preds)


def check_category(category: str, df: pd.DataFrame, window_days: int) -> dict:
    cat_df = df[df["complaint_type"] == category].reset_index(drop=True)
    model, baseline_mape, version = _load_production_model(category)

    if model is None:
        return {
            "category": category, "status": "no_model",
            "trailing_mape": None, "baseline_mape": None,
            "drift_flag": False, "production_version": None,
        }

    if category in PROPHET_CATEGORIES:
        trailing_metrics = _backtest_prophet(model, cat_df, window_days)
    else:
        trailing_metrics = _backtest_lightgbm(model, cat_df, category, window_days)

    trailing_mape = trailing_metrics["mape"]
    drift_flag = False
    if baseline_mape is not None:
        relative_breach = trailing_mape > baseline_mape * DRIFT_RELATIVE_MULTIPLIER
        absolute_breach = (trailing_mape - baseline_mape) > DRIFT_ABSOLUTE_DELTA_POINTS
        drift_flag = relative_breach and absolute_breach

    return {
        "category": category, "status": "ok",
        "trailing_mape": trailing_mape, "baseline_mape": baseline_mape,
        "trailing_mae": trailing_metrics["mae"],
        "drift_flag": drift_flag, "production_version": version,
    }


def run_monitor(window_days: int = 30, log_results: bool = True) -> list:
    setup_mlflow()
    df = load_data()
    categories = sorted(df["complaint_type"].unique())

    checked_at = datetime.now(timezone.utc).isoformat()
    records = []
    for cat in categories:
        record = check_category(cat, df, window_days)
        record["checked_at"] = checked_at
        record["window_days"] = window_days
        records.append(record)

    if log_results:
        _append_drift_log(records)
        _log_to_mlflow(records)

    return records


def _append_drift_log(records: list):
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    if DRIFT_LOG_PATH.exists():
        with open(DRIFT_LOG_PATH) as f:
            existing = json.load(f)
    existing.extend(records)
    existing = existing[-MAX_LOG_ENTRIES:]
    with open(DRIFT_LOG_PATH, "w") as f:
        json.dump(existing, f, indent=2)


def _log_to_mlflow(records: list):
    metrics = {}
    for r in records:
        if r["trailing_mape"] is None:
            continue
        slug = r["category"].replace("/", "_").replace(" ", "_")
        metrics[f"trailing_mape__{slug}"] = r["trailing_mape"]
    if not metrics:
        return
    with mlflow.start_run(run_name=f"monitor-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                           tags={"run_type": "monitor"}):
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check model drift against recent actuals.")
    parser.add_argument("--window-days", type=int, default=30,
                        help="Trailing window of actuals to backtest against (default: 30)")
    args = parser.parse_args()

    results = run_monitor(window_days=args.window_days)
    flagged = [r for r in results if r["drift_flag"]]

    for r in results:
        if r["status"] == "no_model":
            print(f"[no model  ] {r['category']}")
            continue
        flag = "DRIFT" if r["drift_flag"] else "ok"
        print(
            f"[{flag:<5}] {r['category']:<35} "
            f"trailing_mape={r['trailing_mape']:6.1f}%  baseline_mape={r['baseline_mape']:6.1f}%"
        )

    print(f"\n{len(flagged)} of {len(results)} categories flagged for drift.")
