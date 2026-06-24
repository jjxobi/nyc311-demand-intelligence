"""
data.py -- shared data loading for the NYC 311 Demand Intelligence dashboard.
To be used as an module to import / not visible ...
(sys.path must include the dashboard/ directory, which app.py and each page
handle via sys.path.insert at the top of each file.)
"""

import os
import pickle
from pathlib import Path

import duckdb
import holidays as hols
import numpy as np
import pandas as pd
import streamlit as st


def _find_project_root() -> Path:
    if "PROJ_ROOT" in os.environ:
        return Path(os.environ["PROJ_ROOT"])
    candidate = Path(__file__).resolve().parent
    for _ in range(6):
        if (candidate / "pipeline").exists() and (candidate / "forecasting").exists():
            return candidate
        candidate = candidate.parent
    raise RuntimeError(
        "Could not locate project root. "
        "Set the PROJ_ROOT environment variable to the repo root path."
    )


PROJECT_ROOT = _find_project_root()
WAREHOUSE_PATH = PROJECT_ROOT / "pipeline" / "warehouse" / "nyc311.duckdb"
MODELS_DIR     = PROJECT_ROOT / "forecasting" / "models"

PROPHET_CATEGORIES = ["HEAT/HOT WATER", "Illegal Parking", "Noise - Street/Sidewalk"]

FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_28", "roll_std_7", "roll_std_28",
    "day_of_week", "month", "year", "is_weekend", "is_holiday",
]

US_HOLIDAYS = set(hols.US(years=range(2023, 2028)).keys())


@st.cache_data(ttl=3600)
def load_historical() -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    df = con.execute("""
        SELECT request_date, complaint_type, category_group,
               borough, request_count, day_of_week, is_weekend, month, year
        FROM fct_daily_demand
        ORDER BY request_date, complaint_type, borough
    """).fetchdf()
    con.close()
    cutoff = df["request_date"].max()
    df = df[df["request_date"] < cutoff].copy()
    df["request_date"] = pd.to_datetime(df["request_date"])
    return df


@st.cache_data(ttl=3600)
def get_categories() -> list:
    return sorted(load_historical()["complaint_type"].unique().tolist())


@st.cache_data(ttl=3600)
def get_boroughs() -> list:
    return sorted(load_historical()["borough"].unique().tolist())


def _cat_slug(category: str) -> str:
    return category.replace("/", "_").replace(" ", "_")


def load_model(category: str):
    prefix = "prophet" if category in PROPHET_CATEGORIES else "lgbm"
    path = MODELS_DIR / f"{prefix}_{_cat_slug(category)}.pkl"
    if not path.exists():
        st.error(
            f"Model not found: `{path.name}`\n\n"
            "Run `python forecasting/src/train.py` from the project root."
        )
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def generate_forecast(category: str, days: int = 90) -> pd.DataFrame:
    model = load_model(category)
    if model is None:
        return pd.DataFrame()

    df = load_historical()
    cat_agg = (
        df[df["complaint_type"] == category]
        .groupby("request_date")["request_count"]
        .sum()
        .reset_index()
        .rename(columns={"request_date": "ds", "request_count": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )

    last_date = cat_agg["ds"].max()

    if category in PROPHET_CATEGORIES:
        # Prophet was trained on data ending at the train/test split (~Mar 2026).
        # make_future_dataframe extends from the training end, not from last_date.
        # Calculate how many periods are needed to reach last_date + forecast days.
        train_end = pd.Timestamp(model.history_dates.max())
        days_to_cover = (last_date - train_end).days + days + 1
        
        future   = model.make_future_dataframe(periods=days_to_cover)
        forecast = model.predict(future)
        
        # Take exactly `days` rows after last_date
        result = forecast[forecast["ds"] > last_date].head(days)[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()
        
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            result[col] = np.maximum(result[col], 0)
        return result.reset_index(drop=True)

    else:
        history      = cat_agg["y"].tolist()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=days
        )
        rows = []
        for fd in future_dates:
            h   = history
            row = {
                "lag_1":        h[-1]  if len(h) >= 1  else 0,
                "lag_7":        h[-7]  if len(h) >= 7  else 0,
                "lag_14":       h[-14] if len(h) >= 14 else 0,
                "lag_28":       h[-28] if len(h) >= 28 else 0,
                "roll_mean_7":  float(np.mean(h[-7:]))  if len(h) >= 7  else float(np.mean(h)),
                "roll_mean_28": float(np.mean(h[-28:])) if len(h) >= 28 else float(np.mean(h)),
                "roll_std_7":   float(np.std(h[-7:]))   if len(h) >= 7  else 0.0,
                "roll_std_28":  float(np.std(h[-28:]))  if len(h) >= 28 else 0.0,
                "day_of_week":  fd.dayofweek,
                "month":        fd.month,
                "year":         fd.year,
                "is_weekend":   int(fd.dayofweek >= 5),
                "is_holiday":   int(fd.date() in US_HOLIDAYS),
            }
            pred = float(np.maximum(
                model.predict(pd.DataFrame([row])[FEATURES])[0], 0
            ))
            rows.append({"ds": fd, "yhat": pred})
            history.append(pred)

        result = pd.DataFrame(rows)
        result["yhat_lower"] = result["yhat"] * 0.85
        result["yhat_upper"] = result["yhat"] * 1.15
        return result