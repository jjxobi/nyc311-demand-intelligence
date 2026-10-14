"""
Page 4: Model Monitoring

Reads the MLflow tracking store (mlflow.db + mlartifacts/) and the
drift/retrain JSON logs written by monitor.py and retrain_trigger.py.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "forecasting" / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mlflow_utils import EXPERIMENT_NAME, TRACKING_URI, get_client, setup_mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MONITORING_DIR = PROJECT_ROOT / "forecasting" / "monitoring"
DRIFT_LOG_PATH = MONITORING_DIR / "drift_log.json"
RETRAIN_LOG_PATH = MONITORING_DIR / "retrain_log.json"

st.set_page_config(page_title="Model Monitoring", layout="wide")

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
    border-right: 2px solid #2d6a5a;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("Model Monitoring")
st.markdown(
    "<p style='color:#555;margin-top:-12px;'>MLflow experiment tracking, model registry "
    "state, drift detection, and the automated retraining history behind the forecasts.</p>",
    unsafe_allow_html=True
)
st.markdown("---")


@st.cache_data(ttl=300)
def load_json_log(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def mlflow_client():
    setup_mlflow()
    return get_client()


try:
    client = mlflow_client()
    registered_models = list(client.search_registered_models())
except Exception as e:
    st.error(
        f"Could not reach the MLflow tracking store at `{TRACKING_URI}`. "
        f"Run `python forecasting/src/train.py` at least once to create it.\n\n{e}"
    )
    st.stop()

if not registered_models:
    st.info(
        "No models registered yet. Run `python forecasting/src/train.py` "
        "from the project root to populate the MLflow registry."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Registry overview
# ---------------------------------------------------------------------------

st.markdown('<div style="font-size:1rem;font-weight:600;margin-bottom:10px;">Model registry</div>',
            unsafe_allow_html=True)

rows = []
for rm in registered_models:
    versions = client.search_model_versions(f"name='{rm.name}'")
    prod_mv = next((v for v in versions if "production" in (v.aliases or [])), None)
    staging_mv = next((v for v in versions if "staging" in (v.aliases or [])), None)

    prod_metrics = dict(client.get_run(prod_mv.run_id).data.metrics) if prod_mv else {}
    rows.append({
        "Registered model": rm.name,
        "Production version": prod_mv.version if prod_mv else "—",
        "Production MAPE": f"{prod_metrics.get('mape', float('nan')):.1f}%" if prod_metrics else "—",
        "Staging version (pending review)": staging_mv.version if staging_mv else "—",
        "Total versions logged": len(client.search_model_versions(f"name='{rm.name}'")),
    })

registry_df = pd.DataFrame(rows).sort_values("Registered model").reset_index(drop=True)
st.dataframe(registry_df, use_container_width=True, hide_index=True)

st.caption(
    "A version lands in Staging instead of Production when its test MAPE is worse than "
    "the current production model's -- see the promotion logic in `train_one_category()`."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Experiment history
# ---------------------------------------------------------------------------

st.markdown('<div style="font-size:1rem;font-weight:600;margin-bottom:10px;">Training run history</div>',
            unsafe_allow_html=True)

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    [experiment.experiment_id],
    filter_string="tags.mlflow.runName != ''",
    order_by=["attribute.start_time ASC"],
    max_results=2000,
)

history_rows = []
for r in runs:
    cat = r.data.params.get("category")
    mape = r.data.metrics.get("mape")
    if cat is None or mape is None:
        continue
    history_rows.append({
        "category": cat,
        "mape": mape,
        "start_time": pd.to_datetime(r.info.start_time, unit="ms"),
        "model_type": r.data.params.get("model_type"),
        "trigger_reason": r.data.tags.get("trigger_reason", "n/a"),
    })

if history_rows:
    hist_df = pd.DataFrame(history_rows)
    selected_cat = st.selectbox("Category", sorted(hist_df["category"].unique()))
    cat_hist = hist_df[hist_df["category"] == selected_cat].sort_values("start_time")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cat_hist["start_time"], y=cat_hist["mape"],
        mode="lines+markers", name="Test MAPE",
        line=dict(color="#2d6a5a", width=2.5), marker=dict(size=8),
        hovertemplate="%{y:.1f}%%<extra></extra>",
    ))
    fig.update_layout(
        height=360, margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Training run"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Test MAPE (%)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{len(cat_hist)} training run(s) logged for {selected_cat}. "
        "Each point is one MLflow run; hover for exact values."
    )
else:
    st.caption("No per-category run history yet.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Drift monitor
# ---------------------------------------------------------------------------

st.markdown('<div style="font-size:1rem;font-weight:600;margin-bottom:10px;">Drift monitor (trailing 30 days)</div>',
            unsafe_allow_html=True)

drift_log = load_json_log(DRIFT_LOG_PATH)

if not drift_log:
    st.caption("No drift checks logged yet. Run `python forecasting/src/monitor.py`.")
else:
    latest_by_cat = {}
    for r in drift_log:
        latest_by_cat[r["category"]] = r  # log is append-ordered, last wins

    drift_rows = []
    for cat, r in sorted(latest_by_cat.items()):
        if r["status"] == "no_model":
            drift_rows.append({"Category": cat, "Status": "No model", "Trailing MAPE": "—",
                                "Baseline MAPE": "—", "Checked": "—"})
            continue
        drift_rows.append({
            "Category": cat,
            "Status": "🔴 Drift" if r["drift_flag"] else "🟢 OK",
            "Trailing MAPE": f"{r['trailing_mape']:.1f}%",
            "Baseline MAPE": f"{r['baseline_mape']:.1f}%" if r["baseline_mape"] is not None else "—",
            "Checked": pd.to_datetime(r["checked_at"]).strftime("%Y-%m-%d %H:%M UTC"),
        })

    st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, hide_index=True)
    n_drift = sum(1 for r in latest_by_cat.values() if r.get("drift_flag"))
    st.caption(
        f"{n_drift} of {len(latest_by_cat)} categories currently flagged for drift "
        f"(trailing MAPE > {1.35}x baseline AND > 15 points worse -- both conditions, "
        "to avoid flagging low-MAE categories on noise alone)."
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Retrain decision log
# ---------------------------------------------------------------------------

st.markdown('<div style="font-size:1rem;font-weight:600;margin-bottom:10px;">Retraining trigger history</div>',
            unsafe_allow_html=True)

retrain_log = load_json_log(RETRAIN_LOG_PATH)

if not retrain_log:
    st.caption("No retrain decisions logged yet. Run `python forecasting/src/retrain_trigger.py`.")
else:
    decision_rows = []
    for d in reversed(retrain_log):
        decision_rows.append({
            "Timestamp": pd.to_datetime(d["timestamp"]).strftime("%Y-%m-%d %H:%M UTC"),
            "Mode": "forced" if d.get("forced") else ("dry-run" if d.get("dry_run") else "auto"),
            "Candidates": ", ".join(d["candidates"].keys()) if d["candidates"] else "none",
            "Retrained": ", ".join(d["retrained"].keys()) if d.get("retrained") else "—",
        })
    st.dataframe(pd.DataFrame(decision_rows), use_container_width=True, hide_index=True)

    with st.expander("Why did each run do what it did?"):
        for d in reversed(retrain_log):
            ts = pd.to_datetime(d["timestamp"]).strftime("%Y-%m-%d %H:%M UTC")
            st.markdown(f"**{ts}**")
            if not d["candidates"]:
                st.markdown("- No categories met the drift or staleness threshold.")
            else:
                for cat, reason in d["candidates"].items():
                    st.markdown(f"- `{cat}`: {reason}")
