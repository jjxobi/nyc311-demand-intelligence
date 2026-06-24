"""
Page 2: Staffing Recommendation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data import generate_forecast, get_categories

st.set_page_config(page_title="Staffing Recommendation", layout="wide")

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
    border-right: 2px solid #2d6a5a;
    }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("Staffing Recommendation")
st.markdown(
    "<p style='color:#555;margin-top:-12px;'>Translate forecasted 311 demand into "
    "FTE guidance. Adjust assumptions to reflect your operational context.</p>",
    unsafe_allow_html=True
)
st.markdown("---")

st.sidebar.header("Capacity Assumptions")

selected_category = st.sidebar.selectbox(
    "Complaint Category", get_categories(),
    index=get_categories().index("Illegal Parking")
)
requests_per_staff_day = st.sidebar.number_input(
    "Requests handled per staff-day",
    min_value=1, max_value=500, value=25, step=5
)
current_capacity = st.sidebar.number_input(
    "Current FTE capacity", min_value=1, max_value=2000, value=50, step=5
)
forecast_days = st.sidebar.selectbox("Forecast horizon (days)", [30, 60, 90, 180], index=2)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.76rem;color:#444;line-height:1.7;'>
<b>FTE Required</b> = ⌈Daily Forecast ÷ Requests per Staff-Day⌉<br>
Upper bound uses the 95% forecast confidence interval.<br>
Red shading = days exceeding current capacity.
</div>
""", unsafe_allow_html=True)

with st.spinner("Generating forecast..."):
    forecast = generate_forecast(selected_category, days=forecast_days)

if forecast.empty:
    st.error(
        "Could not load forecast model. "
        "Run `python forecasting/src/train.py` from the project root first."
    )
    st.stop()

forecast = forecast.copy()
forecast["fte_required"]  = np.ceil(forecast["yhat"]       / requests_per_staff_day).astype(int)
forecast["fte_upper"]     = np.ceil(forecast["yhat_upper"] / requests_per_staff_day).astype(int)
forecast["capacity_gap"]  = forecast["fte_required"] - current_capacity
forecast["over_capacity"] = forecast["capacity_gap"] > 0

days_over   = int(forecast["over_capacity"].sum())
max_gap     = max(int(forecast["capacity_gap"].max()), 0)
mean_fte    = forecast["fte_required"].mean()
pct_at_risk = days_over / len(forecast) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Mean FTE required", f"{mean_fte:.0f}",
              delta=f"{mean_fte - current_capacity:+.0f} vs. current",
              delta_color="inverse")
with col2:
    st.metric("Days over capacity", f"{days_over} / {forecast_days}",
              delta=f"{pct_at_risk:.0f}% of horizon",
              delta_color="inverse" if days_over > 0 else "off")
with col3:
    st.metric("Peak FTE gap", f"+{max_gap}",
              delta="above capacity" if max_gap > 0 else "within capacity",
              delta_color="inverse" if max_gap > 0 else "off")
with col4:
    status = "At risk" if pct_at_risk > 20 else ("Monitor" if pct_at_risk > 0 else "OK")
    st.metric("Overall status", status)

st.markdown("---")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["fte_required"],
    name="FTE Required", line=dict(color="#e07b39", width=2.5),
    hovertemplate="%{y} FTE<extra>Required</extra>"
))
fig.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["fte_upper"],
    name="FTE (upper bound)", line=dict(color="#e07b39", width=1, dash="dot"),
    hovertemplate="%{y} FTE<extra>Upper bound</extra>"
))
fig.add_hline(
    y=current_capacity, line_dash="dash",
    line_color="#2d6a5a", line_width=2,
    annotation_text=f"Current capacity: {current_capacity} FTE",
    annotation_font_color="#2d6a5a",
    annotation_position="top left"
)

for _, row in forecast[forecast["over_capacity"]].iterrows():
    fig.add_vrect(
        x0=row["ds"] - pd.Timedelta(hours=12),
        x1=row["ds"] + pd.Timedelta(hours=12),
        fillcolor="rgba(192,57,43,0.1)", line_width=0
    )

fig.update_layout(
    height=460, hovermode="x unified",
    legend=dict(
    orientation="h",
    yanchor="top", y=0.99,
    xanchor="left", x=0.01,
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="#e0e0e0",
    borderwidth=1
),
margin=dict(l=0, r=0, t=56, b=0),
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
    xaxis=dict(showgrid=True, gridcolor="#f0f0f0", color="#333"),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="FTE Required", color="#333"),
    title=dict(
        text=f"Staffing Requirement — {selected_category}",
        font=dict(size=16, color="#1a1a1a")
    )
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Red shading indicates days where forecasted demand exceeds current capacity. "
    "Upper bound uses the 95% forecast confidence interval."
)

st.markdown("---")
st.subheader("Weekly summary")

forecast["week"] = forecast["ds"].dt.to_period("W").apply(lambda r: r.start_time)
weekly = (
    forecast.groupby("week")
    .agg(
        avg_fte=("fte_required", "mean"),
        peak_fte=("fte_required", "max"),
        days_over=("over_capacity", "sum"),
        avg_requests=("yhat", "mean"),
    )
    .reset_index()
)
weekly["avg_fte"]      = weekly["avg_fte"].round(1)
weekly["avg_requests"] = weekly["avg_requests"].round(0).astype(int)
weekly["week"]         = weekly["week"].dt.strftime("%d %b %Y")
weekly["Status"] = weekly["days_over"].apply(
    lambda x: "At risk" if x > 2 else ("Monitor" if x > 0 else "OK")
)

st.dataframe(
    weekly.rename(columns={
        "week": "Week of", "avg_fte": "Avg FTE Required",
        "peak_fte": "Peak FTE", "days_over": "Days Over Capacity",
        "avg_requests": "Avg Daily Requests",
    }),
    use_container_width=True, hide_index=True
)