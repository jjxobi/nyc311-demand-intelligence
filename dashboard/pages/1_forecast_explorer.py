"""
Page 1: Forecast Explorer
"""

import sys
from pathlib import Path

# Add dashboard/ to path so 'data' module is importable directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data import generate_forecast, get_boroughs, get_categories, load_historical

st.set_page_config(page_title="Forecast Explorer", layout="wide")

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
    border-right: 2px solid #2d6a5a;
    }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    /* Selectbox label text */
    div[data-testid="stSelectbox"] label {
        color: #2a2a2a !important;
        font-weight: 500;
    }

    /* Selected value text */
    div[data-baseweb="select"] {
        color: #2a2a2a !important;
    }

    /* Dropdown menu items */
    div[data-baseweb="popover"] * {
        color: #2a2a2a !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Forecast Explorer")
st.markdown(
    "<p style='color:#555;margin-top:-12px;'>Historical demand and 90-day forward "
    "forecast by complaint category and borough.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

categories = get_categories()
boroughs   = get_boroughs()

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    selected_category = st.selectbox(
        "Complaint Category", categories,
        index=categories.index("Illegal Parking")
    )
with col2:
    selected_borough = st.selectbox("Borough", ["All (Citywide)"] + boroughs)
with col3:
    forecast_days = st.selectbox("Horizon (days)", [30, 60, 90, 180], index=2)

df = load_historical()
borough_filter = selected_borough != "All (Citywide)"

hist = (
    df[
        (df["complaint_type"] == selected_category) &
        (df["borough"] == selected_borough if borough_filter else True)
    ]
    .groupby("request_date")["request_count"]
    .sum()
    .reset_index()
    .rename(columns={"request_date": "ds", "request_count": "y"})
)
hist["ds"] = pd.to_datetime(hist["ds"])

with st.spinner("Generating forecast..."):
    forecast = generate_forecast(selected_category, days=forecast_days)

if borough_filter and not forecast.empty and not hist.empty:
    citywide = (
        df[df["complaint_type"] == selected_category]
        .groupby("request_date")["request_count"].sum()
    )
    bor = hist.set_index("ds")["y"]
    common = citywide.index.intersection(bor.index)
    if len(common) > 0:
        scale = bor[common].mean() / citywide[common].mean()
        forecast = forecast.copy()
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast[col] = forecast[col] * scale

hist_plot = hist.tail(180)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_plot["ds"], y=hist_plot["y"],
    name="Historical", line=dict(color="#2d6a5a", width=1.5),
    hovertemplate="%{y:,.0f}<extra>Historical</extra>"
))

if not forecast.empty:
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        name="Forecast", line=dict(color="#e07b39", width=2.5),
        hovertemplate="%{y:,.0f}<extra>Forecast</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"].iloc[::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(224,123,57,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI", hoverinfo="skip"
    ))
    fig.add_vline(
        x=hist_plot["ds"].max(), line_dash="dot", line_color="#aaaaaa",
        annotation_text="Forecast start", annotation_font_color="#666"
    )

fig.update_layout(
    height=480, hovermode="x unified",
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
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Daily Requests", color="#333"),
    title=dict(
        text=f"{selected_category} — {'Citywide' if not borough_filter else selected_borough}",
        font=dict(size=16, color="#1a1a1a")
    )
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Historical daily mean", f"{hist['y'].mean():,.0f}")
with col2:
    st.metric("Historical daily max",  f"{hist['y'].max():,.0f}")
with col3:
    if not forecast.empty:
        st.metric("Forecast daily mean", f"{forecast['yhat'].mean():,.0f}")
with col4:
    if not forecast.empty:
        trend = forecast["yhat"].iloc[-7:].mean() - forecast["yhat"].iloc[:7].mean()
        st.metric(
            "Trend over horizon",
            f"{'↑' if trend > 0 else '↓'} {abs(trend):,.0f}/day",
            delta_color="normal"
        )

with st.expander("Borough breakdown (historical averages)"):
    breakdown = (
        df[df["complaint_type"] == selected_category]
        .groupby("borough")["request_count"]
        .agg(["mean", "sum"])
        .rename(columns={"mean": "Daily Avg", "sum": "3yr Total"})
        .round(1)
        .sort_values("3yr Total", ascending=False)
        .reset_index()
        .rename(columns={"borough": "Borough"})
    )
    st.dataframe(breakdown, use_container_width=True, hide_index=True)