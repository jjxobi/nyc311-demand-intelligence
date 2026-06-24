"""
app.py -- NYC 311 Demand Intelligence Dashboard
Landing page and shared configuration.

Run from the project root:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure dashboard/ is importable by pages
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="NYC 311 Demand Intelligence",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>

    section[data-testid="stSidebar"] {
    border-right: 2px solid #2d6a5a;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #111111;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }

    .main-subtitle {
        font-size: 1rem;
        color: #555555;
        margin-top: 4px;
        margin-bottom: 20px;
    }

    hr.accent-rule {
        border: none;
        border-top: 3px solid #2d6a5a;
        margin: 4px 0 24px 0;
        width: 56px;
    }

    .metric-card {
        background: #e8f0ee;
        border-left: 4px solid #2d6a5a;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }

    .metric-card h3 {
        color: #2d6a5a;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
    }

    .metric-card p {
        color: #444444;
        font-size: 0.82rem;
        margin: 3px 0 0 0;
    }

    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #111111;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 6px;
        margin: 24px 0 14px 0;
    }

    .pipeline-step {
        display: inline-block;
        background: #2d6a5a;
        color: #ffffff !important;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
        margin-right: 6px;
        margin-bottom: 8px;
    }

    .body-text {
        color: #2a2a2a;
        font-size: 0.95rem;
        line-height: 1.65;
    }

    .muted-note {
        font-size: 0.78rem;
        color: #666666;
        margin-top: 8px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("""
<div style='padding:8px 0 12px 0;'>
    <div style='font-size:1.05rem;font-weight:700;color:#2d6a5a;'>
        NYC 311 Demand Intelligence
    </div>
    <div style='font-size:0.8rem;color:#444;margin-top:4px;'>
        Jesse O'Brien
    </div>
</div>
""", unsafe_allow_html=True)

# Footer CSS
st.markdown("""
<style>
.sidebar-footer {
    position: fixed;
    bottom: 1rem;
    width: 17rem;
    font-size: 0.76rem;
    color: #555;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# Footer content
st.sidebar.markdown("""
<div class="sidebar-footer">
<hr>
<b>Data</b>: NYC Open Data · Socrata API<br>
<b>Warehouse</b>: DuckDB · dbt<br>
<b>Models</b>: Prophet · LightGBM<br>
<b>Window</b>: Jun 2023 – Jun 2026<br>
<b>Records</b>: 6.2M service requests
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

st.markdown('<div class="main-title">NYC 311 Demand Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Operational forecasting and staffing guidance '
    'for NYC 311 service request volume</div>',
    unsafe_allow_html=True
)
st.markdown('<hr class="accent-rule">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">About this project</div>', unsafe_allow_html=True)

st.markdown("""
 <div class="body-text">
    This platform ingests NYC 311 service request data from the Socrata Open Data API,
    transforms it through a production-style dbt pipeline into a DuckDB warehouse, and
    applies a hybrid forecasting model (Prophet + LightGBM) to predict daily complaint
    volumes by category across all five NYC boroughs.<br><br>
    The staffing recommendation page translates those forecasts into operational FTE
    guidance, surfacing weeks where forecasted demand is likely to exceed current capacity
    at configurable throughput assumptions.<br><br>
    Built as a portfolio project demonstrating end-to-end data engineering, time series
    forecasting, and applied ML deployment. Full source code and methodology on GitHub.
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">Pipeline</div>', unsafe_allow_html=True)

st.markdown("""
<div style="
    text-align:center;
    max-width:700px;
    margin:0 auto;
    line-height:1.9;
">

<span class="pipeline-step">1 · Ingest</span><br>
Socrata Open Data API · Incremental loading<br><br>

<span class="pipeline-step">2 · Transform</span><br>
dbt staging → marts · DuckDB warehouse<br><br>

<span class="pipeline-step">3 · Model</span><br>
SARIMA baseline · Prophet · LightGBM<br><br>

<span class="pipeline-step">4 · Deploy</span><br>
Streamlit Community Cloud<br><br>

<span class="pipeline-step">5 · Query</span><br>
Natural Language → SQL · Gemini API

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header"> </div>', unsafe_allow_html=True)

# KPI cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>6.2M</h3>
        <p>Service requests ingested</p>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>14</h3>
        <p>Complaint categories modelled</p>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>27.6%</h3>
        <p>Mean MAPE · hybrid model</p>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>90 days</h3>
        <p>Forward forecast horizon</p>
    </div>""", unsafe_allow_html=True)


# Model performance table
st.markdown('<div class="section-header">Model performance (90-day held-out test window)</div>', unsafe_allow_html=True)

perf = pd.DataFrame({
    "Category": [
        "Illegal Parking", "Blocked Driveway", "PAINT/PLASTER",
        "Noise - Residential", "PLUMBING", "Noise - Commercial",
        "Air Quality", "Water System", "Sewer",
        "Street Condition", "Noise - Street/Sidewalk",
        "Traffic Signal Condition", "Damaged Tree", "HEAT/HOT WATER"
    ],
    "Model": [
        "Prophet", "LightGBM", "LightGBM",
        "LightGBM", "LightGBM", "LightGBM",
        "LightGBM", "LightGBM", "LightGBM",
        "LightGBM", "Prophet",
        "LightGBM", "LightGBM", "Prophet"
    ],
    "MAPE": [
        "7.1%", "8.5%", "10.5%",
        "12.3%", "17.4%", "20.1%",
        "24.0%", "27.3%", "27.9%",
        "35.3%", "39.9%",
        "48.2%", "48.7%", "59.2%"
    ],
    "Notes": [
        "Smooth weekly seasonality", "Steady high-volume", "Stable demand",
        "Weekend-driven", "Consistent pattern", "Near-tie with Prophet",
        "Low volume — absolute MAE=5 requests/day", "Moderate seasonality",
        "Moderate seasonality", "Event-driven", "Near-tie with LightGBM",
        "Spiky demand", "Storm-driven — limited forecastability",
        "Yearly on/off cycle — Prophet structural advantage"
    ]
})

st.dataframe(perf, use_container_width=True, hide_index=True)

st.markdown("""
<div class="muted-note">
Hybrid model (Prophet for HEAT/HOT WATER, Illegal Parking, Noise - Street/Sidewalk;
LightGBM for remaining 11 categories) achieves <b>27.6% mean MAPE</b> vs. 40.8% Prophet-only
and 28.4% LightGBM-only. SARIMA citywide baseline: 15.87% MAPE (aggregate only, not
directly comparable to category-level metrics).
</div>
""", unsafe_allow_html=True)