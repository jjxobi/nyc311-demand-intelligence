# Dashboard (Component C)

Streamlit app with three pages. Reads directly from the DuckDB marts layer.
All data loading cached with `@st.cache_data` (1-hour TTL) to avoid
redundant queries on widget interaction.

## Pages

**Forecast Explorer**
Select a complaint category and borough, view the last 180 days of
historical volume, and a forward forecast with confidence intervals.
Borough-level forecasts are scaled proportionally from citywide models
(models were trained on citywide aggregates, not per-borough series).

**Staffing Recommendation**
Translates forecast volume into FTE guidance using a configurable
requests-per-staff-day assumption. Surfaces days and weeks where
forecasted demand exceeds current capacity. Weekly summary table
shows average FTE required, peak FTE, and days at risk per week.

**Ask the Data**
Plain-English question interface powered by Gemini. Generates SQL,
validates it (read-only, no DDL/DML), executes against DuckDB, and
returns a conversational answer with supporting data table and chart.
See `assistant/README.md` for full details on the safety approach.

## Running locally

```bash
streamlit run dashboard/app.py
```

Run from the project root, not from inside `dashboard/`. The app uses
`_find_project_root()` in `data.py` to locate the DuckDB warehouse and
model files regardless of working directory, but Streamlit's page
routing requires the entry point to be `dashboard/app.py` relative to
the root.

## Theme

Simple & basic theme configured via `.streamlit/config.toml`. Using the primary color `#2d6a5a` (deep
slate-green) applied to all native Streamlit widgets. Custom CSS handles
metric cards, pipeline badges, and section headers that the theme system
doesn't touch directly.

## Requirements

Models must be trained before running the dashboard:

```bash
python forecasting/src/train.py
```

A `.env` file with `GEMINI_API_KEY` is required for the Ask the Data page.
The remaining two pages work without it.