"""
Page 3: Ask the Data

Natural language to SQL via Gemini API, executed against
the DuckDB marts layer. Read-only, with validation before execution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import re
import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai
from data import PROJECT_ROOT, WAREHOUSE_PATH

load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(page_title="Ask the Data", layout="wide")

st.markdown("""
<style>
    section[data-testid="stSidebar"] { border-right: 2px solid #2d6a5a; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .sql-box {
        background: #f8f8f8;
        border-left: 3px solid #2d6a5a;
        border-radius: 4px;
        padding: 12px 16px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #1a1a1a;
        white-space: pre-wrap;
        margin: 8px 0 16px 0;
    }
    .answer-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #2d6a5a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .interpretation {
        font-size: 1.05rem;
        color: #1a1a1a;
        line-height: 1.6;
        padding: 12px 0 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Schema context
# ---------------------------------------------------------------------------

SCHEMA_CONTEXT = """
You are a SQL expert. Generate a single, valid DuckDB SQL query to answer
the user's question about NYC 311 service requests.

The database has one primary table: fct_daily_demand

Schema:
  fct_daily_demand (
      request_date          DATE,
      complaint_category_id VARCHAR,
      complaint_type        VARCHAR,
      category_group        VARCHAR,
      borough               VARCHAR,     -- 'BROOKLYN', 'MANHATTAN', 'QUEENS', 'BRONX', 'STATEN ISLAND', 'UNSPECIFIED'
      request_count         INTEGER,
      closed_count          INTEGER,
      avg_resolution_hours  FLOAT,
      day_of_week           INTEGER,     -- 0=Monday, 6=Sunday
      is_weekend            BOOLEAN,
      month                 INTEGER,     -- 1-12
      year                  INTEGER      -- 2023, 2024, 2025, 2026
  )

Available complaint_type values:
'Air Quality', 'Blocked Driveway', 'Damaged Tree', 'HEAT/HOT WATER',
'Illegal Parking', 'Noise - Commercial', 'Noise - Residential',
'Noise - Street/Sidewalk', 'PAINT/PLASTER', 'PLUMBING', 'Sewer',
'Street Condition', 'Traffic Signal Condition', 'Water System'

Rules:
- Return ONLY the SQL query, no explanation, no markdown, no backticks.
- Use only SELECT statements. No INSERT, UPDATE, DELETE, DROP, CREATE, or ALTER.
- Always use SUM(request_count) when aggregating volume -- never COUNT(*).
- Date filtering: use request_date >= DATE '2023-06-01' style syntax.
- 'Last summer' means June-August of the most recent full year (2025).
- 'Last year' means year = 2025.
- 'This year' means year = 2026.
- Borough names are uppercase: 'BROOKLYN', 'MANHATTAN', etc.
- When returning month numbers, replace the numeric month with a readable name
  using a CASE statement. Name the column 'month'. Do NOT return both a number
  and a name -- return only the named column.
- When returning day_of_week numbers, replace with day names using a CASE
  statement. Name the column 'day_of_week'. Do NOT return both.
- If no year is specified, aggregate across all years unless context implies otherwise.
- Limit results to 50 rows unless the user asks for more.
"""

# ---------------------------------------------------------------------------
# SQL safety validation
# ---------------------------------------------------------------------------

FORBIDDEN = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE)\b',
    re.IGNORECASE
)

def validate_sql(sql: str) -> tuple[bool, str]:
    if FORBIDDEN.search(sql):
        return False, "Query contains forbidden DDL/DML statement."
    if not sql.strip().upper().startswith("SELECT"):
        return False, "Only SELECT statements are permitted."
    return True, "OK"


# ---------------------------------------------------------------------------
# Gemini SQL generation
# ---------------------------------------------------------------------------

def generate_sql(question: str) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Add it to your .env file.")
        return None

    client = genai.Client(api_key=api_key)
    prompt = f"{SCHEMA_CONTEXT}\n\nQuestion: {question}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        sql = response.text.strip()
        sql = re.sub(r'^```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'^```\s*', '', sql)
        sql = re.sub(r'\s*```$', '', sql)
        return sql.strip()
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Conversational interpretation
# ---------------------------------------------------------------------------

def generate_interpretation(question: str, sql: str, result: pd.DataFrame) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    client = genai.Client(api_key=api_key)
    result_str = result.head(10).to_string(index=False)

    prompt = f"""
You are a data analyst. A user asked the following question about NYC 311 service requests:

Question: {question}

The query returned this data:
{result_str}

Write a concise, conversational response (2-3 sentences max) that directly answers
the question using the specific numbers from the data. Be specific -- use the actual
values. Do not mention SQL, databases, or technical terms. Speak as if explaining
to a city operations manager, not a developer.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def execute_query(sql: str) -> pd.DataFrame | None:
    try:
        con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
        result = con.execute(sql).fetchdf()
        con.close()
        return result
    except Exception as e:
        st.error(f"Query execution error: {e}")
        return None


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_result(df: pd.DataFrame) -> pd.DataFrame:
    month_map = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    day_map = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
        4: "Friday", 5: "Saturday", 6: "Sunday"
    }
    df = df.copy()
    for col in df.columns:
        if col.lower() == "month" and df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].map(month_map).fillna(df[col].astype(str))
        if col.lower() == "day_of_week" and df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].map(day_map).fillna(df[col].astype(str))

    # Drop redundant named columns if the base column already exists and is readable
    if "month" in df.columns and "month_name" in df.columns:
        df = df.drop(columns=["month_name"])
    if "day_of_week" in df.columns and "day_name" in df.columns:
        df = df.drop(columns=["day_name"])

    return df


# ---------------------------------------------------------------------------
# Auto chart
# ---------------------------------------------------------------------------

def auto_chart(df: pd.DataFrame) -> None:
    if df.empty or len(df) < 2:
        return

    cols         = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols    = [c for c in cols if "date" in c.lower()]
    cat_cols     = [c for c in cols if df[c].dtype == object]

    if not numeric_cols:
        return

    y_col  = numeric_cols[0]
    colors = ["#2d6a5a", "#e07b39", "#5a8fd4", "#c0392b", "#8e44ad"]

    if date_cols:
        fig = px.line(
            df, x=date_cols[0], y=y_col,
            color=cat_cols[0] if cat_cols else None,
            title=f"{y_col} over time",
            color_discrete_sequence=colors
        )
    elif cat_cols and len(df) <= 20:
        fig = px.bar(
            df.sort_values(y_col, ascending=False),
            x=cat_cols[0], y=y_col,
            title=f"{y_col} by {cat_cols[0]}",
            color_discrete_sequence=["#2d6a5a"]
        )
    else:
        return

    fig.update_layout(
        height=380,
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Ask the Data")
st.markdown(
    "<p style='color:#555;margin-top:-12px;'>Ask a plain-English question "
    "about NYC 311 demand. Gemini generates the SQL, we validate and run it.</p>",
    unsafe_allow_html=True
)
st.markdown("---")

with st.expander("Example questions"):
    st.markdown("""
    - Which borough had the most noise complaints last summer?
    - What were the top 5 complaint types in Brooklyn in 2025?
    - How did HEAT/HOT WATER complaints change month by month in winter 2024-25?
    - What is the average resolution time by complaint type?
    - Which day of the week has the highest illegal parking complaints?
    - Compare total requests by borough for 2024 vs 2025.
    """)

st.markdown(
    "<p style='font-weight:600;color:#1a1a1a;margin-bottom:2px;'>"
    "Type your question</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='font-size:0.78rem;color:#888;margin-bottom:6px;'>"
    "Powered by Gemini API (free tier). Occasional rate limiting may apply "
    "— if a request fails, wait a few seconds and try again.</p>",
    unsafe_allow_html=True
)

question = st.text_area(
    "question",
    placeholder="e.g. Which borough had the most noise complaints last summer?",
    label_visibility="collapsed",
    height=80
)

run = st.button("Ask", type="primary", use_container_width=True)

if run and question.strip():
    st.markdown("---")

    with st.spinner("Generating SQL..."):
        sql = generate_sql(question)

    if sql:
        is_safe, reason = validate_sql(sql)

        if not is_safe:
            st.error(f"Query blocked: {reason}")
        else:
            with st.spinner("Running query..."):
                result = execute_query(sql)

            if result is not None and not result.empty:
                formatted = format_result(result)

                with st.spinner("Interpreting results..."):
                    interpretation = generate_interpretation(
                        question, sql, formatted
                    )

                if interpretation:
                    st.markdown(
                        f'<div class="interpretation">{interpretation}</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("---")

                with st.expander("View generated SQL"):
                    st.markdown(
                        f'<div class="sql-box">{sql}</div>',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    '<div class="answer-label">Data</div>',
                    unsafe_allow_html=True
                )
                st.dataframe(formatted, use_container_width=True, hide_index=True)
                st.caption(
                    f"{len(result)} row{'s' if len(result) != 1 else ''} returned."
                )

                auto_chart(formatted)

            elif result is not None and result.empty:
                st.info("The query ran successfully but returned no results.")

elif run and not question.strip():
    st.warning("Please enter a question.")