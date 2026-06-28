# LLM Assistant (Component D)

Natural language to SQL interface over the DuckDB marts layer, powered
by the Gemini API. Users ask plain-English questions; the assistant
generates SQL, validates it, executes it, and returns a conversational
answer alongside the raw data.

## How it works

1. User submits a question via the Streamlit interface
2. Gemini receives the question plus injected schema context (full
   `fct_daily_demand` schema, all valid complaint_type values, and
   explicit date interpretation rules)
3. Generated SQL is validated before execution
4. Validated query runs against a read-only DuckDB connection
5. Results are passed back to Gemini for a plain-English interpretation
6. Conversational answer, data table, and auto-generated chart returned

## Safety validation

All generated SQL is validated before execution:

- Only `SELECT` statements permitted — any DDL or DML (`INSERT`, `UPDATE`,
  `DELETE`, `DROP`, `CREATE`, `ALTER`) is blocked outright
- DuckDB connection opened in `read_only=True` mode — writes are impossible
  at the connection level regardless of what SQL is generated
- Query length and result size bounded to prevent runaway queries

This validation step is documented intentionally. LLM-generated SQL
execution is a real risk in production systems — the combination of
prompt-level constraints, application-level validation, and connection-level
read-only enforcement is a defence-in-depth approach rather than relying on
any single layer.

## Schema context

The system prompt injects:
- Full `fct_daily_demand` schema with column descriptions
- All 14 valid `complaint_type` values (prevents hallucinated categories)
- Simple date interpretation rules ("last summer" = June-August 2025, etc.)
- Borough name format (uppercase: 'BROOKLYN', 'MANHATTAN', etc.)
- Aggregation rules (always `SUM(request_count)`, never `COUNT(*)`)

## Running locally

Add your Gemini API key to `.env` in the project root:

```
GEMINI_API_KEY=your_key_here
```

Then run the dashboard:

```bash
streamlit run dashboard/app.py
```

Navigate to the "Ask the Data" page.

## Model

Uses `gemini-2.5-flash`. The assistant makes two API calls per question:
one for SQL generation, one for conversational interpretation of results.
Both are lightweight calls — the schema context is the largest input,
not the user question.