"""
bootstrap_raw_from_warehouse.py

data/raw/ is gitignored (too big to commit), but the committed DuckDB
warehouse already holds everything that was ever loaded into it. This
rebuilds data/raw/ from raw_311_requests so a fresh clone -- or a CI
runner with nothing on disk -- can run `fetch_311_data.py --incremental`
without re-pulling three years of history from Socrata first.

Usage:
    python pipeline/ingest/bootstrap_raw_from_warehouse.py
"""

from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE_PATH = PROJECT_ROOT / "pipeline" / "warehouse" / "nyc311.duckdb"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)

    row_count = con.execute("SELECT COUNT(*) FROM raw_311_requests").fetchone()[0]
    print(f"Exporting {row_count} rows from raw_311_requests to {RAW_DATA_DIR}...")

    con.execute(f"""
        COPY (
            SELECT unique_key, created_date, closed_date, complaint_type,
                   borough, agency, status, incident_zip, year, month
            FROM raw_311_requests
        ) TO '{RAW_DATA_DIR.as_posix()}' (
            FORMAT PARQUET, PARTITION_BY (year, month), OVERWRITE_OR_IGNORE true
        )
    """)
    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
