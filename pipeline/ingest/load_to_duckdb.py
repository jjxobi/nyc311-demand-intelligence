"""
load_to_duckdb.py

Loads raw partitioned parquet files into a DuckDB table as the source
for dbt staging models.
"""

import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_GLOB = str(PROJECT_ROOT / "data" / "raw" / "**" / "*.parquet")
WAREHOUSE_PATH = str(PROJECT_ROOT / "pipeline" / "warehouse" / "nyc311.duckdb")


def main():
    Path(WAREHOUSE_PATH).parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(WAREHOUSE_PATH)

    con.execute(f"""
        CREATE OR REPLACE TABLE raw_311_requests AS
        SELECT * FROM read_parquet('{RAW_DATA_GLOB}', hive_partitioning=true)
    """)

    row_count = con.execute("SELECT COUNT(*) FROM raw_311_requests").fetchone()[0]
    print(f"Loaded {row_count} rows into raw_311_requests at {WAREHOUSE_PATH}")

    print("\nSchema:")
    print(con.execute("DESCRIBE raw_311_requests").fetchdf())

    con.close()


if __name__ == "__main__":
    main()