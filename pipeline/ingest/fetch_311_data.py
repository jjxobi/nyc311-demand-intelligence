"""
fetch_311_data.py

This script pulls the NYC 311 Service Request data from the Socrata Open Data API,
handles pagination and rate limits, and writes raw data to partitioned
parquet files for downstream dbt processing.

Supports incremental loading: on each run, only fetches records created
after the most recent record already on disk.

Usage:
    python fetch_311_data.py --start-date 2021-01-01 --end-date 2026-06-19
    python fetch_311_data.py --incremental   # fetch only new records since last run
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# --- Configuration ---------------------------------------------------------

SOCRATA_ENDPOINT = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
PAGE_SIZE = 50_000          # Socrata's practical max per request
RATE_LIMIT_SLEEP_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Top complaint types by volume — keeps v1 scope manageable per the brief.
# (Determined by an exploratory query; documented in pipeline/README.md)
TARGET_COMPLAINT_TYPES = [
    "Noise - Residential",
    "Illegal Parking",
    "HEAT/HOT WATER",
    "Blocked Driveway",
    "Street Condition",
    "Noise - Street/Sidewalk",
    "Water System",
    "Noise - Commercial",
    "Sanitation Condition",
    "Sewer",
    "Air Quality",
    "Traffic Signal Condition",
    "Damaged Tree",
    "PAINT/PLASTER",
    "PLUMBING",
]

FIELDS = [
    "unique_key",
    "created_date",
    "closed_date",
    "complaint_type",
    "borough",
    "agency",
    "status",
    "incident_zip",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --- Core fetch logic --------------------------------------------------------

def fetch_page(offset: int, start_date: str, end_date: str) -> list[dict]:
    """Fetch a single page of results from the Socrata API with retry logic."""
    where_clause = (
        f"created_date between '{start_date}T00:00:00' and '{end_date}T23:59:59' "
        f"AND complaint_type IN ({','.join(repr(c) for c in TARGET_COMPLAINT_TYPES)})"
    )

    params = {
        "$select": ",".join(FIELDS),
        "$where": where_clause,
        "$limit": PAGE_SIZE,
        "$offset": offset,
        "$order": "created_date ASC",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(SOCRATA_ENDPOINT, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Request failed (attempt %d/%d) at offset %d: %s",
                attempt, MAX_RETRIES, offset, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)  # exponential-ish backoff
            else:
                raise

    return []  # unreachable, satisfies type checkers


def fetch_all(start_date: str, end_date: str) -> pd.DataFrame:
    """Page through the full result set for a date range and return a DataFrame."""
    all_records = []
    offset = 0

    while True:
        logger.info("Fetching offset=%d ...", offset)
        page = fetch_page(offset, start_date, end_date)

        if not page:
            break

        all_records.extend(page)
        offset += PAGE_SIZE
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)  # be a polite API citizen

        if len(page) < PAGE_SIZE:
            break  # last page was partial -> we're done

    logger.info("Fetched %d total records for %s to %s.", len(all_records), start_date, end_date)

    if not all_records:
        return pd.DataFrame(columns=FIELDS)

    df = pd.DataFrame.from_records(all_records)
    return df


# --- Partitioned write -------------------------------------------------------

def write_partitioned_parquet(df: pd.DataFrame) -> None:
    """Write a DataFrame to raw/year=YYYY/month=MM/ partitioned parquet files."""
    if df.empty:
        logger.info("No records to write.")
        return

    df["created_date"] = pd.to_datetime(df["created_date"])
    df["_year"] = df["created_date"].dt.year
    df["_month"] = df["created_date"].dt.month

    for (year, month), group in df.groupby(["_year", "_month"]):
        partition_dir = RAW_DATA_DIR / f"year={year}" / f"month={month:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        out_path = partition_dir / f"311_requests_{year}{month:02d}_{int(time.time())}.parquet"
        group.drop(columns=["_year", "_month"]).to_parquet(out_path, index=False)
        logger.info("Wrote %d records to %s", len(group), out_path)


# --- Incremental load helper --------------------------------------------------

def get_last_loaded_date() -> str | None:
    """Find the most recent created_date already present in the raw data on disk."""
    if not RAW_DATA_DIR.exists():
        return None

    parquet_files = list(RAW_DATA_DIR.rglob("*.parquet"))
    if not parquet_files:
        return None

    max_date = None
    for f in parquet_files:
        df = pd.read_parquet(f, columns=["created_date"])
        file_max = pd.to_datetime(df["created_date"]).max()
        if max_date is None or file_max > max_date:
            max_date = file_max

    return max_date.strftime("%Y-%m-%d") if max_date is not None else None


# --- CLI entry point ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch NYC 311 data from Socrata API.")
    parser.add_argument("--start-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Fetch only records newer than the latest record already on disk.",
    )
    args = parser.parse_args()

    if args.incremental:
        last_date = get_last_loaded_date()
        if last_date is None:
            logger.error("No existing data found for incremental load. Run a full load first with --start-date.")
            return
        start_date = last_date
        logger.info("Incremental load: fetching records since %s", start_date)
    elif args.start_date:
        start_date = args.start_date
    else:
        parser.error("Either --start-date or --incremental is required.")
        return

    df = fetch_all(start_date, args.end_date)
    write_partitioned_parquet(df)


if __name__ == "__main__":
    main()