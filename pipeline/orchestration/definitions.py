"""
definitions.py -- Dagster orchestration for the NYC 311 pipeline.

Wires ingest -> dbt build -> drift check/retrain into one daily job.

    dagster dev -f pipeline/orchestration/definitions.py

That runs both the webserver (trigger daily_pipeline_job manually) and
the daemon that fires daily_pipeline_schedule
"""

import subprocess
import sys
from pathlib import Path

from dagster import (
    Definitions,
    In,
    Out,
    ScheduleDefinition,
    job,
    op,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def _run(args: list[str], cwd: Path | None = None):
    result = subprocess.run(
        args, cwd=cwd or PROJECT_ROOT, capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(args)}")
    return result.stdout


@op(out=Out(bool))
def ingest_new_data(context) -> bool:
    """Pulls new 311 records since the last incremental load."""
    context.log.info("Fetching latest 311 data from Socrata...")
    _run([PYTHON, "pipeline/ingest/fetch_311_data.py", "--incremental"])
    return True


@op(ins={"ready": In(bool)}, out=Out(bool))
def load_and_build_warehouse(context, ready: bool) -> bool:
    context.log.info("Loading new parquet into DuckDB...")
    _run([PYTHON, "pipeline/ingest/load_to_duckdb.py"])
    context.log.info("Running dbt build...")
    _run(["dbt", "run"], cwd=PROJECT_ROOT / "pipeline" / "dbt_project")
    _run(["dbt", "test"], cwd=PROJECT_ROOT / "pipeline" / "dbt_project")
    return True


@op(ins={"ready": In(bool)})
def check_drift_and_retrain(context, ready: bool):
    """Runs monitor.py + the retrain decision logic in-process (not a
    subprocess) so Dagster's own run log captures what got retrained and
    why, straight from retrain_trigger.run()."""
    sys.path.insert(0, str(PROJECT_ROOT / "forecasting" / "src"))
    from retrain_trigger import run as run_retrain_trigger

    decision = run_retrain_trigger(window_days=30, force=False, dry_run=False)
    context.log.info(f"Retrain decision: {decision['candidates']}")
    if decision["retrained"]:
        context.log.info(f"Retrained categories: {list(decision['retrained'].keys())}")
    else:
        context.log.info("No categories needed retraining.")


@job
def daily_pipeline_job():
    ready = load_and_build_warehouse(ingest_new_data())
    check_drift_and_retrain(ready)


daily_pipeline_schedule = ScheduleDefinition(
    job=daily_pipeline_job,
    cron_schedule="0 6 * * *",  # 6am daily
    execution_timezone="America/New_York",
)


defs = Definitions(
    jobs=[daily_pipeline_job],
    schedules=[daily_pipeline_schedule],
)
