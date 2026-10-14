"""
retrain_trigger.py

Decision layer between monitoring and training. Runs the drift check,
retrains whatever needs it, records what happened. A category is a
candidate if monitor.py flagged drift, or its production model is older
than STALENESS_DAYS (catches decay too slow to trip the drift check).

Meant to run on a schedule (see pipeline/orchestration), but it's a
normal CLI script too.

Usage:
    python forecasting/src/retrain_trigger.py
    python forecasting/src/retrain_trigger.py --dry-run
    python forecasting/src/retrain_trigger.py --force   # retrain everything
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlflow_utils import STALENESS_DAYS, get_client, get_production_version, registered_model_name
from monitor import run_monitor
from train import main as train_main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITORING_DIR = PROJECT_ROOT / "forecasting" / "monitoring"
RETRAIN_LOG_PATH = MONITORING_DIR / "retrain_log.json"
MAX_LOG_ENTRIES = 200


def _days_since_production(category: str) -> float | None:
    """Age in days of the current production version's registration
    timestamp, or None if there isn't one yet (always a candidate)."""
    prod = get_production_version(category)
    if prod is None:
        return None
    client = get_client()
    mv = client.get_model_version(registered_model_name(category), prod.version)
    created_ms = mv.creation_timestamp
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc)
    return age.total_seconds() / 86400


def decide_candidates(drift_records: list, force: bool) -> dict:
    """Returns {category: reason} for categories that should be retrained."""
    candidates = {}
    for r in drift_records:
        cat = r["category"]
        if force:
            candidates[cat] = "forced"
            continue
        if r["status"] == "no_model":
            candidates[cat] = "no production model registered yet"
            continue
        if r["drift_flag"]:
            candidates[cat] = (
                f"drift: trailing MAPE {r['trailing_mape']:.1f}% vs "
                f"baseline {r['baseline_mape']:.1f}%"
            )
            continue
        age_days = _days_since_production(cat)
        if age_days is not None and age_days > STALENESS_DAYS:
            candidates[cat] = f"stale: production model is {age_days:.0f} days old"
    return candidates


def run(window_days: int = 30, force: bool = False, dry_run: bool = False) -> dict:
    drift_records = run_monitor(window_days=window_days, log_results=not dry_run)
    candidates = decide_candidates(drift_records, force=force)

    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "forced": force,
        "dry_run": dry_run,
        "candidates": candidates,
        "retrained": {},
    }

    if not candidates:
        print("No categories need retraining right now.")
        if not dry_run:
            _append_retrain_log(decision)
        return decision

    print(f"Retraining {len(candidates)} categor{'y' if len(candidates) == 1 else 'ies'}:")
    for cat, reason in candidates.items():
        print(f"  - {cat}: {reason}")

    if dry_run:
        print("\n--dry-run set, not actually retraining.")
        return decision

    trigger_reason = "auto-forced" if force else "auto-drift-or-stale"
    new_metrics = train_main(
        test_days=90,
        output_dir=PROJECT_ROOT / "forecasting" / "models",
        categories=list(candidates.keys()),
        trigger_reason=trigger_reason,
        auto_promote=True,
    )
    decision["retrained"] = new_metrics
    _append_retrain_log(decision)
    return decision


def _append_retrain_log(decision: dict):
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    if RETRAIN_LOG_PATH.exists():
        with open(RETRAIN_LOG_PATH) as f:
            existing = json.load(f)
    existing.append(decision)
    existing = existing[-MAX_LOG_ENTRIES:]
    with open(RETRAIN_LOG_PATH, "w") as f:
        json.dump(existing, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check drift and retrain flagged categories.")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--force", action="store_true",
                        help="Retrain every category regardless of drift/staleness.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be retrained without training anything.")
    args = parser.parse_args()

    run(window_days=args.window_days, force=args.force, dry_run=args.dry_run)
