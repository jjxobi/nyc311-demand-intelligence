"""
mlflow_utils.py

Shared MLflow tracking/registry config. Local sqlite store, zero paid
infra. Real deployment would swap this for a hosted MLflow server
(Postgres + S3) -- that's the only thing that'd need to change.
"""

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.lightgbm
import mlflow.prophet
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# mlflow 3 put the old file-store (mlruns/) into maintenance mode, sqlite
# is the recommended path now. still zero infra, just a file in the repo.
DB_PATH = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"
ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"

EXPERIMENT_NAME = "nyc311-demand-forecasting"

# how far a trailing-window MAPE can drift from a model's original test MAPE
# before monitor.py calls it drift instead of noise
DRIFT_RELATIVE_MULTIPLIER = 1.35
DRIFT_ABSOLUTE_DELTA_POINTS = 15.0

# if a category's production model hasn't been retrained in this many days,
# retrain_trigger.py will retrain it regardless of drift status
STALENESS_DAYS = 30


def slugify_category(category: str) -> str:
    return category.replace("/", "_").replace(" ", "_")


def registered_model_name(category: str) -> str:
    return f"nyc311-{slugify_category(category)}"


def setup_mlflow() -> str:
    """Point mlflow at the local tracking store and make sure the
    experiment exists. Returns the experiment id."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=f"file:{ARTIFACT_DIR.as_posix()}"
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient(tracking_uri=TRACKING_URI)


def log_and_register_model(category: str, model_type: str, model, params: dict,
                            metrics: dict, extra_tags: dict | None = None):
    """Logs params/metrics/model to the active run and registers a new
    version under nyc311-<category-slug>. Call inside an active run."""
    mlflow.log_params({"category": category, "model_type": model_type, **params})
    mlflow.log_metrics(metrics)
    if extra_tags:
        mlflow.set_tags(extra_tags)

    if model_type == "prophet":
        model_info = mlflow.prophet.log_model(model, name="model")
    else:
        model_info = mlflow.lightgbm.log_model(model, name="model")

    name = registered_model_name(category)
    client = get_client()
    try:
        client.create_registered_model(name)
    except mlflow.exceptions.MlflowException:
        pass  # already exists

    version = client.create_model_version(
        name=name, source=model_info.model_uri, run_id=mlflow.active_run().info.run_id,
    )
    return version


def get_production_version(category: str):
    """Model version aliased 'production', or None. Uses registry aliases,
    not the old Staging/Production stages concept (deprecated by mlflow)."""
    client = get_client()
    name = registered_model_name(category)
    try:
        return client.get_model_version_by_alias(name, "production")
    except mlflow.exceptions.MlflowException:
        return None


def get_staging_version(category: str):
    client = get_client()
    name = registered_model_name(category)
    try:
        return client.get_model_version_by_alias(name, "staging")
    except mlflow.exceptions.MlflowException:
        return None


def get_run_metrics(run_id: str) -> dict:
    client = get_client()
    run = client.get_run(run_id)
    return dict(run.data.metrics)


def promote_to_production(category: str, version: str):
    """Points 'production' at this version -- replaces whatever it
    pointed to before, no separate archive step needed."""
    client = get_client()
    name = registered_model_name(category)
    client.set_registered_model_alias(name, "production", version)


def set_staging(category: str, version: str):
    client = get_client()
    name = registered_model_name(category)
    client.set_registered_model_alias(name, "staging", version)


def load_production_model(category: str, is_prophet: bool, models_dir: Path):
    """Loads the model aliased 'production'. Falls back to the local pkl
    under models_dir when nothing's registered yet, or when the registry
    entry's artifact path was baked in by a different machine/CI runner
    and doesn't resolve here (mlflow's local file store has no portable
    path mode). Returns (model, version) -- version is None on fallback."""
    prod = get_production_version(category)
    if prod is not None:
        try:
            if is_prophet:
                model = mlflow.prophet.load_model(f"models:/{prod.name}@production")
            else:
                model = mlflow.lightgbm.load_model(f"models:/{prod.name}@production")
            return model, prod.version
        except (OSError, mlflow.exceptions.MlflowException):
            pass

    import pickle
    prefix = "prophet" if is_prophet else "lgbm"
    path = models_dir / f"{prefix}_{slugify_category(category)}.pkl"
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        return pickle.load(f), None
