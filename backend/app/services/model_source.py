"""The champion pipeline in the MLflow Model Registry (DagsHub in production).

Only reads the registry, through Wiem's `app.registry` code. Off unless MLFLOW_TRACKING_URI is set.

Loading a registered model runs the code pickled in it: whoever can write to the registry can run code in
the API. Also, `app` is already imported here, so the loaded model runs the deployed code: what a version
really changes is its experiment config.
"""
import os

# MLflow's defaults (120 s, 7 retries with backoff) would keep the API "loading" for many minutes
# when the registry is unreachable. An explicit value in the environment wins.
HTTP_LIMITS = {"MLFLOW_HTTP_REQUEST_TIMEOUT": "15", "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2"}


def enabled() -> bool:
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


def champion_version() -> str | None:
    """The version behind the champion alias (or the role tag), None when there is none.

    Raises when the registry cannot be reached.
    """
    for name, value in HTTP_LIMITS.items():
        os.environ.setdefault(name, value)
    from mlflow.tracking import MlflowClient

    from app.registry.promote import current_champion

    version = current_champion(MlflowClient())
    return None if version is None else str(version.version)


def load_version(version: str):
    """The RAGPipeline inside that version of the model."""
    import mlflow.pyfunc

    from app.registry.promote import MODEL_NAME

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    return model.unwrap_python_model().pipeline
