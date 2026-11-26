"""Log the RAG pipeline as an MLflow model and register a new version (W4.3).

Called from run_experiment.py with --register, inside the evaluation run, so each
registered version is linked to the run that holds its parameters and metrics.
"""
from __future__ import annotations

from pathlib import Path

import mlflow

from app.core.experiment_config import ExperimentConfig
from app.registry.promote import MODEL_NAME

APP_DIR = Path(__file__).resolve().parents[1]  # backend/app, shipped with the model

PIP_REQUIREMENTS = [
    "mlflow", "pydantic>=2", "pyyaml", "pymupdf", "qdrant-client", "sentence-transformers", "groq", "numpy",
]


def register_pipeline(cfg: ExperimentConfig, config_path: Path, metrics: dict[str, float]) -> str:
    """Must run inside an active MLflow run. Returns the new model version number."""
    from app.registry.pyfunc_model import LegalEaseRAGModel

    info = mlflow.pyfunc.log_model(
        name="rag_pipeline",
        python_model=LegalEaseRAGModel(),
        artifacts={"config": str(config_path)},
        code_paths=[str(APP_DIR)],
        pip_requirements=PIP_REQUIREMENTS,
        registered_model_name=MODEL_NAME,
    )
    version = str(info.registered_model_version)

    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(MODEL_NAME, version, "experiment_config", cfg.name)
    client.set_model_version_tag(MODEL_NAME, version, "retrieval_mode", cfg.retrieval.mode)
    for key in ("recall_at_5", "mrr", "abstention_correct_rate", "citation_hit_rate"):
        if key in metrics:
            client.set_model_version_tag(MODEL_NAME, version, key, f"{metrics[key]:.3f}")
    client.update_model_version(
        MODEL_NAME, version,
        description=f"{cfg.name}: embeddings={cfg.embedding.model}, mode={cfg.retrieval.mode}, "
                    f"rewrite={cfg.query_transform.enabled}, rerank={cfg.reranker.enabled}, llm={cfg.llm.model}",
    )
    print(f"Registered {MODEL_NAME} v{version} ({cfg.name})")
    return version
