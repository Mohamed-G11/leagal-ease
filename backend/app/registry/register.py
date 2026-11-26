"""Log the RAG pipeline as an MLflow model and register a new version (W4.3).

Called from run_experiment.py with --register, inside the evaluation run, so each
registered version is linked to the run that holds its parameters and metrics.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import mlflow

from app.core.experiment_config import ExperimentConfig
from app.registry.promote import MODEL_NAME

APP_DIR = Path(__file__).resolve().parents[1]  # backend/app

# Only the packages the pipeline needs travel with the model: no legacy app/rag (PDFs, Chroma),
# no API, no database code, no caches.
MODEL_PACKAGES = ["core", "ingestion", "retrieval", "generation", "registry"]
MODEL_FILES = ["__init__.py", "pipeline.py"]
CORE_FILES = ["__init__.py", "schemas.py", "experiment_config.py"]  # not core/config.py (app settings)


def build_code_dir(dest: Path) -> Path:
    """Copy a minimal `app` package into dest and return its path (passed to code_paths)."""
    pkg = dest / "app"
    pkg.mkdir(parents=True)
    for name in MODEL_FILES:
        if (APP_DIR / name).exists():
            shutil.copy2(APP_DIR / name, pkg / name)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pdf", "*.db")
    for name in MODEL_PACKAGES:
        if name == "core":
            (pkg / "core").mkdir()
            for f in CORE_FILES:
                if (APP_DIR / "core" / f).exists():
                    shutil.copy2(APP_DIR / "core" / f, pkg / "core" / f)
        elif (APP_DIR / name).is_dir():
            shutil.copytree(APP_DIR / name, pkg / name, ignore=ignore)
    return pkg

PIP_REQUIREMENTS = [
    "mlflow", "pydantic>=2", "pyyaml", "pymupdf", "qdrant-client", "sentence-transformers", "groq", "numpy",
]


def register_pipeline(cfg: ExperimentConfig, config_path: Path, metrics: dict[str, float]) -> str:
    """Must run inside an active MLflow run. Returns the new model version number."""
    from app.registry.pyfunc_model import LegalEaseRAGModel

    with tempfile.TemporaryDirectory() as tmp:
        info = mlflow.pyfunc.log_model(
            name="rag_pipeline",
            python_model=LegalEaseRAGModel(),
            artifacts={"config": str(config_path)},
            code_paths=[str(build_code_dir(Path(tmp)))],
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
