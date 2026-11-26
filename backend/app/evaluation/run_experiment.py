"""Run one experiment config end to end and log everything to MLflow.

    python -m app.evaluation.run_experiment \
        --config ../experiments/E0_naive_rag/config.yaml \
        --dataset ../data/evaluation/seed.jsonl \
        --pdf-dir ../data/raw --reindex --generation

MLflow server: Mohamed's docker-compose (set MLFLOW_TRACKING_URI=http://localhost:5000).
Without it, runs go to a local SQLite file (view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import mlflow

from app.core.experiment_config import ExperimentConfig, load_config
from app.evaluation.retrieval_eval import evaluate_generation, evaluate_retrieval, load_dataset

EXPERIMENT = "legalease-rag"


def flatten(d: dict, parent: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else k
        out.update(flatten(v, key) if isinstance(v, dict) else {key: v})
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(cfg: ExperimentConfig, config_path: Path, dataset_path: Path, pdf_dir: Path | None,
        reindex: bool, generation: bool, retriever=None, pipeline=None, embedder=None,
        register: bool = False, promote: bool = False) -> dict:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    dataset = load_dataset(dataset_path)

    with mlflow.start_run(run_name=cfg.name + ("_gen" if generation else "")):
        mlflow.set_tags({"stage": cfg.name.split("_")[0], "retrieval_mode": cfg.retrieval.mode,
                         "with_generation": str(generation), "git_commit": git_commit(),
                         "dataset": dataset_path.name, "owner": "wiem"})
        mlflow.log_params({k: str(v) for k, v in flatten(cfg.model_dump()).items()})
        mlflow.log_param("n_questions", len(dataset))
        mlflow.log_artifact(str(config_path))

        if reindex:
            from app.ingestion.temp_ingest import ingest
            mlflow.log_metric("n_chunks", ingest(pdf_dir, cfg, embedder=embedder))

        if retriever is None:
            from app.retrieval.hybrid import get_retriever
            retriever = get_retriever(cfg, embedder=embedder)
        metrics, rows = evaluate_retrieval(retriever, dataset)

        gen_rows = []
        if generation:
            if pipeline is None:
                from app.pipeline import RAGPipeline
                pipeline = RAGPipeline(cfg, retriever=retriever)
            gen_metrics, gen_rows = evaluate_generation(pipeline, dataset)
            metrics.update(gen_metrics)

        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:  # per-question results for error analysis
            write_jsonl(rows, Path(tmp) / "retrieval_results.jsonl")
            mlflow.log_artifact(str(Path(tmp) / "retrieval_results.jsonl"))
            if gen_rows:
                write_jsonl(gen_rows, Path(tmp) / "generation_results.jsonl")
                mlflow.log_artifact(str(Path(tmp) / "generation_results.jsonl"))

        if register:  # inside the run: the model version is linked to these params and metrics
            from app.registry.register import register_pipeline
            version = register_pipeline(cfg, config_path, metrics)

    if register and promote:
        from app.registry.promote import promote as promote_version
        promote_version(version)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--pdf-dir", type=Path, default=Path("../data/raw"))
    p.add_argument("--reindex", action="store_true", help="rebuild the collection with this config's model")
    p.add_argument("--generation", action="store_true", help="also call the LLM (uses Groq quota)")
    p.add_argument("--register", action="store_true", help="log the pipeline as a new version of 'legalease-rag'")
    p.add_argument("--promote", action="store_true", help="after --register, apply the promotion rule")
    a = p.parse_args()
    run(load_config(a.config), a.config, a.dataset, a.pdf_dir, a.reindex, a.generation,
        register=a.register, promote=a.promote)
