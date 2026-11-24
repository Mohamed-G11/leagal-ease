"""E0 pipeline: retrieve -> generate. This is what Mohamed's /chat calls (W6).

Run:
    python -m app.pipeline "Quel est le rôle du RNE ?" \
        --config ../experiments/E0_naive_rag/config.yaml
"""
import argparse
from pathlib import Path

from app.core.experiment_config import ExperimentConfig, load_config
from app.core.schemas import Answer
from app.generation.generator import Generator
from app.retrieval.dense import DenseRetriever


class RAGPipeline:
    def __init__(self, cfg: ExperimentConfig, retriever=None, generator=None):
        self.cfg = cfg
        self.retriever = retriever or DenseRetriever(cfg)
        self.generator = generator or Generator(cfg.llm)

    def answer(self, question: str, filters: dict | None = None) -> Answer:
        chunks = self.retriever.retrieve(question, k=self.cfg.retrieval.top_k, filters=filters)
        return self.generator.generate(question, chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--config", type=Path, default=Path("../experiments/E0_naive_rag/config.yaml"))
    args = parser.parse_args()

    result = RAGPipeline(load_config(args.config)).answer(args.question)
    print(result.model_dump_json(indent=2))
