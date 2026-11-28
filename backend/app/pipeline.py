"""RAG pipeline: retrieve -> generate.

E0 dense / E1 hybrid / E2 advanced retrieval, chosen by config. This is what Mohamed's /chat calls (W6).

Run:
    python -m app.pipeline "Quel est le rôle du RNE ?" \
        --config ../experiments/E0_naive_rag/config.yaml
"""
import argparse
from pathlib import Path

from app.core.experiment_config import ExperimentConfig, load_config
from app.core.schemas import Answer, ChatTurn, RetrievalResult, recent_history
from app.generation.confidence import compute_confidence
from app.generation.generator import Generator
from app.retrieval.hybrid import get_retriever


class RAGPipeline:
    def __init__(self, cfg: ExperimentConfig, retriever=None, generator=None):
        self.cfg = cfg
        self.retriever = retriever or get_retriever(cfg)
        self.generator = generator or Generator(cfg.llm)

    def answer(self, question: str, filters: dict | None = None,
               history: list[ChatTurn] | list[dict] | None = None) -> Answer:
        """Thread-safe: everything about this request stays in local variables.

        history: previous messages of the conversation, oldest first (the API loads them from the DB).
        """
        turns = recent_history(history, self.cfg.llm.max_history_messages)
        result = self._retrieve(question, filters, turns)
        # result.lang comes from the E2 query transformer (detects Darija); None for E0/E1
        answer = self.generator.generate(question, result.chunks, lang=result.lang, history=turns)
        return answer.model_copy(update={
            "retrieval_score": result.top_score,
            "confidence": compute_confidence(answer, result.chunks, result.top_score),
        })

    def _retrieve(self, question: str, filters: dict | None, turns: list[ChatTurn]) -> RetrievalResult:
        k = self.cfg.retrieval.top_k
        if hasattr(self.retriever, "retrieve_with_meta"):  # E2 / E3
            return self.retriever.retrieve_with_meta(question, k=k, filters=filters, history=turns)
        # E0 / E1: no rewriting, so a follow-up question is searched as typed
        return RetrievalResult(chunks=self.retriever.retrieve(question, k=k, filters=filters))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--config", type=Path, default=Path("../experiments/E0_naive_rag/config.yaml"))
    args = parser.parse_args()

    result = RAGPipeline(load_config(args.config)).answer(args.question)
    print(result.model_dump_json(indent=2))
