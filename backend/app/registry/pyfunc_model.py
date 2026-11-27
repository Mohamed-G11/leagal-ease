"""The whole RAG pipeline packaged as an MLflow model (W4.2).

A RAG system does not train weights, so what we version in the Model Registry is the
pipeline itself: the code (`app/`), the experiment config (models, top-k, reranker...)
and, on the registering run, its evaluation metrics.

The Qdrant index is NOT inside the model: it is data, rebuilt by the ingestion flow
and reached through QDRANT_URL (or the local path written in the config).

    model = mlflow.pyfunc.load_model("models:/legalease-rag@champion")
    model.predict({"question": ["Quel est le délai d'inscription au RNE ?"]})
"""
from __future__ import annotations

from typing import Any

import mlflow.pyfunc


def _to_questions(model_input: Any) -> list[str]:
    """Accept a string, a list of strings, a dict or a pandas DataFrame with a 'question' column."""
    if isinstance(model_input, str):
        return [model_input]
    if isinstance(model_input, dict):
        q = model_input.get("question")
        return [q] if isinstance(q, str) else list(q or [])
    if hasattr(model_input, "columns"):  # pandas DataFrame
        return [str(q) for q in model_input["question"].tolist()]
    return [str(q) for q in model_input]


class LegalEaseRAGModel(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline=None):
        self.pipeline = pipeline  # injected in tests; built from the config otherwise

    def load_context(self, context) -> None:
        if self.pipeline is not None:
            return
        from app.core.experiment_config import load_config
        from app.pipeline import RAGPipeline

        self.pipeline = RAGPipeline(load_config(context.artifacts["config"]))

    def predict(self, context, model_input, params=None) -> list[dict]:
        params = params or {}
        filters, history = params.get("filters"), params.get("history")
        return [self.pipeline.answer(q, filters=filters, history=history).model_dump()
                for q in _to_questions(model_input)]
