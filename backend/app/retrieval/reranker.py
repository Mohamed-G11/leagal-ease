"""E2 — Cross-encoder reranker (BGE reranker v2 m3, multilingual).

A bi-encoder (BGE-M3) encodes the question and the chunk separately, then compares vectors.
A cross-encoder reads the question AND the chunk together, so it judges relevance much better,
but it is too slow to run on the whole corpus: we only give it the top candidates.
"""
import math

from app.core.experiment_config import RerankerConfig
from app.core.schemas import ScoredChunk

_CACHE: dict[str, object] = {}


class Reranker:
    def __init__(self, cfg: RerankerConfig, model=None):
        self.cfg = cfg
        if model is None:
            if cfg.model not in _CACHE:
                from sentence_transformers import CrossEncoder

                _CACHE[cfg.model] = CrossEncoder(cfg.model, max_length=1024)
            model = _CACHE[cfg.model]
        self.model = model

    def score(self, queries: list[str], chunks: list[ScoredChunk]) -> list[float]:
        """Relevance in [0, 1]; with several query variants, keep the best one per chunk."""
        if not chunks:
            return []
        pairs = [(q, sc.chunk.text) for q in queries for sc in chunks]
        raw = self.model.predict(pairs, batch_size=self.cfg.batch_size, show_progress_bar=False)
        raw = [float(x) for x in raw]
        # sentence-transformers usually applies a sigmoid already; if we got logits, apply it here
        probs = raw if all(0.0 <= x <= 1.0 for x in raw) else [1 / (1 + math.exp(-x)) for x in raw]
        n = len(chunks)
        return [max(probs[i + j * n] for j in range(len(queries))) for i in range(n)]

    def rerank(self, queries: list[str], chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        scores = self.score(queries, chunks)
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [ScoredChunk(chunk=sc.chunk, score=s) for sc, s in ranked[:top_k]]
