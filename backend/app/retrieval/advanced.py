"""E2 — Advanced retriever: query transformation -> multi-query hybrid search -> rerank -> compression.

Wraps any base retriever (dense / sparse / hybrid) and keeps the same interface:
    retrieve(query, k, filters) -> list[ScoredChunk]
Each brick can be switched off in the config, which gives the E2 ablations for free.
"""
from app.core.experiment_config import ExperimentConfig
from app.core.schemas import ChatTurn, RetrievalResult, ScoredChunk

RRF_K = 60


def rrf_merge(result_lists: list[list[ScoredChunk]]) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion across the result lists of the different query variants."""
    scores: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    for results in result_lists:
        for rank, sc in enumerate(results, 1):
            cid = sc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            best.setdefault(cid, sc)
    order = sorted(scores, key=scores.get, reverse=True)
    return [ScoredChunk(chunk=best[c].chunk, score=scores[c]) for c in order]


class AdvancedRetriever:
    def __init__(self, cfg: ExperimentConfig, base, transformer=None, reranker=None):
        self.cfg = cfg
        self.base = base
        self.transformer = transformer
        self.reranker = reranker
        # No per-request state on self: one instance serves concurrent API requests.

    def retrieve(self, query: str, k: int | None = None, filters: dict | None = None) -> list[ScoredChunk]:
        """Team interface (used by the evaluation): chunks only."""
        return self.retrieve_with_meta(query, k=k, filters=filters).chunks

    def retrieve_with_meta(self, query: str, k: int | None = None, filters: dict | None = None,
                           history: list[ChatTurn] | None = None) -> RetrievalResult:
        """Chunks + detected language + queries used, all local to this call (thread-safe)."""
        # 1. Query transformation (the history turns a follow-up into a standalone question)
        queries, lang = [query], None
        if self.transformer is not None:
            queries, lang = self.transformer.transform(query, history=history)
        chunks = self.retrieve_queries(queries, k=k, filters=filters)
        top = max((c.score for c in chunks), default=None) if self.reranker is not None else None
        return RetrievalResult(chunks=chunks, lang=lang, queries=queries, top_score=top)

    def retrieve_queries(self, queries: list[str], k: int | None = None,
                         filters: dict | None = None) -> list[ScoredChunk]:
        """Steps 2-4 for queries already rewritten (also used by the E3 agent)."""
        k = k or self.cfg.retrieval.top_k

        # 2. Wider candidate pool from every variant, fused with RRF
        pool_k = self.cfg.reranker.candidates_k if self.reranker is not None else k
        lists = [self.base.retrieve(q, k=pool_k, filters=filters) for q in queries]
        candidates = rrf_merge(lists) if len(lists) > 1 else lists[0]

        # 3. Rerank
        if self.reranker is None:
            return candidates[:k]
        ranked = self.reranker.rerank(queries, candidates[:pool_k], top_k=k)

        # 4. Compression: drop what the reranker considers irrelevant
        comp = self.cfg.compression
        if comp.enabled:
            kept = [sc for sc in ranked if sc.score >= comp.min_score]
            ranked = kept if len(kept) >= comp.keep_at_least else ranked[: comp.keep_at_least]
        return ranked

    @property
    def client(self):
        """The Qdrant client of the wrapped retriever (used by the API health check)."""
        return self.base.client
