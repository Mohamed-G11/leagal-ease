"""E1 retrievers: BM25 only (ablation) and hybrid dense + BM25 fused with RRF.

Same interface as DenseRetriever: retrieve(query, k, filters) -> list[ScoredChunk].
Fusion is done by Qdrant (prefetch both branches, then Reciprocal Rank Fusion),
so the scores returned are RRF scores, not cosine similarities.
"""
from qdrant_client import models

from app.core.experiment_config import ExperimentConfig
from app.core.schemas import Chunk, ScoredChunk
from app.retrieval.sparse import BM25Encoder
from app.retrieval.store import DENSE, SPARSE, build_filter, get_client


def _to_scored(points) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=Chunk(**p.payload), score=p.score) for p in points]


class SparseRetriever:
    """BM25 only — used to measure what the lexical branch brings on its own."""

    def __init__(self, cfg: ExperimentConfig, client=None):
        self.cfg = cfg
        self.client = client or get_client(cfg.qdrant)
        self.bm25 = BM25Encoder(**cfg.retrieval.bm25.model_dump())

    def retrieve(self, query: str, k: int | None = None, filters: dict | None = None) -> list[ScoredChunk]:
        result = self.client.query_points(
            collection_name=self.cfg.qdrant.collection,
            query=self.bm25.encode_query(query),
            using=SPARSE,
            limit=k or self.cfg.retrieval.top_k,
            query_filter=build_filter(filters),
            with_payload=True,
        )
        return _to_scored(result.points)


class HybridRetriever:
    def __init__(self, cfg: ExperimentConfig, embedder=None, client=None):
        from app.retrieval.embedder import get_embedder

        self.cfg = cfg
        self.embedder = embedder or get_embedder(cfg.embedding)
        self.client = client or get_client(cfg.qdrant)
        self.bm25 = BM25Encoder(**cfg.retrieval.bm25.model_dump())

    def retrieve(self, query: str, k: int | None = None, filters: dict | None = None) -> list[ScoredChunk]:
        k = k or self.cfg.retrieval.top_k
        prefetch_k = max(self.cfg.retrieval.prefetch_k, k)
        flt = build_filter(filters)
        result = self.client.query_points(
            collection_name=self.cfg.qdrant.collection,
            prefetch=[
                models.Prefetch(query=self.embedder.embed_query(query), using=DENSE,
                                limit=prefetch_k, filter=flt),
                models.Prefetch(query=self.bm25.encode_query(query), using=SPARSE,
                                limit=prefetch_k, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )
        return _to_scored(result.points)


def get_retriever(cfg: ExperimentConfig, embedder=None, client=None):
    """Pick the retriever from config: retrieval.mode = dense | sparse | hybrid."""
    mode = cfg.retrieval.mode
    if mode == "sparse":
        return SparseRetriever(cfg, client=client)
    if mode == "hybrid":
        return HybridRetriever(cfg, embedder=embedder, client=client)
    from app.retrieval.dense import DenseRetriever

    return DenseRetriever(cfg, embedder=embedder, client=client)
