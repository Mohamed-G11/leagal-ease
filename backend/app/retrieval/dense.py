"""W3 — E0 dense retriever (BGE-M3 + Qdrant).

Implements the team interface:
    retrieve(query, k, filters) -> list[ScoredChunk]
"""
from app.core.experiment_config import ExperimentConfig
from app.core.schemas import Chunk, ScoredChunk
from app.retrieval.store import DENSE, build_filter, get_client


class DenseRetriever:
    def __init__(self, cfg: ExperimentConfig, embedder=None, client=None):
        from app.retrieval.embedder import get_embedder

        self.cfg = cfg
        self.embedder = embedder or get_embedder(cfg.embedding)
        self.client = client or get_client(cfg.qdrant)

    def retrieve(self, query: str, k: int | None = None, filters: dict | None = None) -> list[ScoredChunk]:
        result = self.client.query_points(
            collection_name=self.cfg.qdrant.collection,
            query=self.embedder.embed_query(query),
            using=DENSE,
            limit=k or self.cfg.retrieval.top_k,
            query_filter=build_filter(filters),
            with_payload=True,
        )
        return [ScoredChunk(chunk=Chunk(**p.payload), score=p.score) for p in result.points]
