"""Experiment config loader (W5). Secrets come from the environment only."""
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    batch_size: int = 16
    device: str | None = None  # "cuda", "cpu" or None (auto)
    query_prefix: str = ""  # e.g. "query: " for multilingual-e5
    passage_prefix: str = ""  # e.g. "passage: " for multilingual-e5


class QdrantConfig(BaseModel):
    url: str | None = None  # e.g. "http://localhost:6333" (docker)
    path: str | None = "./qdrant_local"  # local mode, no docker needed
    collection: str = "legalease_e0"


class ChunkingConfig(BaseModel):
    max_tokens: int = 500
    overlap_tokens: int = 50


class BM25Config(BaseModel):
    k1: float = 1.2
    b: float = 0.75
    avg_len: float = 256.0


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "sparse", "hybrid"] = "dense"  # E0 = dense, E1 = hybrid
    top_k: int = 5
    prefetch_k: int = 20  # candidates taken from each branch before RRF fusion
    bm25: BM25Config = BM25Config()


class LLMConfig(BaseModel):
    provider: str = "groq"
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.0
    max_tokens: int = 2048


class ExperimentConfig(BaseModel):
    name: str = "E0_naive_rag"
    embedding: EmbeddingConfig = EmbeddingConfig()
    qdrant: QdrantConfig = QdrantConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()


def load_config(path: str | Path) -> ExperimentConfig:
    """QDRANT_URL in the environment (docker-compose) overrides the local-mode path."""
    with open(path, encoding="utf-8") as f:
        cfg = ExperimentConfig(**(yaml.safe_load(f) or {}))
    if os.getenv("QDRANT_URL") and not cfg.qdrant.url:
        cfg.qdrant.url = os.getenv("QDRANT_URL")
    return cfg


def get_groq_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set (put it in .env, never in code).")
    return key
