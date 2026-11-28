"""Shared data contracts for the whole team (W1).

Every retriever (E0 -> E3) returns list[ScoredChunk];
every generator returns an Answer. Do not change these without a team PR.
"""
from typing import Literal

from pydantic import BaseModel, Field

Lang = Literal["fr", "ar", "en", "darija"]


class Chunk(BaseModel):
    chunk_id: str  # f"{doc_id}_p{page}_{i}"
    doc_id: str  # stable per source file
    doc_title: str
    page: int  # 1-based page number
    section: str | None = None  # e.g. "Article 12" (filled by Hend's parser)
    lang: Lang
    text: str
    metadata: dict = Field(default_factory=dict)  # law number, date, category, in_force


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class Citation(BaseModel):
    chunk_id: str
    doc_title: str
    page: int
    quote: str


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    status: Literal["answered", "abstained"]
    confidence: float | None = None
    lang: Lang
    # best reranker probability of the chunks sent to the LLM (E2+ only, None otherwise)
    retrieval_score: float | None = None


class ChatTurn(BaseModel):
    """One previous message of the conversation, passed by the API (oldest first)."""
    role: Literal["user", "assistant"]
    content: str


class RetrievalResult(BaseModel):
    """What a retrieval call produced for ONE request. Returned, never stored on the retriever,
    so concurrent requests sharing a pipeline cannot overwrite each other's language."""
    chunks: list[ScoredChunk]
    lang: Lang | None = None  # detected by the query transformer (E2), e.g. "darija"
    queries: list[str] = Field(default_factory=list)
    top_score: float | None = None  # best reranker probability (only meaningful with a reranker)


def recent_history(history: list[ChatTurn] | list[dict] | None, max_messages: int,
                   max_chars: int = 600) -> list[ChatTurn]:
    """Keep the last messages only, each cut to max_chars, so the prompt size stays bounded."""
    if not history or max_messages <= 0:
        return []
    turns = [h if isinstance(h, ChatTurn) else ChatTurn(**h) for h in history][-max_messages:]
    return [ChatTurn(role=t.role, content=t.content[:max_chars]) for t in turns if t.content.strip()]
