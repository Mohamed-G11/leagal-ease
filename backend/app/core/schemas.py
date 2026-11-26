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
