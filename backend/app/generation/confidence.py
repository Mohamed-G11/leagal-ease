"""Confidence indicator shown in the UI (0 to 1). It is an indicator, not a probability.

    grounding = share of citations whose quote really appears in the cited chunk
    confidence = mean(grounding, best reranker probability)   with a reranker (E2+)
               = grounding                                     without (E0/E1: scores not comparable)
An answer without any verifiable citation gets at most 0.2; an abstention gets no value.
"""
import re
import unicodedata

from app.core.schemas import Answer, ScoredChunk


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\W+", " ", text).strip()


def quote_in_chunk(quote: str, chunk_text: str, min_overlap: float = 0.6) -> bool:
    """Exact (normalized) match, or enough of the quote's words present in the chunk."""
    q, c = _norm(quote), _norm(chunk_text)
    if not q:
        return False
    if q in c:
        return True
    words = [w for w in q.split() if len(w) > 2]
    return bool(words) and sum(w in c.split() for w in words) / len(words) >= min_overlap


def grounding_ratio(answer: Answer, chunks: list[ScoredChunk]) -> float:
    by_id = {sc.chunk.chunk_id: sc.chunk.text for sc in chunks}
    if not answer.citations:
        return 0.0
    ok = sum(quote_in_chunk(c.quote, by_id.get(c.chunk_id, "")) for c in answer.citations)
    return ok / len(answer.citations)


def compute_confidence(answer: Answer, chunks: list[ScoredChunk], retrieval_score: float | None) -> float | None:
    if answer.status != "answered":
        return None
    grounding = grounding_ratio(answer, chunks)
    value = grounding if retrieval_score is None else (grounding + retrieval_score) / 2
    if grounding == 0:
        value = min(value, 0.2)
    return round(max(0.0, min(1.0, value)), 2)
