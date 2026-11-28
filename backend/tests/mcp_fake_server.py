"""A stdio MCP server with a fake pipeline and index: no model, Groq, Qdrant, PostgreSQL or network.

Used by the client smoke test. Not collected by pytest. Its first search-type call answers "loading",
like a real server that has just started.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def payload(doc_id, page, i, lang="fr"):
    return {"chunk_id": f"{doc_id}_p{page}_{i}", "doc_id": doc_id, "doc_title": f"Titre {doc_id}", "page": page,
            "lang": lang, "text": f"texte {doc_id} {page} {i}"}


class FakeIndex:
    """Just enough of the Qdrant client: scroll with a filter, in pages of two."""

    def __init__(self, payloads):
        self.payloads = payloads

    def scroll(self, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        wanted = {c.key: c.match.value for c in scroll_filter.must} if scroll_filter else {}
        rows = [p for p in self.payloads if all(p[key] == value for key, value in wanted.items())]
        start = offset or 0
        points = [SimpleNamespace(payload=p) for p in rows[start : start + 2]]
        return points, (start + 2 if start + 2 < len(rows) else None)


def main() -> None:
    # before anything from app is imported: no real settings, whatever .env says
    os.environ.update(DATABASE_URL="sqlite://", SECRET_KEY="test", QDRANT_URL="http://fake.test:6333",
                      MLFLOW_TRACKING_URI="", LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from app import mcp_server
    from app.core.errors import APIError
    from app.core.experiment_config import ExperimentConfig
    from app.core.schemas import Answer, Chunk, Citation, ScoredChunk
    from app.services.rag import retrieved_chunks

    chunk = Chunk(**payload("loi", 2, 0))

    def answer(question):
        retrieved_chunks.set((ScoredChunk(chunk=chunk, score=0.5),))
        return Answer(answer="Réponse d'essai [1].", status="answered", lang="fr", confidence=1.0,
                      citations=[Citation(chunk_id=chunk.chunk_id, doc_title=chunk.doc_title, page=2, quote="texte")])

    pipeline = SimpleNamespace(
        cfg=ExperimentConfig(retrieval={"mode": "hybrid"}), answer=answer,
        retriever=SimpleNamespace(
            retrieve=lambda query, k=None, filters=None: [ScoredChunk(chunk=chunk, score=0.5)],
            client=FakeIndex([payload("loi", 2, 0), payload("loi", 3, 0), payload("code", 1, 0, "ar")]),
        ),
    )
    calls = []

    def ready_pipeline():
        calls.append(1)
        if len(calls) == 1:
            raise APIError(503, "rag_loading", "loading")
        return pipeline

    mcp_server.rag_service.start = lambda: None
    mcp_server.rag_service.ready_pipeline = ready_pipeline
    mcp_server.record = lambda *args: None  # no request log
    mcp_server.main([])


if __name__ == "__main__":
    main()
