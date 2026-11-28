"""Fixes reported by Mohamed: malformed LLM output, confidence, safe re-ingestion."""
import json
from pathlib import Path

import pytest

from app.core.schemas import Answer, Chunk, Citation, ScoredChunk
from app.generation.confidence import compute_confidence, quote_in_chunk
from app.generation.generator import parse_answer
from app.ingestion.temp_ingest import ingest
from tests.test_e0 import setup  # noqa: F401

CHUNKS = [ScoredChunk(chunk=Chunk(chunk_id="d_p2_0", doc_id="d", doc_title="Loi", page=2, lang="fr",
                                  text="Toute société doit être inscrite dans un délai de dix jours."), score=0.8)]


@pytest.mark.parametrize("raw", [
    "[1, 2, 3]",                                                  # JSON list
    json.dumps({"status": "answered", "answer": ["a", "b"]}),     # answer not a string
    json.dumps({"status": "answered", "answer": 42}),
    "null", "not json", "", None,
])
def test_malformed_llm_output_becomes_abstention(raw):
    ans = parse_answer(raw, CHUNKS, "fr")
    assert isinstance(ans, Answer) and ans.status == "abstained"


def test_bad_citation_fields_are_tolerated():
    raw = json.dumps({"status": "answered", "answer": "Dix jours [1].",
                      "citations": [{"source": 1, "quote": {"x": 1}}, "junk", {"source": "abc"}, 5]})
    ans = parse_answer(raw, CHUNKS, "fr")
    assert ans.status == "answered" and len(ans.citations) == 1
    assert ans.citations[0].quote.startswith("Toute société")  # non-string quote -> chunk excerpt
    raw = json.dumps({"status": "answered", "answer": "Dix jours.", "citations": {"source": 1}})
    assert parse_answer(raw, CHUNKS, "fr").citations == []  # citations not a list


def _answer(quote):
    return Answer(answer="Dix jours [1].", status="answered", lang="fr",
                  citations=[Citation(chunk_id="d_p2_0", doc_title="Loi", page=2, quote=quote)])


def test_confidence_values():
    assert quote_in_chunk("délai de dix jours", CHUNKS[0].chunk.text)
    assert compute_confidence(_answer("dans un délai de dix jours"), CHUNKS, None) == 1.0  # E0/E1
    assert compute_confidence(_answer("dans un délai de dix jours"), CHUNKS, 0.6) == 0.8  # E2: mean
    assert compute_confidence(_answer("le capital minimum est de 1000 dinars"), CHUNKS, 0.9) == 0.2  # cap
    assert compute_confidence(Answer(answer="x", status="abstained", lang="fr"), CHUNKS, 0.9) is None


def test_reingest_upserts_and_recreate_is_explicit(setup, tmp_path, capsys):  # noqa: F811
    import app.retrieval.store as store
    from tests.test_e0 import FakeEmbedder, make_pdf

    cfg, emb, client, n = setup
    pdf_dir = tmp_path / "more"
    pdf_dir.mkdir()
    make_pdf(pdf_dir / "Autre loi.pdf", ["Article 1. Un autre texte de loi suffisamment long pour un chunk."])
    original = store.get_client
    store.get_client = lambda _cfg: client
    try:
        ingest(pdf_dir, cfg, embedder=FakeEmbedder())  # default: keep existing points
        assert client.count(cfg.qdrant.collection).count == n + 1
        assert "upserting" in capsys.readouterr().out
        ingest(pdf_dir, cfg, embedder=FakeEmbedder(), recreate=True)  # explicit rebuild
        assert client.count(cfg.qdrant.collection).count == 1
    finally:
        store.get_client = original


def test_run_experiment_docstring_points_to_dagshub():
    doc = (Path(__file__).parents[1] / "app/evaluation/run_experiment.py").read_text(encoding="utf-8")
    assert "dagshub.com" in doc and "Mohamed's docker-compose" not in doc


def test_recreate_empties_collection_even_if_old_points_come_back(tmp_path):
    """Simulates the Windows local-mode behaviour: delete_collection leaves the points behind."""
    from qdrant_client import QdrantClient, models

    from app.retrieval.store import DENSE, ensure_collection

    client = QdrantClient(path=str(tmp_path / "q"))
    ensure_collection(client, "c", 4)
    client.upsert("c", [models.PointStruct(id=i, vector={DENSE: [1.0, 0, 0, 0]}) for i in range(3)])
    client.delete_collection = lambda name: None  # the drop "silently fails", as on Windows
    client.create_collection = lambda **kw: None
    assert ensure_collection(client, "c", 4, recreate=True) == "recreated"
    assert client.count("c").count == 0
    client.close()
