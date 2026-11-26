"""E0 tests that run without downloading BGE-M3 or calling Groq (fakes)."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymupdf
import pytest

from app.core.experiment_config import ExperimentConfig, QdrantConfig
from app.core.schemas import Answer
from app.generation.generator import Generator, detect_query_lang, parse_answer
from app.ingestion.temp_ingest import chunk_text, detect_lang, ingest, slugify
from app.pipeline import RAGPipeline
from app.retrieval.dense import DenseRetriever
from app.retrieval.store import get_client


class FakeEmbedder:
    """Bag-of-words hashing: same words -> similar vectors. Enough to test the plumbing."""
    dim = 256

    def count_tokens(self, text):
        return len(text.split())

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
            out[i] /= np.linalg.norm(out[i]) or 1
        return out

    def embed_query(self, q):
        return self.embed([q])[0].tolist()


class FakeGroq:
    def __init__(self, payload):
        self.payload = payload
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        msg = SimpleNamespace(content=json.dumps(self.payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def make_pdf(path: Path, pages: list[str]):
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((50, 72), text, fontsize=10)
    doc.save(path)


@pytest.fixture
def setup(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    make_pdf(pdf_dir / "Loi n° 2018-52 RNE.pdf", [
        "Article 1. Le registre national des entreprises est un registre public.\n"
        "Il centralise les informations relatives aux entreprises et aux personnes morales.",
        "Article 5. Toute societe doit etre inscrite au registre dans un delai de dix jours.\n"
        "Le non respect de ce delai expose la societe a une amende administrative.",
    ])
    cfg = ExperimentConfig(qdrant=QdrantConfig(path=str(tmp_path / "qdrant"), collection="test_e0"))
    client = get_client(cfg.qdrant)
    emb = FakeEmbedder()
    # reuse the same local client (Qdrant local mode allows one client per path)
    import app.retrieval.store as store
    original = store.get_client
    store.get_client = lambda _cfg: client
    try:
        n = ingest(pdf_dir, cfg, embedder=emb)
    finally:
        store.get_client = original
    return cfg, emb, client, n


def test_ingestion_keeps_doc_id_and_pages(setup):
    cfg, emb, client, n = setup
    assert n == 2
    points, _ = client.scroll(cfg.qdrant.collection, limit=10, with_payload=True)
    pages = sorted(p.payload["page"] for p in points)
    assert pages == [1, 2]
    assert {p.payload["doc_id"] for p in points} == {"loi_n_2018_52_rne"}


def test_dense_retrieval_and_filters(setup):
    cfg, emb, client, _ = setup
    r = DenseRetriever(cfg, embedder=emb, client=client)
    hits = r.retrieve("delai inscription registre societe", k=2)
    assert hits[0].chunk.page == 2
    assert hits[0].score >= hits[1].score
    assert r.retrieve("registre", k=5, filters={"lang": "ar"}) == []


def test_pipeline_end_to_end_with_citations(setup):
    cfg, emb, client, _ = setup
    fake = FakeGroq({"status": "answered",
                     "answer": "La société doit s'inscrire dans un délai de dix jours [1].",
                     "citations": [{"source": 1, "quote": "dans un delai de dix jours"},
                                   {"source": 99, "quote": "invalid"}]})
    pipe = RAGPipeline(cfg, retriever=DenseRetriever(cfg, embedder=emb, client=client),
                       generator=Generator(cfg.llm, client=fake))
    ans = pipe.answer("Quel est le delai inscription registre pour une societe ?")
    assert isinstance(ans, Answer) and ans.status == "answered" and ans.lang == "fr"
    assert len(ans.citations) == 1 and ans.citations[0].page == 2  # source 99 dropped
    assert fake.last_kwargs["response_format"] == {"type": "json_object"}


def test_abstention_paths():
    assert parse_answer('{"status": "abstained"}', [], "fr").status == "abstained"
    assert parse_answer("not json", [], "ar").status == "abstained"
    gen = Generator(ExperimentConfig().llm, client=FakeGroq({}))
    assert gen.generate("question ?", []).status == "abstained"


def test_helpers():
    assert slugify("Loi n° 2018-52 du 29 octobre 2018.pdf") == "loi_n_2018_52_du_29_octobre_2018"
    assert detect_lang("الفصل الأول يحدث سجل وطني للمؤسسات") == "ar"
    assert detect_query_lang("كيفاش نسجل شركة ؟") == "ar"
    assert detect_query_lang("Quel est le capital minimum pour une SARL ?") == "fr"
    chunks = chunk_text("Phrase un. " * 300, max_tokens=100, overlap=10, count=lambda s: len(s.split()))
    assert len(chunks) > 1 and all(len(c.split()) <= 110 for c in chunks)


def test_run_experiment_logs_to_mlflow(setup, tmp_path, monkeypatch):
    import mlflow
    from app.evaluation import run_experiment as rx

    cfg, emb, client, _ = setup
    ds = tmp_path / "seed.jsonl"
    ds.write_text("\n".join(json.dumps(x) for x in [
        {"id": "q1", "question": "delai inscription registre societe", "doc_id": "loi_n_2018_52_rne",
         "pages": [2], "lang": "fr", "category": "factual"},
        {"id": "q2", "question": "salaire des pilotes", "doc_id": None, "pages": [], "lang": "fr",
         "category": "unanswerable"},
    ]), encoding="utf-8")
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("name: E0_test\n", encoding="utf-8")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    retriever = DenseRetriever(cfg, embedder=emb, client=client)
    fake = FakeGroq({"status": "abstained"})
    pipe = RAGPipeline(cfg, retriever=retriever, generator=Generator(cfg.llm, client=fake))
    m = rx.run(cfg, cfg_file, ds, None, reindex=False, generation=True, retriever=retriever, pipeline=pipe)

    assert m["recall_at_5"] == 1.0 and m["n_retrieval_questions"] == 1
    assert m["abstention_correct_rate"] == 1.0 and m["false_abstention_rate"] == 1.0
    runs = mlflow.search_runs(experiment_names=[rx.EXPERIMENT])
    assert len(runs) == 1 and runs.iloc[0]["metrics.recall_at_5"] == 1.0
