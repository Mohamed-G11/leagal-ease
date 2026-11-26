"""E2 tests: query transformation, reranking, compression, multi-query fusion (offline fakes)."""
from types import SimpleNamespace

from app.core.experiment_config import QueryTransformConfig, RerankerConfig
from app.core.schemas import Chunk, ScoredChunk
from app.generation.generator import Generator
from app.pipeline import RAGPipeline
from app.retrieval.advanced import AdvancedRetriever, rrf_merge
from app.retrieval.hybrid import HybridRetriever, get_retriever
from app.retrieval.query_transform import QueryTransformer
from app.retrieval.reranker import Reranker
from tests.test_e0 import FakeGroq, setup  # noqa: F401


class FakeCrossEncoder:
    """Relevance = share of query words found in the chunk (returns logits to test the sigmoid)."""

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        out = []
        for q, text in pairs:
            words = {w for w in q.lower().split() if len(w) > 3}
            hit = sum(w in text.lower() for w in words) / (len(words) or 1)
            out.append(8 * hit - 4)  # logit in [-4, 4]
        return out


def sc(cid, text, page=1, score=0.0):
    return ScoredChunk(chunk=Chunk(chunk_id=cid, doc_id="d", doc_title="D", page=page, lang="fr", text=text),
                       score=score)


def test_query_transformer_keeps_original_and_detects_darija():
    fake = FakeGroq({"lang": "darija", "fr": "Quel est le délai de réponse pour le label startup ?",
                     "ar": "ما هو أجل الرد على مطلب علامة المؤسسة الناشئة؟"})
    queries, lang = QueryTransformer(QueryTransformConfig(enabled=True), client=fake).transform("قداش يستنى؟")
    assert queries[0] == "قداش يستنى؟" and len(queries) == 3 and lang == "darija"


def test_query_transformer_never_fails():
    class Broken:
        chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **k: (_ for _ in ()).throw(RuntimeError())))
    assert QueryTransformer(QueryTransformConfig(), client=Broken()).transform("q") == (["q"], None)


def test_rrf_merge_rewards_agreement():
    a, b, c = sc("a", "x"), sc("b", "y"), sc("c", "z")
    merged = rrf_merge([[a, b], [b, c]])
    assert [m.chunk.chunk_id for m in merged][0] == "b"


def test_reranker_sorts_and_uses_best_query():
    r = Reranker(RerankerConfig(), model=FakeCrossEncoder())
    chunks = [sc("a", "rien a voir"), sc("b", "delai inscription registre entreprises")]
    out = r.rerank(["question floue", "delai inscription registre"], chunks, top_k=2)
    assert out[0].chunk.chunk_id == "b" and 0 <= out[1].score < out[0].score <= 1


def test_advanced_retriever_end_to_end(setup):  # noqa: F811
    cfg, emb, client, _ = setup
    cfg = cfg.model_copy(deep=True)
    cfg.retrieval.mode = "hybrid"
    cfg.query_transform.enabled = cfg.reranker.enabled = cfg.compression.enabled = True
    cfg.compression.min_score = 0.5
    fake_llm = FakeGroq({"lang": "darija", "fr": "delai inscription registre societe amende", "ar": ""})
    retr = get_retriever(cfg, embedder=emb, client=client,
                         transformer=QueryTransformer(cfg.query_transform, client=fake_llm),
                         reranker=Reranker(cfg.reranker, model=FakeCrossEncoder()))
    assert isinstance(retr, AdvancedRetriever) and isinstance(retr.base, HybridRetriever)
    hits = retr.retrieve("9adech 3andi wa9t bech nsajel ?", k=2)
    assert hits[0].chunk.page == 2  # found thanks to the French rewrite
    assert len(hits) == 1  # page 1 dropped by compression
    assert retr.last_lang == "darija" and len(retr.last_queries) == 2

    gen = FakeGroq({"status": "answered", "answer": "Dix jours [1].",
                    "citations": [{"source": 1, "quote": "dix jours"}]})
    ans = RAGPipeline(cfg, retriever=retr, generator=Generator(cfg.llm, client=gen)).answer("9adech ?")
    assert ans.lang == "darija" and "Tunisian Darija" in gen.last_kwargs["messages"][0]["content"]


def test_disabled_bricks_return_base_retriever(setup):  # noqa: F811
    cfg, emb, client, _ = setup
    assert not isinstance(get_retriever(cfg, embedder=emb, client=client), AdvancedRetriever)


def test_reranker_best_query_mode_uses_french_rewrite_only():
    calls = []

    class Spy(FakeCrossEncoder):
        def predict(self, pairs, **kw):
            calls.extend(q for q, _ in pairs)
            return super().predict(pairs, **kw)

    r = Reranker(RerankerConfig(query_mode="best"), model=Spy())
    r.rerank(["قداش؟", "délai de réponse label", "أجل الرد"], [sc("a", "délai de réponse"), sc("b", "x")], top_k=2)
    assert set(calls) == {"délai de réponse label"} and len(calls) == 2
