"""E1 tests: BM25 tokenizer/encoder, sparse and hybrid retrieval (offline, fake embedder)."""
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever, SparseRetriever, get_retriever
from app.retrieval.sparse import BM25Encoder, token_id, tokenize
from tests.test_e0 import setup  # noqa: F401  (reuse the ingestion fixture)


def test_tokenize_normalizes_french_and_arabic():
    assert tokenize("Les Sociétés") == ["societe"]
    # hamza / ta marbuta / article prefix normalized -> same token
    assert tokenize("المؤسسة") == tokenize("مؤسسه") == tokenize("الموسسة")
    assert tokenize("في من الى") == []


def test_bm25_vectors_are_stable_and_saturated():
    enc = BM25Encoder()
    v = enc.encode_document("registre registre registre entreprise")
    w = dict(zip(v.indices, v.values, strict=False))
    assert w[token_id("registre")] > w[token_id("entreprise")]
    assert w[token_id("registre")] < 3 * w[token_id("entreprise")]  # tf saturation
    q = enc.encode_query("registre registre entreprise")
    assert q.values == [1.0, 1.0]


def test_sparse_and_hybrid_retrieval(setup):  # noqa: F811
    cfg, emb, client, _ = setup
    sparse = SparseRetriever(cfg, client=client).retrieve("amende administrative delai", k=2)
    assert sparse and sparse[0].chunk.page == 2
    hybrid = HybridRetriever(cfg, embedder=emb, client=client).retrieve("amende administrative delai", k=2)
    assert hybrid[0].chunk.page == 2 and len(hybrid) == 2
    assert HybridRetriever(cfg, embedder=emb, client=client).retrieve("registre", k=2, filters={"lang": "ar"}) == []


def test_factory_picks_mode(setup):  # noqa: F811
    cfg, emb, client, _ = setup
    for mode, cls in [("dense", DenseRetriever), ("sparse", SparseRetriever), ("hybrid", HybridRetriever)]:
        c = cfg.model_copy(deep=True)
        c.retrieval.mode = mode
        assert isinstance(get_retriever(c, embedder=emb, client=client), cls)
