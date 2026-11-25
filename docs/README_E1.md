# E1 — Hybrid RAG (Wiem, Sprint 2)

BGE-M3 dense + BM25 sparse in the same Qdrant collection, fused with Reciprocal Rank Fusion.

| File | Role |
|---|---|
| `backend/app/retrieval/sparse.py` | Tokenizer (FR/EN/AR normalization, light stemming, stopwords incl. darija) + BM25 encoder |
| `backend/app/retrieval/hybrid.py` | `SparseRetriever` (BM25 only), `HybridRetriever` (dense + BM25 → RRF), `get_retriever(cfg)` |
| `backend/app/retrieval/store.py` | Collection now has `dense` + `bm25` (IDF computed by Qdrant) |
| `experiments/E1_hybrid_rag/` | `config.yaml` (hybrid) + `ablation_dense.yaml` + `ablation_sparse.yaml` + `run_e1.bat` |
| `backend/tests/test_e1.py` | Offline tests |

`retrieval.mode` in the config selects `dense | sparse | hybrid`; the pipeline and `run_experiment`
use `get_retriever(cfg)`, so E0 configs keep working unchanged.

Run (from `backend/`): `..\experiments\E1_hybrid_rag\run_e1.bat` → 3 MLflow runs on the same index.

Note: collections built before E1 have no `bm25` vector — rebuild with `--reindex`.
