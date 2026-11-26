# E2 — Advanced RAG (Wiem, Sprint 3)

Question → **query transformation** → **multi-query hybrid search** (RRF) → **cross-encoder rerank** → **compression** → LLM.

| File | Role |
|---|---|
| `backend/app/retrieval/query_transform.py` | `gpt-oss-20b` (JSON mode) detects the language **incl. Darija** and rewrites the question in formal French + MSA. Original question always kept; any error → original only |
| `backend/app/retrieval/reranker.py` | `BAAI/bge-reranker-v2-m3` cross-encoder scores (question, chunk) pairs; with several variants, best score per chunk |
| `backend/app/retrieval/advanced.py` | `AdvancedRetriever`: wraps the E1 hybrid retriever; RRF across variants → top `candidates_k` → rerank → drop chunks under `min_score` |
| `backend/app/retrieval/hybrid.py` | `get_retriever(cfg)` wraps the base retriever with the bricks enabled in the config |
| `backend/app/pipeline.py` | Passes the detected language (Darija) to the generator |
| `experiments/E2_advanced_rag/` | `config.yaml` (all bricks) + 3 ablations + `run_e2.bat` |
| `backend/tests/test_e2.py` | Offline tests (fake LLM, fake cross-encoder) |

Ablations (same index `legalease_e1`, no reindex):

| Config | Rewrite | Rerank | Compression |
|---|---|---|---|
| `E2_advanced_rag` | ✓ | ✓ | ✓ |
| `E2_ablation_no_compression` | ✓ | ✓ | – |
| `E2_ablation_rerank_only` | – | ✓ | – |
| `E2_ablation_rewrite_only` | ✓ | – | – |

Compare with `E1_hybrid_rag` (no brick). Runs with `--generation` are named `<config>_gen`.
Cost: one extra `gpt-oss-20b` call per question; reranker ≈ 2.2 GB download, slow on CPU (20 candidates × variants).
