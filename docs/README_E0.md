# E0 — Naive RAG (Wiem, Sprint 1)

Dense retrieval (BGE-M3) + Qdrant + Groq LLM. Covers tasks W1–W5; W6 = Mohamed's `/chat` calls `RAGPipeline.answer()`.

## Files
| Task | File |
|---|---|
| W1 | `backend/app/core/schemas.py` — shared contracts (Chunk, ScoredChunk, Citation, Answer) |
| W2 | `backend/app/ingestion/temp_ingest.py` — PDF page by page → chunks (doc_id + page) → Qdrant |
| W3 | `backend/app/retrieval/dense.py` (+ `embedder.py`, `store.py`) — `retrieve(query, k, filters)` |
| W4 | `backend/app/generation/generator.py` — numbered sources, [n] citations, abstention, JSON → `Answer` |
| W5 | `experiments/E0_naive_rag/config.yaml` + `backend/app/core/experiment_config.py` (separate from the app settings in `core/config.py`) |
| — | `backend/app/pipeline.py` — retrieve → generate (entry point for `/chat`) |
| — | `backend/tests/test_e0.py` — runs offline (fake embedder + fake LLM) |

## Run
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r ../requirements-dev.txt              # root requirements.txt + pytest and ruff
export GROQ_API_KEY=...                             # Windows: set GROQ_API_KEY=...

python -m pytest -q tests                           # 5 tests, offline

# put the PDFs in ../data/raw/ (the 10 PDFs of the old LegalEase repo are fine for now)
python -m app.ingestion.temp_ingest --pdf-dir ../data/raw --config ../experiments/E0_naive_rag/config.yaml
python -m app.pipeline "Quel est le délai d'inscription au RNE ?" --config ../experiments/E0_naive_rag/config.yaml
```
First run downloads BGE-M3 (~2.3 GB). Qdrant runs in local mode (`./qdrant_local`) until
docker-compose is up — then `QDRANT_URL` from `.env` (http://localhost:6333) is used automatically.

## Notes
- Point IDs are deterministic (uuid5 of chunk_id): re-ingesting overwrites, no duplicates.
- Named vector `dense` → E1 adds `sparse` in the same collection.
- E0 is intentionally naive: no grading, no grounding check, no retry (added in E2/E3).
- Arabic script → `ar`; Darija vs MSA detection belongs to the E2 query router.

## Model comparison + MLflow (Wiem)

`backend/app/evaluation/retrieval_eval.py` + `run_experiment.py`: one config = one MLflow run
(experiment `legalease-rag`) with all config params, metrics and per-question results as artifacts.

| Config | What changes |
|---|---|
| `E0_emb_minilm` | old LegalEase embeddings (baseline) |
| `E0_emb_e5_large` | multilingual-e5-large (with `query:` / `passage:` prefixes) |
| `E0_emb_bge_m3` | BGE-M3 |
| `E0_llm_gptoss20b` / `E0_llm_gptoss120b` / `E0_llm_qwen3_27b` | generator LLM, same BGE-M3 retrieval |

Metrics: `recall_at_{1,3,5,10}`, `mrr`, `{lang}_recall_at_5`, retrieval latency p50/p95;
with `--generation`: `abstention_correct_rate`, `false_abstention_rate`, `citation_hit_rate`, latency.
Faithfulness / answer relevancy = Hend's RAGAS run.

```cmd
cd backend
..\experiments\model_comparison\run_all.bat
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Dataset: Hend's `data/evaluation/seed.jsonl` (format example in `data/evaluation/seed_example.jsonl`;
`doc_id` must match the ingestion slug of the PDF file name, e.g. `loi_2002_32`).
Once Mohamed's MLflow server runs: `set MLFLOW_TRACKING_URI=http://localhost:5000`.
