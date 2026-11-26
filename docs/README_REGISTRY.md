# Model Registry — `legalease-rag` (Wiem, Sprint 4)

A RAG system trains no weights, so the registry versions **the whole pipeline**: code (`backend/app`),
experiment config (embeddings, retrieval mode, rewrite, reranker, LLM, top-k) and, through the
registering run, its evaluation metrics. The Qdrant index is data: it is rebuilt by ingestion and
reached through `QDRANT_URL`.

| File | Role |
|---|---|
| `backend/app/registry/pyfunc_model.py` | `LegalEaseRAGModel` (`mlflow.pyfunc`): `predict({"question": [...]})` → list of `Answer` dicts |
| `backend/app/registry/register.py` | Logs the pipeline in the evaluation run and registers a new version (tags: config, mode, metrics; description) |
| `backend/app/registry/promote.py` | Promotion rule: champion only if ≥ champion on every gate metric (tolerance) and > on `recall_at_5` or `mrr`; otherwise `challenger`. Decision + reasons stored as version tags |
| `backend/app/evaluation/run_experiment.py` | `--register` and `--promote` flags |
| `backend/tests/test_registry.py` | Promotion rule + register / load / predict / promote on a local MLflow |

Gate metrics (tolerance): `recall_at_5` (0.02), `mrr` (0.02), `abstention_correct_rate` (0.05),
`citation_hit_rate` (0.05). Generation metrics are only compared when both versions have them.

Register the history (from `backend/`, DagsHub variables set):
```
python -m app.evaluation.run_experiment --config ..\experiments\E0_naive_rag\config.yaml --dataset ..\data\evaluation\seed.jsonl --register --promote
python -m app.evaluation.run_experiment --config ..\experiments\E1_hybrid_rag\config.yaml --dataset ..\data\evaluation\seed.jsonl --register --promote
python -m app.evaluation.run_experiment --config ..\experiments\E2_advanced_rag\fast.yaml --dataset ..\data\evaluation\seed.jsonl --register --promote
```
Serve the champion: `mlflow.pyfunc.load_model("models:/legalease-rag@champion")`.
If the server refuses aliases, `promote.py` falls back to a `role` tag on the version.
