@echo off
REM Run from wiem-e0\backend with the venv activated and GROQ_API_KEY set.
set DS=..\data\evaluation\seed.jsonl
set CMP=..\experiments\model_comparison

REM 1) Embedding models: retrieval only (no Groq quota), each rebuilds its own collection
python -m app.evaluation.run_experiment --config %CMP%\E0_emb_minilm.yaml   --dataset %DS% --reindex
python -m app.evaluation.run_experiment --config %CMP%\E0_emb_e5_large.yaml --dataset %DS% --reindex
python -m app.evaluation.run_experiment --config %CMP%\E0_emb_bge_m3.yaml   --dataset %DS% --reindex

REM 2) LLMs: same retrieval (bge-m3 collection built above), generation on
python -m app.evaluation.run_experiment --config %CMP%\E0_llm_gptoss20b.yaml  --dataset %DS% --generation
python -m app.evaluation.run_experiment --config %CMP%\E0_llm_gptoss120b.yaml --dataset %DS% --generation
python -m app.evaluation.run_experiment --config %CMP%\E0_llm_qwen3_27b.yaml  --dataset %DS% --generation

echo Done. Open MLflow: mlflow ui --backend-store-uri sqlite:///mlflow.db
