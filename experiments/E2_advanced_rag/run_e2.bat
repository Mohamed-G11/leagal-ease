@echo off
REM Run from backend\ with the venv activated and GROQ_API_KEY set.
REM Uses the E1 index (legalease_e1): run ..\experiments\E1_hybrid_rag\run_e1.bat first if it does not exist.
set DS=..\data\evaluation\seed.jsonl
set E2=..\experiments\E2_advanced_rag
python -m app.evaluation.run_experiment --config %E2%\config.yaml                     --dataset %DS%
python -m app.evaluation.run_experiment --config %E2%\ablation_rewrite_only.yaml      --dataset %DS%
python -m app.evaluation.run_experiment --config %E2%\ablation_rerank_only.yaml       --dataset %DS%
python -m app.evaluation.run_experiment --config %E2%\ablation_no_compression.yaml    --dataset %DS%
REM Full pipeline with generation (abstention, citations) — uses more Groq quota
python -m app.evaluation.run_experiment --config %E2%\config.yaml                     --dataset %DS% --generation
echo Done.
