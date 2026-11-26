@echo off
REM Run from backend\ with the venv activated.
set DS=..\data\evaluation\seed.jsonl
set E1=..\experiments\E1_hybrid_rag
REM Build the E1 collection once (dense + bm25), then compare the 3 modes on the same index
python -m app.evaluation.run_experiment --config %E1%\config.yaml          --dataset %DS% --reindex
python -m app.evaluation.run_experiment --config %E1%\ablation_dense.yaml  --dataset %DS%
python -m app.evaluation.run_experiment --config %E1%\ablation_sparse.yaml --dataset %DS%
echo Done. mlflow ui --backend-store-uri sqlite:///mlflow.db
