# LegalEase-RAG

A RAG assistant for Tunisian legal texts and administrative procedures: FastAPI backend, React/Vite frontend,
Qdrant for vector search, PostgreSQL for application data, and Llama 3.3 70B on Groq for answers.

## Requirements

- Python 3.11
- Docker (for PostgreSQL 16 and Qdrant)
- Node.js (frontend only)

## Setup

```bash
# 1. Virtual environment (repo root)
py -3.11 -m venv .venv            # Windows
# python3.11 -m venv .venv        # Linux / macOS
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

# 2. Dependencies (runtime + pytest + ruff)
pip install -r requirements-dev.txt

# 3. Configuration
cp .env.example .env              # then edit the values (SECRET_KEY, GROQ_API_KEY, passwords)

# 4. PostgreSQL and Qdrant
docker compose up -d

# 5. Database schema and API
cd backend
alembic upgrade head
uvicorn app.main:app --reload
```

`DATABASE_URL` must start with `postgresql+psycopg2://`. SQLAlchemy 2.1 maps a plain `postgresql://` to psycopg v3,
which is not installed.

Docker publishes PostgreSQL on port **5433**, so it does not clash with a PostgreSQL installed on your machine
(which usually holds 5432). If you already have a `.env`, change the port in `DATABASE_URL` to 5433 and run
`docker compose up -d` again to recreate the container.

### RAG pipeline

The API loads the pipeline in the background at startup. `GET /health` shows its progress.

- `RAG_CONFIG` selects the experiment config, relative to the repo root. The default is
  `experiments/E1_hybrid_rag/config.yaml` (hybrid retrieval).
- The app should use the Docker Qdrant: set `QDRANT_URL=http://localhost:6333` in `.env`. Without it, Qdrant runs
  in local mode, in a `qdrant_local/` folder under the directory you start from.
- The collection must be ingested **into that server**. An index built in local mode (`qdrant_local/`) is not
  visible to it. The ingestion command does not read `.env`, so set `QDRANT_URL` in the shell first:

```bash
cd backend
export QDRANT_URL=http://localhost:6333      # PowerShell: $env:QDRANT_URL = "http://localhost:6333"
python -m app.ingestion.temp_ingest --pdf-dir ../data/raw --config ../experiments/E1_hybrid_rag/config.yaml
```

The RAG endpoints are being migrated from Chroma to Qdrant. Until then the chat endpoint answers with HTTP 503.

## Checks

```bash
ruff check .          # lint
pytest                # tests (run from the repo root)
```

## Dependencies

- `requirements.txt`: pinned direct runtime dependencies.
- `requirements-dev.txt`: adds pytest, httpx and ruff.
- `requirements.lock`: full `pip freeze` of the dev environment, for reference.

PyTorch installs as the CPU build by default, so CI never downloads the multi-GB CUDA wheels.

### Optional: CUDA PyTorch for a local GPU

After the normal install, replace the CPU build with a CUDA build. The CUDA indexes lag behind PyPI, so this uses
the newest version they publish (2.14.1 at the time of writing).

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
python -c "import torch; print(torch.cuda.is_available())"
```

Do not commit this change to `requirements.txt`.
