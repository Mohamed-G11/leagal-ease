# LegalEase-RAG

[![CI](https://github.com/Mohamed-G11/leagal-ease/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Mohamed-G11/leagal-ease/actions/workflows/ci.yml?query=branch%3Amain)

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
docker compose up -d postgres qdrant

# 5. Database schema and API
cd backend
alembic upgrade head
uvicorn app.main:app --reload
```

After pulling, run `alembic upgrade head` from `backend/`.

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


### Chat API

Swagger UI is at `http://localhost:8000/docs`. Both endpoints need a login token and use the same pipeline:

- `POST /api/v1/chat` returns the whole answer as JSON, with citations.
- `POST /api/v1/ai/chat/stream` sends it as server-sent events (used by the chat widget).

Each question is answered on its own: the conversation history is saved but not sent to the model. Until the
pipeline and the index are ready, both return HTTP 503.

## Run the API in Docker

```bash
cp .env.example .env       # then set SECRET_KEY, GROQ_API_KEY and the PostgreSQL password
docker compose up -d --build
```

This starts PostgreSQL, Qdrant and the API on `http://localhost:8000` (the API applies the migrations first).
The API, PostgreSQL (5433) and Qdrant (6333, 6334) ports are bound to `127.0.0.1`, so other machines on the
network cannot reach them.

- Models are not baked into the image. The first start downloads BGE-M3 (about 2.3 GB) into the `hf_cache` volume;
  until then `/health` reports `"pipeline": "loading"`. Follow it with `docker compose logs -f api`.
- The API needs an indexed collection. With Qdrant running, ingest once from your machine (see "RAG pipeline").
- Compose overrides `DATABASE_URL` and `QDRANT_URL` from `.env` with the container addresses. `.env` is read at run
  time and is not copied into the image.
- To run the API from source instead, start only the services: `docker compose up -d postgres qdrant`.

### Published image

Every push to `main` builds the API image and publishes it to GitHub Container Registry (`.github/workflows/cd.yml`),
tagged with the full commit SHA and `latest`. Nothing is deployed.

```bash
docker pull ghcr.io/mohamed-g11/leagal-ease-api:latest
```

## Checks

```bash
ruff check .          # lint
pytest                # tests (run from the repo root)
```

## Dependencies

- `requirements.txt`: pinned direct runtime dependencies.
- `requirements-dev.txt`: adds pytest, httpx and ruff.
- `requirements-mlops.txt`: adds Prefect, RAGAS and Evidently (offline tools). CI and the API image skip it.
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

## Request log

Every call to `/api/v1/chat` and `/api/v1/ai/chat/stream` writes one row in `ai_query_logs`: question (first 500
characters), language, `status`, number of citations, latency, reranker score (only with a config that enables a
reranker) and `message_id`. Feedback rows in `ai_feedback` carry the same `message_id`, so they can be joined:

```sql
SELECT l.question, l.status, l.latency_ms, f.rating
FROM ai_query_logs l LEFT JOIN ai_feedback f ON f.message_id = l.message_id;
```

`/api/v1/ai/chat/public` (anonymous visitors) logs only the question, language, `status` (`answered` or `error`)
and latency, with `endpoint = 'public'`: no IP address or visitor identifier.

The admin answer rate and unanswered-questions list leave out these public rows.

Deleting a conversation erases the question text in its log rows and the question and answer text in its
`ai_feedback` rows. The other fields and the ratings stay; the `message_id` link is removed.

Run `alembic upgrade head` after pulling.

## Monitoring

Set `GRAFANA_ADMIN_PASSWORD` in `.env`, then `docker compose up -d --build`.

| Service | Address | Notes |
|---|---|---|
| Metrics | `http://localhost:8000/metrics` | Served by the API |
| Prometheus | `http://localhost:9090` | Scrapes the API every 15 s |
| Grafana | `http://localhost:3000` | User `admin`. Dashboard "LegalEase API" in the LegalEase folder |

- The datasource and dashboard are provisioned from `monitoring/`. Edit the files, not the Grafana UI.
- Grafana stores the password at its first start. Changing `.env` later does not change it.
- Chat metrics: `legalease_chat_requests_total{outcome="answered|abstained|error"}` and
  `legalease_chat_duration_seconds`. `/chat/public` is not counted.
- CPU and memory are those of the API process only, not of the whole machine or the other containers.
- `/metrics` has **no authentication**. Do not expose port 8000 to the internet without blocking that path.
