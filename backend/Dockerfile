# Build from the repo root:  docker build -f backend/Dockerfile -t legalease-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home --uid 10001 app

# Same layout as the repository: the app resolves RAG_CONFIG from the folder above backend/.
WORKDIR /srv/backend

# Dependencies first, so that code changes do not reinstall them.
COPY requirements.txt requirements.lock /srv/
RUN pip install -r /srv/requirements.txt -c /srv/requirements.lock

COPY experiments /srv/experiments
COPY backend/alembic.ini ./
COPY backend/alembic ./alembic
COPY backend/scripts ./scripts
COPY backend/app ./app

# The only places the app writes to. Volumes mounted here inherit this ownership.
RUN mkdir -p uploads "$HF_HOME" && chown -R app:app uploads /home/app/.cache

USER app
EXPOSE 8000

# Models are downloaded on first start, not at build time: the first start takes minutes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# One worker: each worker would load its own copy of the embedding model.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
