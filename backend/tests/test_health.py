"""/health and pipeline loading, with fakes: no PostgreSQL, Qdrant, model or Groq."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import health
from app.core.database import get_db
from app.core.errors import APIError
from app.services.rag import RagService


class FakeQdrant:
    def __init__(self, points=3, error=None):
        self.points, self.error = points, error

    def collection_exists(self, name):
        if self.error:
            raise self.error
        return self.points is not None

    def count(self, name):
        return SimpleNamespace(count=self.points)


def make_service(state="ready", qdrant=None):
    """A service as it looks after loading in local mode."""
    service = RagService("unused.yaml")
    service.state = state
    if state == "ready":
        service.cfg = SimpleNamespace(qdrant=SimpleNamespace(url=None, collection="test"))
        service.pipeline = SimpleNamespace(retriever=SimpleNamespace(client=qdrant or FakeQdrant()))
    return service


def get_health(monkeypatch, service, db_error=None, groq_key="test-key"):
    def execute(statement):
        if db_error:
            raise db_error

    app = FastAPI()
    app.include_router(health.router)
    app.dependency_overrides[get_db] = lambda: SimpleNamespace(execute=execute)
    monkeypatch.setattr(health, "rag_service", service)
    monkeypatch.setenv("GROQ_API_KEY", groq_key)
    return TestClient(app).get("/health")


def test_all_ok(monkeypatch):
    resp = get_health(monkeypatch, make_service())
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "checks": {
        "api": "ok", "postgres": "ok", "qdrant": "ok", "pipeline": "ready", "groq_key": "present"}}


@pytest.mark.parametrize("service,qdrant,pipeline", [
    (make_service("loading"), "loading", "loading"),
    (make_service("failed"), "down", "failed"),
    (make_service(qdrant=FakeQdrant(points=0)), "empty", "ready"),
    (make_service(qdrant=FakeQdrant(points=None)), "empty", "ready"),
    (make_service(qdrant=FakeQdrant(error=ConnectionError())), "down", "ready"),
])
def test_rag_problems_degrade(monkeypatch, service, qdrant, pipeline):
    resp = get_health(monkeypatch, service)
    body = resp.json()
    assert resp.status_code == 200 and body["status"] == "degraded"
    assert (body["checks"]["qdrant"], body["checks"]["pipeline"]) == (qdrant, pipeline)


def test_missing_groq_key_degrades(monkeypatch):
    body = get_health(monkeypatch, make_service(), groq_key="").json()
    assert body["status"] == "degraded" and body["checks"]["groq_key"] == "missing"


def test_postgres_down_returns_503(monkeypatch):
    resp = get_health(monkeypatch, make_service(), db_error=ConnectionError("secret"))
    assert resp.status_code == 503 and resp.json()["status"] == "down"
    assert "secret" not in resp.text


@pytest.mark.parametrize("service,code", [
    (make_service("loading"), "rag_loading"),
    (make_service("failed"), "rag_unavailable"),
    (make_service(qdrant=FakeQdrant(points=0)), "rag_unavailable"),
    (make_service(qdrant=FakeQdrant(error=ConnectionError())), "rag_unavailable"),
])
def test_ready_pipeline_refuses(service, code):
    with pytest.raises(APIError) as exc:
        service.ready_pipeline()
    assert (exc.value.status_code, exc.value.code) == (503, code)


def test_ready_pipeline_recovers_after_ingestion():
    qdrant = FakeQdrant(points=0)
    service = make_service(qdrant=qdrant)
    with pytest.raises(APIError):
        service.ready_pipeline()
    qdrant.points = 5
    assert service.ready_pipeline() is service.pipeline


def test_load_builds_the_pipeline_once(monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr("app.pipeline.RAGPipeline", lambda cfg: built.append(cfg) or "pipeline")
    config = tmp_path / "cfg.yaml"
    config.write_text("name: test\n", encoding="utf-8")
    service = RagService(str(config))
    service.start()
    service.start()
    service._thread.join(timeout=30)
    assert (service.state, service.pipeline, len(built)) == ("ready", "pipeline", 1)


def test_load_failure_is_reported():
    service = RagService("does/not/exist.yaml")
    service.load()
    assert service.state == "failed" and service.pipeline is None


def test_default_config_exists():
    from app.core.config import Settings

    assert RagService(Settings.model_fields["RAG_CONFIG"].default).config_path.is_file()
