"""/metrics and the chat outcome counters, with a fake pipeline and an in-memory database."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.core.errors import APIError
from app.core.schemas import Answer
from app.main import app

# Import every model module so that relationships resolve.
from app.models import analytics, chat, collaboration, procedure, user  # noqa: F401
from app.services.chat import ask
from app.services.rag import rag_service


def outcome_count(outcome):
    return REGISTRY.get_sample_value("legalease_chat_requests_total", {"outcome": outcome})


def duration_count():
    return REGISTRY.get_sample_value("legalease_chat_duration_seconds_count")


@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def use_pipeline(monkeypatch, result):
    def answer(question, filters=None):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(rag_service, "ready_pipeline", lambda: SimpleNamespace(answer=answer))


@pytest.mark.parametrize("status", ["answered", "abstained"])
def test_successful_exchange_is_counted(db, monkeypatch, status):
    use_pipeline(monkeypatch, Answer(answer="ok", status=status, lang="fr"))
    before, timed = outcome_count(status), duration_count()
    ask(db, 1, "question ?", None, "fr")
    assert outcome_count(status) == before + 1 and duration_count() == timed + 1


def test_failed_exchange_is_counted_as_error(db, monkeypatch):
    use_pipeline(monkeypatch, ValueError("boom"))
    before = outcome_count("error")
    with pytest.raises(APIError):
        ask(db, 1, "question ?", None, "fr")
    assert outcome_count("error") == before + 1


def test_metrics_endpoint_reports_requests_by_exact_status():
    client = TestClient(app)  # no `with`: the lifespan (scheduler, model loading) does not run
    client.get("/")
    client.get("/no-such-page")
    body = client.get("/metrics").text
    assert 'http_requests_total{handler="/",method="GET",status="200"}' in body
    assert 'status="404"' in body
    assert "http_request_duration_highr_seconds_bucket" in body
    assert 'legalease_chat_requests_total{outcome="abstained"}' in body
    assert 'handler="/metrics"' not in body and 'handler="/health"' not in body
    assert "/metrics" not in app.openapi()["paths"]
