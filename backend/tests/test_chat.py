"""/chat and /chat/stream with a fake pipeline: no model, Groq, Qdrant or PostgreSQL."""
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from groq import APIConnectionError
from qdrant_client.http.exceptions import ResponseHandlingException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.core.dependencies import get_current_user
from app.core.errors import APIError, register_error_handlers
from app.core.schemas import Answer, Citation

# Import every model module so that relationships resolve.
from app.models import analytics, chat, collaboration, procedure, user  # noqa: F401
from app.services.rag import rag_service
from app.v1.endpoints import ai
from app.v1.endpoints import chat as chat_endpoint

CITATION = Citation(chunk_id="rne_p2_0", doc_title="Loi RNE", page=2, quote="dans un delai de dix jours")
SAVED_CITATION = {**CITATION.model_dump(), "number": None}  # no retriever here
ANSWERED = Answer(answer="Dix jours [1]. " * 40, citations=[CITATION], status="answered", lang="fr")
QUESTION = {"message": "Quel est le delai ?"}


@pytest.fixture
def env(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, autoflush=False)

    def override_db():
        with sessions() as db:
            yield db

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(chat_endpoint.router, prefix="/api/v1")
    app.include_router(ai.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1)

    def use(result):
        """Make the pipeline return `result`, or raise it."""
        def answer(question, filters=None):
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(rag_service, "ready_pipeline", lambda: SimpleNamespace(answer=answer))

    def rows():
        with sessions() as db:
            return db.query(chat.ChatMessage).order_by(chat.ChatMessage.id).all()

    use(ANSWERED)
    return SimpleNamespace(app=app, client=TestClient(app), use=use, rows=rows)


def test_chat_returns_the_answer_and_saves_both_messages(env):
    resp = env.client.post("/api/v1/chat", json=QUESTION)
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == ANSWERED.answer and body["status"] == "answered"
    assert body["citations"] == [SAVED_CITATION]

    question, answer = env.rows()
    assert (question.role.value, question.content, question.status) == ("user", QUESTION["message"], "ok")
    assert (answer.role.value, answer.status, answer.error_code) == ("assistant", "ok", None)
    assert answer.id == body["message_id"] and answer.conversation_id == body["conversation_id"]
    assert answer.citations == [SAVED_CITATION]


def test_abstention_is_saved_as_such(env):
    env.use(Answer(answer="Je ne trouve pas.", status="abstained", lang="fr"))
    assert env.client.post("/api/v1/chat", json=QUESTION).json()["status"] == "abstained"
    answer = env.rows()[1]
    assert answer.status == "abstained" and answer.citations is None


@pytest.mark.parametrize("error,status,code", [
    (APIConnectionError(request=httpx.Request("POST", "http://groq.test")), 502, "llm_error"),
    (ResponseHandlingException(ConnectionError("secret")), 503, "rag_unavailable"),
    (ValueError("secret"), 500, "internal_error"),
])
def test_pipeline_failure_is_saved_and_does_not_leak(env, error, status, code):
    env.use(error)
    resp = env.client.post("/api/v1/chat", json=QUESTION)
    assert resp.status_code == status and resp.json()["error"]["code"] == code
    assert "secret" not in resp.text

    question, failure = env.rows()
    assert question.content == QUESTION["message"]
    assert (failure.role.value, failure.status, failure.error_code) == ("assistant", "error", code)
    assert failure.content == resp.json()["error"]["message"]
    assert resp.json()["error"]["details"] == {"conversation_id": failure.conversation_id}


def test_pipeline_not_ready_is_saved(env, monkeypatch):
    def not_ready():
        raise APIError(503, "rag_loading", "Still starting.")

    monkeypatch.setattr(rag_service, "ready_pipeline", not_ready)
    resp = env.client.post("/api/v1/chat", json=QUESTION)
    assert resp.status_code == 503 and resp.json()["error"]["code"] == "rag_loading"
    assert [m.error_code for m in env.rows()] == [None, "rag_loading"]


@pytest.mark.parametrize("path", ["/api/v1/chat", "/api/v1/ai/chat/stream"])
def test_retry_after_a_failed_first_message_continues_the_conversation(env, path):
    env.use(ValueError("boom"))
    conv_id = env.client.post(path, json=QUESTION).json()["error"]["details"]["conversation_id"]

    env.use(ANSWERED)
    assert env.client.post(path, json={**QUESTION, "conversation_id": conv_id}).status_code == 200
    assert {m.conversation_id for m in env.rows()} == {conv_id}
    assert [m.status for m in env.rows()] == ["ok", "error", "ok", "ok"]


@pytest.mark.parametrize("path", ["/api/v1/chat", "/api/v1/ai/chat/stream"])
def test_message_length_limit(env, path):
    assert env.client.post(path, json={"message": "a" * 4000}).status_code == 200
    resp = env.client.post(path, json={"message": "a" * 4001})
    assert resp.status_code == 422 and resp.json()["error"]["code"] == "validation_error"
    assert len(env.rows()) == 2  # the rejected message is not saved


def test_conversation_is_reused_only_by_its_owner(env):
    first = env.client.post("/api/v1/chat", json=QUESTION).json()["conversation_id"]
    follow_up = {**QUESTION, "conversation_id": first}
    assert env.client.post("/api/v1/chat", json=follow_up).json()["conversation_id"] == first

    env.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=2)
    assert env.client.post("/api/v1/chat", json=follow_up).json()["conversation_id"] != first


def test_invalid_message_saves_nothing(env):
    resp = env.client.post("/api/v1/chat", json={"message": "   "})
    assert resp.status_code == 422 and resp.json()["error"]["code"] == "validation_error"
    assert env.rows() == []


@pytest.mark.parametrize("path", ["/api/v1/chat", "/api/v1/ai/chat/stream"])
def test_chat_requires_auth(env, path):
    del env.app.dependency_overrides[get_current_user]
    resp = env.client.post(path, json=QUESTION)
    assert resp.status_code == 401 and resp.json()["error"]["code"] == "unauthorized"


def events(resp):
    return [line.removeprefix("data: ") for line in resp.text.split("\n\n") if line]


def test_stream_keeps_the_event_format_the_frontend_reads(env):
    resp = env.client.post("/api/v1/ai/chat/stream", json=QUESTION)
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")

    *data, last = events(resp)
    assert last == "[DONE]"  # the widget stops reading lines at [DONE], so nothing may follow it
    payloads = [json.loads(d) for d in data]
    chunks = [p["chunk"] for p in payloads if "chunk" in p]
    assert len(chunks) > 1 and "".join(chunks) == ANSWERED.answer
    sources = next(p["sources"] for p in payloads if "sources" in p)
    assert sources == [{"title": "Loi RNE, p. 2", "source": "Loi RNE", "page": 2}]
    assert payloads[-1] == {"conversation_id": env.rows()[1].conversation_id}


def test_stream_failure_is_a_json_error_and_is_saved(env):
    env.use(ValueError("secret"))
    resp = env.client.post("/api/v1/ai/chat/stream", json=QUESTION)
    assert resp.status_code == 500 and resp.json()["detail"] == "Internal server error."
    assert env.rows()[1].status == "error"


def test_history_returns_status_and_citations(env):
    conv_id = env.client.post("/api/v1/chat", json=QUESTION).json()["conversation_id"]
    messages = env.client.get(f"/api/v1/ai/conversations/{conv_id}").json()["messages"]
    assert [m["status"] for m in messages] == ["ok", "ok"]
    assert messages[1]["citations"] == [SAVED_CITATION] and messages[0]["citations"] == []


def test_swagger_documents_chat():
    from app.main import app

    schema = app.openapi()
    operation = schema["paths"]["/api/v1/chat"]["post"]
    assert {"200", "401", "422", "500", "502", "503"} <= set(operation["responses"])
    assert "ErrorResponse" in schema["components"]["schemas"]
    assert "/health" in schema["paths"] and TestClient(app).get("/docs").status_code == 200
