"""Langfuse tracing, with the real SDK writing to memory. Fake Groq, retriever and reranker: no network."""
import json
import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import tracing
from app.core.database import Base, get_db
from app.core.dependencies import get_current_user
from app.core.errors import register_error_handlers
from app.core.experiment_config import ExperimentConfig
from app.core.schemas import Chunk, ScoredChunk
from app.generation.generator import Generator

# Import every model module so that relationships resolve.
from app.models import analytics, chat, collaboration, procedure, user  # noqa: F401
from app.pipeline import RAGPipeline
from app.retrieval.advanced import AdvancedRetriever
from app.retrieval.reranker import Reranker
from app.services.rag import RagService, rag_service
from app.v1.endpoints import ai
from app.v1.endpoints import chat as chat_endpoint

QUESTION = "SECRET-QUESTION quelle durée ?"
REPLY = "SECRET-ANSWER huit ans [1]."


class FakeGroq:
    def __init__(self, error=None):
        def create(**kwargs):
            if error:
                raise error
            content = json.dumps({"status": "answered", "answer": REPLY, "citations": [{"source": 1, "quote": "q"}]})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def build_pipeline(groq=None):
    """The real pipeline, E2-style retriever and generator, around fakes."""
    chunk = Chunk(chunk_id="loi_p1_0", doc_id="loi", doc_title="Loi", page=1, lang="fr", text="SECRET-CHUNK huit ans")
    base = SimpleNamespace(retrieve=lambda query, k=None, filters=None: [ScoredChunk(chunk=chunk, score=0.5)])
    cfg = ExperimentConfig(name="E_traced", reranker={"enabled": True})
    reranker = Reranker(cfg.reranker, model=SimpleNamespace(predict=lambda pairs, **kw: [0.9] * len(pairs)))
    retriever = AdvancedRetriever(cfg, base, reranker=reranker)
    return RAGPipeline(cfg, retriever=retriever, generator=Generator(cfg.llm, client=groq or FakeGroq()))


def serve(monkeypatch, pipeline):
    service = RagService("unused.yaml")
    service._serve(pipeline, pipeline.cfg, "registry", "7")  # wraps it, as at startup
    monkeypatch.setattr(rag_service, "ready_pipeline", lambda: pipeline)
    monkeypatch.setattr(rag_service, "model_info", service.model_info)


@pytest.fixture
def client():
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
    ai.limiter.reset()
    yield TestClient(app)
    ai.limiter.reset()


@pytest.fixture
def traces(monkeypatch):
    """Tracing on. Returns a function giving the finished spans as {name: (span, attributes)}."""
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    keys = {"public_key": f"pk-test-{uuid.uuid4().hex}", "secret_key": "sk-test"}  # the SDK caches by public key
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", keys["public_key"])
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", keys["secret_key"])
    exporter = InMemorySpanExporter()
    langfuse = Langfuse(**keys, base_url="http://127.0.0.1:1", span_exporter=exporter,
                        tracer_provider=TracerProvider())
    monkeypatch.setattr(tracing, "_client", langfuse)

    def spans():
        langfuse.flush()
        finished = exporter.get_finished_spans()
        return {span.name: (span, dict(span.attributes)) for span in finished}

    yield spans
    langfuse.shutdown()


def meta(attributes, key):
    """A metadata value: the SDK stores them as JSON."""
    return json.loads(attributes[f"langfuse.observation.metadata.{key}"])


def everything_sent(spans) -> str:
    return json.dumps([attributes for _, attributes in spans.values()], default=str)


def test_a_chat_request_is_traced_step_by_step(client, traces, monkeypatch):
    serve(monkeypatch, build_pipeline())
    body = client.post("/api/v1/chat", json={"message": QUESTION}).json()
    assert body["status"] == "answered"

    spans = traces()
    assert set(spans) == {"chat", "retrieval", "reranking", "generation"}
    parent = {name: span.parent.span_id if span.parent else None for name, (span, _) in spans.items()}
    ids = {name: span.context.span_id for name, (span, _) in spans.items()}
    assert parent["retrieval"] == parent["generation"] == ids["chat"] and parent["reranking"] == ids["retrieval"]

    root = spans["chat"][1]
    assert root["session.id"] == str(body["conversation_id"])
    assert meta(root, "status") == "answered"
    assert meta(root, "citations_count") == 1
    assert meta(root, "version") == "7"
    generation = spans["generation"][1]
    assert generation["langfuse.observation.model.name"] == "openai/gpt-oss-120b"
    assert json.loads(generation["langfuse.observation.usage_details"]) == {"input": 120, "output": 30}
    assert "loi_p1_0" in spans["retrieval"][1]["langfuse.observation.output"]
    assert "SECRET" not in everything_sent(spans)  # no question, prompt, chunk or answer text by default


def test_text_is_sent_only_when_asked_for(client, traces, monkeypatch):
    monkeypatch.setenv("LANGFUSE_CAPTURE_TEXT", "true")
    serve(monkeypatch, build_pipeline())
    client.post("/api/v1/chat", json={"message": QUESTION})
    spans = traces()
    assert "SECRET-QUESTION" in spans["chat"][1]["langfuse.observation.input"]
    assert "SECRET-ANSWER" in spans["chat"][1]["langfuse.observation.output"]
    assert "SECRET-CHUNK" in spans["generation"][1]["langfuse.observation.input"]


def test_a_failed_request_is_traced_without_the_error_message(client, traces, monkeypatch):
    serve(monkeypatch, build_pipeline(FakeGroq(error=ValueError("SECRET-QUESTION in an error"))))
    assert client.post("/api/v1/chat", json={"message": QUESTION}).status_code == 500
    spans = traces()
    assert spans["generation"][1]["langfuse.observation.level"] == "ERROR"
    assert meta(spans["chat"][1], "status") == "error"
    assert meta(spans["chat"][1], "error_code") == "internal_error"
    assert "SECRET" not in everything_sent(spans)
    assert all(not span.events for span, _ in spans.values())  # no recorded exception, which would carry text


def test_public_chat_never_sends_text(client, traces, monkeypatch):
    events = [
        {"choices": [{"delta": {"content": "SECRET-ANSWER"}}]},
        {"choices": [{"delta": {}}], "x_groq": {"usage": {"prompt_tokens": 11, "completion_tokens": 5}}},
    ]
    stream = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    real = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=stream))
    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    monkeypatch.setattr(ai.settings, "GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LANGFUSE_CAPTURE_TEXT", "true")  # does not apply to visitors

    assert "SECRET-ANSWER" in client.post("/api/v1/ai/chat/public", json={"message": QUESTION}).text
    spans = traces()
    attributes = spans["chat_public"][1]
    assert meta(attributes, "status") == "answered"
    assert json.loads(attributes["langfuse.observation.usage_details"]) == {"input": 11, "output": 5}
    assert "SECRET" not in everything_sent(spans)


def test_nothing_happens_without_keys(client, monkeypatch):
    def forbidden():
        raise AssertionError("no Langfuse client without keys")

    monkeypatch.setattr(tracing, "_get_client", forbidden)
    serve(monkeypatch, build_pipeline())
    assert client.post("/api/v1/chat", json={"message": QUESTION}).status_code == 200


class Broken:
    """A Langfuse client, observation and context manager that fails at every step."""

    def __init__(self, fail_at):
        self.fail_at = fail_at

    def _step(self, name, result=None):
        if name == self.fail_at:
            raise RuntimeError(f"langfuse broke at {name}")
        return result

    def start_as_current_observation(self, **kwargs):
        return self._step("start", self)

    def start_observation(self, **kwargs):
        return self._step("start", self)

    def __enter__(self):
        return self._step("enter", self)

    def __exit__(self, *args):
        self._step("exit")

    def update(self, **kwargs):
        self._step("update")

    def end(self):
        self._step("end")


@pytest.mark.parametrize("fail_at", ["client", "start", "enter", "update", "exit", "end"])
def test_langfuse_failures_never_break_a_chat(client, monkeypatch, fail_at):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    def get_client():
        if fail_at == "client":
            raise RuntimeError("cannot create the client")
        return Broken(fail_at)

    monkeypatch.setattr(tracing, "_get_client", get_client)
    serve(monkeypatch, build_pipeline())
    body = client.post("/api/v1/chat", json={"message": QUESTION}).json()
    assert body["status"] == "answered" and body["citations"][0]["number"] == 1

    real = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="data: [DONE]\n\n"))
    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    monkeypatch.setattr(ai.settings, "GROQ_API_KEY", "test-key")
    assert client.post("/api/v1/ai/chat/public", json={"message": "hi"}).status_code == 200
