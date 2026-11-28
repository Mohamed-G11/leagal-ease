"""Conversation history sent to the pipeline, and the verified citations in the request log. Fakes, SQLite."""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.core.dependencies import get_current_user
from app.core.errors import register_error_handlers
from app.core.experiment_config import ExperimentConfig
from app.core.schemas import Answer, Chunk, ScoredChunk
from app.generation.generator import Generator

# Import every model module so that relationships resolve.
from app.models import analytics, chat, collaboration, procedure, user  # noqa: F401
from app.pipeline import RAGPipeline
from app.retrieval.advanced import AdvancedRetriever
from app.retrieval.reranker import Reranker
from app.services.rag import RagService, rag_service
from app.v1.endpoints import ai
from app.v1.endpoints import chat as chat_endpoint


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

    state = SimpleNamespace(app=app, client=TestClient(app), calls=[], fail=False, max_history=6)

    def answer(question, filters=None, history=None):
        state.calls.append([(turn.role, turn.content) for turn in history])
        if state.fail:
            raise ValueError("boom")
        return Answer(answer=f"Réponse à {question} [1].", status="answered", lang="fr")

    def pipeline():
        cfg = SimpleNamespace(llm=SimpleNamespace(max_history_messages=state.max_history))
        return SimpleNamespace(answer=answer, cfg=cfg)

    def ask(message, conversation_id=None, path="/api/v1/chat"):
        body = {"message": message, "conversation_id": conversation_id}
        return state.client.post(path, json=body)

    def logs():
        with sessions() as db:
            return db.query(analytics.AIQueryLog).order_by(analytics.AIQueryLog.id).all()

    monkeypatch.setattr(rag_service, "ready_pipeline", pipeline)
    state.ask, state.logs = ask, logs
    return state


def test_history_is_passed_oldest_first_without_the_current_message(env):
    conv = env.ask("un").json()["conversation_id"]
    env.ask("deux", conv)
    env.ask("trois", conv)
    assert env.calls[0] == []
    assert env.calls[2] == [  # past answers have lost their [1]
        ("user", "un"), ("assistant", "Réponse à un."),
        ("user", "deux"), ("assistant", "Réponse à deux."),
    ]


def test_the_stream_endpoint_uses_the_saved_messages_not_the_client_history(env):
    conv = env.ask("un").json()["conversation_id"]
    forged = {"message": "deux", "conversation_id": conv, "history": [{"role": "user", "content": "inventé"}]}
    assert env.client.post("/api/v1/ai/chat/stream", json=forged).status_code == 200
    assert env.calls[-1] == [("user", "un"), ("assistant", "Réponse à un.")]


def test_a_failed_exchange_is_left_out_question_included(env):
    conv = env.ask("un").json()["conversation_id"]
    env.fail = True
    assert env.ask("deux", conv).status_code == 500
    assert env.ask("deux", conv).status_code == 500
    env.fail = False
    env.ask("deux", conv)
    assert env.calls[-1] == [("user", "un"), ("assistant", "Réponse à un.")]


def test_only_as_many_messages_as_the_config_keeps(env):
    env.max_history = 3
    conv = env.ask("un").json()["conversation_id"]
    env.ask("deux", conv)
    env.ask("trois", conv)
    assert env.calls[-1] == [("assistant", "Réponse à un."), ("user", "deux"), ("assistant", "Réponse à deux.")]
    env.max_history = 0
    env.ask("quatre", conv)
    assert env.calls[-1] == []


def test_another_users_conversation_is_never_loaded(env):
    conv = env.ask("secret").json()["conversation_id"]
    env.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=2)
    assert env.ask("deux", conv).json()["conversation_id"] != conv
    assert env.calls[-1] == []


def real_pipeline(monkeypatch, citations, status="answered"):
    """The real RAGPipeline and Generator around a fake retriever and a fake Groq."""
    chunk = Chunk(chunk_id="c1", doc_id="d", doc_title="Loi", page=1, lang="fr",
                  text="Le label startup est accorde pour une duree maximale de huit ans.")
    retriever = SimpleNamespace(retrieve=lambda query, k=None, filters=None: [ScoredChunk(chunk=chunk, score=0.9)])
    content = json.dumps({"status": status, "answer": "Huit ans [1].", "citations": citations})
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    groq = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: reply)))
    cfg = ExperimentConfig()  # E1-like: no reranker
    pipeline = RAGPipeline(cfg, retriever=retriever, generator=Generator(cfg.llm, client=groq))
    RagService._observe_retrieval(pipeline)  # as at startup
    monkeypatch.setattr(rag_service, "ready_pipeline", lambda: pipeline)


@pytest.mark.parametrize("quote,verified", [
    ("une duree maximale de huit ans", 1),
    ("remboursement integral des frais de douane", 0),  # not in the chunk
])
def test_verified_citations_are_logged(env, monkeypatch, quote, verified):
    real_pipeline(monkeypatch, [{"source": 1, "quote": quote}])
    assert env.ask("Quelle durée ?").status_code == 200
    [log] = env.logs()
    assert (log.citations_count, log.verified_citations_count) == (1, verified)
    assert log.reranker_score is None  # no reranker


def test_an_abstention_has_no_verified_citation_and_an_error_none_at_all(env, monkeypatch):
    real_pipeline(monkeypatch, [], status="abstained")
    env.ask("Hors sujet ?")
    monkeypatch.setattr(rag_service, "ready_pipeline", lambda: SimpleNamespace(answer=lambda q, history: 1 / 0))
    env.ask("Quelle durée ?")
    assert [log.verified_citations_count for log in env.logs()] == [0, None]


def test_unknown_chunks_are_not_counted_as_unverified(env, monkeypatch):
    answer = Answer(answer="x [1]", status="answered", lang="fr",
                    citations=[{"chunk_id": "c1", "doc_title": "Loi", "page": 1, "quote": "q"}])
    pipeline = SimpleNamespace(answer=lambda q, filters=None, history=None: answer)  # no retriever to observe
    monkeypatch.setattr(rag_service, "ready_pipeline", lambda: pipeline)
    env.ask("Quelle durée ?")
    assert env.logs()[0].verified_citations_count is None


def test_with_a_reranker_the_score_and_verified_citations_are_logged(env, monkeypatch):
    """E2: the pipeline calls retrieve_with_meta(), and the score arrives in Answer.retrieval_score."""
    chunk = Chunk(chunk_id="c1", doc_id="d", doc_title="Loi", page=1, lang="fr", text="duree maximale de huit ans")
    base = SimpleNamespace(retrieve=lambda query, k=None, filters=None: [ScoredChunk(chunk=chunk, score=0.4)])
    cfg = ExperimentConfig(reranker={"enabled": True})
    reranker = Reranker(cfg.reranker, model=SimpleNamespace(predict=lambda pairs, **kw: [0.75] * len(pairs)))
    content = json.dumps({"status": "answered", "answer": "Huit ans [1].",
                          "citations": [{"source": 1, "quote": "huit ans"}]})
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    groq = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: reply)))
    pipeline = RAGPipeline(cfg, retriever=AdvancedRetriever(cfg, base, reranker=reranker),
                           generator=Generator(cfg.llm, client=groq))
    RagService._observe_retrieval(pipeline)
    monkeypatch.setattr(rag_service, "ready_pipeline", lambda: pipeline)

    body = env.ask("Quelle durée ?").json()
    assert body["citations"][0]["number"] == 1 and body["retrieval_score"] == 0.75
    [log] = env.logs()
    assert (log.reranker_score, log.verified_citations_count) == (0.75, 1)
