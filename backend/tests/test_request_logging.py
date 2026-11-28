"""Every chat request is logged, and feedback is linked to the answer it rates. Fake pipeline, SQLite."""
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.analytics.tracker import get_overview, get_unanswered_queries
from app.core.database import Base, engine, get_db
from app.core.dependencies import get_current_user
from app.core.errors import register_error_handlers
from app.core.schemas import Answer, Chunk, Citation, ScoredChunk

# Import every model module so that relationships resolve.
from app.models import analytics, chat, collaboration, procedure, user  # noqa: F401
from app.services import chat as chat_service
from app.services.rag import RagService, rag_service, retrieved_chunks
from app.v1.endpoints import ai, feedback
from app.v1.endpoints import chat as chat_endpoint

CITATION = Citation(chunk_id="c1", doc_title="Loi", page=2, quote="q")
ANSWERED = Answer(answer="Huit ans [1].", citations=[CITATION], status="answered", lang="fr")
QUESTION = {"message": "Quelle durée ?"}


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
    for router in (chat_endpoint.router, ai.router, feedback.router):
        app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1)

    def use(result, score=None):
        def answer(question, filters=None, history=None):
            if isinstance(result, Exception):
                raise result
            return result.model_copy(update={"retrieval_score": score})

        monkeypatch.setattr(rag_service, "ready_pipeline", lambda: SimpleNamespace(answer=answer))

    def rows(model):
        with sessions() as db:
            return db.query(model).order_by(model.id).all()

    use(ANSWERED)
    return SimpleNamespace(app=app, client=TestClient(app), use=use, rows=rows, sessions=sessions)


def logs(env):
    return env.rows(analytics.AIQueryLog)


@pytest.mark.parametrize("path,endpoint", [("/api/v1/chat", "chat"), ("/api/v1/ai/chat/stream", "stream")])
def test_answered_request_is_logged(env, path, endpoint):
    env.use(ANSWERED, score=0.91)
    assert env.client.post(path, json=QUESTION).status_code == 200
    [log] = logs(env)
    assert (log.user_id, log.question, log.lang, log.endpoint) == (1, QUESTION["message"], "fr", endpoint)
    assert (log.status, log.answered, log.error_code, log.citations_count) == ("answered", True, None, 1)
    assert log.latency_ms >= 0 and log.reranker_score == 0.91
    assert log.message_id == env.rows(chat.ChatMessage)[1].id


def test_abstained_and_failed_requests_are_logged(env):
    env.use(Answer(answer="Je ne sais pas.", status="abstained", lang="fr"))
    env.client.post("/api/v1/chat", json=QUESTION)
    env.use(ValueError("boom"), score=0.5)
    env.client.post("/api/v1/chat", json=QUESTION)

    abstained, failed = logs(env)
    assert (abstained.status, abstained.answered, abstained.citations_count) == ("abstained", False, 0)
    assert abstained.reranker_score is None
    assert (failed.status, failed.error_code, failed.answered) == ("error", "internal_error", False)
    assert failed.message_id == env.rows(chat.ChatMessage)[-1].id


def test_each_request_logs_its_own_score(env):
    env.use(ANSWERED, score=0.8)
    env.client.post("/api/v1/chat", json=QUESTION)
    env.use(ANSWERED)  # E1: no reranker, no score
    env.client.post("/api/v1/chat", json=QUESTION)
    assert [log.reranker_score for log in logs(env)] == [0.8, None]


def test_a_logging_failure_does_not_break_the_chat(env, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("log table is gone")

    monkeypatch.setattr(chat_service, "log_query", broken)
    assert env.client.post("/api/v1/chat", json=QUESTION).status_code == 200


def rate(env, **body):
    assert env.client.post("/api/v1/ai/feedback", json={"rating": "up", **body}).status_code == 200
    return env.rows(analytics.AIFeedback)[-1].message_id


def test_feedback_is_linked_to_the_rated_answer(env):
    first = env.client.post("/api/v1/chat", json=QUESTION).json()
    conv = first["conversation_id"]
    # an older client: no message_id
    assert rate(env, conversation_id=conv, answer=ANSWERED.answer, message_index=1) == first["message_id"]
    assert rate(env, message_id=first["message_id"]) == first["message_id"]
    assert rate(env, conversation_id=conv, answer="something else") is None
    assert rate(env) is None


def test_message_id_wins_over_what_the_client_says(env):
    first = env.client.post("/api/v1/chat", json=QUESTION).json()
    second = env.client.post("/api/v1/chat", json=QUESTION).json()  # same answer text, other conversation
    body = {"message_id": first["message_id"], "conversation_id": second["conversation_id"], "answer": "edited"}
    assert rate(env, **body) == first["message_id"]
    assert env.rows(analytics.AIFeedback)[-1].conversation_id == first["conversation_id"]


def test_unknown_message_id_is_not_replaced_by_a_text_match(env):
    first = env.client.post("/api/v1/chat", json=QUESTION).json()
    body = {"message_id": 9999, "conversation_id": first["conversation_id"], "answer": ANSWERED.answer}
    assert rate(env, **body) is None


def test_feedback_cannot_point_at_another_users_message(env):
    first = env.client.post("/api/v1/chat", json=QUESTION).json()
    env.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=2)
    assert rate(env, message_id=first["message_id"]) is None
    assert rate(env, conversation_id=first["conversation_id"], answer=ANSWERED.answer) is None


def test_deleting_a_conversation_erases_its_questions_only(env):
    env.use(Answer(answer="Je ne sais pas.", status="abstained", lang="fr"), score=0.4)
    deleted = env.client.post("/api/v1/chat", json=QUESTION).json()["conversation_id"]
    env.client.post("/api/v1/chat", json={**QUESTION, "conversation_id": deleted})
    env.client.post("/api/v1/chat", json={"message": "Autre question"})
    latencies = [log.latency_ms for log in logs(env)]

    assert env.client.delete(f"/api/v1/ai/conversations/{deleted}").status_code == 200

    assert [log.question for log in logs(env)] == ["", "", "Autre question"]
    assert [log.latency_ms for log in logs(env)] == latencies
    assert {(log.status, log.reranker_score, log.user_id) for log in logs(env)} == {("abstained", 0.4, 1)}
    with env.sessions() as db:
        assert [gap["question"] for gap in get_unanswered_queries(db)] == ["Autre question"]


def test_another_user_cannot_erase_questions(env):
    conv = env.client.post("/api/v1/chat", json=QUESTION).json()["conversation_id"]
    env.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=2)
    assert env.client.delete(f"/api/v1/ai/conversations/{conv}").status_code == 404
    assert logs(env)[0].question == QUESTION["message"]


def test_deleting_a_conversation_erases_rated_text_but_keeps_the_rating(env):
    first = env.client.post("/api/v1/chat", json=QUESTION).json()
    other = env.client.post("/api/v1/chat", json=QUESTION).json()
    conv = first["conversation_id"]
    text = {"question": QUESTION["message"], "answer": ANSWERED.answer}
    rate(env, conversation_id=conv, **text)  # what the widget sends
    rate(env, message_id=first["message_id"], **text)  # no conversation_id
    rate(env, conversation_id=other["conversation_id"], **text)
    env.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=2)
    rate(env, conversation_id=conv, **text)  # someone else claiming this conversation
    env.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1)

    assert env.client.delete(f"/api/v1/ai/conversations/{conv}").status_code == 200

    rows = env.rows(analytics.AIFeedback)
    assert [row.rating for row in rows] == ["up"] * 4
    assert [(row.question, row.answer) for row in rows[:2]] == [(None, None)] * 2
    assert [(row.question, row.answer) for row in rows[2:]] == [(text["question"], text["answer"])] * 2


def test_public_rows_are_left_out_of_the_answer_rate_and_gaps(env):
    env.use(ANSWERED)
    env.client.post("/api/v1/chat", json=QUESTION)
    env.use(Answer(answer="Je ne sais pas.", status="abstained", lang="fr"))
    env.client.post("/api/v1/chat", json={"message": "Autre question"})
    with env.sessions() as db:
        for status in ("answered", "answered", "error"):
            chat_service.record_public_request(db, "visitor", "fr", status, 10)

        assert get_overview(db) == {
            "total_queries": 5, "answered_queries": 1, "unanswered_queries": 1, "answer_rate": 50.0,
        }
        assert [gap["question"] for gap in get_unanswered_queries(db)] == ["Autre question"]


def test_sql_statements_are_not_echoed():
    assert not engine.echo  # echo prints parameters, including question text


def test_database_errors_hide_query_parameters():
    assert engine.hide_parameters


def scored(chunk_id, score, text="texte"):
    chunk = Chunk(chunk_id=chunk_id, doc_id="d", doc_title="Loi", page=1, lang="fr", text=text)
    return ScoredChunk(chunk=chunk, score=score)


def test_retrieved_chunks_are_not_shared_between_concurrent_requests():
    def retrieve(query, k=None, filters=None):
        return [scored(query, 0.5)]

    service = RagService("unused.yaml")
    service.pipeline = SimpleNamespace(retriever=SimpleNamespace(retrieve=retrieve))
    service._observe_retrieval(service.pipeline)
    both_retrieved = threading.Barrier(2, timeout=5)

    def request(query):
        service.pipeline.retriever.retrieve(query)
        both_retrieved.wait()  # read only after both requests have written
        return [sc.chunk.chunk_id for sc in retrieved_chunks.get()]

    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(request, "ab")) == [["a"], ["b"]]


def test_retrieved_chunks_are_recorded_from_the_retriever():
    chunks = [scored("c1", 0.7), scored("c2", 0.2)]
    retriever = SimpleNamespace(retrieve=lambda query, k=None, filters=None: chunks)
    RagService._observe_retrieval(SimpleNamespace(retriever=retriever))

    retrieved_chunks.set(())
    assert retriever.retrieve("q", k=2) == chunks
    assert retrieved_chunks.get() == tuple(chunks)


def test_citations_are_numbered_like_the_answer_markers_and_kept_in_history(env, monkeypatch):
    def citation(chunk_id):
        return Citation(chunk_id=chunk_id, doc_title="Loi", page=1, quote="q")

    answer = Answer(
        answer="A [3]. B [1].", citations=[citation("c3"), citation("c1"), citation("unseen")],
        status="answered", lang="fr", confidence=0.8,
    )

    def run(question, filters=None, history=None):
        retrieved_chunks.set(tuple(scored(i, 0.5) for i in ("c1", "c2", "c3")))  # what the wrapper records
        return answer

    monkeypatch.setattr(rag_service, "ready_pipeline", lambda: SimpleNamespace(answer=run))
    body = env.client.post("/api/v1/chat", json=QUESTION).json()
    assert [(c["chunk_id"], c["number"]) for c in body["citations"]] == [("c3", 3), ("c1", 1), ("unseen", None)]
    assert body["confidence"] == 0.8

    saved = env.client.get(f"/api/v1/ai/conversations/{body['conversation_id']}").json()["messages"][1]
    assert (saved["id"], saved["status"], saved["confidence"]) == (body["message_id"], "ok", 0.8)
    assert saved["citations"] == body["citations"]


def test_numbers_of_a_previous_request_are_not_reused(env):
    retrieved_chunks.set((scored("c1", 0.5),))
    assert env.client.post("/api/v1/chat", json=QUESTION).json()["citations"][0]["number"] is None
