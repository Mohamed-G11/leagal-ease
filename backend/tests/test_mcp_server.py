"""The MCP tools, called through the SDK's own client. Fakes only: no model, Groq, Qdrant or PostgreSQL."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import mcp_server
from app.analytics.tracker import get_overview, get_unanswered_queries
from app.core.database import Base
from app.core.errors import APIError
from app.core.experiment_config import ExperimentConfig
from app.core.schemas import Answer, Chunk, Citation, ScoredChunk
from app.models import analytics
from app.services.rag import RagService, rag_service, retrieved_chunks
from tests.mcp_fake_server import FakeIndex, payload

TOOLS = {"search_legal_texts", "ask_legal_question", "get_document_page", "list_documents"}


def call(tool, arguments=None):
    """(is_error, structured result or error text), through a real MCP session in memory."""

    async def run():
        async with Client(mcp_server.server) as client:
            return await client.call_tool(tool, arguments or {})

    result = anyio.run(run)
    return result.is_error, (result.content[0].text if result.is_error else result.structured_content)


@pytest.fixture
def env(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(mcp_server, "SessionLocal", sessions)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setattr(mcp_server, "ask_limit", mcp_server.RateLimit(mcp_server.ASK_PER_MINUTE))

    state = SimpleNamespace(sessions=sessions, searches=[], found=True, error=None)
    chunk = Chunk(**payload("loi", 2, 0))

    def retrieve(query, k=None, filters=None):
        state.searches.append((query, k, filters))
        if state.error:
            raise state.error
        return [ScoredChunk(chunk=chunk, score=0.5)] if state.found else []

    def answer(question):
        if state.error:
            raise state.error
        retrieved_chunks.set((ScoredChunk(chunk=Chunk(**payload("x", 1, 0)), score=0.1),
                              ScoredChunk(chunk=chunk, score=0.5)))
        return Answer(answer="Huit ans [2].", status="answered", lang="fr", confidence=1.0,
                      citations=[Citation(chunk_id=chunk.chunk_id, doc_title="Titre loi", page=2, quote="q")])

    index = FakeIndex([payload("loi", 2, 1), payload("loi", 2, 0), payload("loi", 3, 0),
                       payload("code", 1, 0, "ar"), payload("code", 1, 1, "ar")])
    pipeline = SimpleNamespace(cfg=ExperimentConfig(retrieval={"mode": "hybrid"}), answer=answer,
                               retriever=SimpleNamespace(retrieve=retrieve, client=index))
    state.pipeline = pipeline
    monkeypatch.setattr(rag_service, "ready_pipeline", lambda: pipeline)

    def logs():
        with sessions() as db:
            return db.query(analytics.AIQueryLog).order_by(analytics.AIQueryLog.id).all()

    state.logs = logs
    return state


def test_the_four_tools_are_listed_with_their_limits():
    async def run():
        async with Client(mcp_server.server) as client:
            return (await client.list_tools()).tools

    tools = {tool.name: tool for tool in anyio.run(run)}
    assert set(tools) == TOOLS
    search = tools["search_legal_texts"].input_schema["properties"]
    assert (search["k"]["maximum"], search["query"]["maxLength"]) == (10, 1000)
    assert all(tool.description for tool in tools.values())


def test_search_returns_passages_and_passes_the_arguments(env):
    failed, body = call("search_legal_texts", {"query": "label startup", "k": 3, "lang": "fr"})
    assert not failed
    assert env.searches == [("label startup", 3, {"lang": "fr"})]
    assert body["score_type"] == "reciprocal rank fusion"
    assert body["passages"] == [{"doc_id": "loi", "doc_title": "Titre loi", "page": 2, "lang": "fr",
                                 "score": 0.5, "text": "texte loi 2 0"}]


@pytest.mark.parametrize("arguments", [
    {"query": "q", "k": 11}, {"query": "q", "k": 0}, {"query": "x" * 1001}, {"query": ""},
    {"query": "q", "lang": "darija"},
])
def test_invalid_arguments_are_refused_before_searching(env, arguments):
    assert call("search_legal_texts", arguments)[0]
    assert env.searches == [] and env.logs() == []


def test_ask_returns_the_answer_with_numbered_citations(env):
    failed, body = call("ask_legal_question", {"question": "Quelle durée ?"})
    assert not failed
    assert (body["status"], body["confidence"], body["citations"][0]["number"]) == ("answered", 1.0, 2)


def test_ask_is_limited_per_minute(env):
    assert not any(call("ask_legal_question", {"question": "q"})[0] for _ in range(10))
    failed, message = call("ask_legal_question", {"question": "q"})
    assert failed and "Rate limit" in message
    assert len(env.logs()) == 10  # a refused call is not logged
    assert not call("search_legal_texts", {"query": "q"})[0]  # only questions are limited


def test_rate_limit_window_slides():
    clock = SimpleNamespace(now=0.0)
    limit = mcp_server.RateLimit(2, window=60, clock=lambda: clock.now)
    limit.check()
    limit.check()
    with pytest.raises(Exception, match="Rate limit"):
        limit.check()
    clock.now = 60.0
    limit.check()


@pytest.mark.parametrize("tool,arguments", [
    ("search_legal_texts", {"query": "q"}), ("ask_legal_question", {"question": "q"}),
    ("get_document_page", {"doc_id": "loi", "page": 1}), ("list_documents", {}),
])
def test_tools_say_loading_until_the_model_is_ready(env, monkeypatch, tool, arguments):
    service = RagService("unused.yaml")  # as at startup: state "loading"
    monkeypatch.setattr(mcp_server, "rag_service", service)
    failed, message = call(tool, arguments)
    assert failed and "still loading" in message and "retry" in message


def test_without_qdrant_url_nothing_is_loaded_or_searched(env, monkeypatch):
    started = []
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setattr(rag_service, "start", lambda: started.append(True))
    monkeypatch.setattr(mcp_server.server, "run", lambda *args, **kwargs: None)
    mcp_server.main([])
    failed, message = call("search_legal_texts", {"query": "q"})
    assert started == [] and failed and "QDRANT_URL" in message and env.searches == []


def test_start_loads_in_the_background_and_never_captures_text(monkeypatch):
    events = []
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setenv("LANGFUSE_CAPTURE_TEXT", "true")
    monkeypatch.setattr(rag_service, "start", lambda: events.append("start"))
    monkeypatch.setattr(mcp_server.server, "run",
                        lambda transport, **kwargs: events.append((transport, os.environ["LANGFUSE_CAPTURE_TEXT"])))
    mcp_server.main([])
    mcp_server.main(["--http", "--host", "0.0.0.0"])
    assert events == ["start", ("stdio", "false"), "start", ("streamable-http", "false")]


def test_get_document_page_returns_the_passages_in_order(env):
    failed, body = call("get_document_page", {"doc_id": "loi", "page": 2})
    assert not failed and body["passages"] == ["texte loi 2 0", "texte loi 2 1"]
    failed, message = call("get_document_page", {"doc_id": "loi", "page": 9})
    assert failed and "list_documents" in message
    assert call("get_document_page", {"doc_id": "loi", "page": 0})[0]


def test_list_documents_reads_every_page_of_the_index(env):
    failed, body = call("list_documents")
    assert not failed
    assert body["documents"] == [
        {"doc_id": "code", "doc_title": "Titre code", "languages": ["ar"], "pages_with_text": 1, "last_page": 1,
         "passages": 2},
        {"doc_id": "loi", "doc_title": "Titre loi", "languages": ["fr"], "pages_with_text": 2, "last_page": 3,
         "passages": 3},
    ]


def test_questions_are_logged_without_any_identifier(env):
    call("search_legal_texts", {"query": "comment créer une startup ?"})
    env.found = False
    call("search_legal_texts", {"query": "rien"})
    call("ask_legal_question", {"question": "Quelle durée ?"})
    call("get_document_page", {"doc_id": "loi", "page": 2})
    call("list_documents")

    first, second, third = env.logs()  # only the two tools that carry a question
    assert (first.endpoint, first.question, first.lang, first.status) == (
        "mcp_search", "comment créer une startup ?", "fr", "answered")
    assert second.status == "abstained" and (third.endpoint, third.status) == ("mcp_ask", "answered")
    for row in (first, second, third):
        assert (row.user_id, row.message_id, row.citations_count, row.reranker_score) == (None,) * 4
        assert row.latency_ms >= 0


def test_mcp_questions_stay_out_of_the_answer_rate_and_gaps(env):
    env.found = False
    call("search_legal_texts", {"query": "rien"})
    with env.sessions() as db:
        overview = get_overview(db)
        assert (overview["total_queries"], overview["unanswered_queries"]) == (1, 0)
        assert get_unanswered_queries(db) == []


@pytest.mark.parametrize("error,message", [
    (APIError(503, "rag_unavailable", "The document index is not reachable."), "not reachable"),
    (ValueError("SECRET question text"), "Internal server error"),
])
def test_failures_are_short_logged_and_leak_nothing(env, caplog, error, message):
    env.error = error
    failed, text = call("search_legal_texts", {"query": "SECRET question"})
    assert failed and message in text and "SECRET" not in text
    assert "SECRET" not in caplog.text
    assert [row.status for row in env.logs()] == ["error"]


def test_a_logging_failure_does_not_fail_the_tool(env, monkeypatch, caplog):
    def broken():
        raise ConnectionError("PostgreSQL is off: SECRET question")

    monkeypatch.setattr(mcp_server, "SessionLocal", broken)
    assert not call("search_legal_texts", {"query": "SECRET question"})[0]
    assert "Could not log the request (ConnectionError)" in caplog.text and "SECRET" not in caplog.text


def test_stdio_server_started_from_another_directory(tmp_path):
    """The launcher as Claude Desktop starts it: a clean protocol on stdout, no model needed to connect."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_mcp.py"
    environment = {
        **os.environ, "QDRANT_URL": "", "MLFLOW_TRACKING_URI": "", "LANGFUSE_PUBLIC_KEY": "",
        "LANGFUSE_SECRET_KEY": "", "DATABASE_URL": f"sqlite:///{(tmp_path / 'mcp.db').as_posix()}",
        "SECRET_KEY": "test",
    }
    parameters = StdioServerParameters(command=sys.executable, args=[str(script)], env=environment, cwd=tmp_path)

    async def run():
        with anyio.fail_after(90):
            async with Client(stdio_client(parameters)) as client:
                tools = (await client.list_tools()).tools
                result = await client.call_tool("list_documents", {})
                return {tool.name for tool in tools}, result

    names, result = anyio.run(run)
    assert names == TOOLS and result.is_error and "QDRANT_URL" in result.content[0].text


BACKEND = Path(__file__).resolve().parents[1]


def run_client(tmp_path, server, **environment):
    """scripts/mcp_client.py as a user runs it: it starts the server itself and speaks stdio."""
    env = {**os.environ, "PYTHONPATH": str(BACKEND), **environment}
    command = [sys.executable, str(BACKEND / "scripts" / "mcp_client.py"), "--server", str(server),
               "--server-log", str(tmp_path / "server.log")]
    return subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
                          timeout=180)


def test_client_script_calls_every_tool_over_stdio(tmp_path):
    done = run_client(tmp_path, BACKEND / "tests" / "mcp_fake_server.py")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "model still loading, retrying" in done.stdout  # the first call came too early
    for expected in ("== list_documents: 2 documents", "== search_legal_texts: 1 passages",
                     "== get_document_page: Titre loi, p. 2", "== ask_legal_question: answered",
                     "[1] Titre loi, p. 2", "all four tools answered"):
        assert expected in done.stdout


def test_client_script_fails_when_a_tool_fails(tmp_path):
    done = run_client(
        tmp_path, BACKEND / "scripts" / "run_mcp.py", QDRANT_URL="", MLFLOW_TRACKING_URI="", LANGFUSE_PUBLIC_KEY="",
        LANGFUSE_SECRET_KEY="", DATABASE_URL=f"sqlite:///{(tmp_path / 'mcp.db').as_posix()}", SECRET_KEY="test",
    )
    assert done.returncode == 1 and "list_documents failed" in done.stderr and "QDRANT_URL" in done.stderr
