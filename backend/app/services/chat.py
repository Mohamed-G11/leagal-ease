"""One chat exchange: save the question, ask the RAG pipeline, save the answer or the error."""
import logging
import re
import time

from groq import GroqError
from qdrant_client.http.exceptions import ApiException as QdrantApiException
from sqlalchemy.orm import Session

from app import tracing
from app.analytics.tracker import log_query
from app.core.errors import APIError
from app.core.experiment_config import LLMConfig
from app.core.schemas import Answer, ChatTurn
from app.generation.confidence import quote_in_chunk
from app.models.chat import ChatConversation, ChatMessage, MessageRole
from app.monitoring import chat_duration, chat_requests
from app.schemas.chat import ChatOut
from app.services.rag import rag_service, retrieved_chunks

logger = logging.getLogger(__name__)

# [3], [1][2], [1, 2] or 【3】 in a past answer, with the space before it
SOURCE_MARKERS = re.compile(r"[ \t]*(?:[\[【]\d+(?:\s*,\s*\d+)*[\]】])+")


def auto_title(message: str) -> str:
    title = message.strip()[:60]
    return title + "..." if len(message) > 60 else title


def get_or_create_conversation(
    db: Session, user_id: int, conversation_id: int | None, first_message: str, lang: str
) -> ChatConversation:
    """An unknown id, or one belonging to another user, starts a new conversation."""
    if conversation_id:
        conv = db.query(ChatConversation).filter(
            ChatConversation.id == conversation_id,
            ChatConversation.user_id == user_id,
        ).first()
        if conv:
            return conv
    conv = ChatConversation(user_id=user_id, title=auto_title(first_message), lang=lang)
    db.add(conv)
    db.commit()
    return conv


def save_message(db: Session, conversation_id: int, role: MessageRole, content: str, **fields) -> int:
    msg = ChatMessage(conversation_id=conversation_id, role=role, content=content, **fields)
    db.add(msg)
    db.flush()
    message_id = msg.id
    db.commit()  # also releases the connection while the pipeline runs
    return message_id


def load_history(db: Session, conversation_id: int, before_id: int, limit: int) -> list[ChatTurn]:
    """The last `limit` messages before this one, oldest first.

    A failed exchange is left out, question included: a retry would otherwise ask it twice. Past answers lose
    their [n] markers, which point at the sources of that turn, not at the ones the model is about to receive.
    """
    rows = (
        db.query(ChatMessage.role, ChatMessage.content, ChatMessage.status)
        .filter(ChatMessage.conversation_id == conversation_id, ChatMessage.id < before_id)
        .order_by(ChatMessage.id.desc())
        .all()
    )
    db.commit()  # releases the connection while the pipeline runs
    turns, failed_questions = [], 0
    for role, content, status in rows:  # newest first
        if len(turns) >= limit:
            break
        if status == "error":
            failed_questions += 1
        elif role == MessageRole.user and failed_questions:
            failed_questions -= 1
        else:
            if role == MessageRole.assistant:
                content = SOURCE_MARKERS.sub("", content)
            turns.append(ChatTurn(role=role.value, content=content))
    return turns[::-1]


def history_limit(pipeline) -> int:
    """As many messages as the served config keeps."""
    cfg = getattr(pipeline, "cfg", None)
    return (cfg.llm if cfg else LLMConfig()).max_history_messages


def _to_api_error(exc: Exception) -> APIError:
    if isinstance(exc, APIError):
        return exc
    if isinstance(exc, GroqError):
        return APIError(502, "llm_error", "The language model could not answer. Try again.")
    if isinstance(exc, QdrantApiException):
        return APIError(503, "rag_unavailable", "The document index is not reachable.")
    return APIError(500, "internal_error", "Internal server error.")


def record_request(db: Session, **fields) -> None:
    """Write the request log. A failure here must not cost the user their answer."""
    try:
        log_query(db, category=None, docs_found=0, **fields)
    except Exception:
        db.rollback()
        logger.warning("Could not log the chat request", exc_info=True)


def record_public_request(db: Session, question: str, lang: str, status: str, latency_ms: int) -> None:
    """Anonymous visitors: no user id, IP address or anything else about who asked."""
    record_request(
        db, question=question, lang=lang, endpoint="public", status=status, answered=status == "answered",
        latency_ms=latency_ms,
    )


def numbered_citations(answer: Answer) -> list[dict]:
    """Each citation with the [n] the answer text uses for it: its position among the retrieved chunks."""
    numbers = {}
    for number, scored in enumerate(retrieved_chunks.get(), 1):
        numbers.setdefault(scored.chunk.chunk_id, number)
    return [{**c.model_dump(), "number": numbers.get(c.chunk_id)} for c in answer.citations]


def verified_citations(answer: Answer) -> int | None:
    """How many quotes really appear in the chunk they cite. None when the chunks are not known."""
    if not answer.citations:
        return 0
    texts = {scored.chunk.chunk_id: scored.chunk.text for scored in retrieved_chunks.get()}
    if not texts:
        return None
    return sum(quote_in_chunk(c.quote, texts.get(c.chunk_id, "")) for c in answer.citations)


def ask(
    db: Session, user_id: int, message: str, conversation_id: int | None, lang: str, endpoint: str = "chat"
) -> ChatOut:
    """One traced exchange. Raises APIError when the pipeline cannot answer, after saving that failure.

    The error carries the conversation_id, so that a retry can continue the same conversation.
    """
    conv_id = get_or_create_conversation(db, user_id, conversation_id, message, lang).id
    question_id = save_message(db, conv_id, MessageRole.user, message)

    with tracing.observe("chat", session_id=str(conv_id), input=tracing.text(message),
                         metadata={"endpoint": endpoint, "lang": lang, **rag_service.model_info()}) as trace:
        return _answer(db, trace, user_id, message, question_id, conv_id, lang, endpoint)


def _answer(
    db: Session, trace: tracing.Observation, user_id: int, message: str, question_id: int, conv_id: int,
    lang: str, endpoint: str,
) -> ChatOut:
    retrieved_chunks.set(())
    started = time.perf_counter()
    try:
        with chat_duration.time():
            pipeline = rag_service.ready_pipeline()
            history = load_history(db, conv_id, question_id, history_limit(pipeline))
            answer = pipeline.answer(message, history=history)
    except Exception as exc:
        chat_requests.labels(outcome="error").inc()
        error = _to_api_error(exc)
        if error is not exc:
            logger.exception("RAG pipeline failed in conversation %s", conv_id)
        latency_ms = round((time.perf_counter() - started) * 1000)
        message_id = save_message(
            db, conv_id, MessageRole.assistant, error.detail, status="error", error_code=error.code
        )
        record_request(
            db, question=message, lang=lang, user_id=user_id, endpoint=endpoint, answered=False,
            status="error", error_code=error.code, citations_count=0, latency_ms=latency_ms,
            message_id=message_id,
        )
        trace.update(metadata={"status": "error", "error_code": error.code, "citations_count": 0,
                               "message_id": message_id})
        details = {"conversation_id": conv_id}
        raise APIError(error.status_code, error.code, error.detail, details) from exc

    latency_ms = round((time.perf_counter() - started) * 1000)
    chat_requests.labels(outcome=answer.status).inc()
    citations = numbered_citations(answer)
    message_id = save_message(
        db, conv_id, MessageRole.assistant, answer.answer,
        status="abstained" if answer.status == "abstained" else "ok",
        citations=citations or None,
        confidence=answer.confidence,
    )
    record_request(
        db, question=message, lang=answer.lang, user_id=user_id, endpoint=endpoint,
        answered=answer.status == "answered", status=answer.status, citations_count=len(answer.citations),
        latency_ms=latency_ms, reranker_score=answer.retrieval_score,
        verified_citations_count=verified_citations(answer), message_id=message_id,
    )
    trace.update(output=tracing.text(answer.answer), metadata={
        "status": answer.status, "citations_count": len(answer.citations), "message_id": message_id,
    })
    return ChatOut(
        **answer.model_dump(exclude={"citations"}), citations=citations,
        conversation_id=conv_id, message_id=message_id,
    )
