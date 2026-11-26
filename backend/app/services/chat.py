"""One chat exchange: save the question, ask the RAG pipeline, save the answer or the error."""
import logging

from groq import GroqError
from qdrant_client.http.exceptions import ApiException as QdrantApiException
from sqlalchemy.orm import Session

from app.core.errors import APIError
from app.models.chat import ChatConversation, ChatMessage, MessageRole
from app.schemas.chat import ChatOut
from app.services.rag import rag_service

logger = logging.getLogger(__name__)


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


def _to_api_error(exc: Exception) -> APIError:
    if isinstance(exc, APIError):
        return exc
    if isinstance(exc, GroqError):
        return APIError(502, "llm_error", "The language model could not answer. Try again.")
    if isinstance(exc, QdrantApiException):
        return APIError(503, "rag_unavailable", "The document index is not reachable.")
    return APIError(500, "internal_error", "Internal server error.")


def ask(db: Session, user_id: int, message: str, conversation_id: int | None, lang: str) -> ChatOut:
    """Raises APIError when the pipeline cannot answer, after saving that failure.

    The error carries the conversation_id, so that a retry can continue the same conversation.
    """
    conv_id = get_or_create_conversation(db, user_id, conversation_id, message, lang).id
    save_message(db, conv_id, MessageRole.user, message)

    try:
        answer = rag_service.ready_pipeline().answer(message)
    except Exception as exc:
        error = _to_api_error(exc)
        if error is not exc:
            logger.exception("RAG pipeline failed in conversation %s", conv_id)
        save_message(db, conv_id, MessageRole.assistant, error.detail, status="error", error_code=error.code)
        details = {"conversation_id": conv_id}
        raise APIError(error.status_code, error.code, error.detail, details) from exc

    message_id = save_message(
        db, conv_id, MessageRole.assistant, answer.answer,
        status="abstained" if answer.status == "abstained" else "ok",
        citations=[c.model_dump() for c in answer.citations] or None,
    )
    return ChatOut(**answer.model_dump(), conversation_id=conv_id, message_id=message_id)
