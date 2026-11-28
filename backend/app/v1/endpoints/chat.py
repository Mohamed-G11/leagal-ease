"""POST /api/v1/chat: one question, one complete JSON answer."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.core.errors import ERROR_RESPONSES
from app.models.user import User
from app.schemas.chat import ChatIn, ChatOut
from app.services.chat import ask
from app.v1.endpoints.ai import detect_language

router = APIRouter(tags=["Chat"])


# A plain `def`: the pipeline blocks, so FastAPI must run this in its thread pool.
@router.post("/chat", response_model=ChatOut, responses=ERROR_RESPONSES, summary="Ask the legal assistant")
def chat(req: ChatIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Answers from the indexed Tunisian legal documents, with citations.

    `status` is `abstained` when the documents do not contain the answer. Every exchange is saved,
    including failures. The last messages of the conversation are given to the model as context
    (without failed exchanges).
    """
    return ask(db, current_user.id, req.message, req.conversation_id, detect_language(req.message))
