# app/v1/endpoints/feedback.py
# Add this router to ai.py or register separately in main.py

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.analytics import AIFeedback
from app.models.chat import ChatConversation, ChatMessage, MessageRole
from app.models.user import User

router = APIRouter(prefix="/ai/feedback", tags=["AI Feedback"])


class FeedbackRequest(BaseModel):
    rating:          str            # "up" or "down"
    question:        str | None  = None
    answer:          str | None  = None
    conversation_id: int | None  = None
    message_index:   int | None  = None
    message_id:      int | None  = None


def rated_message_id(db: Session, user_id: int, req: FeedbackRequest) -> int | None:
    """The assistant message being rated, only if it belongs to this user.

    The chat widget sends no message_id yet, so the answer is also looked up by its text in the conversation.
    """
    query = (
        db.query(ChatMessage.id)
        .join(ChatConversation)
        .filter(ChatConversation.user_id == user_id, ChatMessage.role == MessageRole.assistant)
    )
    if req.message_id is not None:
        query = query.filter(ChatMessage.id == req.message_id)
    elif req.conversation_id is not None and req.answer:
        query = query.filter(ChatConversation.id == req.conversation_id, ChatMessage.content == req.answer)
    else:
        return None
    row = query.order_by(ChatMessage.id.desc()).first()
    return row.id if row else None


@router.post("")
async def submit_feedback(
    req: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save 👍/👎 feedback on an AI response."""
    if req.rating not in ("up", "down"):
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="rating must be 'up' or 'down'")

    feedback = AIFeedback(
        user_id         = current_user.id,
        conversation_id = req.conversation_id,
        message_index   = req.message_index,
        question        = req.question[:500] if req.question else None,
        answer          = req.answer[:1000]  if req.answer  else None,
        rating          = req.rating,
        message_id      = rated_message_id(db, current_user.id, req),
    )
    db.add(feedback)
    db.commit()
    return {"status": "saved", "rating": req.rating}


@router.get("/stats")
async def feedback_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Feedback statistics — thumbs up vs down rate."""
    from sqlalchemy import func
    total = db.query(func.count(AIFeedback.id)).scalar() or 0
    ups   = db.query(func.count(AIFeedback.id)).filter(AIFeedback.rating == "up").scalar()   or 0
    downs = db.query(func.count(AIFeedback.id)).filter(AIFeedback.rating == "down").scalar() or 0

    # Get worst-rated answers (thumbs down)
    worst = (
        db.query(AIFeedback)
        .filter(AIFeedback.rating == "down")
        .order_by(AIFeedback.created_at.desc())
        .limit(10)
        .all()
    )

    return {
        "total":      total,
        "thumbs_up":  ups,
        "thumbs_down": downs,
        "satisfaction_rate": round(ups / total * 100, 1) if total else 0,
        "worst_answers": [
            {
                "question":   f.question,
                "answer":     f.answer[:200] if f.answer else None,
                "created_at": f.created_at.isoformat() if f.created_at else "",
            }
            for f in worst
        ],
    }
