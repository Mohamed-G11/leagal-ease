# app/analytics/tracker.py
# Adapted from colleague's analytics/tracker.py — persisted to PostgreSQL
from collections import defaultdict

from sqlalchemy import desc, func, or_, select
from sqlalchemy.orm import Session

from app.models.analytics import AIFeedback, AIQueryLog
from app.models.chat import ChatMessage


def log_query(
    db: Session,
    question: str,
    lang: str,
    docs_found: int,
    category: str | None,
    answered: bool,
    user_id: int | None = None,
    endpoint: str = "stream",
    **details,  # status, error_code, citations_count, verified_citations_count, latency_ms, reranker_score,
    #             message_id
) -> None:
    """Log every AI query to DB for analytics."""
    entry = AIQueryLog(
        user_id    = user_id,
        question   = question[:500],   # truncate very long questions
        lang       = lang,
        category   = category,
        docs_found = docs_found,
        answered   = answered,
        endpoint   = endpoint,
        **details,
    )
    db.add(entry)
    db.commit()


# /chat/public rows: "answered" only means Groq replied, not that the documents held an answer.
NOT_PUBLIC = AIQueryLog.endpoint.is_distinct_from("public")


def erase_conversation_text(db: Session, conversation_id: int, user_id: int) -> None:
    """Erase the question and answer text kept outside the conversation. The caller commits.

    Log rows keep their other fields, feedback rows keep the rating.
    """
    messages = select(ChatMessage.id).where(ChatMessage.conversation_id == conversation_id)
    db.query(AIQueryLog).filter(AIQueryLog.message_id.in_(messages)).update(
        {AIQueryLog.question: ""}, synchronize_session=False
    )
    # conversation_id comes from the client, so only the owner's ratings are matched.
    db.query(AIFeedback).filter(
        AIFeedback.user_id == user_id,
        or_(AIFeedback.conversation_id == conversation_id, AIFeedback.message_id.in_(messages)),
    ).update({AIFeedback.question: None, AIFeedback.answer: None}, synchronize_session=False)


def get_overview(db: Session) -> dict:
    """General stats — adapted from colleague's get_overview().

    total_queries counts every row; the answered figures and the rate leave out /chat/public.
    """
    total    = db.query(func.count(AIQueryLog.id)).scalar() or 0
    rated    = db.query(func.count(AIQueryLog.id)).filter(NOT_PUBLIC).scalar() or 0
    answered = (
        db.query(func.count(AIQueryLog.id)).filter(NOT_PUBLIC, AIQueryLog.answered.is_(True)).scalar() or 0
    )

    return {
        "total_queries":      total,
        "answered_queries":   answered,
        "unanswered_queries": rated - answered,
        "answer_rate":        round(answered / rated * 100, 1) if rated else 0,
    }


def get_language_stats(db: Session) -> dict:
    """Language distribution — adapted from colleague's get_language_stats()."""
    rows = (
        db.query(AIQueryLog.lang, func.count(AIQueryLog.id).label("count"))
        .group_by(AIQueryLog.lang)
        .order_by(desc("count"))
        .all()
    )
    return {
        "language_distribution": {r.lang or "unknown": r.count for r in rows}
    }


def get_procedure_stats(db: Session) -> dict:
    """Procedure category hits — adapted from colleague's get_procedure_stats()."""
    rows = (
        db.query(AIQueryLog.category, func.count(AIQueryLog.id).label("count"))
        .filter(AIQueryLog.category.isnot(None))
        .group_by(AIQueryLog.category)
        .order_by(desc("count"))
        .all()
    )
    return {
        "procedure_hits": {r.category: r.count for r in rows}
    }


def get_keyword_stats(db: Session, top_n: int = 20) -> dict:
    """Top keywords — adapted from colleague's get_keyword_stats()."""
    rows = db.query(AIQueryLog.question).all()
    keyword_counter: dict = defaultdict(int)
    for (question,) in rows:
        for word in question.lower().split():
            if len(word) > 3:
                keyword_counter[word] += 1
    top = sorted(keyword_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return {"top_keywords": dict(top)}


def get_recent_queries(db: Session, limit: int = 20) -> list:
    """Recent queries — adapted from colleague's get_recent_queries()."""
    rows = (
        db.query(AIQueryLog)
        .order_by(desc(AIQueryLog.created_at))
        .limit(limit)
        .all()
    )
    return [
        {
            "id":         r.id,
            "question":   r.question,
            "lang":       r.lang,
            "category":   r.category,
            "docs_found": r.docs_found,
            "answered":   r.answered,
            "endpoint":   r.endpoint,
            "status":          r.status,
            "error_code":      r.error_code,
            "citations_count": r.citations_count,
            "verified_citations_count": r.verified_citations_count,
            "latency_ms":      r.latency_ms,
            "reranker_score":  r.reranker_score,
            "message_id":      r.message_id,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        }
        for r in rows
    ]


def get_unanswered_queries(db: Session, limit: int = 20) -> list:
    """Queries the RAG couldn't answer — useful to detect dataset gaps."""
    rows = (
        db.query(AIQueryLog)
        .filter(NOT_PUBLIC, AIQueryLog.answered.is_(False), AIQueryLog.question != "")  # "" = erased by the user
        .order_by(desc(AIQueryLog.created_at))
        .limit(limit)
        .all()
    )
    return [
        {
            "question":   r.question,
            "lang":       r.lang,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        }
        for r in rows
    ]


def get_daily_stats(db: Session, days: int = 7) -> list:
    """Queries per day for the last N days."""
    from sqlalchemy import Date, cast
    rows = (
        db.query(
            cast(AIQueryLog.created_at, Date).label("day"),
            func.count(AIQueryLog.id).label("count"),
        )
        .group_by("day")
        .order_by("day")
        .limit(days)
        .all()
    )
    return [{"day": str(r.day), "count": r.count} for r in rows]
