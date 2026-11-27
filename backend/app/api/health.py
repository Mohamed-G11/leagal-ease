"""GET /health: API, PostgreSQL, Qdrant, pipeline, Groq key and served model. Groq itself is never called."""
import logging
import os
from typing import Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.rag import rag_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


class HealthChecks(BaseModel):
    api: Literal["ok"]
    postgres: Literal["ok", "down"]
    qdrant: Literal["ok", "down", "empty", "loading"]
    pipeline: Literal["ready", "loading", "failed"]
    groq_key: Literal["present", "missing"]


class ServedModel(BaseModel):
    source: Literal["registry", "config"] | None  # null while nothing is loaded
    version: str | None  # model registry version
    config: str | None  # experiment config name


class Health(BaseModel):
    status: Literal["ok", "degraded", "down"]
    checks: HealthChecks
    model: ServedModel


def postgres_status(db: Session) -> str:
    try:
        db.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        logger.warning("PostgreSQL check failed", exc_info=True)
        return "down"


@router.get("/health", response_model=Health, responses={503: {"model": Health, "description": "PostgreSQL is down"}})
def health(response: Response, db: Session = Depends(get_db)):
    checks = HealthChecks(
        api="ok",
        postgres=postgres_status(db),
        qdrant=rag_service.qdrant_status(),
        pipeline=rag_service.state,
        groq_key="present" if os.getenv("GROQ_API_KEY") else "missing",
    )
    model = ServedModel(**rag_service.model_info())
    if checks.postgres == "down":
        response.status_code = 503
        return Health(status="down", checks=checks, model=model)
    rag_ok = (checks.qdrant, checks.pipeline, checks.groq_key) == ("ok", "ready", "present")
    return Health(status="ok" if rag_ok else "degraded", checks=checks, model=model)
