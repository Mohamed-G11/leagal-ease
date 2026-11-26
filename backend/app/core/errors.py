"""One JSON shape for every API error.

    {"error": {"code": "not_found", "message": "..."}, "detail": "..."}

`detail` mirrors FastAPI's default so the frontend keeps working.
"""
import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
from sqlalchemy.exc import InterfaceError, OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

CODES_BY_STATUS = {
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    422: "validation_error",
    429: "rate_limited",
}
# slowapi's own message is the limit itself ("10 per 1 minute"). Our limits are all per minute.
RATE_LIMIT_MESSAGE = "Too many requests. Please wait a minute and try again."


class ErrorInfo(BaseModel):
    code: str
    message: str
    details: Any = None  # validation errors, or {"conversation_id": ...} for a failed chat


class ErrorResponse(BaseModel):
    error: ErrorInfo
    detail: Any


# For the `responses=` argument of a route, so that Swagger shows the error shape.
ERROR_RESPONSES = {
    status: {"model": ErrorResponse, "description": description}
    for status, description in {
        401: "unauthorized",
        422: "validation_error",
        500: "internal_error",
        502: "llm_error",
        503: "rag_loading, rag_unavailable or database_unavailable",
    }.items()
}


class APIError(HTTPException):
    """HTTPException with an explicit error code."""

    def __init__(self, status_code: int, code: str, message: str, details: dict | None = None):
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.details = details


def error_response(status_code: int, code: str, message: str, detail=None, details=None, headers=None) -> JSONResponse:
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    body = {"error": error, "detail": message if detail is None else detail}
    return JSONResponse(body, status_code=status_code, headers=headers)


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    status = exc.status_code
    code = getattr(exc, "code", None) or CODES_BY_STATUS.get(status) or (
        "bad_request" if status < 500 else "internal_error"
    )
    detail = RATE_LIMIT_MESSAGE if isinstance(exc, RateLimitExceeded) else exc.detail
    message = detail if isinstance(detail, str) else HTTPStatus(status).phrase
    details = getattr(exc, "details", None)
    return error_response(status, code, message, detail=detail, details=details, headers=exc.headers)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = jsonable_encoder(exc.errors())
    return error_response(422, "validation_error", "Invalid request.", detail=errors, details=errors)


async def database_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Database error on %s %s", request.method, request.url.path, exc_info=exc)
    return error_response(503, "database_unavailable", "The database is not available.")


async def catch_unhandled_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return error_response(500, "internal_error", "Internal server error.")


def register_error_handlers(app: FastAPI) -> None:
    """Call this before adding CORSMiddleware, so that 500 responses also get CORS headers."""
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(OperationalError, database_error_handler)
    app.add_exception_handler(InterfaceError, database_error_handler)
    # A middleware, not an `Exception` handler: Starlette runs the latter outside CORSMiddleware.
    app.middleware("http")(catch_unhandled_errors)
