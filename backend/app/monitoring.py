"""Prometheus metrics: HTTP metrics from the instrumentator, plus chat outcomes.

CPU and memory come from prometheus_client's process collector (process_cpu_seconds_total,
process_resident_memory_bytes), which works on Linux, so inside the container.
"""
from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

CHAT_OUTCOMES = ("answered", "abstained", "error")

chat_requests = Counter("legalease_chat_requests_total", "Chat exchanges by outcome.", ["outcome"])
for _outcome in CHAT_OUTCOMES:
    chat_requests.labels(outcome=_outcome)  # export 0 before the first exchange

chat_duration = Histogram(
    "legalease_chat_duration_seconds",
    "Time spent in the RAG pipeline per chat exchange, including failures.",
    buckets=(0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 60),
)


def setup_metrics(app: FastAPI) -> None:
    """Instrument every route and serve GET /metrics (no authentication, see the README)."""
    Instrumentator(
        should_group_status_codes=False,  # 401, 429, 503... instead of 4xx / 5xx
        excluded_handlers=["^/metrics$", "^/health$"],  # frequent probes would hide real traffic
    ).instrument(app).expose(app, include_in_schema=False)
