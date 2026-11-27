"""Langfuse tracing of chat requests. Does nothing unless both keys are set, and never fails a request.

Sent for each request: timings, model names, token counts, retrieved chunk ids and scores, status,
number of citations, conversation and message ids.

Question, prompt and answer text are sent only with LANGFUSE_CAPTURE_TEXT=true, and never for /chat/public:
what is in Langfuse is not erased when a user deletes a conversation.

The pipeline's own code is not modified: `instrument()` wraps methods on its objects.
"""
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_client = None


def enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def capture_text() -> bool:
    return os.getenv("LANGFUSE_CAPTURE_TEXT", "").strip().lower() == "true"


def text(value):
    """The value when text may be sent, else None."""
    return value if capture_text() else None


def _get_client():
    global _client
    if _client is None:
        from langfuse import Langfuse

        _client = Langfuse()  # reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL
    return _client


def _warn() -> None:
    logger.warning("Langfuse tracing failed (the request is not affected)", exc_info=True)


class Observation:
    """A Langfuse observation, or nothing. Every call is safe."""

    def __init__(self, span=None):
        self.span = span

    def update(self, **fields) -> None:
        if self.span is None:
            return
        try:
            self.span.update(**fields)
        except Exception:
            _warn()

    def end(self) -> None:
        if self.span is None:
            return
        try:
            self.span.end()
        except Exception:
            _warn()


def start(name: str, as_type: str = "span", **fields) -> Observation:
    """An observation that the caller ends. For async code, where a "current" observation does not fit."""
    if not enabled():
        return Observation()
    try:
        return Observation(_get_client().start_observation(name=name, as_type=as_type, **fields))
    except Exception:
        _warn()
        return Observation()


@contextmanager
def observe(name: str, as_type: str = "span", session_id: str | None = None, **fields):
    """Time a block. Observations opened inside it become its children."""
    opened = []
    observation = Observation()
    if enabled():
        try:
            client = _get_client()
            if session_id:
                from langfuse import propagate_attributes

                attributes = propagate_attributes(session_id=session_id, trace_name=name)
                attributes.__enter__()
                opened.append(attributes)
            current = client.start_as_current_observation(name=name, as_type=as_type, **fields)
            observation = Observation(current.__enter__())
            opened.append(current)
        except Exception:
            _warn()
    try:
        yield observation
    except Exception as exc:
        # Only the class name: error messages can quote the question
        observation.update(level="ERROR", status_message=type(exc).__name__)
        raise
    finally:
        for manager in reversed(opened):
            try:
                manager.__exit__(None, None, None)
            except Exception:
                _warn()


def shutdown() -> None:
    """Send what is still queued."""
    if _client is not None:
        try:
            _client.shutdown()
        except Exception:
            _warn()


def instrument(pipeline) -> None:
    """Trace retrieval, query rewriting, reranking and generation of a RAGPipeline."""
    retriever = getattr(pipeline, "retriever", None)
    _trace_chunks(retriever, "retrieve", "retrieval", "retriever")
    _trace_chunks(getattr(retriever, "reranker", None), "rerank", "reranking", "span")
    _trace_llm(getattr(retriever, "transformer", None), "query_rewrite")
    _trace_llm(getattr(pipeline, "generator", None), "generation")


def _trace_chunks(owner, method: str, name: str, as_type: str) -> None:
    """Wrap a method that returns scored chunks. Its arguments hold the question: they are not sent."""
    original = getattr(owner, method, None)
    if original is None:
        return

    def traced(*args, **kwargs):
        with observe(name, as_type) as observation:
            chunks = original(*args, **kwargs)
            if observation.span is not None:
                try:
                    found = [{"chunk_id": c.chunk.chunk_id, "score": round(float(c.score), 4)} for c in chunks]
                    observation.update(output=found, metadata={"chunks": len(found)})
                except Exception:
                    _warn()
            return chunks

    setattr(owner, method, traced)


def _trace_llm(owner, name: str) -> None:
    """Wrap the Groq call of the generator or the query rewriter, for the model and token counts."""
    completions = getattr(getattr(getattr(owner, "client", None), "chat", None), "completions", None)
    if completions is None:
        return
    original = completions.create

    def traced(*args, **kwargs):
        parameters = {k: kwargs[k] for k in ("temperature", "max_tokens") if k in kwargs}
        with observe(name, "generation", model=kwargs.get("model"), model_parameters=parameters,
                     input=text(kwargs.get("messages"))) as observation:
            response = original(*args, **kwargs)
            if observation.span is not None:
                try:
                    usage = response.usage
                    observation.update(
                        usage_details={"input": usage.prompt_tokens, "output": usage.completion_tokens},
                        output=text(response.choices[0].message.content),
                    )
                except Exception:
                    _warn()
            return response

    completions.create = traced
