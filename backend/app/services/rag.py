"""The served RAG pipeline: loaded in the background at startup, switched by an admin reload.

It comes from the model registry (the champion version) when MLFLOW_TRACKING_URI is set, and from the
RAG_CONFIG file otherwise, or when the registry cannot provide a usable model.
"""
import logging
import threading
from contextvars import ContextVar
from pathlib import Path

from app import tracing
from app.core.config import settings  # also loads .env, so QDRANT_URL and GROQ_API_KEY are in the environment
from app.core.errors import APIError
from app.services import model_source

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

# The chunks retrieved for the current request, in order. Answer carries neither their texts nor the source
# numbers. A context variable: concurrent requests do not see each other's value.
retrieved_chunks: ContextVar[tuple] = ContextVar("retrieved_chunks", default=())  # of ScoredChunk


class RagService:
    def __init__(self, config_path: str):
        self.config_path = REPO_ROOT / config_path  # an absolute path stays as it is
        self.state = "loading"  # loading -> ready | failed
        self.cfg = None
        self.pipeline = None
        self.source = None  # "registry" | "config"
        self.version = None  # registry version, when the source is the registry
        self._thread = None
        self._probe = None
        self._loading = threading.Lock()  # one load or reload at a time

    def start(self) -> None:
        """Load in a background thread so the server answers while the embedding model loads."""
        if self._thread is None:
            self._thread = threading.Thread(target=self.load, name="rag-loader", daemon=True)
            self._thread.start()

    def load(self) -> None:
        """The registry champion, else the config file."""
        with self._loading:
            problem = self._load_champion()
            if problem:
                logger.info("Serving the config file: %s", problem)
                self._load_config()

    def reload(self) -> str | None:
        """Switch to the current champion. Returns why the champion is not served, or None.

        What is being served keeps serving until the new pipeline is ready, and stays if it cannot be built.
        """
        if not self._loading.acquire(blocking=False):
            raise APIError(409, "conflict", "The assistant is already loading a model.")
        try:
            problem = self._load_champion()
            if problem and self.pipeline is None:
                self._load_config()
            return problem
        finally:
            self._loading.release()

    def _load_champion(self) -> str | None:
        if not model_source.enabled():
            return "MLFLOW_TRACKING_URI is not set"
        try:
            version = model_source.champion_version()
            if version is None:
                return "the registry has no champion"
            if (self.source, self.version) != ("registry", version):
                pipeline = model_source.load_version(version)
                self._serve(pipeline, pipeline.cfg, "registry", version)
            return None
        except Exception:
            logger.warning("Could not load the registry champion", exc_info=True)
            return "the registry champion could not be loaded (see the server log)"

    def _load_config(self) -> None:
        try:
            from app.core.experiment_config import load_config
            from app.pipeline import RAGPipeline

            cfg = load_config(self.config_path)
            self._serve(RAGPipeline(cfg), cfg, "config", None)
        except Exception:
            self.state = "failed"
            logger.exception("RAG pipeline failed to load from %s", self.config_path)

    def _serve(self, pipeline, cfg, source: str, version: str | None) -> None:
        self._observe_retrieval(pipeline)  # both before requests can see this pipeline
        tracing.instrument(pipeline)
        self.cfg, self.pipeline, self._probe = cfg, pipeline, None
        self.source, self.version, self.state = source, version, "ready"
        logger.info("RAG pipeline ready (%s from the %s, version %s)", getattr(cfg, "name", "?"), source, version)

    @staticmethod
    def _observe_retrieval(pipeline) -> None:
        """Wrap the retrieval calls to remember which chunks the generator receives.

        It numbers its sources in this order, which gives the [n] of each citation, and the chunk texts
        are what the quotes are checked against.
        """
        retriever = getattr(pipeline, "retriever", None)
        if retriever is None:
            return  # citations then have no number and are not checked
        retrieve = retriever.retrieve

        def retrieve_and_record(*args, **kwargs):
            chunks = retrieve(*args, **kwargs)
            retrieved_chunks.set(tuple(chunks))
            return chunks

        retriever.retrieve = retrieve_and_record

        # E2+: the pipeline calls retrieve_with_meta() (it returns the detected language too)
        if hasattr(retriever, "retrieve_with_meta"):
            retrieve_meta = retriever.retrieve_with_meta

            def retrieve_meta_and_record(*args, **kwargs):
                result = retrieve_meta(*args, **kwargs)
                retrieved_chunks.set(tuple(result.chunks))
                return result

            retriever.retrieve_with_meta = retrieve_meta_and_record

    def _qdrant_client(self):
        if self.cfg is None:
            return None
        if not self.cfg.qdrant.url:
            # Local mode allows one client per folder: reuse the pipeline's.
            return self.pipeline.retriever.client if self.pipeline else None
        if self._probe is None:
            from qdrant_client import QdrantClient

            self._probe = QdrantClient(url=self.cfg.qdrant.url, timeout=2, check_compatibility=False)
        return self._probe

    def qdrant_status(self) -> str:
        """ok | empty (collection missing or without points) | down | loading"""
        try:
            client = self._qdrant_client()
            if client is None:
                return "loading" if self.state == "loading" else "down"
            name = self.cfg.qdrant.collection
            if not client.collection_exists(name) or client.count(name).count == 0:
                return "empty"
            return "ok"
        except Exception:
            logger.warning("Qdrant check failed", exc_info=True)
            return "down"

    def model_info(self) -> dict:
        return {"source": self.source, "version": self.version, "config": getattr(self.cfg, "name", None)}

    def ready_pipeline(self):
        """The pipeline, or a 503 explaining why it cannot answer. Qdrant is checked on every call."""
        if self.state == "loading":
            raise APIError(503, "rag_loading", "The assistant is still starting. Try again in a moment.")
        if self.state == "failed":
            raise APIError(503, "rag_unavailable", "The assistant failed to start.")
        status = self.qdrant_status()
        if status != "ok":
            reason = "has no indexed documents" if status == "empty" else "is not reachable"
            raise APIError(503, "rag_unavailable", f"The document index {reason}.")
        return self.pipeline


rag_service = RagService(settings.RAG_CONFIG)
