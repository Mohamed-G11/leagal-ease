"""Qdrant helpers. Uses a named vector "dense" so E1 can add a "sparse" vector
to the same collection without breaking E0."""
import logging
import uuid
import warnings

from qdrant_client import QdrantClient, models

from app.core.experiment_config import QdrantConfig

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"


def get_client(cfg: QdrantConfig) -> QdrantClient:
    if cfg.url:
        return QdrantClient(url=cfg.url)
    return QdrantClient(path=cfg.path)  # local mode, no server needed


def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool = False) -> str:
    """Returns "created", "recreated" or "kept". Never drops data unless recreate=True."""
    if client.collection_exists(name):
        if not recreate:
            return "kept"
        dropped = client.count(name).count
        logger.warning("Dropping collection '%s' (%d points) before re-indexing", name, dropped)
        client.delete_collection(name)
        _create(client, name, dim)
        # Qdrant local mode on Windows can keep the old files open and reload them on create:
        # make sure the rebuilt collection is really empty.
        if client.count(name).count:
            client.delete(name, points_selector=models.FilterSelector(filter=models.Filter()))
        left = client.count(name).count
        if left:
            raise RuntimeError(f"Could not empty collection '{name}' ({left} points left); "
                               "stop the API or use the Qdrant server (QDRANT_URL)")
        return "recreated"
    _create(client, name, dim)
    return "created"


def recreate_collection(client: QdrantClient, name: str, dim: int) -> None:
    """Kept for compatibility: explicit drop + create."""
    ensure_collection(client, name, dim, recreate=True)


def _create(client: QdrantClient, name: str, dim: int) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        # IDF is computed by Qdrant over the whole collection (BM25, E1)
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    # payload indexes make metadata filtering fast
    with warnings.catch_warnings():  # local mode ignores indexes; server mode uses them
        warnings.simplefilter("ignore")
        for field in ("doc_id", "lang"):
            client.create_payload_index(name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)


def point_id(chunk_id: str) -> str:
    """Deterministic UUID: re-ingesting the same chunk overwrites it instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def build_filter(filters: dict | None) -> models.Filter | None:
    """{"lang": "fr", "doc_id": ["a", "b"]} -> Qdrant filter (AND between keys)."""
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        match = models.MatchAny(any=value) if isinstance(value, list) else models.MatchValue(value=value)
        conditions.append(models.FieldCondition(key=key, match=match))
    return models.Filter(must=conditions)
