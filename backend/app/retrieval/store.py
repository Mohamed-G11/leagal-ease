"""Qdrant helpers. Uses a named vector "dense" so E1 can add a "sparse" vector
to the same collection without breaking E0."""
import uuid
import warnings

from qdrant_client import QdrantClient, models

from app.core.experiment_config import QdrantConfig

DENSE = "dense"


def get_client(cfg: QdrantConfig) -> QdrantClient:
    if cfg.url:
        return QdrantClient(url=cfg.url)
    return QdrantClient(path=cfg.path)  # local mode, no server needed


def recreate_collection(client: QdrantClient, name: str, dim: int) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
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
