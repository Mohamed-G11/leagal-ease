"""BGE-M3 dense embedder (singleton). E0 uses dense vectors only;
sparse vectors are added in E1 so the comparison stays clean."""
import numpy as np

from app.core.experiment_config import EmbeddingConfig


class DenseEmbedder:
    def __init__(self, cfg: EmbeddingConfig):
        from sentence_transformers import SentenceTransformer  # heavy import, lazy

        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model, device=cfg.device)
        self.dim = (
            self.model.get_embedding_dimension()
            if hasattr(self.model, "get_embedding_dimension")
            else self.model.get_sentence_embedding_dimension()
        )

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.tokenize(text))

    def embed(self, texts: list[str], prefix: str | None = None) -> np.ndarray:
        prefix = self.cfg.passage_prefix if prefix is None else prefix
        return self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,  # cosine similarity == dot product
            show_progress_bar=len(texts) > 50,
        )

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query], prefix=self.cfg.query_prefix)[0].tolist()


_CACHE: dict[str, DenseEmbedder] = {}


def get_embedder(cfg: EmbeddingConfig) -> DenseEmbedder:
    """One instance per model config (loading BGE-M3 twice would waste ~2 GB of RAM)."""
    key = cfg.model_dump_json()
    if key not in _CACHE:
        _CACHE[key] = DenseEmbedder(cfg)
    return _CACHE[key]
