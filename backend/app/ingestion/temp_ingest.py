"""W2 — Temporary ingestion for E0/E1 (replaced by Hend's parser later).

Page by page -> chunks of ~max_tokens with overlap -> BGE-M3 dense + BM25 sparse -> Qdrant.
Each chunk keeps doc_id + page so answers can cite their source.

Run:
    python -m app.ingestion.temp_ingest --pdf-dir ../data/raw \
        --config ../experiments/E0_naive_rag/config.yaml
"""
import argparse
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path

import pymupdf
from qdrant_client import models

from app.core.experiment_config import ExperimentConfig, load_config
from app.core.schemas import Chunk

ARABIC = re.compile(r"[؀-ۿ]")
FRENCH_HINTS = {" le ", " la ", " les ", " des ", " du ", " et ", " article ", " pour ", " dans "}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟;:])\s+|\n{2,}")


def slugify(name: str) -> str:
    """Stable doc_id from a file name: 'Loi n° 2018-52.pdf' -> 'loi_n_2018_52'."""
    name = unicodedata.normalize("NFKD", Path(name).stem).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return slug or "doc"


def detect_lang(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if letters and sum(bool(ARABIC.match(c)) for c in letters) / len(letters) > 0.3:
        return "ar"
    padded = f" {text.lower()} "
    return "fr" if sum(h in padded for h in FRENCH_HINTS) >= 3 else "en"


def clean(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"-\n(\w)", r"\1", text)  # words cut at line end
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_tokens: int, overlap: int, count: Callable[[str], int]) -> list[str]:
    """Greedy sentence packing; overlap = trailing sentences carried to the next chunk."""
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s and s.strip()]
    chunks, current, current_tokens = [], [], 0
    for sent in sentences:
        n = count(sent)
        if current and current_tokens + n > max_tokens:
            chunks.append(" ".join(current))
            # keep the last sentences up to `overlap` tokens
            carry, carry_tokens = [], 0
            for s in reversed(current):
                t = count(s)
                if carry_tokens + t > overlap:
                    break
                carry.insert(0, s)
                carry_tokens += t
            current, current_tokens = carry, carry_tokens
        current.append(sent)
        current_tokens += n
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c) > 50]


def pdf_to_chunks(pdf_path: Path, cfg: ExperimentConfig, count: Callable[[str], int]) -> list[Chunk]:
    doc_id = slugify(pdf_path.name)
    chunks: list[Chunk] = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            text = clean(page.get_text("text"))
            if len(text) < 50:
                continue  # empty or scanned page (OCR comes later)
            lang = detect_lang(text)
            for i, piece in enumerate(
                chunk_text(text, cfg.chunking.max_tokens, cfg.chunking.overlap_tokens, count)
            ):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_p{page_index + 1}_{i}",
                        doc_id=doc_id,
                        doc_title=pdf_path.stem,
                        page=page_index + 1,
                        lang=lang,
                        text=piece,
                        metadata={"source_file": pdf_path.name},
                    )
                )
    return chunks


def ingest(pdf_dir: Path, cfg: ExperimentConfig, embedder=None) -> int:
    from app.retrieval.embedder import get_embedder
    from app.retrieval.sparse import BM25Encoder
    from app.retrieval.store import DENSE, SPARSE, get_client, point_id, recreate_collection

    embedder = embedder or get_embedder(cfg.embedding)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {pdf_dir}")

    all_chunks: list[Chunk] = []
    for pdf in pdfs:
        chunks = pdf_to_chunks(pdf, cfg, embedder.count_tokens)
        print(f"  {pdf.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    bm25 = BM25Encoder(**cfg.retrieval.bm25.model_dump())
    client = get_client(cfg.qdrant)
    recreate_collection(client, cfg.qdrant.collection, embedder.dim)

    batch = 64
    for start in range(0, len(all_chunks), batch):
        part = all_chunks[start : start + batch]
        vectors = embedder.embed([c.text for c in part])
        client.upsert(
            collection_name=cfg.qdrant.collection,
            points=[
                models.PointStruct(
                    id=point_id(c.chunk_id),
                    vector={DENSE: v.tolist(), SPARSE: bm25.encode_document(c.text)},
                    payload=c.model_dump(),
                )
                for c, v in zip(part, vectors)
            ],
        )
    print(f"Indexed {len(all_chunks)} chunks from {len(pdfs)} PDFs into '{cfg.qdrant.collection}'")
    return len(all_chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("../experiments/E0_naive_rag/config.yaml"))
    args = parser.parse_args()
    ingest(args.pdf_dir, load_config(args.config))
