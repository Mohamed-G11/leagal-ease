"""Model evaluation for Wiem's experiments (retrieval + light generation checks).

Hend owns the reference dataset and the RAGAS evaluation; this module only reads
her `seed.jsonl` format:
    {"id", "question", "reference_answer", "doc_id", "pages", "lang", "category"}
A retrieved chunk is relevant if it comes from the gold doc_id AND one of the gold pages.
Questions with category "unanswerable" are excluded from retrieval metrics and used
to measure abstention.
"""
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from app.core.schemas import ScoredChunk

UNANSWERABLE = "unanswerable"


def load_dataset(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_relevant(sc: ScoredChunk, item: dict) -> bool:
    return sc.chunk.doc_id == item.get("doc_id") and sc.chunk.page in set(item.get("pages") or [])


def first_relevant_rank(hits: list[ScoredChunk], item: dict) -> int | None:
    for rank, sc in enumerate(hits, 1):
        if is_relevant(sc, item):
            return rank
    return None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def evaluate_retrieval(retriever, dataset: list[dict], ks=(1, 3, 5, 10)) -> tuple[dict, list[dict]]:
    """Recall@k (hit rate), MRR, per-language breakdown and latency."""
    max_k = max(ks)
    rows, latencies = [], []
    for item in dataset:
        if item.get("category") == UNANSWERABLE:
            continue
        t0 = time.perf_counter()
        hits = retriever.retrieve(item["question"], k=max_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        rank = first_relevant_rank(hits, item)
        rows.append({
            "id": item["id"], "lang": item.get("lang"), "category": item.get("category"),
            "rank": rank, "top_chunks": [h.chunk.chunk_id for h in hits[:5]],
        })

    metrics = aggregate(rows, ks, prefix="")
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
    for lang, lang_rows in by_lang.items():
        metrics.update(aggregate(lang_rows, ks, prefix=f"{lang}_", only=(5,)))
    metrics["retrieval_latency_p50_ms"] = percentile(latencies, 0.5)
    metrics["retrieval_latency_p95_ms"] = percentile(latencies, 0.95)
    metrics["n_retrieval_questions"] = len(rows)
    return metrics, rows


def aggregate(rows: list[dict], ks, prefix: str, only=None) -> dict:
    if not rows:
        return {}
    out = {}
    for k in only or ks:
        out[f"{prefix}recall_at_{k}"] = sum(r["rank"] is not None and r["rank"] <= k for r in rows) / len(rows)
    if only is None:
        out[f"{prefix}mrr"] = statistics.mean(1 / r["rank"] if r["rank"] else 0 for r in rows)
    return out


def evaluate_generation(pipeline, dataset: list[dict]) -> tuple[dict, list[dict]]:
    """Cheap generation metrics (no LLM judge): abstention, citation hit, latency.
    Faithfulness / answer relevancy come from Hend's RAGAS run."""
    rows, latencies = [], []
    for item in dataset:
        t0 = time.perf_counter()
        ans = pipeline.answer(item["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        gold_pages = set(item.get("pages") or [])
        cited_gold = any(c.page in gold_pages and c.chunk_id.startswith(f"{item.get('doc_id')}_p")
                         for c in ans.citations)
        rows.append({
            "id": item["id"], "lang": item.get("lang"), "category": item.get("category"),
            "status": ans.status, "n_citations": len(ans.citations), "cited_gold": cited_gold,
            "answer": ans.answer,
        })

    unans = [r for r in rows if r["category"] == UNANSWERABLE]
    ans_rows = [r for r in rows if r["category"] != UNANSWERABLE]
    metrics = {
        "gen_latency_p50_ms": percentile(latencies, 0.5),
        "gen_latency_p95_ms": percentile(latencies, 0.95),
    }
    if unans:  # should abstain
        metrics["abstention_correct_rate"] = sum(r["status"] == "abstained" for r in unans) / len(unans)
    if ans_rows:  # should answer, and cite the right page
        metrics["false_abstention_rate"] = sum(r["status"] == "abstained" for r in ans_rows) / len(ans_rows)
        answered = [r for r in ans_rows if r["status"] == "answered"]
        if answered:
            metrics["citation_hit_rate"] = sum(r["cited_gold"] for r in answered) / len(answered)
    return metrics, rows
