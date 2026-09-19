"""Hybrid retrieval: BM25 + dense fused with reciprocal rank fusion (RRF).

:func:`hybrid_search` runs both legs over the pilot index chunks and fuses
them; it is the entry point the smoke run and (later) the router slice use.
"""

from __future__ import annotations

from costsmart.retrieval.bm25 import bm25_search
from costsmart.retrieval.dense import dense_search

# RRF constant. Pinned for the pilot.
RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[dict]], k: int = RRF_K, top_k: int = 10
) -> list[dict]:
    """Fuse ranked lists of ``{chunk_id, doc_id, score}`` dicts via RRF."""
    fused: dict[str, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            cid = hit["chunk_id"]
            entry = fused.setdefault(
                cid, {"chunk_id": cid, "doc_id": hit["doc_id"], "score": 0.0}
            )
            entry["score"] += 1.0 / (k + rank + 1)
    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:top_k]


def hybrid_search(
    query: str,
    chunks: list[dict],
    bm25_stats: dict,
    vectors: list[list[float]],
    top_k: int = 10,
    dense_k: int = 50,
    bm25_k: int = 50,
) -> list[dict]:
    """Run dense + BM25 legs and fuse with RRF."""
    dense_hits = dense_search(query, chunks, vectors, top_k=dense_k)
    bm25_hits = bm25_search(query, chunks, bm25_stats, top_k=bm25_k)
    return reciprocal_rank_fusion([dense_hits, bm25_hits], top_k=top_k)
