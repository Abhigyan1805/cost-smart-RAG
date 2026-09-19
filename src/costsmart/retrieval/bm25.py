"""Okapi BM25 over the pilot chunks (stdlib only).

:class:`BM25Index` is built from :func:`build_bm25_stats` output, which is
also serialised into the JSON pilot index so retrieval needs no rebuild.
"""

from __future__ import annotations

import math
from collections import Counter

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_stats(texts: list[str]) -> dict:
    """Precompute per-doc term frequencies, doc frequencies, and avg length."""
    tokenized = [tokenize(t) for t in texts]
    doc_freq: dict[str, int] = {}
    term_freqs: list[dict[str, int]] = []
    doc_lens: list[int] = []
    for toks in tokenized:
        tf = dict(Counter(toks))
        term_freqs.append(tf)
        doc_lens.append(len(toks))
        for term in tf:
            doc_freq[term] = doc_freq.get(term, 0) + 1
    return {
        "num_docs": len(texts),
        "avgdl": (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0,
        "doc_freq": doc_freq,
        "term_freqs": term_freqs,
        "doc_lens": doc_lens,
        "k1": K1,
        "b": B,
    }


def _idf(term: str, stats: dict) -> float:
    n = stats["num_docs"]
    df = stats["doc_freq"].get(term, 0)
    return math.log(1 + (n - df + 0.5) / (df + 0.5))


def bm25_search(
    query: str, chunks: list[dict], stats: dict, top_k: int = 10
) -> list[dict]:
    """Rank ``chunks`` with Okapi BM25; ties broken by chunk order."""
    qterms = tokenize(query)
    k1, b = stats.get("k1", K1), stats.get("b", B)
    avgdl = stats.get("avgdl", 0.0) or 1.0
    scored: list[dict] = []
    for rank, (chunk, tf, dl) in enumerate(
        zip(chunks, stats["term_freqs"], stats["doc_lens"])
    ):
        score = 0.0
        for term in qterms:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + k1 * (1 - b + b * dl / avgdl)
            score += _idf(term, stats) * f * (k1 + 1) / denom
        scored.append(
            {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"], "score": score}
        )
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]
