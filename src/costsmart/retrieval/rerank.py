"""Cross-encoder reranking with a pinned bge-reranker model.

Pinned reranker: ``BAAI/bge-reranker-base``.

When ``sentence-transformers`` is installed, the real ``CrossEncoder`` is
used. Otherwise :func:`rerank` falls back to a query-term overlap scorer
(stdlib only) so the pipeline and the ``rerank_agreement`` router feature
work offline. Both paths return the input hits re-sorted with fresh
``rerank_score`` values; original positions are kept in ``base_rank``.
"""

from __future__ import annotations

# Pinned cross-encoder. Pinned; change needs a firstmate decision.
RERANKER_MODEL = "BAAI/bge-reranker-base"


def _overlap_scores(query: str, texts: list[str]) -> list[float]:
    qtokens = set(query.lower().split())
    scores: list[float] = []
    for text in texts:
        ttokens = set(text.lower().split())
        inter = len(qtokens & ttokens)
        union = len(qtokens | ttokens) or 1
        # Jaccard plus a recall-flavoured term so longer gold passages win.
        scores.append(inter / union + 0.5 * inter / (len(qtokens) or 1))
    return scores


def _ce_scores(query: str, texts: list[str]) -> list[float] | None:
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ImportError:
        return None
    try:
        model = CrossEncoder(RERANKER_MODEL)
        return list(map(float, model.predict([(query, t) for t in texts])))
    except Exception:
        return None


def rerank(query: str, hits: list[dict], chunk_lookup: dict[str, str]) -> list[dict]:
    """Rerank ``hits`` (each with ``chunk_id``) against ``query``.

    ``chunk_lookup`` maps ``chunk_id`` -> chunk text. Returns new dicts with
    ``rerank_score`` and ``base_rank`` (0-based position before reranking),
    sorted by ``rerank_score`` descending.
    """
    texts = [chunk_lookup.get(h["chunk_id"], "") for h in hits]
    scores = _ce_scores(query, texts)
    if scores is None:
        scores = _overlap_scores(query, texts)
    reranked = [
        {**h, "rerank_score": s, "base_rank": i} for i, (h, s) in enumerate(zip(hits, scores))
    ]
    reranked.sort(key=lambda r: r["rerank_score"], reverse=True)
    return reranked
