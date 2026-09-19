"""Post-retrieval router features (free difficulty signals).

Retrieval runs before routing by design, so this module exposes the score-
distribution signals the router slice consumes -- computed from retrieval
outputs only, with no extra model calls:

  * ``top1_score``         -- top hit score
  * ``mean_topk``          -- mean of the top-k scores
  * ``score_std``          -- population std of the top-k scores
  * ``score_gap``          -- top1 score minus top5 score (0.0 when k < 5)
  * ``n_distinct_docs``    -- distinct ``doc_id`` values in the top-k
  * ``score_entropy``      -- entropy of the softmax-normalised top-k scores
  * ``rerank_agreement``   -- Spearman rho between base rank and reranked
                              rank over the reranked set
  * ``max_passage_overlap``-- max query/chunk token overlap ratio in top-k

:func:`extract_routing_features` returns all eight in one dict.
"""

from __future__ import annotations

import math

FEATURE_NAMES = [
    "top1_score",
    "mean_topk",
    "score_std",
    "score_gap",
    "n_distinct_docs",
    "score_entropy",
    "rerank_agreement",
    "max_passage_overlap",
]


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (average ranks for ties), stdlib only."""
    n = len(xs)
    if n < 2:
        return 1.0

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based average rank
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return 1.0 if vx == vy else 0.0
    return cov / math.sqrt(vx * vy)


def token_overlap_ratio(query: str, text: str) -> float:
    """Fraction of distinct query tokens appearing in ``text``."""
    qtokens = set(query.lower().split())
    if not qtokens:
        return 0.0
    ttokens = set(text.lower().split())
    return len(qtokens & ttokens) / len(qtokens)


def extract_routing_features(
    query: str,
    hits: list[dict],
    reranked: list[dict] | None = None,
    chunk_lookup: dict[str, str] | None = None,
    top_k: int = 10,
) -> dict:
    """Compute the eight router difficulty signals for one query.

    ``hits`` are base retrieval results with ``score`` (and ``doc_id``);
    ``reranked`` optionally carries ``base_rank`` + ``rerank_score`` from
    :func:`costsmart.retrieval.rerank.rerank` for ``rerank_agreement``.
    ``chunk_lookup`` maps ``chunk_id`` -> text for ``max_passage_overlap``.
    """
    hits = hits[:top_k]
    scores = [float(h.get("score", 0.0)) for h in hits]
    n = len(scores)

    top1 = scores[0] if scores else 0.0
    mean = (sum(scores) / n) if n else 0.0
    var = (sum((s - mean) ** 2 for s in scores) / n) if n else 0.0
    gap = (scores[0] - scores[4]) if n >= 5 else 0.0
    n_docs = len({h.get("doc_id") for h in hits})
    probs = _softmax(scores)
    entropy = -sum(p * math.log(p) for p in probs if p > 0)

    if reranked:
        # Align by chunk_id: base position vs reranked position.
        base_pos = {h["chunk_id"]: i for i, h in enumerate(hits)}
        xs, ys = [], []
        for j, r in enumerate(reranked):
            cid = r["chunk_id"]
            if cid in base_pos:
                xs.append(float(base_pos[cid]))
                ys.append(float(j))
        agreement = _spearman(xs, ys) if len(xs) >= 2 else 1.0
    else:
        agreement = 1.0

    lookup = chunk_lookup or {}
    overlap = 0.0
    for h in hits:
        text = lookup.get(h.get("chunk_id", ""), "")
        if text:
            overlap = max(overlap, token_overlap_ratio(query, text))

    return {
        "top1_score": top1,
        "mean_topk": mean,
        "score_std": math.sqrt(var),
        "score_gap": gap,
        "n_distinct_docs": n_docs,
        "score_entropy": entropy,
        "rerank_agreement": agreement,
        "max_passage_overlap": overlap,
    }
