"""Self-consistency signal: sample N answers, score by majority agreement.

A confident model converges on one answer across samples; dispersion signals
uncertainty. Score = share of samples matching the majority (normalized)
answer. Cost = ``num_samples * per_sample_cost`` passed in by the caller, so
the agent loop can budget it.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Sequence

from . import SignalResult


def normalize(answer: str) -> str:
    """Normalise an answer for comparison: lowercase, strip, collapse space."""
    return re.sub(r"\s+", " ", (answer or "").strip().lower())


def agreement_score(
    samples: Sequence[str],
    normalize_fn: Callable[[str], str] = normalize,
    per_sample_cost: float = 1.0,
) -> SignalResult:
    """Score the agreement across sampled answers (majority share)."""
    normed = [normalize_fn(s) for s in samples]
    if not normed:
        return SignalResult(signal="self_consistency", score=0.0, cost=0.0,
                            details={"num_samples": 0})
    top, count = Counter(normed).most_common(1)[0]
    score = count / len(normed)
    return SignalResult(
        signal="self_consistency", score=score,
        cost=len(normed) * per_sample_cost,
        details={"num_samples": len(normed), "num_unique": len(set(normed)),
                 "majority": top, "majority_count": count},
    )
