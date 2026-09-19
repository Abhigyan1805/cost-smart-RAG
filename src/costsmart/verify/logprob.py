"""Logprob signal: token-likelihood confidence (cheapest verifier).

Maps the mean token logprob of the generated answer to P(correct) via
``exp(mean_logprob)`` (already in [0, 1]). Optional length penalty downweights
long, hedging answers. Pure arithmetic: ~zero cost.
"""

from __future__ import annotations

import math
from typing import Sequence

from . import SignalResult


def confidence_from_logprobs(
    token_logprobs: Sequence[float],
    length_penalty: float = 0.0,
    cost: float = 0.0,
) -> SignalResult:
    """Score an answer from its per-token logprobs.

    ``length_penalty`` multiplies the score by ``n ** -penalty`` to discount
    verbose answers; 0 disables it. Empty input scores 0.0 (no evidence).
    """
    toks = [float(t) for t in token_logprobs]
    if not toks:
        return SignalResult(signal="logprob", score=0.0, cost=cost,
                            details={"n_tokens": 0})
    mean_lp = sum(toks) / len(toks)
    score = math.exp(mean_lp)  # in (0, 1]
    if length_penalty:
        score *= len(toks) ** -length_penalty
    score = max(min(score, 1.0), 0.0)
    return SignalResult(
        signal="logprob", score=score, cost=cost,
        details={"n_tokens": len(toks), "mean_logprob": mean_lp,
                 "length_penalty": length_penalty},
    )
