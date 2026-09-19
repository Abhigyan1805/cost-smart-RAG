"""NLI signal: does the retrieved evidence entail the answer?

Pluggable :class:`NLIModel` protocol (any entailment scorer returning a
probability); ships with a token-overlap heuristic fallback so the loop runs
with no model weights. Score = entailment probability of (evidence -> claim).
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from . import SignalResult

_TOKEN = re.compile(r"[a-z0-9]+")


class NLIModel(Protocol):
    """Entailment scorer: P(premise entails hypothesis) in [0, 1]."""

    def entails(self, premise: str, hypothesis: str) -> float: ...


def heuristic_entailment(evidence: str, claim: str) -> float:
    """Token-recall fallback: share of claim tokens covered by the evidence."""
    claim_toks = set(_TOKEN.findall((claim or "").lower()))
    if not claim_toks:
        return 0.0
    ev_toks = set(_TOKEN.findall((evidence or "").lower()))
    stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
            "to", "and", "or", "it", "that", "this"}
    content = {t for t in claim_toks if t not in stop} or claim_toks
    return len(content & ev_toks) / len(content)


def score_nli(
    evidence: Sequence[str] | str,
    answer: str,
    model: NLIModel | None = None,
    cost: float = 0.0,
) -> SignalResult:
    """Score whether *evidence* entails *answer* (max over evidence chunks)."""
    chunks = [evidence] if isinstance(evidence, str) else list(evidence)
    if not chunks or not (answer or "").strip():
        return SignalResult(signal="nli", score=0.0, cost=cost,
                            details={"num_chunks": len(chunks)})
    scorer = model.entails if model is not None else heuristic_entailment
    best = max(float(scorer(chunk, answer)) for chunk in chunks)
    best = max(min(best, 1.0), 0.0)
    return SignalResult(
        signal="nli", score=best, cost=cost,
        details={"num_chunks": len(chunks),
                 "model": type(model).__name__ if model else "heuristic-overlap"},
    )
