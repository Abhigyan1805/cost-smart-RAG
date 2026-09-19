"""Critic signal: LLM-judge grade of the (query, evidence, answer) trace.

Pluggable :class:`CriticGrader` protocol -- bring any judge (strong LLM,
reward model, rubric script). Without a grader this raises
:data:`NotImplementedError` rather than fabricating a score, so ungraded
answers can never masquerade as verified. Score = grader's P(answer correct).
"""

from __future__ import annotations

from typing import Protocol, Sequence

from . import SignalResult


class CriticGrader(Protocol):
    """Judge: grade P(correct) for a (query, evidence, answer) trace."""

    def grade(self, query: str, evidence: Sequence[str], answer: str) -> float: ...


def score_critic(
    query: str,
    evidence: Sequence[str] | str,
    answer: str,
    grader: CriticGrader | None = None,
    cost: float = 20.0,
) -> SignalResult:
    """Grade the answer with *grader* (required; no heuristic fallback)."""
    if grader is None:
        raise NotImplementedError(
            "critic signal needs a grader (LLM judge): pass "
            "score_critic(..., grader=...) implementing "
            "CriticGrader.grade(query, evidence, answer) -> float in [0, 1]."
        )
    chunks = [evidence] if isinstance(evidence, str) else list(evidence)
    score = max(min(float(grader.grade(query, chunks, answer)), 1.0), 0.0)
    return SignalResult(
        signal="critic", score=score, cost=cost,
        details={"grader": type(grader).__name__, "num_chunks": len(chunks)},
    )
