"""Verifier signals: each returns a calibrated score + cost annotation.

Every signal returns :class:`SignalResult` with ``score`` in [0, 1] (higher =
more likely correct) plus the ``cost`` incurred to compute it. Scores are raw
model outputs meant for later AUROC/ECE analysis -- see
:mod:`costsmart.verify.ensemble` for calibration helpers and
:func:`~costsmart.verify.ensemble.to_record` for the logging schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SignalResult:
    """One verifier signal: calibrated score plus cost annotation."""

    signal: str  # e.g. "logprob", "self_consistency", "nli", "critic", "ensemble"
    score: float  # P(answer correct), in [0, 1]
    cost: float = 0.0  # cost units spent computing this signal
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")
        if self.cost < 0:
            raise ValueError(f"cost must be >= 0, got {self.cost}")
