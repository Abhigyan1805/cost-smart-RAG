"""Ensemble + calibration: fuse signals, then calibrate for AUROC/ECE analysis.

- :func:`combine`: cost-aware weighted mean of signal scores. Each signal's
  weight defaults to ``1 / (1 + cost)`` so expensive judges don't dominate
  unless the caller says so via explicit ``weights``.
- :class:`Calibrator`: temperature scaling fitted on a validation set of
  (score, label) pairs (grid search on NLL); :meth:`Calibrator.predict`
  maps raw scores to calibrated probabilities.
- :func:`auroc` / :func:`ece`: stdlib AUROC (Mann-Whitney) and expected
  calibration error (uniform bins) for the project's core measurement.
- :func:`to_record`: logging schema binding a score to its label later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from . import SignalResult


def combine(
    signals: Sequence[SignalResult],
    weights: Mapping[str, float] | None = None,
) -> SignalResult:
    """Fuse *signals* into one ensemble score (cost-aware weighted mean)."""
    if not signals:
        return SignalResult(signal="ensemble", score=0.0, cost=0.0,
                            details={"num_signals": 0})
    total_w, total = 0.0, 0.0
    parts: dict[str, float] = {}
    for sig in signals:
        w = float(weights[sig.signal]) if weights and sig.signal in weights else 1.0 / (1.0 + sig.cost)
        parts[sig.signal] = sig.score
        total_w += w
        total += w * sig.score
    score = total / total_w if total_w else 0.0
    return SignalResult(
        signal="ensemble", score=score,
        cost=sum(s.cost for s in signals),
        details={"num_signals": len(signals), "parts": parts},
    )


@dataclass
class Calibrator:
    """Temperature scaling: calibrated P = sigmoid(logit(score) / T).

    Fit ``T`` on validation (score, label) pairs by NLL grid search.
    ``T > 1`` softens overconfidence, ``T < 1`` sharpens underconfidence.
    """

    temperature: float = 1.0
    n_grid: int = 50
    fitted: bool = False
    history: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def _logit(p: float) -> float:
        p = max(min(p, 1 - 1e-6), 1e-6)
        return math.log(p / (1 - p))

    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(min(z, 500.0), -500.0)))

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "Calibrator":
        """Fit temperature on validation scores/labels (raises on empty input)."""
        if len(scores) != len(labels) or not scores:
            raise ValueError("scores and labels must be non-empty and parallel")
        logits = [self._logit(s) for s in scores]
        best_t, best_nll = 1.0, math.inf
        for i in range(1, self.n_grid + 1):
            t = 0.05 + (5.0 - 0.05) * i / self.n_grid
            nll = 0.0
            for z, y in zip(logits, labels):
                p = self._sigmoid(z / t)
                nll -= y * math.log(max(p, 1e-12)) + (1 - y) * math.log(max(1 - p, 1e-12))
            if nll < best_nll:
                best_t, best_nll = t, nll
        self.temperature = best_t
        self.fitted = True
        self.history = {"temperature": best_t, "nll": best_nll, "n": len(scores)}
        return self

    def predict(self, score: float | Sequence[float]) -> float | list[float]:
        """Map raw score(s) to calibrated probabilities."""
        single = isinstance(score, (int, float))
        out = [self._sigmoid(self._logit(float(s)) / self.temperature)
               for s in ([score] if single else score)]
        return out[0] if single else out


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC via the Mann-Whitney rank statistic (0.5 = chance)."""
    pos = sorted((s for s, y in zip(scores, labels) if y == 1))
    n_pos, n_neg = len(pos), len(labels) - len(pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auroc needs both positive and negative labels")
    all_sorted = sorted(scores)
    # Tie-safe path: average ranks (correct for small eval sets).
    ranks: dict[float, float] = {}
    i = 0
    while i < len(all_sorted):
        j = i
        while j < len(all_sorted) and all_sorted[j] == all_sorted[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[all_sorted[i]] = avg
        i = j
    rank_sum = sum(ranks[s] for s in pos)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ece(
    scores: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> float:
    """Expected calibration error with uniform bins."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and parallel")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for idx, s in enumerate(scores):
        b = min(int(float(s) * n_bins), n_bins - 1)
        bins[b].append(idx)
    total = len(scores)
    err = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(labels[i] for i in b) / len(b)
        conf = sum(scores[i] for i in b) / len(b)
        err += len(b) / total * abs(acc - conf)
    return err


def to_record(
    signal: SignalResult,
    label: int | None = None,
    query_id: str | None = None,
) -> dict:
    """Logging schema for later AUROC/ECE analysis (label filled in at eval)."""
    return {"query_id": query_id, "signal": signal.signal,
            "y_score": signal.score, "y_true": label,
            "cost": signal.cost, "details": dict(signal.details)}
