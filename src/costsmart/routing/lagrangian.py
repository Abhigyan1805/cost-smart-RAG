"""Lagrangian budget selector: feasibility filter + lambda selection.

Each routing candidate carries ``(expected_quality, expected_cost)``. For a
price of quality ``lam`` the Lagrangian score ``quality - lam * cost`` ranks
candidates; raising ``lam`` shifts choices toward cheaper routes. Given an
average per-query budget, :func:`select_lambda` finds the smallest ``lam``
whose induced choices fit the budget (bisection on a monotone curve).

An :class:`AdaptiveLambdaController` stub tracks online spend and nudges
``lam`` up/down; the full closed-loop controller lands with telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    """One routable option for a query (e.g. a local or cloud route)."""

    name: str
    expected_quality: float  # 0..1, e.g. validation accuracy for this tier
    expected_cost: float  # same cost units as the budget


def feasibility_filter(
    candidates: list[Candidate], hard_budget: float
) -> list[Candidate]:
    """Drop candidates whose expected cost exceeds the hard per-query cap."""
    return [c for c in candidates if c.expected_cost <= hard_budget]


def lagrangian_score(quality: float, cost: float, lam: float) -> float:
    """Lagrangian ranking score: quality minus lambda-weighted cost."""
    return quality - lam * cost


def select_route(
    candidates: list[Candidate],
    lam: float,
    hard_budget: float | None = None,
) -> Candidate:
    """Pick the max Lagrangian-score candidate (feasibility-filtered first).

    Ties break toward the cheaper candidate. Raises :class:`ValueError` when
    nothing is feasible.
    """
    pool = feasibility_filter(candidates, hard_budget) if hard_budget is not None else list(candidates)
    if not pool:
        raise ValueError("no feasible candidate under hard_budget")
    return min(pool, key=lambda c: (-lagrangian_score(c.expected_quality, c.expected_cost, lam), c.expected_cost))


def mean_cost_for_lambda(
    workload: list[list[Candidate]], lam: float
) -> float:
    """Mean per-query cost if every query in *workload* routes via *lam*."""
    if not workload:
        return 0.0
    return sum(select_route(cands, lam).expected_cost for cands in workload) / len(workload)


def select_lambda(
    workload: list[list[Candidate]],
    budget: float,
    lam_lo: float = 0.0,
    lam_hi: float = 10.0,
    tol: float = 1e-4,
) -> float:
    """Smallest lambda whose induced mean cost fits *budget* (bisection).

    ``lam=0`` always picks max quality; if even that fits, returns 0. If even
    ``lam_hi`` overshoots, ``lam_hi`` is returned (caller should widen it or
    cut scope) -- the overshoot is the caller's signal, not an exception.
    """
    if not workload:
        raise ValueError("workload is empty")
    if mean_cost_for_lambda(workload, lam_lo) <= budget:
        return lam_lo
    lo, hi = lam_lo, lam_hi
    while mean_cost_for_lambda(workload, hi) > budget and hi < 1e6:
        hi *= 2.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if mean_cost_for_lambda(workload, mid) <= budget:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


@dataclass
class AdaptiveLambdaController:
    """Online lambda nudger (stub): tracks spend vs budget and adapts.

    Full closed-loop control (windowing, smoothing, anti-windup) lands with
    the telemetry slice; this stub implements the proportional core so the
    agent loop can already call it.
    """

    lam: float = 0.1
    step: float = 0.01
    lam_min: float = 0.0
    lam_max: float = 10.0
    history: list[float] = field(default_factory=list, repr=False)

    def update(self, spent: float, budget: float) -> float:
        """Nudge lambda up when over budget, down when under; return new lam."""
        if spent > budget:
            self.lam = min(self.lam + self.step, self.lam_max)
        elif spent < budget:
            self.lam = max(self.lam - self.step, self.lam_min)
        self.history.append(self.lam)
        return self.lam
