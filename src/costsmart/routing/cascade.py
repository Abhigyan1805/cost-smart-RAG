"""v3 cascade router: try cheap first, escalate on low confidence.

The cascade wraps any weak (cheap) routing policy: if the weak policy's
confidence clears ``escalate_below``, its decision stands; otherwise the query
escalates to the strong route. Escalations and their cost delta are recorded
so the eval harness can measure cascade savings vs always-strong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .heuristic import CLOUD, COST, LOCAL, RouteDecision, route as heuristic_route

WeakPolicy = Callable[[str], RouteDecision]


@dataclass
class CascadeResult:
    decision: RouteDecision
    escalated: bool
    weak_confidence: float
    extra_cost: float  # cost above the weak route actually incurred


@dataclass
class CascadeRouter:
    """Cheap-first router with confidence-triggered escalation."""

    weak_policy: WeakPolicy = heuristic_route
    escalate_below: float = 0.65
    weak_route: str = LOCAL
    strong_route: str = CLOUD
    history: list[CascadeResult] = field(default_factory=list, repr=False)

    def route(self, query: str) -> CascadeResult:
        weak = self.weak_policy(query)
        if weak.confidence >= self.escalate_below and weak.route == self.weak_route:
            res = CascadeResult(weak, escalated=False,
                                weak_confidence=weak.confidence, extra_cost=0.0)
        else:
            strong = RouteDecision(
                route=self.strong_route,
                confidence=weak.confidence,
                reasons=[f"cascade-escalation: weak={weak.route} "
                         f"conf={weak.confidence:.3f} < {self.escalate_below}"],
                rule_hits=list(weak.rule_hits),
                estimated_cost=COST[self.strong_route],
            )
            res = CascadeResult(strong, escalated=True,
                                weak_confidence=weak.confidence,
                                extra_cost=COST[self.strong_route] - COST[self.weak_route])
        self.history.append(res)
        return res

    @property
    def escalation_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(1 for r in self.history if r.escalated) / len(self.history)
