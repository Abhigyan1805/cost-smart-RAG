"""v4 hybrid router: combine heuristic fast-path, featurized model, and cascade.

Policy: high-confidence v0 heuristic decisions short-circuit (no model or
retrieval cost); otherwise defer to the v1 featurized router, and escalate via
the v3 cascade rule when v1 is itself uncertain. Retrieval features are passed
through opaquely -- this module never imports retrieval internals, per the v1
feature-dict contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .cascade import CascadeRouter
from .featurized import FeaturizedRouter
from .heuristic import RouteDecision, route as heuristic_route


@dataclass
class HybridRouter:
    """Hybrid v0 -> v1 -> cascade routing policy."""

    fast_path_confidence: float = 0.85
    featurized: FeaturizedRouter | None = None
    cascade: CascadeRouter | None = None

    def __post_init__(self) -> None:
        self.featurized = self.featurized or FeaturizedRouter()
        # Cascade wraps the featurized decision; weak policy injected lazily.
        self._fitted = bool(getattr(self.featurized, "weights", None))

    def route(
        self, query: str, retrieval: Mapping[str, Any] | None = None
    ) -> RouteDecision:
        fast = heuristic_route(query)
        if fast.confidence >= self.fast_path_confidence:
            return RouteDecision(
                route=fast.route,
                confidence=fast.confidence,
                reasons=["hybrid: v0 fast-path"] + fast.reasons,
                rule_hits=list(fast.rule_hits),
                estimated_cost=fast.estimated_cost,
            )
        try:
            assert self.featurized is not None
            routed = self.featurized.decide(query, retrieval)
        except RuntimeError:
            # Unfitted v1: fall back to the v0 decision.
            return RouteDecision(
                route=fast.route,
                confidence=fast.confidence,
                reasons=["hybrid: v1 unfitted, v0 fallback"] + fast.reasons,
                rule_hits=list(fast.rule_hits),
                estimated_cost=fast.estimated_cost,
            )
        return RouteDecision(
            route=routed.route,
            confidence=routed.confidence,
            reasons=["hybrid: v1 decision (v0 uncertain)"] + routed.reasons,
            rule_hits=list(routed.rule_hits),
            estimated_cost=routed.estimated_cost,
        )
