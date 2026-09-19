"""Agent loop: capped state machine over ACCEPT/RETRIEVE/ESCALATE/CLARIFY.

- At most ``max_iterations`` verify-decide cycles (default 3).
- Dedup guard: repeat retrieval queries are served from cache (no second
  charge, logged as ``dedup_hit``) so a looping retriever cannot burn budget.
- Budget exhaustion: when the next step is unaffordable the loop stops and
  returns the best answer so far flagged ``low_confidence`` (never fails).
- Every event is logged with a ``trace_id`` (uuid4, injectable for tests).

Injectable callbacks keep this slice independent of corpus/retrieval and
model clients: ``answer_fn(query, evidence)``, ``verify_fn(answer,
evidence)``, ``retrieve_fn(query)``, ``escalate_fn(query, evidence)``.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..verify import SignalResult
from .actions import Action, ActionDecision

Outcome = str  # ACCEPT | LOW_CONFIDENCE | BUDGET_EXHAUSTED | NEEDS_CLARIFICATION

DEFAULT_COSTS = {"answer": 1.0, "verify": 0.5, "retrieve": 2.0, "escalate": 20.0}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


@dataclass
class TraceEvent:
    trace_id: str
    iteration: int
    action: str
    detail: str
    cost: float = 0.0


@dataclass
class LoopResult:
    trace_id: str
    query: str
    final_answer: str
    outcome: Outcome
    low_confidence: bool
    iterations_used: int
    total_cost: float
    decisions: list[ActionDecision] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    dedup_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id, "query": self.query,
            "final_answer": self.final_answer, "outcome": self.outcome,
            "low_confidence": self.low_confidence,
            "iterations_used": self.iterations_used,
            "total_cost": self.total_cost, "dedup_hits": self.dedup_hits,
            "decisions": [ {"action": d.action.value, "rationale": d.rationale,
                            "confidence": d.confidence, "iteration": d.iteration,
                            "extra": d.extra} for d in self.decisions ],
            "trace": [ {"trace_id": e.trace_id, "iteration": e.iteration,
                        "action": e.action, "detail": e.detail,
                        "cost": e.cost} for e in self.trace ],
        }


class AgentLoop:
    """Capped verify-then-act loop with budget and dedup guards."""

    def __init__(
        self,
        max_iterations: int = 3,
        budget: float = 50.0,
        accept_threshold: float = 0.7,
        costs: dict[str, float] | None = None,
        trace_id: str | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.max_iterations = max_iterations
        self.budget = budget
        self.accept_threshold = accept_threshold
        self.costs = {**DEFAULT_COSTS, **(costs or {})}
        self.trace_id = trace_id or uuid.uuid4().hex

    # -- internals -----------------------------------------------------
    def _log(self, trace: list[TraceEvent], iteration: int, action: str,
             detail: str, cost: float = 0.0) -> None:
        trace.append(TraceEvent(self.trace_id, iteration, action, detail, cost))

    def _affordable(self, spent: float, *steps: str) -> bool:
        return spent + sum(self.costs[s] for s in steps) <= self.budget

    # -- main ----------------------------------------------------------
    def run(
        self,
        query: str,
        answer_fn: Callable[[str, list[str]], str],
        verify_fn: Callable[[str, list[str]], SignalResult],
        retrieve_fn: Callable[[str], Sequence[str]] | None = None,
        escalate_fn: Callable[[str, list[str]], str] | None = None,
        needs_clarification_fn: Callable[[str], bool] | None = None,
    ) -> LoopResult:
        """Run the loop; always returns a result, never raises on budget/caps."""
        trace: list[TraceEvent] = []
        decisions: list[ActionDecision] = []
        evidence: list[str] = []
        seen_queries = {_normalize(query)}
        cache: dict[str, list[str]] = {}
        dedup_hits = 0
        spent = 0.0
        escalated = False
        answer = ""

        def decide(act: Action, rationale: str, conf: float, it: int,
                   **extra: Any) -> None:
            decisions.append(ActionDecision(act, rationale, conf, it, dict(extra)))
            self._log(trace, it, act.value, rationale)

        for it in range(1, self.max_iterations + 1):
            # --- answer + verify (charged each iteration) ---
            if not self._affordable(spent, "answer", "verify"):
                decide(Action.ACCEPT, "budget exhausted before answer/verify; "
                       "returning best-so-far as low_confidence", 0.0, it)
                return self._result(query, answer, "BUDGET_EXHAUSTED", True,
                                    it - 1, spent, decisions, trace, dedup_hits)
            answer = answer_fn(query, evidence)
            spent += self.costs["answer"]
            sig = verify_fn(answer, evidence)
            spent += self.costs["verify"] + sig.cost
            self._log(trace, it, "VERIFY",
                      f"score={sig.score:.3f} ({sig.signal})", sig.cost)

            if sig.score >= self.accept_threshold:
                decide(Action.ACCEPT,
                       f"verifier score {sig.score:.3f} >= {self.accept_threshold}",
                       sig.score, it)
                return self._result(query, answer, "ACCEPT", False,
                                    it, spent, decisions, trace, dedup_hits)

            if it >= self.max_iterations:
                break  # cap reached: fall through to low_confidence return

            # --- choose next action ---
            if needs_clarification_fn and needs_clarification_fn(query):
                decide(Action.CLARIFY, "query flagged ambiguous", sig.score, it)
                return self._result(query, answer, "NEEDS_CLARIFICATION", True,
                                    it, spent, decisions, trace, dedup_hits)

            if retrieve_fn is not None and self._affordable(spent, "retrieve"):
                rq = _normalize(query)
                if rq in cache:
                    dedup_hits += 1
                    self._log(trace, it, "RETRIEVE",
                              f"dedup_hit: {query!r} already fetched; no charge")
                    decide(Action.RETRIEVE, "dedup_hit: no new evidence",
                           sig.score, it, dedup_hit=True)
                    # No new evidence: escalate or finish rather than spin.
                else:
                    docs = list(retrieve_fn(query))
                    cache[rq] = docs
                    seen_queries.add(rq)
                    evidence = docs
                    spent += self.costs["retrieve"]
                    decide(Action.RETRIEVE, f"fetched {len(docs)} docs; re-answer",
                           sig.score, it)
                    continue

            if (not escalated and escalate_fn is not None
                    and self._affordable(spent, "escalate")):
                answer = escalate_fn(query, evidence)
                spent += self.costs["escalate"]
                escalated = True
                self._log(trace, it, "ESCALATE",
                          "re-answered with strong model", self.costs["escalate"])
                decide(Action.ESCALATE, "weak answer failed verification",
                       sig.score, it)
                # Verify the escalated answer next iteration without extra retrieval.
                continue

            if not self._affordable(spent, "retrieve", "escalate"):
                decide(Action.ACCEPT, "budget exhausted; best-so-far low_confidence",
                       sig.score, it)
                return self._result(query, answer, "BUDGET_EXHAUSTED", True,
                                    it, spent, decisions, trace, dedup_hits)
            break  # no useful action left

        decide(Action.ACCEPT, "iteration cap reached; best-so-far low_confidence",
               0.0, self.max_iterations)
        outcome: Outcome = "BUDGET_EXHAUSTED" if spent >= self.budget else "LOW_CONFIDENCE"
        return self._result(query, answer, outcome, True,
                            self.max_iterations, spent, decisions, trace, dedup_hits)

    def _result(self, query: str, answer: str, outcome: Outcome,
                low: bool, it: int, spent: float,
                decisions: list[ActionDecision], trace: list[TraceEvent],
                dedup_hits: int) -> LoopResult:
        return LoopResult(self.trace_id, query, answer, outcome, low,
                          min(max(it, 1), self.max_iterations),
                          round(spent, 4), decisions, trace, dedup_hits)
