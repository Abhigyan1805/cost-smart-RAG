"""Tests: agent-loop caps, dedup guard, budget exhaustion (stdlib unittest).

Run: ``python3 -m unittest discover -s tests -t .`` from the repo root.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from costsmart.agent.loop import AgentLoop
from costsmart.verify import SignalResult


def _low_verify(answer: str, evidence: list[str]) -> SignalResult:
    return SignalResult(signal="stub", score=0.1, cost=0.0)


def _high_verify(answer: str, evidence: list[str]) -> SignalResult:
    return SignalResult(signal="stub", score=0.95, cost=0.0)


class LoopCapsTest(unittest.TestCase):
    def test_iterations_never_exceed_max(self) -> None:
        loop = AgentLoop(max_iterations=3, budget=1000.0)
        calls = {"n": 0}

        def answer_fn(q: str, evidence: list[str]) -> str:
            calls["n"] += 1
            return "answer"

        res = loop.run("q", answer_fn, _low_verify, retrieve_fn=lambda q: ["d"])
        self.assertLessEqual(res.iterations_used, 3)
        self.assertLessEqual(calls["n"], 3)
        self.assertTrue(res.low_confidence)
        self.assertEqual(res.outcome, "LOW_CONFIDENCE")

    def test_accept_on_first_try_when_confident(self) -> None:
        loop = AgentLoop(max_iterations=3, budget=1000.0)
        res = loop.run("q", lambda q, e: "answer", _high_verify)
        self.assertEqual(res.outcome, "ACCEPT")
        self.assertFalse(res.low_confidence)
        self.assertEqual(res.iterations_used, 1)

    def test_trace_id_logged_on_every_event(self) -> None:
        loop = AgentLoop(max_iterations=2, budget=1000.0, trace_id="t-1")
        res = loop.run("q", lambda q, e: "a", _low_verify,
                       retrieve_fn=lambda q: ["d"])
        self.assertEqual(res.trace_id, "t-1")
        self.assertTrue(res.trace)
        self.assertTrue(all(ev.trace_id == "t-1" for ev in res.trace))


class DedupTest(unittest.TestCase):
    def test_repeat_retrieval_served_from_cache(self) -> None:
        loop = AgentLoop(max_iterations=3, budget=1000.0)
        fetches = {"n": 0}

        def retrieve_fn(q: str) -> list[str]:
            fetches["n"] += 1
            return ["doc"]

        res = loop.run("same query", lambda q, e: "a", _low_verify,
                       retrieve_fn=retrieve_fn)
        self.assertEqual(fetches["n"], 1)  # second RETRIEVE is a dedup hit
        self.assertGreaterEqual(res.dedup_hits, 1)


class BudgetTest(unittest.TestCase):
    def test_budget_exhaustion_returns_low_confidence(self) -> None:
        loop = AgentLoop(max_iterations=3, budget=1.0,
                         costs={"answer": 1.0, "verify": 1.0,
                                "retrieve": 5.0, "escalate": 50.0})
        res = loop.run("q", lambda q, e: "a", _low_verify,
                       retrieve_fn=lambda q: ["d"])
        self.assertTrue(res.low_confidence)
        self.assertEqual(res.outcome, "BUDGET_EXHAUSTED")
        self.assertLessEqual(res.total_cost, 1.0 + 1e-9)

    def test_escalation_skipped_when_unaffordable(self) -> None:
        loop = AgentLoop(max_iterations=3, budget=10.0,
                         costs={"answer": 1.0, "verify": 0.5,
                                "retrieve": 2.0, "escalate": 500.0})
        escalated = {"n": 0}

        def escalate_fn(q: str, e: list[str]) -> str:
            escalated["n"] += 1
            return "strong answer"

        res = loop.run("q", lambda q, e: "a", _low_verify,
                       retrieve_fn=lambda q: ["d"], escalate_fn=escalate_fn)
        self.assertEqual(escalated["n"], 0)
        self.assertTrue(res.low_confidence)


if __name__ == "__main__":
    unittest.main()
