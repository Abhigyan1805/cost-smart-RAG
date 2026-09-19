"""Tests: Lagrangian feasibility filter + lambda selection (stdlib unittest).

Run: ``python3 -m unittest discover -s tests -t .`` from the repo root
(or ``PYTHONPATH=src python3 -m unittest discover -s tests -t .``).
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from costsmart.routing.lagrangian import (
    Candidate,
    feasibility_filter,
    lagrangian_score,
    mean_cost_for_lambda,
    select_lambda,
    select_route,
)


def _workload() -> list[list[Candidate]]:
    local = Candidate("local", expected_quality=0.70, expected_cost=1.0)
    cloud = Candidate("cloud", expected_quality=0.92, expected_cost=20.0)
    return [[local, cloud] for _ in range(4)]


class FeasibilityTest(unittest.TestCase):
    def test_over_budget_candidates_dropped(self) -> None:
        cands = [Candidate("local", 0.7, 1.0), Candidate("cloud", 0.92, 20.0)]
        self.assertEqual([c.name for c in feasibility_filter(cands, 5.0)], ["local"])

    def test_all_feasible_when_cap_generous(self) -> None:
        cands = [Candidate("local", 0.7, 1.0), Candidate("cloud", 0.92, 20.0)]
        self.assertEqual(len(feasibility_filter(cands, 100.0)), 2)

    def test_select_route_raises_when_nothing_feasible(self) -> None:
        with self.assertRaises(ValueError):
            select_route([Candidate("cloud", 0.92, 20.0)], lam=0.0, hard_budget=1.0)

    def test_select_route_respects_hard_budget(self) -> None:
        cands = [Candidate("local", 0.7, 1.0), Candidate("cloud", 0.92, 20.0)]
        # lam=0 prefers max quality, but the cap forces local.
        self.assertEqual(select_route(cands, lam=0.0, hard_budget=5.0).name, "local")


class LambdaSelectionTest(unittest.TestCase):
    def test_lagrangian_score_prefers_cheap_when_lambda_high(self) -> None:
        cheap = lagrangian_score(0.70, 1.0, lam=1.0)
        pricey = lagrangian_score(0.92, 20.0, lam=1.0)
        self.assertGreater(cheap, pricey)

    def test_lam_zero_picks_max_quality(self) -> None:
        for cands in _workload():
            self.assertEqual(select_route(cands, lam=0.0).name, "cloud")

    def test_select_lambda_fits_budget(self) -> None:
        workload = _workload()
        lam = select_lambda(workload, budget=5.0)
        self.assertLessEqual(mean_cost_for_lambda(workload, lam), 5.0 + 1e-9)

    def test_select_lambda_zero_when_budget_generous(self) -> None:
        self.assertEqual(select_lambda(_workload(), budget=100.0), 0.0)

    def test_tighter_budget_needs_larger_lambda(self) -> None:
        workload = _workload()
        loose = select_lambda(workload, budget=15.0)
        tight = select_lambda(workload, budget=2.0)
        self.assertGreaterEqual(tight, loose)

    def test_empty_workload_raises(self) -> None:
        with self.assertRaises(ValueError):
            select_lambda([], budget=5.0)


if __name__ == "__main__":
    unittest.main()
