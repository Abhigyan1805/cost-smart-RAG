"""Tests for the headroom 2x2 gate (costheadroom-09; unittest, stdlib-only)."""

import unittest

from costsmart.eval.metrics import (
    build_contingency_table,
    is_correct,
    max_savings_at_no_quality_loss,
    oracle_cheap_cost,
    routable_fraction,
)
from costsmart.eval.stats import (
    bootstrap_paired_diff_ci,
    bootstrap_proportion_ci,
    bootstrap_saving_ci,
    mcnemar,
)


def _row(query_id, route_id, f1, cloud=0.0, amort=0.0):
    return {"query_id": query_id, "route_id": route_id, "token_f1": f1,
            "cloud_spend_usd": cloud, "amortized_usd": amort}


def _attempts():
    # 4 queries: both / strong-only / cheap-only / neither.
    return [
        _row("q1", "L0", 1.0, 0.0, 0.001), _row("q1", "C4", 1.0, 0.01, 0.0),
        _row("q2", "L0", 0.0, 0.0, 0.001), _row("q2", "C4", 1.0, 0.02, 0.0),
        _row("q3", "L0", 0.8, 0.0, 0.001), _row("q3", "C4", 0.0, 0.03, 0.0),
        _row("q4", "L0", 0.0, 0.0, 0.001), _row("q4", "C4", 0.1, 0.04, 0.0),
    ]


class ContingencyTest(unittest.TestCase):
    def test_cells(self):
        table = build_contingency_table(_attempts())
        self.assertEqual(table["n"], 4)
        self.assertEqual(table["both_correct"], 1)
        self.assertEqual(table["strong_only"], 1)
        self.assertEqual(table["cheap_only"], 1)
        self.assertEqual(table["neither"], 1)
        self.assertEqual(table["skipped"], 0)
        self.assertAlmostEqual(table["both_correct_rate"], 0.25)

    def test_missing_route_skipped_not_dropped(self):
        rows = [_row("q1", "L0", 1.0)]
        table = build_contingency_table(rows)
        self.assertEqual(table["n"], 0)
        self.assertEqual(table["skipped"], 1)

    def test_is_correct_threshold(self):
        self.assertTrue(is_correct({"token_f1": 0.5}))
        self.assertFalse(is_correct({"token_f1": 0.49}))
        self.assertFalse(is_correct({}))

    def test_routable_fraction(self):
        table = build_contingency_table(_attempts())
        rf = routable_fraction(table)
        self.assertAlmostEqual(rf["estimate"], 0.5)
        self.assertEqual((rf["k"], rf["n"]), (2, 4))
        # Raw-rows input agrees with prebuilt-table input.
        self.assertAlmostEqual(
            routable_fraction(_attempts())["estimate"], 0.5)

    def test_oracle_cheap_cost_prefers_cheap_correct(self):
        cost, route = oracle_cheap_cost(_attempts(), "q1")
        self.assertEqual(route, "L0")
        self.assertAlmostEqual(cost, 0.001)


class SavingsTest(unittest.TestCase):
    def test_saving_math(self):
        out = max_savings_at_no_quality_loss(_attempts())
        # Strong total = .01+.02+.03+.04=.10; oracle = .001+.02+.001+.001=.023.
        self.assertEqual(out["n"], 4)
        self.assertAlmostEqual(out["strong_cost"], 0.10)
        self.assertAlmostEqual(out["oracle_cost"], 0.023)
        self.assertAlmostEqual(out["saving_fraction"], 0.77)
        self.assertEqual(out["n_quality_loss_queries"], 0)

    def test_queries_missing_strong_route_excluded(self):
        rows = _attempts() + [_row("q9", "L0", 1.0)]
        out = max_savings_at_no_quality_loss(rows)
        self.assertEqual(out["n"], 4)


class StatsTest(unittest.TestCase):
    def test_mcnemar_no_discordant(self):
        self.assertEqual(mcnemar(0, 0)["p_value"], 1.0)

    def test_mcnemar_exact_two_sided(self):
        # n=10, min=1: two-sided p = 2 * (1+10)/1024 = 22/1024.
        self.assertAlmostEqual(mcnemar(9, 1)["p_value"], 22 / 1024)
        self.assertEqual(mcnemar(9, 1)["n_discordant"], 10)

    def test_mcnemar_chi2_branch(self):
        out = mcnemar(20, 5)
        self.assertEqual(out["n_discordant"], 25)
        self.assertLess(out["p_value"], 0.05)

    def test_proportion_ci_shape(self):
        out = bootstrap_proportion_ci(11, 20, n_resamples=1000, seed=0)
        self.assertAlmostEqual(out["estimate"], 0.55)
        self.assertLessEqual(out["lo"], out["estimate"])
        self.assertLessEqual(out["estimate"], out["hi"])
        self.assertTrue(0.0 <= out["lo"] and out["hi"] <= 1.0)

    def test_proportion_ci_deterministic(self):
        a = bootstrap_proportion_ci(11, 20, n_resamples=500, seed=7)
        b = bootstrap_proportion_ci(11, 20, n_resamples=500, seed=7)
        self.assertEqual(a, b)

    def test_paired_diff_ci(self):
        a = [1, 0, 1, 0]
        b = [1, 1, 1, 0]
        out = bootstrap_paired_diff_ci(a, b, n_resamples=1000, seed=0)
        self.assertAlmostEqual(out["estimate"], 0.25)
        self.assertLessEqual(out["lo"], 0.25)
        self.assertLessEqual(0.25, out["hi"])

    def test_paired_diff_rejects_mismatch(self):
        with self.assertRaises(ValueError):
            bootstrap_paired_diff_ci([1], [1, 0])

    def test_saving_ci(self):
        out = bootstrap_saving_ci([0.01, 0.02], [0.001, 0.02],
                                  n_resamples=1000, seed=0)
        self.assertAlmostEqual(out["estimate"], 0.3, places=6)
        self.assertLessEqual(out["lo"], out["estimate"])
        self.assertLessEqual(out["estimate"], out["hi"])


if __name__ == "__main__":
    unittest.main()
