"""Tests for answer graders (unittest, stdlib-only env)."""

import unittest

from costsmart.eval.graders import (
    cohen_kappa,
    exact_match,
    grade,
    judge_stub,
    lenient_exact_match,
    token_f1,
)


class GradersTest(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(exact_match("Paris", "Paris"), 1)
        self.assertEqual(exact_match("paris", "Paris"), 0)

    def test_lenient_em(self):
        self.assertEqual(lenient_exact_match("  The Paris! ", "paris"), 1)
        self.assertEqual(lenient_exact_match("Paris", "London"), 0)

    def test_token_f1_perfect(self):
        self.assertAlmostEqual(token_f1("Paris", "Paris"), 1.0)

    def test_token_f1_partial(self):
        f1 = token_f1("Paris France", "Paris")
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)

    def test_token_f1_empty(self):
        self.assertEqual(token_f1("", "Paris"), 0.0)
        self.assertEqual(token_f1("Paris", ""), 0.0)

    def test_judge_stub_leaves_null(self):
        out = judge_stub("Paris", "Paris")
        self.assertIsNone(out["judge_score"])
        self.assertFalse(out["graded"])
        self.assertTrue(out["judge_model"])

    def test_grade_bundle(self):
        g = grade("Paris", "Paris")
        self.assertEqual(g["exact_match"], 1)
        self.assertEqual(g["lenient_em"], 1)
        self.assertAlmostEqual(g["token_f1"], 1.0)
        self.assertIsNone(g["judge_score"])

    def test_kappa_perfect_agreement(self):
        self.assertAlmostEqual(cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)

    def test_kappa_degenerate(self):
        self.assertEqual(cohen_kappa([], []), 0.0)
        self.assertEqual(cohen_kappa([1, 1], [1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
