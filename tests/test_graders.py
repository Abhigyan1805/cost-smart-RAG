"""Tests for answer graders (unittest, stdlib-only env)."""

import csv
import inspect
import tempfile
import unittest
from pathlib import Path

from costsmart.eval.graders import (
    C2_STRONG_MODEL,
    JUDGE_MODEL,
    JUDGE_RUBRIC,
    JUDGE_TEMPERATURE,
    KAPPA_GATE,
    SECOND_JUDGE_MODEL,
    build_judge_prompt,
    cohen_kappa,
    exact_match,
    grade,
    judge_disagreement_rate,
    judge_stub,
    judge_with_llm,
    kappa_from_file,
    lenient_exact_match,
    parse_judge_output,
    passes_kappa_gate,
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


class EmptyPredictionTest(unittest.TestCase):
    def test_empty_prediction_scores_zero(self):
        self.assertEqual(exact_match("", "Paris"), 0)
        self.assertEqual(lenient_exact_match("", "Paris"), 0)
        self.assertEqual(token_f1("", "Paris"), 0.0)

    def test_empty_reference_scores_zero(self):
        self.assertEqual(exact_match("Paris", ""), 0)
        self.assertEqual(lenient_exact_match("Paris", ""), 0)
        self.assertEqual(token_f1("Paris", ""), 0.0)

    def test_both_empty_pinned(self):
        # Strict EM is pure string equality; F1 has no tokens to overlap.
        self.assertEqual(exact_match("", ""), 1)
        self.assertEqual(token_f1("", ""), 0.0)


class PartialOverlapTest(unittest.TestCase):
    def test_disjoint_tokens_zero(self):
        self.assertEqual(token_f1("Paris", "London"), 0.0)

    def test_repeated_prediction_tokens_count_once(self):
        # Bag semantics: one shared type, precision 1/3, recall 1 -> F1 0.5.
        self.assertAlmostEqual(token_f1("Paris Paris Paris", "Paris"), 0.5)

    def test_extra_tokens_penalize_precision_not_recall(self):
        full = token_f1("Paris", "Paris France")
        # Recall 1/2, precision 1 -> F1 = 2*(1*0.5)/(1.5) = 2/3.
        self.assertAlmostEqual(full, 2 / 3)


class NumericAnswerTest(unittest.TestCase):
    def test_trailing_punctuation_ignored(self):
        self.assertEqual(lenient_exact_match("42", "42."), 1)
        self.assertEqual(lenient_exact_match("1,000", "1000"), 0)  # pinned: no unit math

    def test_distinct_numbers_differ(self):
        self.assertEqual(exact_match("42", "42.0"), 0)
        self.assertEqual(lenient_exact_match("42", "42.0"), 0)
        self.assertEqual(lenient_exact_match("100", "200"), 0)
        self.assertEqual(token_f1("100", "200"), 0.0)

    def test_numeric_phrase_case_punct_variants(self):
        self.assertAlmostEqual(token_f1("100 degrees Celsius", "100 Degrees Celsius!"), 1.0)
        self.assertEqual(lenient_exact_match("100 degrees Celsius", "100 Degrees Celsius!"), 1)


class CasePunctArticleTest(unittest.TestCase):
    def test_case_and_punctuation_variants(self):
        self.assertEqual(exact_match("Paris", "paris!"), 0)
        self.assertEqual(lenient_exact_match("Paris", "paris!"), 1)
        self.assertEqual(lenient_exact_match("  PARIS... ", "paris"), 1)

    def test_articles_ignored(self):
        self.assertEqual(lenient_exact_match("the Eiffel Tower", "Eiffel Tower"), 1)
        self.assertEqual(lenient_exact_match("A dog", "an dog"), 1)

    def test_different_content_stays_zero(self):
        self.assertEqual(lenient_exact_match("the Eiffel Tower", "the Louvre"), 0)


class JudgeStubStabilityTest(unittest.TestCase):
    """Telemetry-slice contract: judge_stub keeps signature + return keys."""

    def test_signature_unchanged(self):
        params = list(inspect.signature(judge_stub).parameters)
        self.assertEqual(params, ["prediction", "reference", "judge_model"])

    def test_return_keys_unchanged(self):
        out = judge_stub("a", "b", judge_model="m")
        self.assertEqual(
            set(out), {"judge_score", "judge_model", "note", "graded"}
        )
        self.assertEqual(out["judge_model"], "m")
        self.assertIsNone(out["judge_score"])
        self.assertFalse(out["graded"])

    def test_grade_default_output_unchanged(self):
        g = grade("Paris", "London")
        self.assertEqual(
            set(g), {"exact_match", "lenient_em", "token_f1", "judge_score", "judge_model"}
        )
        self.assertEqual(g["exact_match"], 0)
        self.assertIsNone(g["judge_score"])

    def test_grade_accepts_backfilled_judge(self):
        judged = {
            "judge_score": 0.5,
            "judge_model": JUDGE_MODEL,
            "graded": True,
        }
        g = grade("Paris", "London", judge_result=judged)
        self.assertEqual(g["judge_score"], 0.5)
        self.assertEqual(g["judge_model"], JUDGE_MODEL)


class JudgeInstrumentTest(unittest.TestCase):
    def test_temperature_zero(self):
        self.assertEqual(JUDGE_TEMPERATURE, 0)

    def test_judge_differs_from_c2_strong_route(self):
        for model in (JUDGE_MODEL, SECOND_JUDGE_MODEL):
            self.assertNotEqual(model, C2_STRONG_MODEL)
            self.assertNotIn("gpt-4o-2024-08-06", model)
        self.assertNotEqual(JUDGE_MODEL, SECOND_JUDGE_MODEL)
        # Primary judge is a different family (no self-preference bias).
        self.assertNotIn("gpt", JUDGE_MODEL.lower())

    def test_rubric_is_explicit(self):
        self.assertIn("1.0", JUDGE_RUBRIC)
        self.assertIn("0.5", JUDGE_RUBRIC)
        self.assertIn("0.0", JUDGE_RUBRIC)
        self.assertIn("score", JUDGE_RUBRIC)
        self.assertIn("rationale", JUDGE_RUBRIC)

    def test_prompt_embeds_pair_and_reference(self):
        prompt = build_judge_prompt("Capital of France?", "Paris", "Paris")
        self.assertIn("Capital of France?", prompt)
        self.assertIn("Paris", prompt)
        self.assertIn("REFERENCE ANSWER: Paris", prompt)

    def test_prompt_marks_missing_reference(self):
        prompt = build_judge_prompt("Why is the sky blue?", "Rayleigh scattering")
        self.assertIn("none provided", prompt)

    def test_parse_valid_output(self):
        score, rationale = parse_judge_output('{"score": 0.5, "rationale": "partly right"}')
        self.assertEqual(score, 0.5)
        self.assertEqual(rationale, "partly right")

    def test_parse_tolerates_prose_and_clamps(self):
        score, _ = parse_judge_output('Verdict: {"score": 2.5, "rationale": "x"} done')
        self.assertEqual(score, 1.0)

    def test_parse_garbage_raises(self):
        with self.assertRaises(ValueError):
            parse_judge_output("looks good to me")

    def test_unwired_judge_leaves_null_with_prompt(self):
        out = judge_with_llm("Q?", "A")
        self.assertIsNone(out["judge_score"])
        self.assertFalse(out["graded"])
        self.assertEqual(out["judge_model"], JUDGE_MODEL)
        self.assertEqual(out["judge_temperature"], 0)
        self.assertIn("Q?", out["prompt"])

    def test_wired_judge_grades_via_client(self):
        class FakeClient:
            def generate(self, prompt, temperature=0):
                assert temperature == 0
                assert "Q?" in prompt

                class Resp:
                    text = '{"score": 1.0, "rationale": "exact"}'

                return Resp()

        out = judge_with_llm("Q?", "A", client=FakeClient())
        self.assertTrue(out["graded"])
        self.assertEqual(out["judge_score"], 1.0)

    def test_unparseable_client_output_leaves_null(self):
        class BadClient:
            def generate(self, prompt, temperature=0):
                class Resp:
                    text = "no json here"

                return Resp()

        out = judge_with_llm("Q?", "A", client=BadClient())
        self.assertFalse(out["graded"])
        self.assertIsNone(out["judge_score"])


class SecondJudgeCheckTest(unittest.TestCase):
    def test_identical_judges_zero_disagreement(self):
        out = judge_disagreement_rate([1.0, 0.5, 0.0], [1.0, 0.5, 0.0])
        self.assertEqual(out["disagreements"], 0)
        self.assertEqual(out["disagreement_rate"], 0.0)
        self.assertEqual(out["n"], 3)

    def test_known_disagreement_fraction(self):
        out = judge_disagreement_rate([1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(out["disagreements"], 1)
        self.assertAlmostEqual(out["disagreement_rate"], 0.25)

    def test_adjacent_levels_within_tol_agree(self):
        out = judge_disagreement_rate([1.0], [0.9], tol=0.25)
        self.assertEqual(out["disagreement_rate"], 0.0)

    def test_empty_lists_zero(self):
        out = judge_disagreement_rate([], [])
        self.assertEqual(out["n"], 0)
        self.assertEqual(out["disagreement_rate"], 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            judge_disagreement_rate([1.0], [1.0, 0.0])


class KappaGateTest(unittest.TestCase):
    def test_gate_threshold(self):
        self.assertEqual(KAPPA_GATE, 0.7)
        self.assertTrue(passes_kappa_gate(0.7))
        self.assertTrue(passes_kappa_gate(0.85))
        self.assertFalse(passes_kappa_gate(0.69))

    def _write_csv(self, directory: str, rows: list[dict]) -> Path:
        path = Path(directory) / "captain_labels.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_kappa_from_file_perfect_agreement(self):
        # Synthetic fixture for the computation only — never captain labels.
        rows = [
            {"pair_id": f"p{i}", "judge": j, "human": h}
            for i, (j, h) in enumerate([(1, 1), (1, 1), (0, 0), (0, 0)])
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = kappa_from_file(self._write_csv(tmp, rows))
        self.assertAlmostEqual(out["kappa"], 1.0)
        self.assertEqual(out["n"], 4)
        self.assertTrue(out["passes_gate"])

    def test_kappa_from_file_disagreement_fails_gate(self):
        rows = [
            {"pair_id": f"p{i}", "judge_label": j, "human_label": h}
            for i, (j, h) in enumerate([(1, 0), (0, 1), (1, 0), (0, 1)])
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = kappa_from_file(self._write_csv(tmp, rows))
        self.assertLess(out["kappa"], KAPPA_GATE)
        self.assertFalse(out["passes_gate"])

    def test_kappa_from_file_threshold_binarizes(self):
        rows = [
            {"pair_id": f"p{i}", "judge": j, "human": h}
            for i, (j, h) in enumerate([(1.0, 1), (0.0, 0), (1.0, 1), (0.0, 0)])
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = kappa_from_file(self._write_csv(tmp, rows), threshold=0.5)
        self.assertAlmostEqual(out["kappa"], 1.0)

    def test_kappa_from_file_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            kappa_from_file("/nonexistent/captain_labels.csv")

    def test_kappa_from_file_missing_columns_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("pair_id,foo\np0,1\n")
            with self.assertRaises(ValueError):
                kappa_from_file(path)


if __name__ == "__main__":
    unittest.main()
