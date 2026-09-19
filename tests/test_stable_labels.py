"""Tests for costfinal-10 stable labels (unittest, stdlib-only).

Covers: repeat cache keys (5-tuple, never colliding with base keys),
majority grading with the documented tie rule (ties count as incorrect),
representative-row agreement, repeat store idempotency, and flip stats.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from costsmart.eval import oracle_sweep as sweep
from costsmart.eval import stability
from costsmart.eval.metrics import is_correct
from costsmart.telemetry.schema import N_REPEATS
from costsmart.telemetry.store import TelemetryStore


def _repeat_row(qid="nq:0000", route="L0", idx=0, f1=1.0, pred="photosynthesis"):
    return {
        "cache_key": sweep.repeat_cache_key(qid, route, "v1", "m", idx),
        "repeat_idx": idx,
        "query_id": qid,
        "route_id": route,
        "prompt_version": "v1",
        "model_version": "m",
        "prediction": pred,
        "reference": "photosynthesis",
        "exact_match": 0,
        "lenient_em": 0,
        "token_f1": f1,
        "judge_score": None,
        "judge_model": "",
        "tokens_in": 10,
        "tokens_out": 3,
        "gpu_seconds": 1.0,
        "concurrency": 1,
        "cloud_spend_usd": 0.0,
        "amortized_usd": 0.001,
        "latency_ms_total": 1000.0,
        "latency_ms_retrieval": 5.0,
        "latency_ms_llm": 995.0,
        "latency_ms_verify": 0.0,
        "git_sha": "sha",
        "config_hash": "cfg",
        "created_at": "t",
        "generator_mode": "measured",
        "retrieval_mode": "measured",
        "temperature": 0,
        "seed": 0,
    }


class RepeatKeyTest(unittest.TestCase):
    def test_stable_and_scoped(self):
        a = sweep.repeat_cache_key("q1", "L0", "v1", "m", 0)
        self.assertEqual(a, sweep.repeat_cache_key("q1", "L0", "v1", "m", 0))
        self.assertEqual(len(a), 64)
        self.assertNotEqual(a, sweep.repeat_cache_key("q1", "L0", "v1", "m", 1))
        self.assertNotEqual(a, sweep.repeat_cache_key("q2", "L0", "v1", "m", 0))
        self.assertNotEqual(a, sweep.repeat_cache_key("q1", "L1", "v1", "m", 0))

    def test_never_collides_with_base_key(self):
        # Even at repeat_idx=0 the 5-tuple hash differs from the base
        # 4-tuple hash: repeat slots can never overwrite base rows.
        for idx in range(N_REPEATS):
            self.assertNotEqual(
                sweep.repeat_cache_key("q1", "L0", "v1", "m", idx),
                sweep.cache_key("q1", "L0", "v1", "m"))


class MajorityTest(unittest.TestCase):
    def test_complete_majorities(self):
        for votes, expected in [([1, 1, 1], True), ([1, 1, 0], True),
                                ([1, 0, 0], False), ([0, 0, 0], False)]:
            out = stability.majority_correct([bool(v) for v in votes])
            self.assertEqual(out["stable_correct"], expected, votes)
            self.assertTrue(out["complete"], votes)
            self.assertFalse(out["tie"], votes)

    def test_ties_count_as_incorrect(self):
        # Even split with 2 draws: no strict majority -> incorrect + tie.
        out = stability.majority_correct([True, False])
        self.assertFalse(out["stable_correct"])
        self.assertTrue(out["tie"])
        # Single draw short of quota -> incorrect + tie.
        out = stability.majority_correct([True])
        self.assertFalse(out["stable_correct"])
        self.assertTrue(out["tie"])
        # No draws at all -> incorrect.
        out = stability.majority_correct([])
        self.assertFalse(out["stable_correct"])

    def test_unoverturnable_pair_counts_correct(self):
        # 2-0 with one draw missing: the missing draw cannot overturn the
        # 2 correct votes, so the label is correct (still flagged
        # provisional via tie=True).
        out = stability.majority_correct([True, True])
        self.assertTrue(out["stable_correct"])
        self.assertTrue(out["tie"])
        self.assertFalse(out["complete"])


class StableLabelTest(unittest.TestCase):
    def test_representative_agrees_with_verdict(self):
        reps = [_repeat_row(idx=0, f1=1.0, pred="photosynthesis"),
                _repeat_row(idx=1, f1=0.0, pred="mitochondria"),
                _repeat_row(idx=2, f1=1.0, pred="Photosynthesis")]
        out = stability.stable_label_for_pair(reps)
        self.assertTrue(out["stable_correct"])
        self.assertTrue(is_correct(out["representative"]))
        self.assertEqual((out["n_correct"], out["n_total"]), (2, 3))

        reps_bad = [_repeat_row(idx=i, f1=0.0, pred="wrong") for i in range(3)]
        out = stability.stable_label_for_pair(reps_bad)
        self.assertFalse(out["stable_correct"])
        self.assertFalse(is_correct(out["representative"]))

    def test_tie_falls_back_to_base_row(self):
        reps = [_repeat_row(idx=0, f1=1.0), _repeat_row(idx=1, f1=0.0)]
        base = dict(_repeat_row(idx=0, f1=0.2, pred="base draw"))
        base.pop("repeat_idx", None)
        out = stability.stable_label_for_pair(reps, base)
        self.assertFalse(out["stable_correct"])
        self.assertTrue(out["tie"])
        self.assertEqual(out["representative"]["prediction"], "base draw")

    def test_grade_repeats_flags(self):
        graded = stability.grade_repeats(["photosynthesis", "mitochondria"],
                                         "photosynthesis")
        self.assertTrue(graded[0]["correct"])
        self.assertFalse(graded[1]["correct"])


class RepeatStoreTest(unittest.TestCase):
    def test_insert_fetch_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TelemetryStore(Path(tmp) / "r.db")
            try:
                row = _repeat_row()
                self.assertTrue(store.insert_repeat(row))
                self.assertFalse(store.insert_repeat(row))
                self.assertTrue(store.has_repeat(row["cache_key"]))
                self.assertEqual(store.count_repeats(), 1)
                fetched = store.fetch_repeats("L0")
                self.assertEqual(len(fetched), 1)
                self.assertEqual(fetched[0]["repeat_idx"], 0)
                # attempts table untouched by repeat writes.
                self.assertEqual(store.count(), 0)
            finally:
                store.close()

    def test_base_and_repeat_keys_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TelemetryStore(Path(tmp) / "r.db")
            try:
                base_key = sweep.cache_key("q1", "L0", "v1", "m")
                self.assertFalse(store.has(base_key))
                self.assertFalse(store.has_repeat(base_key))
            finally:
                store.close()


class FlipStatsTest(unittest.TestCase):
    def test_string_and_label_flips(self):
        s = stability.flip_stats([["a", "a", "a"], ["a", "b", "a"]])
        self.assertEqual(s["n_pairs"], 2)
        self.assertAlmostEqual(s["string_flip_rate"], 0.5)
        self.assertEqual(s["distinct_predictions_hist"], {1: 1, 2: 1})
        lv = stability.label_flip_stats([[True] * 3, [True, False, True],
                                         [False] * 3])
        self.assertAlmostEqual(lv["label_flip_rate"], 1 / 3)
        self.assertEqual(lv["vote_split_hist"]["3-0"], 1)
        self.assertEqual(lv["vote_split_hist"]["2-1"], 1)
        self.assertEqual(lv["vote_split_hist"]["0-3"], 1)
        self.assertEqual(lv["unanimous_correct"], 1)
        self.assertEqual(lv["unanimous_incorrect"], 1)


if __name__ == "__main__":
    unittest.main()
