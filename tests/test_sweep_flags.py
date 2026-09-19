"""costsweep-08: provenance flags, idempotency, zero-cloud-spend guard."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from costsmart.eval import oracle_sweep as sweep
from costsmart.telemetry.store import TelemetryStore


def _query(qid="nq:0000"):
    return {"query_id": qid, "question": "what is photosynthesis known for",
            "reference": "photosynthesis"}


class CacheKeyTest(unittest.TestCase):
    def test_stable_and_tuple_scoped(self):
        a = sweep.cache_key("q1", "L0", "v1", "m")
        b = sweep.cache_key("q1", "L0", "v1", "m")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)
        # Any tuple element change flips the key (no cross-contamination).
        self.assertNotEqual(a, sweep.cache_key("q2", "L0", "v1", "m"))
        self.assertNotEqual(a, sweep.cache_key("q1", "C1", "v1", "m"))
        self.assertNotEqual(a, sweep.cache_key("q1", "L0", "v2", "m"))
        self.assertNotEqual(a, sweep.cache_key("q1", "L0", "v1", "m2"))


class StubFlagsTest(unittest.TestCase):
    def test_stub_attempt_carries_flags(self):
        attempt = sweep.build_attempt(_query(), "C1", "v1", "sha", "cfg",
                                      seed=0, retrieval_ms=6.0)
        self.assertEqual(attempt["generator_mode"], "stub")
        self.assertEqual(attempt["retrieval_mode"], "measured")
        self.assertEqual(attempt["temperature"], 0)
        self.assertEqual(attempt["seed"], 0)

    def test_no_retrieval_is_stub(self):
        attempt = sweep.build_attempt(_query(), "L0", "v1", "sha", "cfg")
        self.assertEqual(attempt["retrieval_mode"], "stub")
        self.assertEqual(attempt["generator_mode"], "stub")


class ZeroCloudSpendTest(unittest.TestCase):
    def test_live_cloud_route_refused(self):
        for route in sweep.CLOUD_ROUTES:
            with self.assertRaises(RuntimeError):
                sweep.execute_live_local(_query(), route, [], object(), "m")

    def test_live_local_route_needs_client(self):
        with self.assertRaises(RuntimeError):
            sweep.execute_live_local(_query(), "L0", [], None, "m")

    def test_live_local_attempt_measured_no_network(self):
        from costsmart.models.base import GenerateResult

        calls = {}

        class FakeColab:
            model_id = "Qwen/Qwen2.5-1.5B-Instruct"

            def generate(self, prompt, **kwargs):
                calls["kwargs"] = kwargs
                self.last_prompt = prompt
                return GenerateResult(
                    text="Final answer: photosynthesis", tokens=3,
                    latency_s=1.5,
                    raw={"response": "Final answer: photosynthesis",
                         "eval_count": 3, "prompt_eval_count": 42,
                         "latency_s": 1.5})

        fake = FakeColab()
        attempt = sweep.build_attempt(
            _query(), "L0", "v1", "sha", "cfg", seed=0,
            retrieval_ms=6.0, live_client=fake,
            live_model_version=fake.model_id, temperature=0)
        self.assertEqual(attempt["generator_mode"], "measured")
        self.assertEqual(attempt["retrieval_mode"], "measured")
        self.assertEqual(attempt["model_version"], fake.model_id)
        self.assertEqual(attempt["prediction"], "photosynthesis")
        self.assertEqual(attempt["tokens_in"], 42)
        self.assertEqual(attempt["tokens_out"], 3)
        self.assertAlmostEqual(attempt["gpu_seconds"], 1.5)
        # Sampling contract enforced on the wire.
        self.assertEqual(calls["kwargs"]["temperature"], 0)
        self.assertEqual(calls["kwargs"]["options"], {"seed": 0, "temperature": 0})
        # Measured model_version changes the cache key (never collides
        # with the stub row for the same query+route).
        stub = sweep.build_attempt(_query(), "L0", "v1", "sha", "cfg")
        self.assertNotEqual(attempt["cache_key"], stub["cache_key"])


class PromptTest(unittest.TestCase):
    def test_k0_has_no_context(self):
        prompt = sweep.build_local_prompt("Q?", [], "direct")
        self.assertIn("Q?", prompt)
        self.assertNotIn("Context passages", prompt)

    def test_cot_extracts_final_line(self):
        text = "Reasoning...\nFinal answer: Paris"
        self.assertEqual(sweep.extract_final_answer(text), "Paris")
        self.assertEqual(sweep.extract_final_answer("plain"), "plain")


class ResumabilityTest(unittest.TestCase):
    def test_insert_twice_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TelemetryStore(Path(tmp) / "t.db")
            try:
                attempt = sweep.build_attempt(_query(), "L0", "v1", "sha",
                                              "cfg", retrieval_ms=5.0)
                self.assertTrue(store.insert_attempt(attempt))
                self.assertFalse(store.insert_attempt(attempt))
                self.assertEqual(store.count(), 1)
            finally:
                store.close()


class ResumableLiveSweepTest(unittest.TestCase):
    def test_completed_live_rows_are_not_regenerated(self):
        from costsmart.models.base import GenerateResult

        class FakeColab:
            model_id = "fake/1.5b"
            calls = 0

            def generate(self, prompt, **kwargs):
                type(self).calls += 1
                return GenerateResult(
                    text="Final answer: Paris", tokens=2, latency_s=0.01,
                    raw={"response": "Final answer: Paris", "eval_count": 2,
                         "prompt_eval_count": 7, "latency_s": 0.01})

        with tempfile.TemporaryDirectory() as tmp:
            store = TelemetryStore(Path(tmp) / "t.db")
            try:
                first = sweep.run_sweep(
                    store=store, config_path=None, limit=2, index_path=None,
                    live_local=True, live_clients={"local-small": FakeColab()},
                    live_routes=("L0", "L1"))
                self.assertEqual(first["generator_measured_attempts"], 2)
                self.assertEqual(FakeColab.calls, 2)
                # Re-run over the same two pairs: stored cache keys are
                # pre-checked, so no live generation is re-spent.
                second = sweep.run_sweep(
                    store=store, config_path=None, limit=2, index_path=None,
                    live_local=True, live_clients={"local-small": FakeColab()},
                    live_routes=("L0", "L1"))
                self.assertEqual(second["inserted"], 0)
                self.assertEqual(second["skipped_existing"], 2)
                self.assertEqual(second["generator_measured_attempts"], 0)
                self.assertEqual(second["retrieval_measured_attempts"], 0)
                self.assertEqual(FakeColab.calls, 2)
            finally:
                store.close()


class LegacyMigrationTest(unittest.TestCase):
    def test_legacy_db_opens_and_flags_default(self):
        from costsmart.telemetry.schema import ATTEMPT_COLUMNS, MIGRATED_COLUMNS

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "legacy.db")
            legacy_cols = [c for c in ATTEMPT_COLUMNS if c not in MIGRATED_COLUMNS]
            coldefs = ", ".join(f"{c} TEXT" for c in legacy_cols)
            conn = sqlite3.connect(db)
            conn.execute(
                "CREATE TABLE attempts (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                f" {coldefs})"
            )
            conn.execute("INSERT INTO attempts (cache_key, query_id)"
                         " VALUES ('k1', 'q1')")
            conn.commit()
            conn.close()
            store = TelemetryStore(db)
            try:
                rows = store.query(
                    "SELECT generator_mode, retrieval_mode FROM attempts")
                self.assertEqual(rows[0]["generator_mode"], "unflagged-legacy")
                self.assertEqual(rows[0]["retrieval_mode"], "unflagged-legacy")
                attempt = sweep.build_attempt(_query("nq:0001"), "L1", "v1",
                                              "sha", "cfg", retrieval_ms=5.0)
                self.assertTrue(store.insert_attempt(attempt))
                self.assertEqual(store.count(), 2)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
