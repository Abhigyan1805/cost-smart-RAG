"""tiersweep-16: cheap-tier break-even analysis contract."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from costsmart.telemetry.store import TelemetryStore

ROOT = Path(__file__).resolve().parents[1]


def _load_tier_sweep():
    path = ROOT / "scripts" / "tier_sweep.py"
    spec = importlib.util.spec_from_file_location("tier_sweep", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tier_sweep"] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(qid, route, model, f1, mode="measured"):
    return {
        "cache_key": f"{qid}:{route}:{model}",
        "query_id": qid, "route_id": route, "prompt_version": "v1",
        "model_version": model, "prediction": "x", "reference": "x",
        "exact_match": int(f1 == 1.0), "lenient_em": int(f1 == 1.0),
        "token_f1": f1, "judge_score": None, "judge_model": "",
        "tokens_in": 10, "tokens_out": 3, "gpu_seconds": 2.0,
        "concurrency": 1, "cloud_spend_usd": 0.0, "amortized_usd": 0.0002,
        "latency_ms_total": 0.0, "latency_ms_retrieval": 0.0,
        "latency_ms_llm": 0.0, "latency_ms_verify": 0.0, "git_sha": "",
        "config_hash": "", "created_at": "", "generator_mode": mode,
        "retrieval_mode": "measured", "temperature": 0, "seed": 0,
        "model_revision": "rev", "weights_sha256": "w",
        "attestation": "server-attested",
    }


class TierAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load_tier_sweep()
        self.model = "Qwen/Qwen2.5-3B-Instruct"
        n = 30
        self.base = []
        self.repeats = []
        for i in range(n):
            qid = f"nq:{i:04d}"
            # C4 strong stub: always correct, priced cloud-side.
            c4 = _row(qid, "C4", "gpt-4o", 1.0, mode="stub")
            c4["cloud_spend_usd"] = 0.0006
            c4["amortized_usd"] = 0.0
            c4["attestation"] = "stub"
            self.base.append(c4)
            # L0 cheap: correct for the first 6 queries (coverage 0.20).
            f1 = 1.0 if i < 6 else 0.0
            cheap = _row(qid, "L0", self.model, f1)
            self.base.append(cheap)
            for k in range(3):
                rep = dict(cheap)
                rep["cache_key"] = f"{cheap['cache_key']}:r{k}"
                rep["repeat_idx"] = k
                self.repeats.append(rep)

    def test_analyse_tier_coverage_and_cost(self):
        stable, meta = self.mod.build_stable_matrix(self.base, self.repeats)
        self.assertEqual(meta["n_repeated_pairs"], 30)
        tier = self.mod.analyse_tier(self.model, self.base, stable,
                                     self.repeats, resamples=200, seed=0)
        self.assertAlmostEqual(tier["params_billions"], 3.0)
        route = tier["routes"]["L0"]
        self.assertAlmostEqual(route["coverage"], 0.20)
        self.assertGreater(route["gpu_seconds_per_query"], 0.0)
        self.assertGreater(route["cost_ratio_cheap_over_strong"], 0.0)
        self.assertEqual(route["mcnemar_c"], 0)

    def test_break_even_identifies_first_go(self):
        stable, _ = self.mod.build_stable_matrix(self.base, self.repeats)
        tier = self.mod.analyse_tier(self.model, self.base, stable,
                                     self.repeats, resamples=200, seed=0)
        be = self.mod.break_even_statement([tier])
        self.assertIsNotNone(be["first_go_l0"])
        self.assertIn("3B", be["first_go_l0"]["model_version"])
        self.assertTrue(be["first_go_l0"]["decision"].startswith("GO"))

    def test_model_param_parsing(self):
        self.assertEqual(self.mod.model_param_billions("Qwen/Qwen2.5-7B-Instruct"),
                         7.0)
        self.assertIsNone(self.mod.model_param_billions("gpt-4o"))


class DiscoverRunsTest(unittest.TestCase):
    def test_discovers_matching_pairs(self):
        mod = _load_tier_sweep()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "sweep-3b.db").write_bytes(b"")
            (tmp / "repeats-3b.db").write_bytes(b"")
            (tmp / "sweep-orphan.db").write_bytes(b"")
            runs = mod.discover_runs(tmp)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0][0], "3b")


class TierTableTest(unittest.TestCase):
    def test_table_lists_every_swept_route(self):
        mod = _load_tier_sweep()
        tier = {
            "model_version": "Qwen/Qwen2.5-3B-Instruct", "params_billions": 3.0,
            "best_cheap_route": "L1", "routes": {
                r: {"coverage": 0.2, "coverage_ci": [0.1, 0.3],
                    "gap_c4_minus_tier": 0.8, "mcnemar_p": 1e-9,
                    "gpu_seconds_per_query": 2.0,
                    "cost_ratio_cheap_over_strong": 0.3,
                    "saving_at_gate": 0.07, "verdict": "GO"}
                for r in ("L0", "L1", "C0")
            },
        }
        be = mod.break_even_statement([tier])
        table = mod.render_table_md([tier], be)
        for route in ("L0", "L1", "C0"):
            self.assertIn(f"| {route} |", table)


if __name__ == "__main__":
    unittest.main()
