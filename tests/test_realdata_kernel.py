"""realdata-15: Kaggle real-data route contract.

Pins the same safety properties as the synthetic costsweep-13 kernel - GPU +
internet script, local routes only - plus the real-data-specific steps:
the kernel must fetch the real corpus and build the real index before the
sweep, and the runbook config must pin the real mix.
"""

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kernels/realdata-15-local-sweep"
META = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config/experiments/sweep-200-realdata15.yaml"
HANDOFF = ROOT / "docs/kaggle-handoff.md"
PROVENANCE = ROOT / "results/realdata-15/kaggle_provenance.json"
RESULTS = ROOT / "results/realdata-15"


class RealDataKernelTest(unittest.TestCase):
    def test_metadata_is_valid_and_gpu_internet_script(self):
        meta = json.loads(META.read_text())
        self.assertEqual(meta["kernel_type"], "script")
        self.assertTrue(meta["enable_gpu"])
        self.assertTrue(meta["enable_internet"])
        self.assertTrue(meta["is_private"])
        self.assertTrue((KERNEL_DIR / meta["code_file"]).is_file())
        self.assertEqual(meta["id"], "abhigyan1818/realdata15-local-sweep")

    def test_kernel_only_runs_local_routes(self):
        src = (KERNEL_DIR / "kernel.py").read_text()
        self.assertIn('"--routes", "L0,L1"', src)
        self.assertIn('"--live-routes", "L0,L1"', src)
        self.assertNotIn('"--live-routes", "L0,L1,C0"', src)

    def test_kernel_fetches_real_corpus_then_builds_real_index(self):
        src = (KERNEL_DIR / "kernel.py").read_text()
        self.assertIn("scripts/fetch_tier_a.py", src)
        self.assertIn('"--mix", "real"', src)
        self.assertIn("results/realdata-15/corpus.json", src)

    def test_handoff_documents_realdata_runbook(self):
        text = HANDOFF.read_text()
        self.assertIn("kernels/realdata-15-local-sweep", text)
        self.assertIn("results/realdata-15/sweep.db", text)


class RealDataConfigTest(unittest.TestCase):
    def test_config_pins_real_source_and_mix(self):
        text = CONFIG.read_text()
        self.assertIn("corpus_source: real", text)
        self.assertIn("query_source: tier-a", text)
        self.assertIn("{nq: 50, hotpotqa: 80, musique: 70}", text)


class RealDataProvenanceTest(unittest.TestCase):
    def _prov(self):
        self.assertTrue(PROVENANCE.is_file(),
                        "realdata-15 provenance missing")
        return json.loads(PROVENANCE.read_text())

    def test_log_hash_and_runbook_summary_recorded(self):
        prov = self._prov()
        self.assertEqual(prov["kernel"]["id"],
                         "abhigyan1818/realdata15-local-sweep")
        log = prov["logs"]["realdata15-local-sweep.log"]
        self.assertEqual(len(log["sha256"]), 64)
        self.assertIn("kaggle kernels output", prov["download"]["command"])
        summary = prov["runbook_summary"]
        self.assertEqual(summary["sweep_rows"], 1400)
        self.assertEqual(summary["repeat_rows"], 1200)
        self.assertEqual(summary["repeat_pairs"], 400)
        # zero-cloud-spend guard: only L0/L1 measured
        measured = {r["route_id"] for r in summary["sweep_by_route_mode"]
                    if r["generator_mode"] == "measured"}
        self.assertEqual(measured, {"L0", "L1"})

    def test_recorded_artifact_hashes_match_committed_dbs(self):
        artifacts = self._prov()["artifacts"]
        for name, meta in artifacts.items():
            digest = hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
            self.assertEqual(digest, meta["sha256"], name)

    def test_committed_corpus_matches_provenance(self):
        prov = self._prov()
        corpus = json.loads((RESULTS / "corpus.json").read_text())
        self.assertEqual(corpus["manifest"]["corpus_sha256"],
                         prov["corpus"]["body_sha256"])
        self.assertTrue(prov["corpus"]["matches_committed"])

    def test_gate_verdict_recorded_no_go(self):
        self.assertTrue((RESULTS / "headroom_stable.json").is_file())
        stable = json.loads((RESULTS / "headroom_stable.json").read_text())
        single = json.loads((RESULTS / "headroom_single.json").read_text())
        self.assertEqual(stable["gate"]["verdict"], "NO-GO")
        self.assertEqual(single["gate"]["verdict"], "NO-GO")
        self.assertEqual(stable["n_paired_queries"], 200)
        # stable majority over the full repeat protocol
        self.assertEqual(stable["contingency"]["cheap_only"], 0)


if __name__ == "__main__":
    unittest.main()
