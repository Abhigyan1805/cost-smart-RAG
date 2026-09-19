"""realdata-15: Kaggle real-data route contract.

Pins the same safety properties as the synthetic costsweep-13 kernel - GPU +
internet script, local routes only - plus the real-data-specific steps:
the kernel must fetch the real corpus and build the real index before the
sweep, and the runbook config must pin the real mix.
"""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kernels/realdata-15-local-sweep"
META = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config/experiments/sweep-200-realdata15.yaml"
HANDOFF = ROOT / "docs/kaggle-handoff.md"


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


if __name__ == "__main__":
    unittest.main()
