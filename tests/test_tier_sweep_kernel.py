"""tiersweep-16: config + Kaggle kernel + runbook contract."""

import json
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kernels/tiersweep-16-local-sweep"
META = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config/experiments/sweep-200-tiersweep16.yaml"
MODELS = ROOT / "config/models.yaml"
HANDOFF = ROOT / "docs/kaggle-handoff.md"


class TierConfigTest(unittest.TestCase):
    def test_three_cheap_tiers_declared(self):
        entries = {e["name"]: e for e in yaml.safe_load(
            MODELS.read_text())["models"]}
        self.assertEqual(entries["local-small"]["model_id"],
                         "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(entries["local-3b"]["model_id"],
                         "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(entries["local-medium"]["model_id"],
                         "Qwen/Qwen2.5-7B-Instruct")

    def test_experiment_config_pins_real_tier_a(self):
        text = CONFIG.read_text()
        self.assertIn("corpus_source: real", text)
        self.assertIn("query_source: tier-a", text)
        self.assertIn("{nq: 50, hotpotqa: 80, musique: 70}", text)


class TierKernelTest(unittest.TestCase):
    def test_metadata_is_valid_and_gpu_internet_script(self):
        meta = json.loads(META.read_text())
        self.assertEqual(meta["kernel_type"], "script")
        self.assertTrue(meta["enable_gpu"])
        self.assertTrue(meta["enable_internet"])
        self.assertTrue(meta["is_private"])
        self.assertTrue((KERNEL_DIR / meta["code_file"]).is_file())
        self.assertEqual(meta["id"], "abhigyan1818/tiersweep16-local-sweep")

    def test_kernel_sweeps_only_local_cheap_routes(self):
        src = (KERNEL_DIR / "kernel.py").read_text()
        self.assertIn('"--live-routes", "L0,L1,C0"', src)
        self.assertIn('"--live-model"', src)
        self.assertIn('"--routes", "L0,L1,C0"', src)
        # Zero-cloud-spend guard: no cloud route is ever a live route.
        self.assertNotIn('"--live-routes", "L0,L1,C0,C1"', src)

    def test_kernel_serves_every_swept_checkpoint(self):
        src = (KERNEL_DIR / "kernel.py").read_text()
        for model in ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
                      "Qwen/Qwen2.5-7B-Instruct"):
            self.assertIn(model, src)

    def test_runbook_documents_tiersweep(self):
        text = HANDOFF.read_text()
        self.assertIn("kernels/tiersweep-16-local-sweep", text)
        self.assertIn("results/tiersweep-16", text)
        self.assertIn("tier_sweep.py", text)

    def test_runbook_cites_reused_1p5b_run(self):
        # The filtered kernel does not produce sweep-1p5b.db, so the runbook
        # must cite the committed realdata-15 1.5B run explicitly instead of
        # silently dropping the tier from the break-even table.
        text = HANDOFF.read_text()
        self.assertIn(
            "--run 1p5b=results/realdata-15/sweep.db:"
            "results/realdata-15/repeats.db", text)


if __name__ == "__main__":
    unittest.main()
