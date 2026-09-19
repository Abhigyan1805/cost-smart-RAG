"""costsweep-13: Kaggle GPU route contract.

The Kaggle kernel is the fallback compute host when the Colab free tier is
exhausted. These pin the two properties that make it usable - a script kernel
with GPU + internet enabled - and that its documented handoff still exists,
so a future edit cannot silently drop the route back to a non-GPU upload.
"""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kernels/costsweep-13-local-sweep"
META = KERNEL_DIR / "kernel-metadata.json"
HANDOFF = ROOT / "docs/kaggle-handoff.md"


class KaggleKernelTest(unittest.TestCase):
    def test_metadata_is_valid_and_gpu_internet_script(self):
        meta = json.loads(META.read_text())
        self.assertEqual(meta["kernel_type"], "script")
        self.assertTrue(meta["enable_gpu"])
        self.assertTrue(meta["enable_internet"])
        self.assertTrue(meta["is_private"])
        self.assertTrue((KERNEL_DIR / meta["code_file"]).is_file())
        self.assertTrue(meta["id"].endswith("costsweep13-local-sweep"))

    def test_handoff_documents_cli_commands(self):
        text = HANDOFF.read_text()
        for command in ("kaggle kernels push", "kaggle kernels status",
                        "kaggle kernels output"):
            self.assertIn(command, text)

    def test_kernel_only_runs_local_routes(self):
        # Zero-cloud-spend guard: the kernel must pass L0,L1 (never C*) as
        # live routes to the sweep/repeats.
        src = (KERNEL_DIR / "kernel.py").read_text()
        self.assertIn('"--routes", "L0,L1"', src)
        self.assertIn('"--live-routes", "L0,L1"', src)


if __name__ == "__main__":
    unittest.main()
