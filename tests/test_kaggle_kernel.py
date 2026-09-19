"""costsweep-13: Kaggle GPU route contract.

The Kaggle kernel is the fallback compute host when the Colab free tier is
exhausted. These pin the two properties that make it usable - a script kernel
with GPU + internet enabled - and that its documented handoff still exists,
so a future edit cannot silently drop the route back to a non-GPU upload.
The provenance test binds the ingested DBs to the persisted kernel log hash
and runbook summary.
"""

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kernels/costsweep-13-local-sweep"
META = KERNEL_DIR / "kernel-metadata.json"
HANDOFF = ROOT / "docs/kaggle-handoff.md"
PROVENANCE = ROOT / "results/costmultihop-12/kaggle_provenance.json"
INGEST_DIR = ROOT / "results/costmultihop-12"


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


class KaggleProvenanceTest(unittest.TestCase):
    def _provenance(self):
        self.assertTrue(PROVENANCE.is_file())
        return json.loads(PROVENANCE.read_text())

    def test_log_hash_and_runbook_summary_recorded(self):
        prov = self._provenance()
        self.assertEqual(prov["kernel"]["id"],
                         "abhigyan1818/costsweep13-local-sweep")
        log = prov["logs"]["costsweep13-local-sweep.log"]
        self.assertEqual(len(log["sha256"]), 64)
        self.assertIn("kaggle kernels output", prov["download"]["command"])
        summary = prov["runbook_summary"]
        self.assertEqual(summary["sweep_rows"], 1400)
        self.assertEqual(summary["repeat_rows"], 1200)
        self.assertEqual(summary["repeat_pairs"], 400)

    def test_recorded_artifact_hashes_match_committed_dbs(self):
        # If a DB is ever regenerated, its provenance record must be updated
        # in the same change; this keeps the resume audit honest.
        artifacts = self._provenance()["artifacts"]
        for name, meta in artifacts.items():
            digest = hashlib.sha256((INGEST_DIR / name).read_bytes()).hexdigest()
            self.assertEqual(digest, meta["sha256"], name)

    def test_provenance_doc_references_record(self):
        doc = (INGEST_DIR / "KAGGLE_PROVENANCE.md").read_text()
        self.assertIn("kaggle_provenance.json", doc)
        self.assertIn("kaggle kernels output", doc)


if __name__ == "__main__":
    unittest.main()
