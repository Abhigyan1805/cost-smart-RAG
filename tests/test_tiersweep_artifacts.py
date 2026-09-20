"""tiersweep-16: committed break-even artifacts + attestation provenance.

Pins the real numbers so a later edit cannot silently change the verdict or
mix unattested rows into the attested tiers.
"""

import hashlib
import json
import sqlite3
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/tiersweep-16"


def _tier(model_suffix):
    data = json.loads((RESULTS / "tiers.json").read_text())
    for tier in data["tiers"]:
        if tier["model_version"].endswith(model_suffix):
            return tier
    raise AssertionError(f"tier {model_suffix} missing from tiers.json")


class TierNumbersTest(unittest.TestCase):
    def test_break_even_is_3b_l0(self):
        data = json.loads((RESULTS / "tiers.json").read_text())
        first = data["break_even"]["first_go_l0"]
        self.assertIsNotNone(first)
        self.assertEqual(first["model_version"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(first["route"], "L0")
        self.assertGreaterEqual(first["coverage_ci"][0], 0.10)
        self.assertTrue(first["decision"].startswith("GO"))

    def test_l0_coverage_by_tier(self):
        self.assertAlmostEqual(_tier("1.5B-Instruct")["routes"]["L0"]["coverage"],
                               0.095, places=3)
        self.assertAlmostEqual(_tier("3B-Instruct")["routes"]["L0"]["coverage"],
                               0.155, places=3)
        self.assertAlmostEqual(_tier("7B-Instruct")["routes"]["L0"]["coverage"],
                               0.150, places=3)

    def test_table_records_break_even_statement(self):
        text = (RESULTS / "TIERS.md").read_text()
        self.assertIn("first clears the gate", text)
        self.assertIn("Qwen/Qwen2.5-3B-Instruct", text)


class TierAttestationTest(unittest.TestCase):
    def test_new_tier_measured_rows_are_server_attested(self):
        for tag in ("3b", "7b"):
            conn = sqlite3.connect(str(RESULTS / f"sweep-{tag}.db"))
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT DISTINCT attestation, model_revision, weights_sha256"
                    " FROM attempts WHERE generator_mode='measured'").fetchall()
            finally:
                conn.close()
            self.assertEqual(len(rows), 1, tag)
            self.assertEqual(rows[0]["attestation"], "server-attested", tag)
            self.assertEqual(len(rows[0]["model_revision"]), 40, tag)
            self.assertEqual(len(rows[0]["weights_sha256"]), 64, tag)

    def test_audit_flags_reused_1p5b_but_not_new_tiers(self):
        audit = json.loads((RESULTS / "attestation_audit.json").read_text())
        for name, entry in audit.items():
            unattested = (entry["attempts"]["unattested_measured"]
                          + entry["repeat_attempts"]["unattested_measured"])
            if "realdata-15" in name:
                self.assertGreater(unattested, 0, name)
            else:
                self.assertEqual(unattested, 0, name)


class TierProvenanceTest(unittest.TestCase):
    def test_artifact_hashes_match_committed_files(self):
        prov = json.loads((RESULTS / "kaggle_provenance.json").read_text())
        for name, meta in prov["artifacts"].items():
            digest = hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
            self.assertEqual(digest, meta["sha256"], name)

    def test_manifest_records_revisions_and_licences(self):
        text = (RESULTS / "MODEL_MANIFEST.md").read_text()
        self.assertIn("aa8e72537993ba99e69dfaafa59ed015b17504d1", text)
        self.assertIn("a09a35458c702b33eeacc393d103063234e8bc28", text)
        self.assertIn("Qwen Research License", text)


if __name__ == "__main__":
    unittest.main()
