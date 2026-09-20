"""tiersweep-16: server-side model attestation.

The audit finding: ``model_version`` was self-declared by the client, so
"measured model X" was not provable. These tests pin the fix - the serving
endpoint's revision/weight hash flows into telemetry, measured rows without
it are flagged (never falsified), and legacy rows migrate cleanly.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from costsmart.eval import oracle_sweep as sweep
from costsmart.models.base import GenerateResult
from costsmart.telemetry import attestation
from costsmart.telemetry.schema import (
    ATTESTATION_ATTESTED,
    ATTESTATION_LEGACY,
    ATTESTATION_STUB,
    ATTESTATION_UNATTESTED,
    ATTEMPT_COLUMNS,
)
from costsmart.telemetry.store import TelemetryStore


def _query(qid="nq:0000"):
    return {"query_id": qid, "question": "what is photosynthesis known for",
            "reference": "photosynthesis"}


class FakeServerClient:
    """Errors out if asked for a cloud route; attests a fake revision."""

    model_id = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, revision="a09a35458c702b33eeacc393d103063234e8bc28",
                 weights="f" * 64):
        self.revision = revision
        self.weights = weights

    def generate(self, prompt, **kwargs):
        raw = {"response": "Final answer: photosynthesis", "eval_count": 3,
               "prompt_eval_count": 42, "latency_s": 1.5}
        if self.revision is not None:
            raw["model_revision"] = self.revision
        if self.weights is not None:
            raw["weights_sha256"] = self.weights
        return GenerateResult(text=raw["response"], tokens=3, latency_s=1.5,
                              raw=raw)


class BuildAttemptAttestationTest(unittest.TestCase):
    def test_measured_row_records_server_revision(self):
        attempt = sweep.build_attempt(
            _query(), "L0", "v1", "sha", "cfg", seed=0, retrieval_ms=6.0,
            live_client=FakeServerClient(),
            live_model_version=FakeServerClient.model_id, temperature=0)
        self.assertEqual(attempt["generator_mode"], "measured")
        self.assertEqual(attempt["attestation"], ATTESTATION_ATTESTED)
        self.assertEqual(attempt["model_revision"],
                         "a09a35458c702b33eeacc393d103063234e8bc28")
        self.assertEqual(attempt["weights_sha256"], "f" * 64)
        self.assertEqual(attempt["model_version"],
                         "Qwen/Qwen2.5-3B-Instruct")

    def test_measured_row_without_attestation_is_flagged(self):
        attempt = sweep.build_attempt(
            _query(), "L0", "v1", "sha", "cfg", seed=0, retrieval_ms=6.0,
            live_client=FakeServerClient(revision=None, weights=None),
            live_model_version=FakeServerClient.model_id, temperature=0)
        self.assertEqual(attempt["generator_mode"], "measured")
        self.assertEqual(attempt["attestation"], ATTESTATION_UNATTESTED)

    def test_stub_row_is_stub_attested(self):
        attempt = sweep.build_attempt(_query(), "C1", "v1", "sha", "cfg",
                                      retrieval_ms=6.0)
        self.assertEqual(attempt["attestation"], ATTESTATION_STUB)


class AttestationStatusTest(unittest.TestCase):
    def test_legacy_measured_row_is_unattested(self):
        row = {"generator_mode": "measured", "attestation": ATTESTATION_LEGACY}
        self.assertEqual(attestation.attestation_status(row),
                         ATTESTATION_UNATTESTED)

    def test_legacy_measured_row_with_revision_is_attested(self):
        row = {"generator_mode": "measured", "attestation": ATTESTATION_LEGACY,
               "model_revision": "abc"}
        self.assertEqual(attestation.attestation_status(row),
                         ATTESTATION_ATTESTED)

    def test_stub_row_stays_stub(self):
        row = {"generator_mode": "stub", "attestation": ATTESTATION_LEGACY}
        self.assertEqual(attestation.attestation_status(row), ATTESTATION_STUB)

    def test_audit_counts_and_examples(self):
        rows = [
            {"generator_mode": "measured", "model_revision": "abc",
             "query_id": "q1", "route_id": "L0"},
            {"generator_mode": "measured", "query_id": "q2", "route_id": "L0"},
            {"generator_mode": "stub", "query_id": "q3", "route_id": "C1"},
        ]
        report = attestation.audit(rows)
        self.assertEqual(report["counts"][ATTESTATION_ATTESTED], 1)
        self.assertEqual(report["counts"][ATTESTATION_UNATTESTED], 1)
        self.assertEqual(report["counts"][ATTESTATION_STUB], 1)
        self.assertEqual(report["unattested_measured"], 1)
        self.assertEqual(report["unattested_examples"][0]["query_id"], "q2")


class AttestationMigrationTest(unittest.TestCase):
    def test_legacy_db_gains_columns_and_flags_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "legacy.db")
            legacy_cols = [c for c in ATTEMPT_COLUMNS
                           if c not in ("model_revision", "weights_sha256",
                                        "attestation")]
            coldefs = ", ".join(f"{c} TEXT" for c in legacy_cols)
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE attempts (id INTEGER PRIMARY KEY"
                         f" AUTOINCREMENT, {coldefs})")
            conn.execute(
                "INSERT INTO attempts (cache_key, query_id, generator_mode)"
                " VALUES ('k1', 'q1', 'measured')")
            conn.commit()
            conn.close()
            store = TelemetryStore(db)
            try:
                cols = {r["name"] for r in store.query(
                    "PRAGMA table_info(attempts)")}
                self.assertTrue({"model_revision", "weights_sha256",
                                 "attestation"} <= cols)
                row = store.query(
                    "SELECT * FROM attempts WHERE cache_key='k1'")[0]
                self.assertEqual(row["attestation"], ATTESTATION_LEGACY)
                # The legacy row is measured but unproven -> flagged.
                self.assertEqual(attestation.attestation_status(row),
                                 ATTESTATION_UNATTESTED)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
