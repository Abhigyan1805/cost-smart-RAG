"""Guards for scripts/stable_oracle.py (unittest, stdlib-only).

The script has two incompatible mixes in the repo (legacy pilot vs
multi-hop) and used to default to the legacy artifacts. These tests pin the
fix: a bare invocation must refuse, and a sweep/repeats pair drawn from
different mixes must refuse before writing anything.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from costsmart.telemetry.schema import ATTEMPT_COLUMNS, REPEAT_COLUMNS
from costsmart.telemetry.store import TelemetryStore

ROOT = Path(__file__).resolve().parents[1]

_NOT_NULL_REPEAT_DEFAULTS = {
    "cache_key": "k", "repeat_idx": 0, "query_id": "not-in-sweep",
    "route_id": "L0", "prompt_version": "v1", "model_version": "m",
    "prediction": "x", "reference": "x", "exact_match": 0, "lenient_em": 0,
    "token_f1": 1.0, "judge_model": "", "tokens_in": 0, "tokens_out": 0,
    "gpu_seconds": 0.0, "concurrency": 1, "cloud_spend_usd": 0.0,
    "amortized_usd": 0.0, "latency_ms_total": 0.0,
    "latency_ms_retrieval": 0.0, "latency_ms_llm": 0.0,
    "latency_ms_verify": 0.0, "git_sha": "", "config_hash": "",
    "created_at": "", "generator_mode": "measured",
    "retrieval_mode": "measured", "temperature": 0, "seed": 0,
}


def _full_repeat_row(**overrides):
    row = {k: _NOT_NULL_REPEAT_DEFAULTS.get(k) for k in REPEAT_COLUMNS}
    row.update(overrides)
    return row


def _full_attempt_row(**overrides):
    row = {k: _NOT_NULL_REPEAT_DEFAULTS.get(k) for k in ATTEMPT_COLUMNS}
    row.update(overrides)
    return row


def _load_stable_oracle():
    path = ROOT / "scripts" / "stable_oracle.py"
    spec = importlib.util.spec_from_file_location("stable_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stable_oracle"] = mod
    spec.loader.exec_module(mod)
    return mod


class StableOracleGuardTest(unittest.TestCase):
    def test_bare_invocation_refuses(self):
        mod = _load_stable_oracle()
        with self.assertRaises(SystemExit):
            mod.main([])

    def test_mismatched_mixes_refuse(self):
        mod = _load_stable_oracle()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sweep = tmp / "sweep.db"
            repeats = tmp / "repeats.db"
            with TelemetryStore(sweep) as store:
                self.assertTrue(store.insert_attempt(
                    _full_attempt_row(query_id="q1", route_id="L0")))
            with TelemetryStore(repeats) as store:
                self.assertTrue(store.insert_repeat(_full_repeat_row()))
            with self.assertRaises(SystemExit):
                mod.main(["--sweep-db", str(sweep),
                          "--repeats-db", str(repeats),
                          "--out-dir", str(tmp / "out")])


if __name__ == "__main__":
    unittest.main()
