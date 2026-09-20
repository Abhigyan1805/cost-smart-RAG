#!/usr/bin/env python3
"""Attestation audit: flag measured rows without server-side model provenance.

Reads one or more telemetry DBs (attempts + repeat_attempts) and reports, per
DB, how many measured rows carry a server-reported revision/weight hash
(``server-attested``) versus how many were measured but cannot prove which
weights produced them (``unattested``). Absent attestation is a flag, not a
falsification: committed legacy rows keep their labels and are counted here.

Usage:
    python scripts/attestation_audit.py --db results/realdata-15/sweep.db
    python scripts/attestation_audit.py --db a.db --db b.db --json out.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from costsmart.telemetry import attestation  # noqa: E402


def _load(db_path: str, table: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        try:
            rows = [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
        except sqlite3.OperationalError:
            return []
    finally:
        conn.close()
    return rows


def audit_db(db_path: str) -> dict:
    attempts = _load(db_path, "attempts")
    repeats = _load(db_path, "repeat_attempts")
    return {
        "db": db_path,
        "attempts": attestation.audit(attempts),
        "repeat_attempts": attestation.audit(repeats),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measured-model attestation audit")
    parser.add_argument("--db", action="append", required=True,
                        help="telemetry DB to audit (repeatable)")
    parser.add_argument("--json", default=None, help="write the full payload here")
    args = parser.parse_args(argv)

    report = {db: audit_db(db) for db in args.db}
    payload = json.dumps(report, indent=2)
    if args.json:
        Path(args.json).write_text(payload)
    print(payload)
    # Non-zero exit when any measured row is unattested: the audit is a gate.
    unattested = sum(
        entry["attempts"]["unattested_measured"]
        + entry["repeat_attempts"]["unattested_measured"]
        for entry in report.values()
    )
    return 1 if unattested else 0


if __name__ == "__main__":
    raise SystemExit(main())
