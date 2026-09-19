#!/usr/bin/env python3
"""Compute judge-vs-human Cohen's kappa from the captain's labels file.

Reads the completed hand-labelling CSV/JSONL (captain supplies it; see
``docs/judge_labelling_guide.md``), prints kappa + the >= 0.7 gate verdict.

Exit codes: 0 = gate passed, 1 = gate failed, 2 = usage/data error.

Usage:
    PYTHONPATH=src python scripts/compute_kappa.py captain_labels.csv
    PYTHONPATH=src python scripts/compute_kappa.py captain_labels.csv --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys

from costsmart.eval.graders import KAPPA_GATE, kappa_from_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge-vs-human kappa gate check.")
    parser.add_argument("labels", help="captain's completed labels file (.csv/.jsonl)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="binarize numeric scores at this threshold (omit for discrete labels)",
    )
    parser.add_argument("--gate", type=float, default=KAPPA_GATE, help="kappa gate (default 0.7)")
    args = parser.parse_args(argv)

    try:
        result = kappa_from_file(args.labels, threshold=args.threshold, gate=args.gate)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    verdict = "PASS" if result["passes_gate"] else "FAIL"
    print(f"kappa={result['kappa']:.3f} n={result['n']} gate={result['gate']} -> {verdict}")
    if not result["passes_gate"]:
        print(
            "gate failed: recalibrate or replace the judge before trusting "
            "judge_score in sweep analysis",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
