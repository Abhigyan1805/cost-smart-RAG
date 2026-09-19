#!/usr/bin/env python3
"""Stable-oracle rebuild + headroom recount for costfinal-10.

Reads the frozen single-run sweep matrix (read-only) plus the measured
repeat draws, grades every live local-tier pair by majority (ties count as
incorrect - see costsmart.eval.stability), rebuilds the oracle matrix on
stable labels, and recomputes the headroom gate (contingency +
routable_fraction + max_savings + bootstrap CIs + McNemar) for BOTH the
single-run matrix and the stable matrix, so the report can state exactly
whether single-run label noise would have changed any conclusion.

Writes results/costfinal-10/:
  stable_labels.json    per-pair votes, verdict, tie flags
  stable_attempts.json  oracle matrix rows (representative repeat draws for
                        repeated routes, base single-run rows otherwise)
  flip_report.json      string-flip + label-flip distributions per route
  headroom_single.json/.svg  gate on single-run labels (the Week-2 baseline)
  headroom_stable.json/.svg  gate on stable majority labels (the recount)
  comparison.json       single-vs-stable label changes + verdict deltas

Usage:
    python scripts/stable_oracle.py --sweep-db results/costsweep-08/sweep.db \\
        --repeats-db results/costfinal-10/repeats.db \\
        --out-dir results/costfinal-10
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from costsmart.eval import stability  # noqa: E402
from costsmart.eval.metrics import CORRECT_THRESHOLD  # noqa: E402
from costsmart.telemetry.schema import N_REPEATS  # noqa: E402
from make_plots import compute_headroom, render_headroom_svg  # noqa: E402


def load_rows(db_path: str, table: str = "attempts") -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stable oracle rebuild")
    parser.add_argument("--sweep-db", default="results/costsweep-08/sweep.db")
    parser.add_argument("--repeats-db",
                        default="results/costfinal-10/repeats.db")
    parser.add_argument("--out-dir", default="results/costfinal-10")
    parser.add_argument("--cheap", default="L0")
    parser.add_argument("--strong", default="C4")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    base_rows = load_rows(args.sweep_db, "attempts")
    try:
        repeat_rows = load_rows(args.repeats_db, "repeat_attempts")
    except sqlite3.OperationalError as exc:
        raise SystemExit(f"cannot load repeats from {args.repeats_db}: {exc}")

    base_by_pair = {(r["query_id"], r["route_id"]): r for r in base_rows}
    repeats_by_pair: dict[tuple[str, str], list[dict]] = {}
    for r in repeat_rows:
        repeats_by_pair.setdefault((r["query_id"], r["route_id"]), []).append(r)

    # Only routes with a full repeat set get stable labels; every other
    # route (cloud stubs, C0 stub) keeps its single-run row - deterministic
    # stubs have no sampling noise to stabilize.
    repeated_routes = sorted({route for (_, route) in repeats_by_pair})
    stable_rows: list[dict] = []
    labels: list[dict] = []
    changed: list[dict] = []
    for (qid, route), reps in sorted(repeats_by_pair.items()):
        base = base_by_pair.get((qid, route))
        if len(reps) < N_REPEATS:
            print(f"warning: {qid}/{route} has {len(reps)}/{N_REPEATS} "
                  f"draws - conservative tie-rule label applies",
                  file=sys.stderr)
        verdict = stability.stable_label_for_pair(reps, base)
        rep = verdict["representative"]
        if rep is None:
            raise SystemExit(f"no representative row for {qid}/{route}")
        stable_row = dict(rep)
        stable_row.update({
            "stable_correct": verdict["stable_correct"],
            "stable_n_correct": verdict["n_correct"],
            "stable_n_total": verdict["n_total"],
            "stable_tie": verdict["tie"],
            "stable_complete": verdict["complete"],
        })
        stable_rows.append(stable_row)
        single_correct = (float((base or {}).get("token_f1") or 0.0)
                          >= CORRECT_THRESHOLD) if base else None
        labels.append({
            "query_id": qid,
            "route_id": route,
            "single_correct": single_correct,
            "stable_correct": verdict["stable_correct"],
            "n_correct": verdict["n_correct"],
            "n_total": verdict["n_total"],
            "complete": verdict["complete"],
            "tie": verdict["tie"],
        })
        if single_correct is not None and bool(single_correct) != verdict["stable_correct"]:
            changed.append({"query_id": qid, "route_id": route,
                            "single_correct": bool(single_correct),
                            "stable_correct": verdict["stable_correct"]})

    # Stable oracle matrix = representative rows for repeated pairs + base
    # single-run rows for everything else (one row per pair, same shape as
    # the sweep matrix so metrics/stats/make_plots run unchanged).
    repeated_pairs = set(repeats_by_pair)
    stable_matrix = list(stable_rows) + [
        r for (qid, route), r in base_by_pair.items()
        if (qid, route) not in repeated_pairs
    ]

    # Flip distributions per repeated route (string-level: what the
    # generator did; label-level: what the oracle sees).
    flip_report: dict[str, dict] = {}
    for route in repeated_routes:
        per_pair_texts: list[list[str]] = []
        vote_lists: list[list[bool]] = []
        for (qid, r), reps in sorted(repeats_by_pair.items()):
            if r != route:
                continue
            ordered = sorted(reps, key=lambda x: x.get("repeat_idx", 0))
            per_pair_texts.append([x.get("prediction", "") for x in ordered])
            vote_lists.append([float(x.get("token_f1") or 0.0) >= CORRECT_THRESHOLD
                               for x in ordered])
        flip_report[route] = {
            "string": stability.flip_stats(per_pair_texts),
            "label": stability.label_flip_stats(vote_lists),
        }

    single_gate = compute_headroom(base_rows, args.cheap, args.strong,
                                   args.resamples, args.seed)
    stable_gate = compute_headroom(stable_matrix, args.cheap, args.strong,
                                   args.resamples, args.seed + 1000)
    comparison = {
        "cheap_route": args.cheap,
        "strong_route": args.strong,
        "n_repeated_pairs": len(repeats_by_pair),
        "repeated_routes": repeated_routes,
        "n_label_changes_single_vs_stable": len(changed),
        "label_changes": changed,
        "single": {
            "routable_fraction": single_gate["routable_fraction"]["estimate"],
            "routable_ci": [single_gate["routable_fraction"]["ci_lo"],
                            single_gate["routable_fraction"]["ci_hi"]],
            "saving_fraction": single_gate["savings"]["saving_fraction"],
            "verdict": single_gate["gate"]["verdict"],
            "mcnemar_p": single_gate["paired"]["mcnemar_p"],
        },
        "stable": {
            "routable_fraction": stable_gate["routable_fraction"]["estimate"],
            "routable_ci": [stable_gate["routable_fraction"]["ci_lo"],
                            stable_gate["routable_fraction"]["ci_hi"]],
            "saving_fraction": stable_gate["savings"]["saving_fraction"],
            "verdict": stable_gate["gate"]["verdict"],
            "mcnemar_p": stable_gate["paired"]["mcnemar_p"],
        },
        "verdict_changed": (single_gate["gate"]["verdict"]
                            != stable_gate["gate"]["verdict"]),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stable_labels.json").write_text(json.dumps(labels, indent=2))
    (out_dir / "stable_attempts.json").write_text(
        json.dumps(stable_matrix, indent=2))
    (out_dir / "flip_report.json").write_text(
        json.dumps(flip_report, indent=2))
    (out_dir / "headroom_single.json").write_text(
        json.dumps(single_gate, indent=2))
    (out_dir / "headroom_single.svg").write_text(
        render_headroom_svg(single_gate))
    (out_dir / "headroom_stable.json").write_text(
        json.dumps(stable_gate, indent=2))
    (out_dir / "headroom_stable.svg").write_text(
        render_headroom_svg(stable_gate))
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
