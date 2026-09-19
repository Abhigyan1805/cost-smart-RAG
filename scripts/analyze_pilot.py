#!/usr/bin/env python3
"""Pilot analysis for costpilot-06 (stdlib only).

Reads the pilot telemetry table and writes the review deliverables:

* ``tokens_by_query_route.csv`` -- measured input/output tokens per query
  per route (deliverable b).
* ``per_route_summary.csv`` -- mean tokens, mean token_f1, mean/sum costs
  and retrieval latency per route.
* ``extrapolation.json`` -- full-sweep (Tier-A 200 queries x 7 routes =
  1400 attempts) cloud-cost extrapolation with the pilot math shown
  (deliverable c).

Usage:
    python scripts/analyze_pilot.py --db results/costpilot-06/pilot.db \\
        --out-dir results/costpilot-06
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

# Full sweep scope the pilot extrapolates to.
FULL_SWEEP_QUERIES = 200
FULL_SWEEP_ROUTES = 7
FULL_SWEEP_ATTEMPTS = FULL_SWEEP_QUERIES * FULL_SWEEP_ROUTES  # 1400


def _rows(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM attempts ORDER BY query_id, route_id")]
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze pilot telemetry")
    parser.add_argument("--db", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    attempts = _rows(args.db)
    if not attempts:
        raise SystemExit(f"no attempts in {args.db}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_pilot = len(attempts)
    n_queries = len({a["query_id"] for a in attempts})
    routes = sorted({a["route_id"] for a in attempts})
    scale = FULL_SWEEP_ATTEMPTS / n_pilot

    # (b) tokens per query per route.
    with open(out_dir / "tokens_by_query_route.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query_id", "route_id", "model_version",
                         "tokens_in", "tokens_out", "token_f1",
                         "cloud_spend_usd", "amortized_usd"])
        for a in attempts:
            writer.writerow([a["query_id"], a["route_id"], a["model_version"],
                             a["tokens_in"], a["tokens_out"], a["token_f1"],
                             a["cloud_spend_usd"], a["amortized_usd"]])

    # Per-route summary.
    per_route = []
    for route in routes:
        rs = [a for a in attempts if a["route_id"] == route]
        n = len(rs)
        mean_in = sum(a["tokens_in"] for a in rs) / n
        mean_out = sum(a["tokens_out"] for a in rs) / n
        cloud = sum(a["cloud_spend_usd"] for a in rs)
        amort = sum(a["amortized_usd"] for a in rs)
        per_route.append({
            "route_id": route,
            "model_version": rs[0]["model_version"],
            "n": n,
            "mean_tokens_in": round(mean_in, 2),
            "mean_tokens_out": round(mean_out, 2),
            "mean_token_f1": round(sum(a["token_f1"] for a in rs) / n, 4),
            "mean_latency_ms_retrieval": round(
                sum(a["latency_ms_retrieval"] for a in rs) / n, 2),
            "pilot_cloud_usd": round(cloud, 6),
            "pilot_amortized_usd": round(amort, 6),
            # Linear extrapolation: same route, 200 queries instead of n.
            "full_sweep_cloud_usd": round(cloud / n * FULL_SWEEP_QUERIES, 4),
            "full_sweep_amortized_usd": round(amort / n * FULL_SWEEP_QUERIES, 4),
        })
    with open(out_dir / "per_route_summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_route[0].keys()))
        writer.writeheader()
        writer.writerows(per_route)

    pilot_cloud = sum(a["cloud_spend_usd"] for a in attempts)
    pilot_amort = sum(a["amortized_usd"] for a in attempts)
    extrapolation = {
        "pilot": {
            "n_queries": n_queries,
            "routes": routes,
            "n_attempts": n_pilot,
            "pilot_cloud_usd": round(pilot_cloud, 6),
            "pilot_amortized_usd": round(pilot_amort, 6),
        },
        "full_sweep": {
            "n_queries": FULL_SWEEP_QUERIES,
            "n_routes": FULL_SWEEP_ROUTES,
            "n_attempts": FULL_SWEEP_ATTEMPTS,
            "scale_factor_vs_pilot": scale,
        },
        "math": (
            f"full = pilot_sum x ({FULL_SWEEP_ATTEMPTS}/{n_pilot})"
            f" = pilot_sum x {scale}; per-route full = "
            f"route_pilot_sum / {n_queries} x {FULL_SWEEP_QUERIES}"
        ),
        "full_sweep_cloud_usd": round(pilot_cloud * scale, 4),
        "full_sweep_amortized_usd_preliminary": round(pilot_amort * scale, 4),
        "per_route": per_route,
        "caveats": [
            "Generator outputs are stub estimates (no API calls, $0 spent); "
            "cloud USD = measured stub tokens x verified per-token prices.",
            "Local-tier gpu_seconds are stubbed until the Colab session "
            "attaches; amortized figures are PRELIMINARY.",
            "Extrapolation assumes per-route token/cost distributions from "
            "the 20-query stratified slice hold for the full 200-query Tier-A set.",
        ],
    }
    (out_dir / "extrapolation.json").write_text(json.dumps(extrapolation, indent=2))
    print(json.dumps({
        "n_attempts": n_pilot,
        "pilot_cloud_usd": round(pilot_cloud, 6),
        "pilot_amortized_usd": round(pilot_amort, 6),
        "full_sweep_cloud_usd": extrapolation["full_sweep_cloud_usd"],
        "full_sweep_amortized_usd_preliminary":
            extrapolation["full_sweep_amortized_usd_preliminary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
