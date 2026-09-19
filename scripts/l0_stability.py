#!/usr/bin/env python3
"""L0 stability check for costsweep-08 (stdlib only + repo imports).

Runs the L0 route (local-small, k=0, direct) ``repeats`` times over
``n_queries`` Tier-A queries and reports the flip rate: the fraction of
queries whose repeat predictions are not all identical (exact string).

* ``--mode stub``: deterministic stub executor (expects flip rate 0.0;
  smoke-tests the harness with $0 spent and no Colab session).
* ``--mode live``: real generations through the Colab session at
  temperature 0 + fixed seed (requires COSTSMART_COLAB_ENDPOINT; this is the
  reported L0 stability number).

Repeats intentionally do NOT write to the sweep telemetry table: stability
repeats share one (query_id, route_id, prompt_version, model_version)
4-tuple, so cache_key idempotency would skip repeats 2..N. Results go to a
standalone JSON report instead.

Usage:
    python scripts/l0_stability.py --mode stub --n-queries 100 --repeats 3
    python scripts/l0_stability.py --mode live --out results/costsweep-08/l0_stability.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from costsmart.eval.oracle_sweep import (  # noqa: E402
    GENERATOR_SEED,
    GENERATOR_TEMPERATURE,
    build_local_prompt,
    extract_final_answer,
    load_tier_a_queries,
)


def _stub_predict(query: dict, repeat: int) -> str:
    """One deterministic stub prediction (hash-seeded, repeat-stable)."""
    from costsmart.eval.oracle_sweep import stub_execute

    return stub_execute(query, "L0", seed=GENERATOR_SEED)["prediction"]


def _live_predict(query: dict, client, model_id: str) -> str:
    prompt = build_local_prompt(query["question"], [], "direct")
    result = client.generate(
        prompt,
        temperature=GENERATOR_TEMPERATURE,
        options={"seed": GENERATOR_SEED, "temperature": GENERATOR_TEMPERATURE},
    )
    return extract_final_answer(result.text)


def flip_rate(predictions: list[list[str]]) -> dict:
    """Flip stats over per-query repeat-prediction lists."""
    n = len(predictions)
    flipped = sum(1 for reps in predictions if len(set(reps)) > 1)
    return {
        "n_queries": n,
        "repeats": len(predictions[0]) if predictions else 0,
        "flipped": flipped,
        "flip_rate": (flipped / n) if n else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L0 stability check (flip rate)")
    parser.add_argument("--mode", choices=("stub", "live"), default="stub")
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--counts", default="nq:40,triviaqa:35,hotpotqa:25",
                        help="stratified Tier-A slice for the 100 queries")
    parser.add_argument("--out", default=None, help="JSON report path")
    args = parser.parse_args(argv)

    counts = {}
    for chunk in args.counts.split(","):
        name, num = chunk.split(":")
        counts[name.strip()] = int(num)
    queries = load_tier_a_queries(counts)
    queries = queries[: args.n_queries]
    if len(queries) < args.n_queries:
        raise SystemExit(
            f"only {len(queries)} queries for counts={counts}, "
            f"wanted {args.n_queries}"
        )

    client = None
    if args.mode == "live":
        from costsmart.models.colab_client import COLAB_ENDPOINT_ENV, get_colab_endpoint
        from costsmart.models.registry import get_client

        if not get_colab_endpoint():
            raise SystemExit(
                f"live mode needs a Colab session: set {COLAB_ENDPOINT_ENV} "
                "per docs/colab-handoff.md"
            )
        client = get_client("local-small")

    per_query = []
    all_reps = []
    for q in queries:
        if args.mode == "live":
            reps = [_live_predict(q, client, "") for _ in range(args.repeats)]
        else:
            reps = [_stub_predict(q, r) for r in range(args.repeats)]
        all_reps.append(reps)
        per_query.append({"query_id": q["query_id"], "predictions": reps,
                          "flipped": len(set(reps)) > 1})

    stats = flip_rate(all_reps)
    report = {
        "mode": args.mode,
        "route": "L0",
        "temperature": GENERATOR_TEMPERATURE,
        "seed": GENERATOR_SEED,
        "query_counts": counts,
        "query_set_sha256": hashlib.sha256(
            json.dumps([q["query_id"] for q in queries]).encode()
        ).hexdigest(),
        **stats,
        "per_query": per_query,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "per_query"}, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
