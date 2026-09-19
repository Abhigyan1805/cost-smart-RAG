"""Evaluate router vs exhaustive oracle on sweep telemetry (``make eval``).

Per query, the oracle route is the cheapest attempt with token_f1 >= 0.5
(see ``baselines.oracle_route``). Every ``always-<route>`` baseline, the
seeded random baseline, and (optionally) a trained v1 router artifact are
scored with routing accuracy + cost regret + quality gap from
``costsmart.eval.metrics``. Costs combine both accounting columns
(cloud_spend_usd + amortized_usd).

The v0 tier-to-route mapping for a trained router is documented, not tuned:
CLOUD tier -> C1 (cloud-small, k=5, direct), LOCAL tier -> L1 (local-small,
k=5, direct). A learned route-level policy lands in a later slice.

Usage:
    python -m costsmart.eval.evaluate --db results/costpilot-06/pilot.db
    python -m costsmart.eval.evaluate --db pilot.db --router router-v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import metrics
from .baselines import BASELINES, oracle_route
from ..telemetry.store import TelemetryStore

# v0 tier-to-route mapping (documented default; see module docstring).
TIER_ROUTE = {"cloud": "C1", "local": "L1"}


def _total_cost(attempt: dict) -> float:
    return (attempt.get("cloud_spend_usd") or 0.0) + (attempt.get("amortized_usd") or 0.0)


def _pick_for_query(query_id: str, attempts: list[dict], route_id: str) -> dict | None:
    cands = [a for a in attempts if a.get("query_id") == query_id and a.get("route_id") == route_id]
    if not cands:
        cands = [a for a in attempts if a.get("query_id") == query_id]
        if not cands:
            return None
        cands = sorted(cands, key=_total_cost)
    return cands[0]


def _load_router(router_path: str | Path | None):
    if not router_path:
        return None
    from ..routing.featurized import FeaturizedRouter

    artifact = json.loads(Path(router_path).read_text())
    router = FeaturizedRouter(
        mode=artifact.get("mode", "pre"),
        backend="builtin",
        threshold=float(artifact.get("threshold", 0.5)),
    )
    router.weights = list(artifact["weights"])
    router.bias = float(artifact["bias"])
    router._mean = list(artifact.get("mean", []))
    router._var = list(artifact.get("var", [1.0] * len(router.weights)))
    router.n_features_ = int(artifact.get("n_features", len(router.weights)))
    return router


def evaluate_db(
    db_path: str | Path,
    router_path: str | Path | None = None,
    seed: int = 0,
) -> dict:
    store = TelemetryStore(db_path)
    try:
        attempts = store.fetch_all()
    finally:
        store.close()
    if not attempts:
        raise RuntimeError(f"no attempts in {db_path}; run the sweep first")

    by_query: dict[str, list[dict]] = {}
    for attempt in attempts:
        by_query.setdefault(attempt["query_id"], []).append(attempt)
    query_ids = sorted(by_query)

    oracle_routes = [oracle_route(qid, by_query[qid]) for qid in query_ids]
    oracle_picks = [_pick_for_query(qid, attempts, route) for qid, route in zip(query_ids, oracle_routes)]
    oracle_costs = [_total_cost(a) for a in oracle_picks]
    oracle_f1 = [(a.get("token_f1") or 0.0) for a in oracle_picks]

    try:
        from ..corpus import loaders as corpus_loaders

        corpus, _ = corpus_loaders.load_pilot_subset(source="synthetic")
        questions = {q["query_id"]: q.get("question", "") for q in corpus}
    except Exception:
        questions = {}

    results: dict[str, dict] = {}
    for name, fn in BASELINES.items():
        if name == "oracle":
            continue
        if name == "random":
            predicted = [fn(qid, seed) for qid in query_ids]
        else:
            predicted = [fn(qid) for qid in query_ids]
        picks = [_pick_for_query(qid, attempts, route) for qid, route in zip(query_ids, predicted)]
        costs = [_total_cost(a) for a in picks]
        f1s = [(a.get("token_f1") or 0.0) for a in picks]
        reg = metrics.regret(costs, oracle_costs, f1s, oracle_f1)
        results[name] = {
            "routing_accuracy": metrics.routing_accuracy(predicted, oracle_routes),
            **reg,
        }

    router = _load_router(router_path)
    if router is not None:
        predicted = []
        for qid in query_ids:
            decision = router.decide(questions.get(qid, qid))
            predicted.append(TIER_ROUTE["cloud" if decision.route == "cloud" else "local"])
        picks = [_pick_for_query(qid, attempts, route) for qid, route in zip(query_ids, predicted)]
        costs = [_total_cost(a) for a in picks]
        f1s = [(a.get("token_f1") or 0.0) for a in picks]
        reg = metrics.regret(costs, oracle_costs, f1s, oracle_f1)
        results["router-v1"] = {
            "routing_accuracy": metrics.routing_accuracy(predicted, oracle_routes),
            "tier_mapping": dict(TIER_ROUTE),
            **reg,
        }

    return {
        "db": str(db_path),
        "n_queries": len(query_ids),
        "n_attempts": len(attempts),
        "oracle_mean_cost_usd": sum(oracle_costs) / len(oracle_costs),
        "oracle_mean_token_f1": sum(oracle_f1) / len(oracle_f1),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate router vs oracle on telemetry")
    parser.add_argument("--db", default="telemetry.db", help="SQLite telemetry path")
    parser.add_argument("--router", default=None, help="Trained router artifact JSON")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random baseline")
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_db(args.db, args.router, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
