"""Train the v1 featurized router on sweep telemetry (``make train-router``).

Labels come from the oracle post-hoc: 1 when the cheapest correct route for
the query is a cloud route, else 0. Query text is re-attached from the Tier-A
corpus slice by query_id (falling back to the query_id string when the corpus
record is absent, e.g. toy sweeps). Default mode is ``pre`` (query-text only,
stdlib-only builtin backend) so training runs anywhere without sklearn.

Usage:
    python -m costsmart.routing.train --db results/costpilot-06/pilot.db \\
        --out results/costpilot-06/router-v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..eval.baselines import oracle_route
from ..eval.oracle_sweep import CLOUD_ROUTES
from ..telemetry.store import TelemetryStore
from .featurized import FeaturizedRouter

DEFAULT_OUT = "router-v1.json"


def _question_lookup() -> dict[str, str]:
    """query_id -> question text from the Tier-A corpus slice (offline)."""
    try:
        from ..corpus import loaders as corpus_loaders

        queries, _ = corpus_loaders.load_pilot_subset(source="synthetic")
        return {q["query_id"]: q.get("question", "") for q in queries}
    except Exception:
        return {}


def train_on_db(
    db_path: str | Path,
    out_path: str | Path = DEFAULT_OUT,
    mode: str = "pre",
    threshold: float = 0.5,
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

    questions = _question_lookup()
    query_ids = sorted(by_query)
    texts = [questions.get(qid, qid) for qid in query_ids]
    oracle_routes = [oracle_route(qid, by_query[qid]) for qid in query_ids]
    labels = [1 if route in CLOUD_ROUTES else 0 for route in oracle_routes]

    router = FeaturizedRouter(mode=mode, backend="builtin", threshold=threshold)
    router.fit(texts, labels)

    artifact = {
        "mode": router.mode,
        "backend": "builtin",
        "threshold": router.threshold,
        "weights": list(router.weights),
        "bias": router.bias,
        "mean": list(getattr(router, "_mean", [])),
        "var": list(getattr(router, "_var", [])),
        "n_features": router.n_features_,
        "trained_on": {
            "db": str(db_path),
            "n_queries": len(query_ids),
            "n_attempts": len(attempts),
            "cloud_label_rate": sum(labels) / len(labels),
        },
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    return {
        "out": str(out),
        "n_queries": len(query_ids),
        "n_attempts": len(attempts),
        "cloud_label_rate": sum(labels) / len(labels),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train v1 router on sweep telemetry")
    parser.add_argument("--db", default="telemetry.db", help="SQLite telemetry path")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Router artifact JSON path")
    parser.add_argument("--mode", default="pre", choices=["pre", "post", "both"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    print(json.dumps(train_on_db(args.db, args.out, args.mode, args.threshold), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
