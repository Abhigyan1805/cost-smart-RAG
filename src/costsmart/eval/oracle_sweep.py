"""Oracle sweep harness: query x route matrix -> telemetry attempts table.

Resumable + idempotent: each attempt's ``cache_key`` is
``sha256(query_id | route_id | prompt_version | model_version)`` with a
UNIQUE constraint, so re-running after an interruption (e.g. an 18k-run
killed halfway) skips recorded attempts and only executes the remainder.

Pilot executor is a deterministic stub (hash-seeded pseudo latency/tokens,
canned predictions): cloud spend stays *estimated, not spent*. Real
retrieval/LLM calls land in a later slice behind the same interface.

Usage:
    python -m costsmart.eval.oracle_sweep --limit 20
    python -m costsmart.eval.oracle_sweep --limit 200 --db telemetry.db \\
        --config config/experiments/E1.yaml --export-parquet attempts.parquet
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from ..telemetry import cost as cost_mod
from ..telemetry.store import TelemetryStore
from . import graders

# Route -> model + tier. L0/L1/C0 run local (no metered cloud spend);
# C1..C4 are cloud routes (metered per-token prices, still estimated here).
#
# Week-1 7-route sweep set (model x retrieval depth x reasoning), the full
# sweep the pilot extrapolates to (Tier-A 200 queries x 7 routes):
#   L0: local-small,  k=0,  direct            (closed-book local baseline)
#   L1: local-small,  k=5,  direct
#   C0: local-medium, k=5,  chain-of-thought
#   C1: cloud-small,  k=5,  direct
#   C2: cloud-small,  k=10, rerank-then-answer
#   C3: cloud-large,  k=5,  direct
#   C4: cloud-large,  k=10, rerank-then-answer
ROUTE_MODELS = {
    "L0": "ollama-qwen2.5-3b",
    "L1": "ollama-qwen2.5-3b",
    "C0": "ollama-llama3.1-8b",
    "C1": "gpt-4o-mini",
    "C2": "gpt-4o-mini",
    "C3": "gpt-4o",
    "C4": "gpt-4o",
}

# Route metadata for reports: (model tier, retrieval depth k, reasoning).
ROUTE_SPECS = {
    "L0": ("local-small", 0, "direct"),
    "L1": ("local-small", 5, "direct"),
    "C0": ("local-medium", 5, "chain-of-thought"),
    "C1": ("cloud-small", 5, "direct"),
    "C2": ("cloud-small", 10, "rerank-then-answer"),
    "C3": ("cloud-large", 5, "direct"),
    "C4": ("cloud-large", 10, "rerank-then-answer"),
}

# Cloud routes answer through metered APIs (estimated here, not spent);
# local routes run on Colab GPUs (stubbed until the captain connects Colab).
CLOUD_ROUTES = ("C1", "C2", "C3", "C4")
LOCAL_ROUTES = ("L0", "L1", "C0")

PROMPT_VERSION = "v1"
DEFAULT_ROUTES = ("L0", "L1", "C0", "C1", "C2", "C3", "C4")
DEFAULT_DB = "telemetry.db"


def cache_key(query_id: str, route_id: str, prompt_version: str, model_version: str) -> str:
    """Stable idempotency key: sha256 of the 4-tuple, pipe-joined."""
    raw = "|".join([query_id, route_id, prompt_version, model_version])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git_sha() -> str:
    """Current git SHA; env override (GIT_SHA) or 'unknown' outside a repo."""
    env = os.environ.get("GIT_SHA")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def config_hash(config_text: str) -> str:
    """sha256 of the canonical experiment config text."""
    return hashlib.sha256(config_text.encode("utf-8")).hexdigest()


def load_config(config_path: str | Path | None) -> tuple[dict, str]:
    """Load experiment YAML (stdlib fallback) -> (config, config_hash)."""
    if config_path is None:
        defaults = default_config()
        text = json.dumps(defaults, sort_keys=True)
        return defaults, config_hash(text)
    path = Path(config_path)
    text = path.read_text()
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(text) or {}
    except ImportError:
        cfg = _mini_yaml(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {config_path} must be a mapping")
    return cfg, config_hash(text)


def _mini_yaml(text: str) -> dict:
    """Tiny key: value / key: [a, b] subset parser (PyYAML absent fallback)."""
    cfg: dict = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip() or line.startswith((" ", "\t")) or ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            cfg[key.strip()] = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
        elif val in ("", "~", "null"):
            cfg[key.strip()] = None
        else:
            try:
                cfg[key.strip()] = int(val)
            except ValueError:
                try:
                    cfg[key.strip()] = float(val)
                except ValueError:
                    cfg[key.strip()] = val.strip("'\"")
    return cfg


def default_config() -> dict:
    return {
        "name": "pilot-default",
        "routes": list(DEFAULT_ROUTES),
        "prompt_version": PROMPT_VERSION,
        "n_queries": 50,
        "seed": 0,
        "query_source": "toy",
    }


# Default Tier-A pilot slice for a 20-query pilot: stratified 8 NQ + 7
# TriviaQA + 5 HotpotQA (same query-id scheme as the full 200-query pilot).
DEFAULT_TIER_A_COUNTS = {"nq": 8, "triviaqa": 7, "hotpotqa": 5}

DEFAULT_INDEX_PATH = "data/index/pilot_index.json"


def load_tier_a_queries(
    counts: dict | None = None,
    source: str = "synthetic",
) -> list[dict]:
    """Stratified Tier-A queries via the corpus slice (offline synthetic default).

    Maps corpus records to sweep query dicts: ``reference`` is the first gold
    answer ("" when the record has none). Raises RuntimeError when a slice is
    empty so a silent empty sweep never looks like a result.
    """
    from ..corpus import loaders as corpus_loaders

    counts = dict(counts or DEFAULT_TIER_A_COUNTS)
    queries: list[dict] = []
    for dataset, n in counts.items():
        ds_queries, _ = corpus_loaders.load_dataset_queries(dataset, n=n, source=source)
        for rec in ds_queries:
            answers = rec.get("answers") or []
            queries.append(
                {
                    "query_id": rec["query_id"],
                    "question": rec.get("question", ""),
                    "reference": answers[0] if answers else "",
                }
            )
    if not queries:
        raise RuntimeError(f"tier-a query load produced 0 queries for counts={counts}")
    return queries


def load_query_set(cfg: dict) -> list[dict]:
    """Resolve the sweep query set from config (tier-a corpus or toy fallback)."""
    if (cfg.get("query_source") or "toy") == "tier-a":
        counts = cfg.get("tier_a_counts") or DEFAULT_TIER_A_COUNTS
        return load_tier_a_queries(counts, source=cfg.get("corpus_source") or "synthetic")
    n_queries = int(cfg.get("n_queries") or 50)
    return pilot_queries(n_queries, int(cfg.get("seed") or 0))


def pilot_queries(n: int, seed: int = 0) -> list[dict]:
    """Deterministic synthetic query set (standalone until corpus slice lands)."""
    topics = [
        ("capital-of-france", "What is the capital of France?", "Paris"),
        ("boiling-point", "At what temperature does water boil at sea level?", "100 degrees Celsius"),
        ("author-1984", "Who wrote the novel 1984?", "George Orwell"),
        ("speed-of-light", "What is the speed of light in vacuum?", "299792458 meters per second"),
        ("largest-planet", "Which is the largest planet in the solar system?", "Jupiter"),
    ]
    queries = []
    for i in range(max(0, n)):
        slug, question, answer = topics[i % len(topics)]
        queries.append(
            {
                "query_id": f"q{i:04d}-{slug}",
                "question": f"{question} (variant {i // len(topics)})",
                "reference": answer,
            }
        )
    return queries


def stub_execute(query: dict, route_id: str, seed: int = 0) -> dict:
    """Deterministic stub execution: no network, no spend - estimates only.

    Local routes answer from the reference with light paraphrase noise;
    cloud routes answer exactly (the oracle gap the sweep measures).
    Latency/tokens derive from sha256(query_id + route_id) for stability.
    """
    digest = hashlib.sha256(f"{seed}:{query['query_id']}:{route_id}".encode()).digest()
    pick = lambda lo, hi, byte: lo + (byte / 255) * (hi - lo)  # noqa: E731
    reference = query["reference"]
    if route_id in CLOUD_ROUTES:
        prediction = reference
    elif route_id == "L0" and digest[0] % 3 == 0:
        # Closed-book local baseline genuinely misses ~1/3 of queries: the
        # oracle gap the sweep measures (cost vs correctness trade-off).
        prediction = "I don't know"
    elif digest[0] % 8 == 0:
        # Retrieval-augmented local tiers return a partial answer on ~1/8 of
        # queries: lower token_f1 but still above the 0.5 oracle threshold.
        words = reference.split()
        prediction = " ".join(words[: max(1, len(words) // 2)])
    elif digest[0] % 4 == 0:
        prediction = reference.lower()
    else:
        prediction = reference
    tokens_in = 80 + digest[1] % 120
    tokens_out = 8 + digest[2] % 40
    lat_retrieval = pick(15, 120, digest[3])
    lat_llm = pick(200, 2500, digest[4]) if route_id in CLOUD_ROUTES else pick(400, 4000, digest[4])
    lat_verify = pick(5, 60, digest[5])
    return {
        "prediction": prediction,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "gpu_seconds": (lat_llm / 1000) if route_id in LOCAL_ROUTES else 0.0,
        "latency_ms_retrieval": lat_retrieval,
        "latency_ms_llm": lat_llm,
        "latency_ms_verify": lat_verify,
        "latency_ms_total": lat_retrieval + lat_llm + lat_verify,
    }


def build_attempt(
    query: dict,
    route_id: str,
    prompt_version: str,
    sha: str,
    cfg_hash: str,
    seed: int = 0,
    retrieval_ms: float | None = None,
) -> dict:
    model_version = ROUTE_MODELS.get(route_id, ROUTE_MODELS["L0"])
    run = stub_execute(query, route_id, seed)
    if retrieval_ms is not None:
        # Measured hybrid-retrieval latency replaces the stub estimate; the
        # generator itself stays a stub (cloud spend estimated, not spent).
        run["latency_ms_retrieval"] = max(0.0, retrieval_ms)
        run["latency_ms_total"] = (
            run["latency_ms_retrieval"] + run["latency_ms_llm"] + run["latency_ms_verify"]
        )
    graded = graders.grade(run["prediction"], query["reference"])
    costs = cost_mod.cost_record(
        run["tokens_in"],
        run["tokens_out"],
        model_version,
        gpu_seconds=run["gpu_seconds"],
    )
    return {
        "cache_key": cache_key(query["query_id"], route_id, prompt_version, model_version),
        "query_id": query["query_id"],
        "route_id": route_id,
        "prompt_version": prompt_version,
        "model_version": model_version,
        "prediction": run["prediction"],
        "reference": query["reference"],
        **graded,
        "tokens_in": costs["tokens_in"],
        "tokens_out": costs["tokens_out"],
        "gpu_seconds": costs["gpu_seconds"],
        "concurrency": costs["concurrency"],
        "cloud_spend_usd": costs["cloud_spend_usd"],
        "amortized_usd": costs["amortized_usd"],
        "latency_ms_total": run["latency_ms_total"],
        "latency_ms_retrieval": run["latency_ms_retrieval"],
        "latency_ms_llm": run["latency_ms_llm"],
        "latency_ms_verify": run["latency_ms_verify"],
        "git_sha": sha,
        "config_hash": cfg_hash,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def _load_retrieval_index(index_path: str | Path | None) -> dict | None:
    """Load the pilot index for measured retrieval latency (None = stub)."""
    if not index_path:
        return None
    path = Path(index_path)
    if not path.exists():
        print(f"warning: index {path} missing; retrieval latency stays stubbed", file=sys.stderr)
        return None
    return json.loads(path.read_text())


def _measured_retrieval_ms(question: str, index: dict, top_k: int = 5) -> float:
    """Time one hybrid_search over the pilot index; also contract-check features."""
    import time as _time

    from ..retrieval.features import FEATURE_NAMES, extract_routing_features
    from ..retrieval.hybrid import hybrid_search

    t0 = _time.perf_counter()
    hits = hybrid_search(
        question,
        index["chunks"],
        index["bm25"],
        index["dense_vectors"],
        top_k=top_k,
    )
    dt_ms = (_time.perf_counter() - t0) * 1000.0
    feats = extract_routing_features(question, hits)
    missing = [k for k in FEATURE_NAMES if k not in feats]
    if missing:
        raise RuntimeError(f"retrieval feature contract broken, missing: {missing}")
    return dt_ms


def run_sweep(
    db_path: str | Path = DEFAULT_DB,
    config_path: str | Path | None = None,
    limit: int | None = 20,
    store: TelemetryStore | None = None,
    index_path: str | Path | None = DEFAULT_INDEX_PATH,
    use_retrieval: bool = True,
) -> dict:
    """Run the query x route matrix; resumable via cache_key skips.

    ``limit=None`` runs the full matrix (pilot config = 20 queries x 7 routes
    = 140 attempts). Retrieval latency is measured against the pilot index
    when available; the generator stays a stub (spend estimated, not spent)
    and local tiers stay stubbed (marked preliminary in reports).
    """
    cfg, cfg_hash = load_config(config_path)
    sha = git_sha()
    routes = cfg.get("routes") or list(DEFAULT_ROUTES)
    prompt_version = cfg.get("prompt_version") or PROMPT_VERSION
    seed = int(cfg.get("seed") or 0)
    queries = load_query_set(cfg)

    matrix = [(q, r) for q in queries for r in routes]
    if limit is not None:
        matrix = matrix[: max(0, limit)]

    index = _load_retrieval_index(index_path) if use_retrieval else None
    retrieval_measured = 0

    own_store = store is None
    store = store or TelemetryStore(db_path)
    inserted = skipped = 0
    try:
        for query, route_id in matrix:
            retrieval_ms = None
            if index is not None and query.get("question"):
                retrieval_ms = _measured_retrieval_ms(query["question"], index)
                retrieval_measured += 1
            attempt = build_attempt(
                query, route_id, prompt_version, sha, cfg_hash, seed,
                retrieval_ms=retrieval_ms,
            )
            if store.insert_attempt(attempt):
                inserted += 1
            else:
                skipped += 1
        total = store.count()
    finally:
        if own_store:
            store.close()
    return {
        "attempted": len(matrix),
        "inserted": inserted,
        "skipped_existing": skipped,
        "total_rows": total,
        "git_sha": sha,
        "config_hash": cfg_hash,
        "query_source": cfg.get("query_source") or "toy",
        "retrieval_latency": "measured" if index is not None else "stubbed",
        "retrieval_measured_attempts": retrieval_measured,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Oracle sweep -> telemetry table")
    parser.add_argument("--limit", type=int, default=20, help="max attempts (default 20)")
    parser.add_argument("--no-limit", action="store_true", help="run the full matrix")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite path")
    parser.add_argument("--config", default=None, help="experiment YAML path")
    parser.add_argument("--export-parquet", default=None, help="also export Parquet here")
    parser.add_argument("--export-csv", default=None, help="also export CSV here")
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH,
                        help="pilot index for measured retrieval latency")
    parser.add_argument("--no-retrieval", action="store_true",
                        help="skip the index; stub all retrieval latency")
    args = parser.parse_args(argv)

    limit = None if args.no_limit else args.limit
    store = TelemetryStore(args.db)
    try:
        summary = run_sweep(
            args.db, args.config, limit, store=store,
            index_path=None if args.no_retrieval else args.index,
            use_retrieval=not args.no_retrieval,
        )
        if args.export_csv:
            store.export_csv(args.export_csv)
        if args.export_parquet:
            try:
                store.export_parquet(args.export_parquet)
            except RuntimeError as exc:
                print(f"warning: {exc}", file=sys.stderr)
    finally:
        store.close()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
