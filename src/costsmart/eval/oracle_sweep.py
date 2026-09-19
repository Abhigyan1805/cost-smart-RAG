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

# Route -> model + tier. L0/C0 run local (no metered cloud spend);
# C1/C2 are cloud routes (metered per-token prices, still estimated here).
ROUTE_MODELS = {
    "L0": "ollama-qwen2.5-3b",
    "C0": "ollama-llama3.1-8b",
    "C1": "gpt-4o-mini",
    "C2": "gpt-4o",
}

PROMPT_VERSION = "v1"
DEFAULT_ROUTES = ("L0", "C0", "C1", "C2")
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
    }


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
    if route_id in ("C1", "C2"):
        prediction = reference
    elif digest[0] % 4 == 0:
        prediction = reference.lower()
    else:
        prediction = reference
    tokens_in = 80 + digest[1] % 120
    tokens_out = 8 + digest[2] % 40
    lat_retrieval = pick(15, 120, digest[3])
    lat_llm = pick(200, 2500, digest[4]) if route_id.startswith("C") else pick(400, 4000, digest[4])
    lat_verify = pick(5, 60, digest[5])
    return {
        "prediction": prediction,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "gpu_seconds": (lat_llm / 1000) if route_id in ("L0", "C0") else 0.0,
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
) -> dict:
    model_version = ROUTE_MODELS.get(route_id, ROUTE_MODELS["L0"])
    run = stub_execute(query, route_id, seed)
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


def run_sweep(
    db_path: str | Path = DEFAULT_DB,
    config_path: str | Path | None = None,
    limit: int | None = 20,
    store: TelemetryStore | None = None,
) -> dict:
    """Run the query x route matrix; resumable via cache_key skips."""
    cfg, cfg_hash = load_config(config_path)
    sha = git_sha()
    routes = cfg.get("routes") or list(DEFAULT_ROUTES)
    prompt_version = cfg.get("prompt_version") or PROMPT_VERSION
    seed = int(cfg.get("seed") or 0)
    n_queries = int(cfg.get("n_queries") or 50)
    queries = pilot_queries(n_queries, seed)

    matrix = [(q, r) for q in queries for r in routes]
    if limit is not None:
        matrix = matrix[: max(0, limit)]

    own_store = store is None
    store = store or TelemetryStore(db_path)
    inserted = skipped = 0
    try:
        for query, route_id in matrix:
            attempt = build_attempt(query, route_id, prompt_version, sha, cfg_hash, seed)
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
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Oracle sweep -> telemetry table")
    parser.add_argument("--limit", type=int, default=20, help="max attempts (default 20)")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite path")
    parser.add_argument("--config", default=None, help="experiment YAML path")
    parser.add_argument("--export-parquet", default=None, help="also export Parquet here")
    parser.add_argument("--export-csv", default=None, help="also export CSV here")
    args = parser.parse_args(argv)

    store = TelemetryStore(args.db)
    try:
        summary = run_sweep(args.db, args.config, args.limit, store=store)
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
