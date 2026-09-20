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
from ..telemetry.schema import (
    ATTESTATION_ATTESTED,
    ATTESTATION_STUB,
    ATTESTATION_UNATTESTED,
)
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

#: Generator sampling contract for every live local-tier call (costsweep-08):
#: temperature 0, fixed seed. The stub path ignores sampling (hash-seeded
#: determinism) but records the same values so rows stay comparable.
GENERATOR_TEMPERATURE = 0
GENERATOR_SEED = 0

#: Telemetry provenance flags recorded per attempt row.
GENERATOR_STUB = "stub"
GENERATOR_MEASURED = "measured"
RETRIEVAL_STUB = "stub"
RETRIEVAL_MEASURED = "measured"


def cache_key(query_id: str, route_id: str, prompt_version: str, model_version: str) -> str:
    """Stable idempotency key: sha256 of the 4-tuple, pipe-joined."""
    raw = "|".join([query_id, route_id, prompt_version, model_version])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def repeat_cache_key(
    query_id: str,
    route_id: str,
    prompt_version: str,
    model_version: str,
    repeat_idx: int,
) -> str:
    """Idempotency key for one stability-repeat draw (costfinal-10).

    Always hashes the 5-tuple (repeat_idx included, even at 0), so a repeat
    row can never collide with a base single-run row for the same
    query+route. ``repeat_idx`` only keys storage - it never changes the
    generator sampling contract (temperature 0, fixed seed, same as the
    sweep draw being repeated).
    """
    raw = "|".join([query_id, route_id, prompt_version, model_version, str(int(repeat_idx))])
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


def build_local_prompt(question: str, passages: list[str], reasoning: str) -> str:
    """Render the generator prompt for one live local-tier call.

    Retrieval passages (when k > 0) are inlined numbered; the reasoning
    strategy selects the instruction block. CoT / rerank prompts require the
    model to put its final answer on a ``Final answer:`` line so
    :func:`extract_final_answer` can grade it deterministically.
    """
    context = ""
    if passages:
        numbered = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        context = f"Context passages:\n{numbered}\n\n"
    if reasoning == "chain-of-thought":
        instruction = (
            "Think step by step, then write your final answer on one line "
            "starting with 'Final answer:'."
        )
    elif reasoning == "rerank-then-answer":
        instruction = (
            "First name the number of the most relevant passage, then write "
            "your final answer on one line starting with 'Final answer:'."
        )
    else:
        instruction = "Answer concisely with just the answer."
    return f"{context}Question: {question}\n{instruction}\nAnswer:"


def extract_final_answer(text: str) -> str:
    """Pull the ``Final answer:`` line when present, else the full text."""
    for line in text.splitlines():
        if line.strip().lower().startswith("final answer:"):
            return line.split(":", 1)[1].strip()
    return text.strip()


def execute_live_local(
    query: dict,
    route_id: str,
    passages: list[str],
    client,
    model_version: str,
    temperature: int = GENERATOR_TEMPERATURE,
    seed: int = GENERATOR_SEED,
) -> dict:
    """Run one local-tier route for real through the Colab session.

    Hard rule: cloud routes NEVER execute live (zero cloud spend by
    construction — there is no cloud client code path in this module).
    Raises RuntimeError for cloud routes and when the client is missing.
    """
    if route_id in CLOUD_ROUTES:
        raise RuntimeError(
            f"refusing live execution for cloud route {route_id}: "
            "cloud-tier routes are stub-estimated only ($0 spent)"
        )
    if client is None:
        raise RuntimeError(
            f"no live client for local route {route_id}: set "
            "COSTSMART_COLAB_ENDPOINT per docs/colab-handoff.md"
        )
    _, k, reasoning = ROUTE_SPECS[route_id]
    prompt = build_local_prompt(query["question"], passages[:k], reasoning)
    result = client.generate(
        prompt,
        temperature=temperature,
        options={"seed": seed, "temperature": temperature},
    )
    raw = result.raw or {}
    server_latency_s = raw.get("latency_s", result.latency_s)
    tokens_out = int(raw.get("eval_count", result.tokens) or 0)
    # Server-tokenized prompt count when the Colab server reports it
    # (scripts/colab_local_tier.py serve); else a whitespace estimate.
    tokens_in = int(raw.get("prompt_eval_count", 0) or len(prompt.split()))
    # Server-side model attestation (tiersweep-16): the endpoint reports the
    # resolved revision / weight hash it actually served, so the row's model
    # identity is proven by the server, not declared by the client.
    return {
        "prediction": extract_final_answer(result.text),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "gpu_seconds": max(0.0, float(server_latency_s)),
        "latency_ms_retrieval": 0.0,  # filled by the caller (measured index time)
        "latency_ms_llm": max(0.0, float(server_latency_s)) * 1000.0,
        "latency_ms_verify": 0.0,
        "latency_ms_total": max(0.0, float(server_latency_s)) * 1000.0,
        "model_version": model_version,
        "model_revision": str(raw.get("model_revision") or ""),
        "weights_sha256": str(raw.get("weights_sha256") or ""),
    }


def build_attempt(
    query: dict,
    route_id: str,
    prompt_version: str,
    sha: str,
    cfg_hash: str,
    seed: int = 0,
    retrieval_ms: float | None = None,
    retrieval_passages: list[str] | None = None,
    live_client=None,
    live_model_version: str | None = None,
    temperature: int = GENERATOR_TEMPERATURE,
) -> dict:
    model_version = ROUTE_MODELS.get(route_id, ROUTE_MODELS["L0"])
    if live_client is not None:
        # Live path is local-tiers only; cloud routes raise inside.
        run = execute_live_local(
            query, route_id, retrieval_passages or [], live_client,
            live_model_version or model_version,
            temperature=temperature, seed=seed,
        )
        model_version = run.pop("model_version")
        generator_mode = GENERATOR_MEASURED
        model_revision = run.pop("model_revision", "")
        weights_sha256 = run.pop("weights_sha256", "")
        attestation = (
            ATTESTATION_ATTESTED if (model_revision or weights_sha256)
            else ATTESTATION_UNATTESTED
        )
    else:
        run = stub_execute(query, route_id, seed)
        generator_mode = GENERATOR_STUB
        model_revision = ""
        weights_sha256 = ""
        attestation = ATTESTATION_STUB
    if retrieval_ms is not None:
        # Measured hybrid-retrieval latency replaces the stub estimate; the
        # generator itself stays a stub (cloud spend estimated, not spent).
        run["latency_ms_retrieval"] = max(0.0, retrieval_ms)
        run["latency_ms_total"] = (
            run["latency_ms_retrieval"] + run["latency_ms_llm"] + run["latency_ms_verify"]
        )
        retrieval_mode = RETRIEVAL_MEASURED
    else:
        retrieval_mode = RETRIEVAL_STUB
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
        "generator_mode": generator_mode,
        "retrieval_mode": retrieval_mode,
        "temperature": temperature,
        "seed": seed,
        "model_revision": model_revision,
        "weights_sha256": weights_sha256,
        "attestation": attestation,
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


def _measured_retrieval(question: str, index: dict, top_k: int = 5) -> tuple[float, list[str]]:
    """Time one hybrid_search over the pilot index; also contract-check features.

    Returns ``(latency_ms, passage_texts)``: the texts back the live
    local-tier prompts (k passages per ROUTE_SPECS), so generation and
    retrieval timing share one index lookup per attempt.
    """
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
    by_id = {c["chunk_id"]: c["text"] for c in index["chunks"]}
    passages = [by_id[h["chunk_id"]] for h in hits if h["chunk_id"] in by_id]
    return dt_ms, passages


def _measured_retrieval_ms(question: str, index: dict, top_k: int = 5) -> float:
    """Back-compat shim: latency only (pilot call sites)."""
    dt_ms, _ = _measured_retrieval(question, index, top_k)
    return dt_ms


def _build_live_clients(
    route_ids: list[str], live_model: str | None = None,
) -> dict[str, object]:
    """Attach Colab clients for the local tiers in this sweep.

    Fails fast when the session is not attached (never a silent stub
    fallback). Cloud routes are excluded by construction — see
    :func:`execute_live_local`.

    ``live_model`` (tiersweep-16) overrides the served model id for every
    live route: the attached endpoint serves ONE model, so a tier sweep runs
    the same routes against progressively stronger checkpoints. The override
    becomes the row's ``model_version`` (hence the cache key), so rows for
    different checkpoints never collide. When ``None`` the historical
    route->tier mapping is used unchanged.
    """
    from ..models.colab_client import COLAB_ENDPOINT_ENV, get_colab_endpoint
    from ..models.registry import create_client, get_client

    tiers = {ROUTE_SPECS[r][0] for r in route_ids if r in LOCAL_ROUTES}
    if not tiers:
        return {}
    if not get_colab_endpoint():
        raise RuntimeError(
            "live local-tier sweep requested but no Colab session is attached: "
            f"set {COLAB_ENDPOINT_ENV} per docs/colab-handoff.md "
            "(the captain connects the Colab session on request)"
        )
    if live_model:
        entry = {"name": "live-served-model", "provider": "colab",
                 "model_id": live_model}
        client = create_client(entry)
        return {tier: client for tier in sorted(tiers)}
    clients: dict[str, object] = {}
    for tier in sorted(tiers):
        clients[tier] = get_client(tier)
    return clients


def run_sweep(
    db_path: str | Path = DEFAULT_DB,
    config_path: str | Path | None = None,
    limit: int | None = 20,
    store: TelemetryStore | None = None,
    index_path: str | Path | None = DEFAULT_INDEX_PATH,
    use_retrieval: bool = True,
    live_local: bool = False,
    live_clients: dict[str, object] | None = None,
    live_routes: tuple[str, ...] = LOCAL_ROUTES,
    live_model: str | None = None,
) -> dict:
    """Run the query x route matrix; resumable via cache_key skips.

    ``limit=None`` runs the full matrix (pilot config = 20 queries x 7 routes
    = 140 attempts). Retrieval latency is measured against the pilot index
    when available; the generator stays a stub (spend estimated, not spent)
    and local tiers stay stubbed (marked preliminary in reports).

    With ``live_local=True``, the ``live_routes`` subset of the local-tier
    routes (default all of L0/L1/C0) executes for real through the Colab
    session at temperature 0 + config seed (``generator_mode='measured'``);
    every other route — all cloud-tier routes ALWAYS — stays stub-estimated
    (``generator_mode='stub'``): zero cloud spend by construction, since no
    live cloud code path exists. Restrict ``live_routes`` to the tiers the
    attached session actually serves (a single-model endpoint must not back
    rows labeled with another model id).
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

    clients: dict[str, object] = {}
    live_set = tuple(r for r in live_routes if r in LOCAL_ROUTES)
    if live_local:
        clients = dict(
            live_clients
            or _build_live_clients(list(live_set), live_model=live_model)
        )

    own_store = store is None
    store = store or TelemetryStore(db_path)
    inserted = skipped = 0
    n_live = n_stub = 0
    try:
        for query, route_id in matrix:
            live_client = None
            live_model_version: str | None = None
            if live_local and route_id in live_set:
                tier = ROUTE_SPECS[route_id][0]
                live_client = clients[tier]
                live_model_version = getattr(live_client, "model_id", tier)
            # Pre-skip rows already recorded: the cache key is computable
            # before execution, so a resumed sweep never re-spends a live
            # local generation (or retrieval lookup) on a completed attempt.
            model_version = live_model_version or ROUTE_MODELS.get(
                route_id, ROUTE_MODELS["L0"])
            if store.has(cache_key(query["query_id"], route_id,
                                   prompt_version, model_version)):
                skipped += 1
                continue
            retrieval_ms = None
            passages: list[str] = []
            if index is not None and query.get("question"):
                retrieval_ms, passages = _measured_retrieval(query["question"], index)
                retrieval_measured += 1
            if live_client is not None:
                n_live += 1
            else:
                n_stub += 1
            attempt = build_attempt(
                query, route_id, prompt_version, sha, cfg_hash, seed,
                retrieval_ms=retrieval_ms,
                retrieval_passages=passages,
                live_client=live_client,
                live_model_version=live_model_version,
                temperature=GENERATOR_TEMPERATURE,
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
        "live_local": live_local,
        "live_routes": list(live_set) if live_local else [],
        "live_model": live_model,
        "generator_measured_attempts": n_live,
        "generator_stubbed_attempts": n_stub,
        "temperature": GENERATOR_TEMPERATURE,
        "seed": seed,
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
    parser.add_argument("--live-local", action="store_true",
                        help="execute local-tier routes (L0/L1/C0) for real via "
                             "the Colab session (temperature 0, config seed); "
                             "cloud routes stay stub-estimated ($0 spent)")
    parser.add_argument("--live-routes", default=",".join(LOCAL_ROUTES),
                        help="comma-separated subset of local routes to run live "
                             "(default all); restrict to the tiers the attached "
                             "session actually serves, e.g. 'L0,L1'")
    parser.add_argument("--live-model", default=None,
                        help="served model id override for every live local route "
                             "(tier sweep: one endpoint, progressively stronger "
                             "checkpoints); recorded as the row's model_version")
    args = parser.parse_args(argv)

    limit = None if args.no_limit else args.limit
    live_routes = tuple(r.strip() for r in args.live_routes.split(",") if r.strip())
    # Store opens inside run_sweep (after the live-attach check), so a
    # refused --live-local run leaves no empty DB artifact behind.
    summary = run_sweep(
        args.db, args.config, limit, store=None,
        index_path=None if args.no_retrieval else args.index,
        use_retrieval=not args.no_retrieval,
        live_local=args.live_local,
        live_routes=live_routes,
        live_model=args.live_model,
    )
    if args.export_csv or args.export_parquet:
        store = TelemetryStore(args.db)
        try:
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
