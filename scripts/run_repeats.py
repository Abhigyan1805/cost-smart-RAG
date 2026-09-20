#!/usr/bin/env python3
"""Stability repeats for costfinal-10 (stdlib only + repo imports).

Re-runs every live local-tier (query, route) pair from the merged sweep
matrix ``N_REPEATS`` (3) times under the same sampling contract as the sweep
(temperature 0, fixed seed; repeat_idx only keys storage, never sampling)
and stores each draw in the ``repeat_attempts`` table of a SEPARATE repeats
DB (default ``results/costfinal-10/repeats.db``) - the frozen sweep matrix
is opened read-only and never modified.

Hard rules (enforced, not just documented):
  * cloud routes NEVER execute: target pairs are enumerated from sweep rows
    with ``generator_mode='measured'`` on local routes only; anything else
    aborts. There is no live cloud code path.
  * live mode NEVER writes stub rows: a draw whose ``generator_mode`` is
    not ``'measured'`` aborts the run instead of being stored.
  * repeat draws NEVER overwrite base rows: separate table + 5-tuple repeat
    keys (see ``repeat_cache_key``), resumable via INSERT OR IGNORE.

``--mode stub`` rehearses enumeration, keying, storage, and the downstream
majority pipeline with the deterministic stub executor ($0, no Colab) into
a scratch DB - it refuses the default live repeats path so rehearsal rows
can never mix with measured repeats.

Usage:
    python scripts/run_repeats.py --mode stub --db /tmp/rehearsal.db --limit 6
    COSTSMART_COLAB_ENDPOINT=<tunnel> python scripts/run_repeats.py --mode live
    COSTSMART_COLAB_ENDPOINT=<tunnel> python scripts/run_repeats.py --mode live \\
        --routes L0 --limit 10   # smoke slice (resumable: re-run to continue)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from costsmart.eval.oracle_sweep import (  # noqa: E402
    GENERATOR_SEED,
    GENERATOR_TEMPERATURE,
    LOCAL_ROUTES,
    build_attempt,
    load_config,
    load_tier_a_queries,
    repeat_cache_key,
)
from costsmart.telemetry.schema import N_REPEATS  # noqa: E402
from costsmart.telemetry.store import TelemetryStore  # noqa: E402

DEFAULT_SWEEP_DB = "results/costsweep-08/sweep.db"
DEFAULT_REPEATS_DB = "results/costfinal-10/repeats.db"
DEFAULT_INDEX_PATH = "data/index/pilot_index.json"


def load_target_pairs(sweep_db: str, routes: list[str]) -> list[dict]:
    """Live local-tier pairs needing repeats, from the merged sweep matrix.

    Only rows with ``generator_mode='measured'`` on local routes qualify
    (sweep-08: L0/L1 on the 1.5B endpoint; C0 stayed stub - no 7B endpoint -
    so it is excluded by construction, not by special-case). Anything
    outside LOCAL_ROUTES aborts: cloud pairs must never enter the repeat
    set.
    """
    for route in routes:
        if route not in LOCAL_ROUTES:
            raise SystemExit(
                f"refusing repeat target {route!r}: repeats cover live "
                f"local-tier pairs only ({', '.join(LOCAL_ROUTES)}); cloud "
                "routes have no live code path ($0 spent)"
            )
    conn = sqlite3.connect(sweep_db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT DISTINCT query_id, route_id, prompt_version, model_version"
            " FROM attempts WHERE generator_mode = 'measured'"
            " AND route_id IN (%s) ORDER BY query_id, route_id"
            % ",".join("?" for _ in routes),
            tuple(routes),
        ).fetchall()
    finally:
        conn.close()
    pairs = [dict(r) for r in rows]
    non_local = [p for p in pairs if p["route_id"] not in LOCAL_ROUTES]
    if non_local:
        raise SystemExit(f"measured non-local pairs in sweep matrix: {non_local[:3]}")
    if not pairs:
        raise SystemExit(
            f"no measured local-tier pairs for routes={routes} in {sweep_db}"
        )
    return pairs


def load_query_map(cfg: dict, needed: set[str]) -> dict[str, dict]:
    """Tier-A question text + reference keyed by query_id (offline synthetic).

    The sweep DB stores references but not question text; reload the frozen
    corpus slice (same counts/source as the sweep config) and require every
    needed query_id to resolve so a silent question mismatch never labels
    the wrong prompt.
    """
    counts = cfg.get("tier_a_counts") or {"nq": 80, "triviaqa": 70, "hotpotqa": 50}
    source = cfg.get("corpus_source") or "synthetic"
    queries = load_tier_a_queries(dict(counts), source=source)
    qmap = {q["query_id"]: q for q in queries}
    missing = needed - set(qmap)
    if missing:
        raise SystemExit(
            f"{len(missing)} sweep query_ids missing from the corpus slice "
            f"(counts={counts}, source={source}), e.g. {sorted(missing)[:3]}"
        )
    return qmap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stability repeat draws")
    parser.add_argument("--mode", choices=("live", "stub"), default="live")
    parser.add_argument("--sweep-db", default=DEFAULT_SWEEP_DB)
    parser.add_argument("--db", default=DEFAULT_REPEATS_DB,
                        help="repeats DB (separate file from the sweep DB)")
    parser.add_argument("--config",
                        default="config/experiments/sweep-200-tier-a.yaml")
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--routes", default="L0,L1",
                        help="comma-separated local routes (default L0,L1)")
    parser.add_argument("--live-model", default=None,
                        help="served model id override for every live local "
                             "route (tier sweep); must match the sweep rows' "
                             "model_version or the run refuses to mix models")
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on (query, route) pairs (smoke slices)")
    parser.add_argument("--retries", type=int, default=3,
                        help="transient-error retries per draw before aborting")
    parser.add_argument("--retry-sleep", type=float, default=30.0,
                        help="seconds between draw retries")
    parser.add_argument("--out", default=None, help="JSON summary path")
    args = parser.parse_args(argv)

    if args.mode == "stub" and Path(args.db) == Path(DEFAULT_REPEATS_DB):
        raise SystemExit(
            "stub rehearsal refuses the live repeats path: pass --db to a "
            "scratch file so stub rows can never mix with measured repeats"
        )
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    pairs = load_target_pairs(args.sweep_db, routes)
    if args.limit is not None:
        pairs = pairs[: max(0, args.limit)]

    cfg, cfg_hash = load_config(args.config)
    from costsmart.eval.oracle_sweep import git_sha
    sha = git_sha()
    seed = int(cfg.get("seed") or 0)
    prompt_default = cfg.get("prompt_version") or "v1"
    qmap = load_query_map(cfg, {p["query_id"] for p in pairs})

    # Retrieval index (same pilot index + timed hybrid lookup the sweep used).
    index = None
    if not args.no_retrieval:
        from costsmart.eval.oracle_sweep import _load_retrieval_index
        index = _load_retrieval_index(args.index)

    # Live clients per tier (fails fast with no Colab session; cloud tiers
    # have no client by construction). Stub mode uses no clients.
    clients: dict[str, object] = {}
    if args.mode == "live":
        from costsmart.eval.oracle_sweep import _build_live_clients
        clients = _build_live_clients(routes, live_model=args.live_model)

    store = TelemetryStore(args.db)
    try:
        import time as _time

        from costsmart.eval.oracle_sweep import ROUTE_SPECS, _measured_retrieval
        inserted = skipped = 0
        draws = 0
        retrieval_cache: dict = {}
        for pair in pairs:
            route_id = pair["route_id"]
            query = qmap[pair["query_id"]]
            tier = ROUTE_SPECS[route_id][0]
            live_client = clients.get(tier) if args.mode == "live" else None
            live_model = (getattr(live_client, "model_id", None)
                          if live_client is not None else None)
            if args.mode == "live":
                # The attached endpoint must serve the SAME model id the
                # sweep measured for this tier; otherwise repeat draws would
                # be mislabeled (the costsweep-08 C0 gap: never grade 1.5B
                # outputs as 7B).
                if live_model != pair["model_version"]:
                    raise SystemExit(
                        f"attached endpoint serves {live_model!r} but sweep "
                        f"pair {pair['query_id']}/{route_id} measured "
                        f"{pair['model_version']!r}: refusing to mix models"
                    )
            for repeat_idx in range(args.repeats):
                key = repeat_cache_key(
                    pair["query_id"], route_id,
                    pair.get("prompt_version") or prompt_default,
                    live_model or pair["model_version"], repeat_idx)
                if store.has_repeat(key):
                    skipped += 1
                    continue
                retrieval_ms = None
                passages: list[str] = []
                if index is not None and query.get("question"):
                    retrieval_ms, passages = _measured_retrieval(
                        query["question"], index, cache=retrieval_cache)
                attempt = None
                last_err: Exception | None = None
                # Transient tunnel/generation errors retry the same draw
                # (same sampling contract); exhaustion aborts loudly rather
                # than storing a partial or stub row. Fatal SystemExits
                # (model mismatch, stub-as-measured) never retry.
                for trial in range(max(1, args.retries) + 1):
                    try:
                        attempt = build_attempt(
                            query, route_id,
                            pair.get("prompt_version") or prompt_default,
                            sha, cfg_hash, seed,
                            retrieval_ms=retrieval_ms,
                            retrieval_passages=passages,
                            live_client=live_client,
                            live_model_version=live_model,
                            temperature=GENERATOR_TEMPERATURE,
                        )
                        break
                    except SystemExit:
                        raise
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:  # noqa: BLE001 - retry then abort
                        last_err = exc
                        if trial < max(1, args.retries):
                            print(f"  retry {trial + 1}/{args.retries} for "
                                  f"{pair['query_id']}/{route_id} "
                                  f"r{repeat_idx}: {exc}",
                                  file=sys.stderr, flush=True)
                            _time.sleep(args.retry_sleep)
                if attempt is None:
                    raise SystemExit(
                        f"draw failed after {args.retries} retries for "
                        f"{pair['query_id']}/{route_id} r{repeat_idx}: "
                        f"{last_err} (completed draws stay stored; re-run "
                        "to resume)"
                    )
                if args.mode == "live" and attempt["generator_mode"] != "measured":
                    raise SystemExit(
                        "live draw came back generator_mode="
                        f"{attempt['generator_mode']!r} for "
                        f"{pair['query_id']}/{route_id} r{repeat_idx}: "
                        "refusing to store a stub row as a measured repeat"
                    )
                attempt["repeat_idx"] = repeat_idx
                attempt["cache_key"] = key
                if store.insert_repeat(attempt):
                    inserted += 1
                else:
                    skipped += 1
                draws += 1
                if draws % 25 == 0:
                    print(f"  ... {draws} draws attempted "
                          f"({inserted} new, {skipped} skipped)",
                          file=sys.stderr, flush=True)
        total = store.count_repeats()
    finally:
        store.close()

    summary = {
        "mode": args.mode,
        "sweep_db": args.sweep_db,
        "repeats_db": args.db,
        "routes": routes,
        "live_model": args.live_model,
        "n_pairs": len(pairs),
        "repeats": args.repeats,
        "draws_expected": len(pairs) * args.repeats,
        "inserted": inserted,
        "skipped_existing": skipped,
        "total_repeat_rows": total,
        "temperature": GENERATOR_TEMPERATURE,
        "seed": seed,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
