#!/usr/bin/env python3
"""Kaggle batch kernel: costsweep-13 L0/L1 live local-tier sweep + repeats.

Kaggle is the GPU route when the Colab free tier is exhausted (see
``docs/kaggle-handoff.md``). This kernel is a thin driver: it clones the
public repo at the task branch, rebuilds the frozen multihop retrieval
index, starts the repo's own transformers server
(``scripts/colab_local_tier.py serve``) on the Kaggle GPU, then drives the
committed runbook against the localhost endpoint:

  1. base matrix  - L0/L1 measured, C0..C4 stub, resume via cache keys
  2. 3x repeats   - one extra-threshold-stable majority label per
                    measured L0/L1 pair, resume via 5-tuple repeat keys

Zero cloud spend by construction: cloud routes are never executed (there is
no live cloud code path in the repo), and no stub row can overwrite a
measured row (INSERT OR IGNORE on the cache key). The sweep and repeats DBs
are committed inputs (preserved partial run) so the kernel only spends GPU
on work that is actually missing; every artifact is copied to
``/kaggle/working`` for ``kaggle kernels output``.

Push / monitor / download (from the repo root)::

    kaggle kernels push   -p kernels/costsweep-13-local-sweep
    kaggle kernels status abhigyan1818/costsweep13-local-sweep
    kaggle kernels output abhigyan1818/costsweep13-local-sweep -p /tmp/kout

See ``docs/kaggle-handoff.md`` for the ingest + recount commands.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_URL = "https://github.com/Abhigyan1805/cost-smart-RAG.git"
BRANCH = "fm/costsweep-13"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PORT = 8000
WORK = "/kaggle/working"
CLONE = os.path.join(WORK, "repo")
ENDPOINT = f"http://127.0.0.1:{PORT}"
ARTIFACTS = (
    "results/costmultihop-12/sweep.db",
    "results/costmultihop-12/repeats.db",
    "results/costmultihop-12/sweep.csv",
    "results/costmultihop-12/repeats_summary.json",
)


def run(cmd: list[str], cwd: str, env: dict, desc: str) -> None:
    print(f"\n=== {desc} ===\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def clone_repo() -> None:
    if os.path.isdir(os.path.join(CLONE, ".git")):
        print(f"reusing existing clone at {CLONE}", flush=True)
        return
    if os.path.isdir(CLONE):
        shutil.rmtree(CLONE)
    run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, CLONE],
        cwd=WORK, env=os.environ.copy(), desc=f"clone {BRANCH}",
    )


def ensure_deps() -> None:
    """Kaggle images ship torch/transformers; upgrade only what is missing."""
    for module, package in (("transformers", "transformers"),
                            ("accelerate", "accelerate"),
                            ("torch", "torch")):
        try:
            __import__(module)
        except ImportError:
            print(f"{module} missing; pip installing {package}", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", package],
                           check=True)
    import transformers

    print(f"transformers {transformers.__version__}", flush=True)


def build_index() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    run(
        [sys.executable, "-m", "costsmart.corpus.build_index",
         "--mix", "multihop", "--source", "synthetic",
         "--out", "data/index/multihop_index.json"],
        cwd=CLONE, env=env, desc="build frozen multihop index",
    )


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    log = open(os.path.join(WORK, "local_tier_server.log"), "w")
    proc = subprocess.Popen(
        [sys.executable, "scripts/colab_local_tier.py", "serve",
         "--model", MODEL, "--port", str(PORT)],
        cwd=CLONE, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    print(f"server pid={proc.pid}; waiting for {ENDPOINT} ...", flush=True)
    return proc


def wait_healthy(proc: subprocess.Popen, timeout_s: float = 2400.0) -> None:
    body = json.dumps({"model": MODEL, "prompt": "Say OK.",
                       "stream": False}).encode()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited early rc={proc.returncode}; see "
                f"{WORK}/local_tier_server.log")
        try:
            req = urllib.request.Request(
                ENDPOINT + "/api/generate", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
            print(f"server healthy: {str(payload)[:120]}", flush=True)
            return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"  not ready yet ({exc}); retrying", flush=True)
            time.sleep(15)
    raise RuntimeError(f"server did not become healthy within {timeout_s}s")


def drive_runbook() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["COSTSMART_COLAB_ENDPOINT"] = ENDPOINT

    # 1) base matrix: L0/L1 measured, C0..C4 stub; stored cache keys skip
    #    the preserved live Colab rows, so this only fills genuine gaps.
    run(
        [sys.executable, "-m", "costsmart.eval.oracle_sweep", "--no-limit",
         "--db", "results/costmultihop-12/sweep.db",
         "--config", "config/experiments/sweep-200-multihop.yaml",
         "--index", "data/index/multihop_index.json",
         "--live-local", "--live-routes", "L0,L1",
         "--export-csv", "results/costmultihop-12/sweep.csv"],
        cwd=CLONE, env=env, desc="base sweep (L0/L1 measured, cloud stub)",
    )

    # 2) stability repeats: 3 draws per measured L0/L1 pair, resumable.
    run(
        [sys.executable, "scripts/run_repeats.py", "--mode", "live",
         "--sweep-db", "results/costmultihop-12/sweep.db",
         "--db", "results/costmultihop-12/repeats.db",
         "--config", "config/experiments/sweep-200-multihop.yaml",
         "--index", "data/index/multihop_index.json",
         "--routes", "L0,L1", "--retries", "3", "--retry-sleep", "10",
         "--out", "results/costmultihop-12/repeats_summary.json"],
        cwd=CLONE, env=env, desc="3x stability repeats",
    )

    # 3) counts / provenance summary for the ingest step.
    import sqlite3

    summary: dict = {}
    sweep_db = os.path.join(CLONE, "results/costmultihop-12/sweep.db")
    repeats_db = os.path.join(CLONE, "results/costmultihop-12/repeats.db")
    conn = sqlite3.connect(sweep_db)
    try:
        summary["sweep_rows"] = conn.execute(
            "SELECT COUNT(*) FROM attempts").fetchone()[0]
        summary["sweep_by_route_mode"] = [
            {"route_id": r[0], "generator_mode": r[1], "n": r[2]}
            for r in conn.execute(
                "SELECT route_id, generator_mode, COUNT(*) FROM attempts"
                " GROUP BY 1,2 ORDER BY 1")]
    finally:
        conn.close()
    conn = sqlite3.connect(repeats_db)
    try:
        summary["repeat_rows"] = conn.execute(
            "SELECT COUNT(*) FROM repeat_attempts").fetchone()[0]
        summary["repeat_pairs"] = conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT query_id, route_id"
            " FROM repeat_attempts)").fetchone()[0]
        summary["repeat_by_route_mode"] = [
            {"route_id": r[0], "generator_mode": r[1], "n": r[2]}
            for r in conn.execute(
                "SELECT route_id, generator_mode, COUNT(*) FROM repeat_attempts"
                " GROUP BY 1,2 ORDER BY 1")]
    finally:
        conn.close()
    print("RUNBOOK SUMMARY: " + json.dumps(summary), flush=True)
    return summary


def collect_outputs(summary: dict) -> None:
    for rel in ARTIFACTS:
        src = os.path.join(CLONE, rel)
        if not os.path.exists(src):
            print(f"WARN missing artifact {rel}", flush=True)
            continue
        dst = os.path.join(WORK, os.path.basename(rel))
        shutil.copy2(src, dst)
        print(f"copied {rel} -> {dst}", flush=True)
    with open(os.path.join(WORK, "kaggle_run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    # Keep kernel output small: drop the clone (DBs already copied above).
    shutil.rmtree(CLONE, ignore_errors=True)


def main() -> int:
    clone_repo()
    ensure_deps()
    build_index()
    proc = start_server()
    try:
        wait_healthy(proc)
        summary = drive_runbook()
        collect_outputs(summary)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
