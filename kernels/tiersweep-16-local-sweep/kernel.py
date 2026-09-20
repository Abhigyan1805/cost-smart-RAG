#!/usr/bin/env python3
"""Kaggle batch kernel: tiersweep-16 cheap-tier break-even sweep.

Sweeps the cheap-tier routes (L0 closed-book, L1 k=5 retrieval, C0 k=5
chain-of-thought) for progressively stronger Qwen2.5 checkpoints
(1.5B -> 3B -> 7B) on the committed real Tier-A corpus, so the routing
break-even can be read off real coverage vs measured GPU-seconds cost.

Same proven route as ``kernels/realdata-15-local-sweep``
(``docs/kaggle-handoff.md``): clone the public branch, use the committed real
corpus + deterministic real index, serve the model with the repo's own
transformers server (which now reports a server-side model revision/weight
hash), and drive the committed runbook:

  1. base sweep: ``oracle_sweep --live-local --live-routes L0,L1,C0
     --live-model <model>`` (cloud routes C1..C4 stay stub-estimated: $0);
  2. 3x stability repeats: ``run_repeats.py --routes L0,L1,C0
     --live-model <model>``.

Both stages are resumable by cache key, and each model writes its own DBs
(``sweep-<tag>.db`` / ``repeats-<tag>.db``). Committed partial DBs are
picked up on a re-push, so an interrupted multi-hour run only re-executes the
remainder. Outputs are copied to ``/kaggle/working`` after every model, so a
timeout still yields the completed tiers.

Zero cloud spend by construction: no live cloud code path exists, and no stub
row can overwrite a measured row (INSERT OR IGNORE on the cache key).

Push / monitor / download (from the repo root)::

    kaggle kernels push   -p kernels/tiersweep-16-local-sweep
    kaggle kernels status abhigyan1818/tiersweep16-local-sweep
    kaggle kernels output abhigyan1818/tiersweep16-local-sweep -p /tmp/kout

See ``docs/kaggle-handoff.md`` for the ingest + recount commands.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_URL = "https://github.com/Abhigyan1805/cost-smart-RAG.git"
BRANCH = "fm/costsmart-tiersweep-16"
PORT = 8000
WORK = "/kaggle/working"
CLONE = os.path.join(WORK, "repo")
ENDPOINT = f"http://127.0.0.1:{PORT}"
RESULTS = "results/tiersweep-16"
CORPUS = "results/realdata-15/corpus.json"
INDEX = "data/index/real_index.json"
CONFIG = "config/experiments/sweep-200-tiersweep16.yaml"

#: Served checkpoints, in ascending capability order.
MODELS = (
    ("Qwen/Qwen2.5-1.5B-Instruct", "1p5b"),
    ("Qwen/Qwen2.5-3B-Instruct", "3b"),
    ("Qwen/Qwen2.5-7B-Instruct", "7b"),
)
#: Which tiers to sweep. Default 3b,7b: the 1.5B tier is already measured on
#: this exact corpus by the committed realdata-15 run (results/realdata-15/),
#: so re-measuring it would just re-spend GPU time. Override with the
#: environment variable COSTSMART_TIERS (e.g. "1p5b,3b,7b") when a fresh
#: 1.5B run is actually wanted.
TARGET_FILTER = tuple(
    t.strip() for t in os.environ.get("COSTSMART_TIERS", "3b,7b").split(",")
    if t.strip()
)


def _artifacts(tag: str) -> tuple[str, ...]:
    return (
        f"{RESULTS}/sweep-{tag}.db",
        f"{RESULTS}/repeats-{tag}.db",
        f"{RESULTS}/sweep-{tag}.csv",
        f"{RESULTS}/repeats-{tag}_summary.json",
        f"{RESULTS}/{tag}-server.log",
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
    for module, package in (("datasets", "datasets"),
                            ("transformers", "transformers"),
                            ("accelerate", "accelerate"),
                            ("torch", "torch")):
        try:
            __import__(module)
        except ImportError:
            print(f"{module} missing; pip installing {package}", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", package],
                           check=True)
    import datasets
    import transformers

    print(f"datasets {datasets.__version__} transformers "
          f"{transformers.__version__}", flush=True)


def verify_corpus() -> None:
    """Use the committed corpus and refuse a checksum mismatch."""
    path = os.path.join(CLONE, CORPUS)
    if not os.path.exists(path):
        raise RuntimeError(f"committed real corpus missing at {CORPUS}")
    with open(path) as fh:
        corpus = json.load(fh)
    freeze = os.path.join(CLONE, "results/realdata-15/splits_freeze.json")
    with open(freeze) as fh:
        expected = json.load(fh).get("corpus_sha256")
    actual = corpus["manifest"]["corpus_sha256"]
    if expected and actual != expected:
        raise RuntimeError(
            f"committed corpus sha {actual} != splits_freeze {expected}; "
            "refusing to run on an unverified corpus")


def build_index() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    run(
        [sys.executable, "-m", "costsmart.corpus.build_index",
         "--mix", "real", "--source", "real", "--out", INDEX],
        cwd=CLONE, env=env, desc="build real Tier-A retrieval index",
    )


def start_server(model: str, tag: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    log_path = os.path.join(CLONE, f"{RESULTS}/{tag}-server.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "scripts/colab_local_tier.py", "serve",
         "--model", model, "--port", str(PORT)],
        cwd=CLONE, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    print(f"[{tag}] server pid={proc.pid}; waiting for {ENDPOINT} ...", flush=True)
    return proc


def wait_healthy(proc: subprocess.Popen, model: str,
                 timeout_s: float = 2400.0) -> dict:
    body = json.dumps({"model": model, "prompt": "Say OK.",
                       "stream": False}).encode()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early rc={proc.returncode}")
        try:
            req = urllib.request.Request(
                ENDPOINT + "/api/generate", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode())
            print(f"server healthy: revision="
                  f"{payload.get('model_revision') or 'unknown'} "
                  f"weights={(payload.get('weights_sha256') or '')[:12]}",
                  flush=True)
            return payload
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"  not ready yet ({exc}); retrying", flush=True)
            time.sleep(15)
    raise RuntimeError(f"server did not become healthy within {timeout_s}s")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()


def drive_runbook(model: str, tag: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["COSTSMART_COLAB_ENDPOINT"] = ENDPOINT
    sweep_db = f"{RESULTS}/sweep-{tag}.db"
    repeats_db = f"{RESULTS}/repeats-{tag}.db"

    # 1) base matrix: L0/L1/C0 measured on this checkpoint; C1..C4 stub.
    run(
        [sys.executable, "-m", "costsmart.eval.oracle_sweep", "--no-limit",
         "--db", sweep_db, "--config", CONFIG, "--index", INDEX,
         "--live-local", "--live-routes", "L0,L1,C0",
         "--live-model", model,
         "--export-csv", f"{RESULTS}/sweep-{tag}.csv"],
        cwd=CLONE, env=env, desc=f"[{tag}] base tier sweep",
    )
    # 2) stability repeats: 3 draws per measured L0/L1/C0 pair.
    run(
        [sys.executable, "scripts/run_repeats.py", "--mode", "live",
         "--sweep-db", sweep_db, "--db", repeats_db, "--config", CONFIG,
         "--index", INDEX, "--routes", "L0,L1,C0", "--live-model", model,
         "--retries", "3", "--retry-sleep", "10",
         "--out", f"{RESULTS}/repeats-{tag}_summary.json"],
        cwd=CLONE, env=env, desc=f"[{tag}] 3x stability repeats",
    )

    conn = sqlite3.connect(os.path.join(CLONE, sweep_db))
    try:
        rows = conn.execute(
            "SELECT route_id, model_version, model_revision, weights_sha256,"
            " attestation, generator_mode, COUNT(*) FROM attempts"
            " GROUP BY 1,2,3,4,5,6 ORDER BY 1").fetchall()
    finally:
        conn.close()
    conn = sqlite3.connect(os.path.join(CLONE, repeats_db))
    try:
        repeat_rows = conn.execute(
            "SELECT COUNT(*) FROM repeat_attempts").fetchone()[0]
        repeat_pairs = conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT query_id, route_id"
            " FROM repeat_attempts)").fetchone()[0]
    finally:
        conn.close()
    return {
        "tag": tag,
        "model": model,
        "sweep_rows": rows,
        "repeat_rows": repeat_rows,
        "repeat_pairs": repeat_pairs,
    }


def collect_outputs(summary: list[dict]) -> None:
    for entry in summary:
        for rel in _artifacts(entry["tag"]):
            src = os.path.join(CLONE, rel)
            if not os.path.exists(src):
                print(f"WARN missing artifact {rel}", flush=True)
                continue
            dst = os.path.join(WORK, os.path.basename(rel))
            shutil.copy2(src, dst)
            print(f"copied {rel} -> {dst}", flush=True)
    with open(os.path.join(WORK, "kaggle_run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)


def main() -> int:
    clone_repo()
    ensure_deps()
    verify_corpus()
    build_index()
    results: list[dict] = []
    for model, tag in MODELS:
        if TARGET_FILTER and tag not in TARGET_FILTER:
            print(f"[{tag}] skipped by COSTSMART_TIERS={TARGET_FILTER}", flush=True)
            continue
        proc = start_server(model, tag)
        try:
            wait_healthy(proc, model)
            summary = drive_runbook(model, tag)
        finally:
            stop_server(proc)
        results.append(summary)
        print("MODEL SUMMARY: " + json.dumps(summary, default=str), flush=True)
        # Copy after every model so a timeout still yields completed tiers.
        collect_outputs([summary])
    # Final copy + full run summary across every completed model.
    collect_outputs(results)
    print("RUNBOOK SUMMARY: " + json.dumps(results, default=str), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
