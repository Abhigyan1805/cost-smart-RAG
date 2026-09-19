#!/usr/bin/env python3
"""Kaggle batch kernel: realdata-15 decisive live sweep on REAL Tier-A data.

This is the real-data counterpart of ``kernels/costsweep-13-local-sweep``
(see ``docs/kaggle-handoff.md``). It clones the public repo at the task
branch, then, in order:

  1. fetches + normalizes the real Tier-A corpus through the repo's
     HuggingFace loader path (``scripts/fetch_tier_a.py``) and writes the
     committed corpus JSON (50 NQ + 80 HotpotQA + 70 MuSiQue);
  2. rebuilds the retrieval index from that corpus (``--mix real``);
  3. starts the repo's own transformers server
     (``scripts/colab_local_tier.py serve``) on the Kaggle GPU;
  4. drives the committed runbook against ``http://127.0.0.1:8000``:
     base matrix (L0/L1 measured, C0..C4 stub) then 3x stability repeats,
     both resumable by cache key;
  5. copies the corpus, index, sweep/repeats DBs, CSV and summaries to
     ``/kaggle/working`` for ``kaggle kernels output``.

Zero cloud spend by construction: cloud routes are never executed (there is
no live cloud code path in the repo), and no stub row can overwrite a
measured row (INSERT OR IGNORE on the cache key).

Push / monitor / download (from the repo root)::

    kaggle kernels push   -p kernels/realdata-15-local-sweep
    kaggle kernels status abhigyan1818/realdata15-local-sweep
    kaggle kernels output abhigyan1818/realdata15-local-sweep -p /tmp/kout

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
BRANCH = "fm/costsmart-realdata-15"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PORT = 8000
WORK = "/kaggle/working"
CLONE = os.path.join(WORK, "repo")
ENDPOINT = f"http://127.0.0.1:{PORT}"
RESULTS = "results/realdata-15"
CORPUS = "results/realdata-15/corpus.json"
INDEX = "data/index/real_index.json"
CONFIG = "config/experiments/sweep-200-realdata15.yaml"
ARTIFACTS = (
    CORPUS,
    INDEX,
    f"{RESULTS}/sweep.db",
    f"{RESULTS}/repeats.db",
    f"{RESULTS}/sweep.csv",
    f"{RESULTS}/repeats_summary.json",
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
    """Kaggle images ship torch/transformers; install only what is missing."""
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


def fetch_corpus() -> dict:
    """Use the committed corpus; refetch only if it is missing.

    The corpus + frozen checksums are committed, so the decisive run must use
    exactly those rows (a refetch would depend on mutable upstream revisions).
    """
    path = os.path.join(CLONE, CORPUS)
    if os.path.exists(path):
        print(f"using committed real corpus {CORPUS}", flush=True)
    else:
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        run(
            [sys.executable, "scripts/fetch_tier_a.py", "--out", CORPUS],
            cwd=CLONE, env=env, desc="fetch + normalize real Tier-A corpus",
        )
    with open(path) as fh:
        corpus = json.load(fh)
    expected = _frozen_corpus_sha()
    if expected and corpus["manifest"]["corpus_sha256"] != expected:
        raise RuntimeError(
            "committed corpus body sha256 does not match "
            f"splits_freeze.json ({corpus['manifest']['corpus_sha256']} != "
            f"{expected}); refusing to run on an unverified corpus")
    return corpus["manifest"]


def _frozen_corpus_sha() -> str | None:
    freeze = os.path.join(CLONE, RESULTS, "splits_freeze.json")
    if not os.path.exists(freeze):
        return None
    with open(freeze) as fh:
        return json.load(fh).get("corpus_sha256")


def build_index() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    run(
        [sys.executable, "-m", "costsmart.corpus.build_index",
         "--mix", "real", "--source", "real", "--out", INDEX],
        cwd=CLONE, env=env, desc="build real Tier-A retrieval index",
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

    # 1) base matrix: L0/L1 measured, C0..C4 stub; resumable by cache key.
    run(
        [sys.executable, "-m", "costsmart.eval.oracle_sweep", "--no-limit",
         "--db", f"{RESULTS}/sweep.db", "--config", CONFIG,
         "--index", INDEX, "--live-local", "--live-routes", "L0,L1",
         "--export-csv", f"{RESULTS}/sweep.csv"],
        cwd=CLONE, env=env, desc="base real-data sweep (L0/L1 measured, cloud stub)",
    )

    # 2) stability repeats: 3 draws per measured L0/L1 pair, resumable.
    run(
        [sys.executable, "scripts/run_repeats.py", "--mode", "live",
         "--sweep-db", f"{RESULTS}/sweep.db",
         "--db", f"{RESULTS}/repeats.db", "--config", CONFIG,
         "--index", INDEX, "--routes", "L0,L1", "--retries", "3",
         "--retry-sleep", "10",
         "--out", f"{RESULTS}/repeats_summary.json"],
        cwd=CLONE, env=env, desc="3x real-data stability repeats",
    )

    # 3) counts / provenance summary for the ingest step.
    import sqlite3

    summary: dict = {}
    conn = sqlite3.connect(os.path.join(CLONE, f"{RESULTS}/sweep.db"))
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
    conn = sqlite3.connect(os.path.join(CLONE, f"{RESULTS}/repeats.db"))
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
    index_meta = json.loads(open(os.path.join(CLONE, INDEX)).read())
    summary["index"] = {
        "num_queries": index_meta.get("num_queries"),
        "num_passages": index_meta.get("num_passages"),
        "num_chunks": index_meta.get("num_chunks"),
        "embedding_model": index_meta.get("embedding_model"),
        "embedding_backend": index_meta.get("embedding_backend"),
        "embedding_dim": index_meta.get("embedding_dim"),
    }
    summary["corpus_manifest"] = json.loads(
        open(os.path.join(CLONE, CORPUS)).read())["manifest"]
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
    shutil.rmtree(CLONE, ignore_errors=True)


def main() -> int:
    clone_repo()
    ensure_deps()
    fetch_corpus()
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
