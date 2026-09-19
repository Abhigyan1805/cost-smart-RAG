# costsmart-rag

Existing LLM routers treat the query as the only input and the model as the
only action. In a RAG setting, retrieval produces strong difficulty signals
for free, and retrieval depth is itself a cheap action - costsmart-rag routes
over the joint space and measures the resulting cost-quality frontier against
an exhaustive oracle.

## How it works

A routed RAG agent: retrieval runs **before** routing, so post-retrieval
signals (score margins, coverage, agreement) become free router features. The
router picks a route from the joint space of **model x retrieval depth x
reasoning strategy**, and every claim is measured against an **exhaustive
oracle** that evaluates all routes per query.

Local model tiers run on a **remote GPU** - Google Colab, or a Kaggle GPU
kernel when the Colab free tier is exhausted - never a local Ollama daemon.
Local execution goes through the Colab-compatible client interface in
`src/costsmart/models/colab_client.py`; handoff details are in
`docs/colab-handoff.md` and `docs/kaggle-handoff.md`. Cloud routes (C1..C4)
are **stub-only by construction**: there is no live cloud code path, so $0 is
spent and the `cloud_spend_usd` columns are *estimates*, not charges.

## What has shipped

- **Corpus + retrieval** (`src/costsmart/corpus`, `src/costsmart/retrieval`):
  synthetic Tier-A query sets (legacy pilot + a 75%-multi-hop mix),
  deterministic index builds, BM25 + hybrid retrieval, rerank, and
  post-retrieval features. The Hugging Face loader path exists but no live
  fetch happens in this task.
- **Telemetry + sweep harness** (`src/costsmart/eval/oracle_sweep.py`,
  `src/costsmart/telemetry`): exhaustive route sweeps into SQLite with
  per-row provenance (`generator_mode` / `retrieval_mode`, temperature, seed,
  git sha, config hash) plus CSV/JSONL export. Committed **1400-attempt
  matrices** in `results/costsweep-08/` (legacy mix) and
  `results/costmultihop-12/` (multi-hop mix: 400 measured L0/L1 + 1000
  cloud/C0 stub rows).
- **Eval + oracle** (`src/costsmart/eval`): deterministic Tier-A graders
  (EM / lenient-EM / token-F1), the headroom gate (contingency,
  routable-fraction, max saving, bootstrap CIs, McNemar) in
  `scripts/make_plots.py`, and the stable-oracle majority recount in
  `scripts/stable_oracle.py`.
- **Stable-oracle recounts** (`results/costfinal-10/`,
  `results/costmultihop-12/`): 3x live repeats per measured pair, strict
  majority grading with ties counted incorrect
  (`src/costsmart/eval/stability.py`). The recount shows single-run label
  noise does not change the verdict.
- **Routing + verification + agent loop** (`src/costsmart/routing`,
  `src/costsmart/verify`, `src/costsmart/agent`): router training and
  baselines, verification critics, and the demo agent loop.
- **Remote GPU route** (Colab, plus `kernels/costsweep-13-local-sweep/` on
  Kaggle): the Kaggle kernel clones the repo, serves
  `Qwen/Qwen2.5-1.5B-Instruct`, and resumes the committed DBs by cache key.
  Its runbook log is hash-pinned in
  `results/costmultihop-12/KAGGLE_PROVENANCE.md`.

## Status at a glance

| Area | Status |
|---|---|
| Skeleton / model interfaces / CLI | shipped |
| Corpus + retrieval | shipped (synthetic; HF loader path exists, no live fetch) |
| Telemetry / sweep harness | shipped (1400-attempt matrices, per-row provenance) |
| Eval + oracle | shipped (headroom gate + stable recount) |
| Routing / router training | shipped (pilot `router-v1`; not yet on a real dataset) |
| Verification / agent loop / demo | shipped |
| Remote GPU route (Colab / Kaggle) | shipped |
| Tier-B judge validation | not started (captain-only hand labels; no reported number depends on it) |

## Current result and its limits

The headroom gate is **NO-GO on both mixes and both labelings** (stable
routable fraction 0.06 [0.03, 0.095] on the legacy mix; 0.045 [0.020, 0.075]
on the multi-hop mix). Read it as a **pipeline smoke signal, not a benchmark
conclusion**: the corpus is offline synthetic and its templates keep the
answer entity in the question text. "Dense" retrieval falls back to the
stdlib hash-embedding in this environment (not the pinned MiniLM), and the
cloud-stub cost columns are estimated, not spent. **A real-dataset re-run is
required before any product decision.** Full framing: `REPORT.md`, "Scope and
claim strength (read first)".

## Reproducing

```sh
# tests (stdlib-only; `make test` needs pytest, which this env may lack)
PYTHONPATH=src python3 -m unittest discover -s tests

# rebuild + recount a gate from committed telemetry (explicit paths required)
PYTHONPATH=src python3 scripts/stable_oracle.py \
  --sweep-db results/costmultihop-12/sweep.db \
  --repeats-db results/costmultihop-12/repeats.db \
  --out-dir /tmp/recount
```

`scripts/stable_oracle.py` requires `--sweep-db`, `--repeats-db`, and
`--out-dir`: it refuses a bare invocation and refuses a sweep/repeats pair
drawn from different mixes, so it cannot silently recount the legacy pilot
mix when the reader means the multi-hop one.

## Make instructions

```sh
make install       # install package + dev extras (needs a networked env)
make index         # build the retrieval index
make sweep         # run the exhaustive route sweep (oracle telemetry)
make train-router  # train the router on sweep telemetry
make eval          # evaluate router vs exhaustive oracle
make report        # render REPORT.md figures
make test          # run pytest (pytest may be absent; use the unittest command above)
make lint          # run ruff
```

Individual CLI entrypoints are also available via
`python -m costsmart.cli <command>`.

Pinned model versions live in `config/models.yaml`; the cost model and pricing
snapshot live in `config/costs.yaml` (cloud routes are stub-estimated, never
executed). Experiment configs are under `config/experiments/`.
