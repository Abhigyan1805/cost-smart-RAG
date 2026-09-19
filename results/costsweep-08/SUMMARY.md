# costsweep-08 summary: zero-spend oracle sweep (Tier-A 200 x 7 = 1400 attempts)

Sweep DB: `sweep.db` (SQLite `attempts`, also `sweep.csv`). Config:
`config/experiments/sweep-200-tier-a.yaml` (seed 0, prompt `v1`,
`config_hash 7fdc85f6…`, `git_sha 7c72e82…`). Query set frozen in
`splits_freeze.json` (set SHA256 `fe00c488…`). Cloud spend: **$0**
(no live cloud code path exists; `execute_live_local` raises for C1..C4).

## Provenance (per-row flags)

| routes | generator | n | mean token_f1 | mean gpu_s | cloud $ (sum) | amortized $ (sum) |
|---|---|---|---|---|---|---|
| L0 (local-small k=0 direct) | **measured** (Colab 1.5B, temp 0, seed 0) | 200 | 0.227 | 1.167 | 0.0 | 0.022698 |
| L1 (local-small k=5 direct) | **measured** (Colab 1.5B, temp 0, seed 0) | 200 | 0.129 | 3.095 | 0.0 | 0.060182 |
| C0 (local-medium k=5 CoT) | stub (no 7B endpoint served — see gap) | 200 | 0.963 | 2.158 | 0.0 | 0.041969 |
| C1 (cloud-small k=5 direct) | stub (pilot calibration) | 200 | 1.000 | — | 0.007409 | 0.0 |
| C2 (cloud-small k=10 rerank) | stub (pilot calibration) | 200 | 1.000 | — | 0.007104 | 0.0 |
| C3 (cloud-large k=5 direct) | stub (pilot calibration) | 200 | 1.000 | — | 0.120970 | 0.0 |
| C4 (cloud-large k=10 rerank) | stub (pilot calibration) | 200 | 1.000 | — | 0.123853 | 0.0 |

All 1400 rows: `retrieval_mode='measured'` (timed hybrid search vs
`data/index/pilot_index.json`). All `cache_key`s unique (1400/1400).
Cloud-stub total $0.259 matches the CALIBRATION.md extrapolation (~$0.25).

Real-vs-stub deltas worth noting (deterministic Tier-A graders, single-word
refs vs verbose 1.5B answers): measured L0/L1 token_f1 (0.23/0.13) lands far
below the stub's modelled gap (0.53/0.95) — verbosity, not wrongness
(L0 answers are factually on-target, e.g. mitochondria in eukaryotic cells).
Measured gpu_s (L0 1.17s, L1 3.10s) replaces the stub assumptions (2.47/2.45s);
`costs.yaml` recalibration to these means is follow-up work.

## L0 stability: 100 queries x 3 repeats → flip_rate **0.88** (`l0_stability.json`)

88/100 queries produced non-identical repeat predictions (slice: 40 NQ + 35
TriviaQA + 25 HotpotQA, set SHA `da71ae72…`). Flips are paraphrase-level
sampling variation (same facts, different wording), NOT garbage — but at
requested temperature 0 the rate should be ~0 (stub mode: 0.0). Cause: the
attached Colab server runs the OLD `colab_local_tier.py` handler, which
ignores `options.seed/temperature` (no `prompt_eval_count` in responses
either — `tokens_in` for live rows is a whitespace fallback). The sampling
contract is therefore requested-but-unenforced server-side. Fix: redeploy
this branch's `scripts/colab_local_tier.py` in the Colab tab (respects
seed/temperature, reports `prompt_eval_count`), then re-run
`scripts/l0_stability.py --mode live`. Sweep L0/L1 predictions stand as
measured single draws regardless.

## Gap: C0 (local-medium) still stubbed

The attached endpoint serves only Qwen2.5-1.5B (self-identified; single
tunnel). Running C0 against it would mislabel 1.5B outputs as 7B, so C0
stayed `generator_mode='stub'` by `--live-routes L0,L1` restriction.
Backfill when a 7B endpoint lands (resumable, no collisions — measured rows
hash the served model id):
`COSTSMART_COLAB_ENDPOINT=<7B-tunnel> python -m costsmart.eval.oracle_sweep
--no-limit --db results/costsweep-08/sweep.db --config
config/experiments/sweep-200-tier-a.yaml --live-local --live-routes C0`.

## Files

- `sweep.db` / `sweep.csv` — 1400 telemetry rows
- `splits_freeze.json` — frozen query set + set hash
- `CALIBRATION.md` — cloud-stub math from pilot-20
- `l0_stability.json` — 100x3 repeat report (flip_rate 0.88)
- `SUMMARY.md` — this file
