# costpilot-06: 20-query pilot findings (first real numbers)

Branch `fm/costpilot-06`. All four Week-1 slices verified working as one
system: corpus -> retrieval -> telemetry -> eval (+ routing, verify, agent
imports). Pilot: **20 Tier-A queries x 7 routes = 140 attempts** in
`results/costpilot-06/pilot.db` (also `pilot.csv`), every row carrying
`git_sha` + `config_hash` + both cost columns (`cloud_spend_usd`,
`amortized_usd`).

Status labels used below: **measured** (ran here), **estimated** (stub math x
verified prices, $0 spent), **preliminary** (stub stands in for Colab until
the captain connects it).

## (a) Pilot telemetry table

- `results/costpilot-06/pilot.db` (SQLite, `attempts` table, 140 rows) and
  `results/costpilot-06/pilot.csv` (same rows, reviewable).
- Query set: stratified Tier-A slice, 8 NQ + 7 TriviaQA + 5 HotpotQA
  (`config/experiments/pilot-20.yaml`, `query_source: tier-a`, offline
  synthetic source). Full Tier-A pilot is 200 queries (80/70/50).
- 7-route Week-1 sweep set: L0 (local-small, k=0, direct), L1 (local-small,
  k=5, direct), C0 (local-medium, k=5, CoT), C1 (cloud-small, k=5, direct),
  C2 (cloud-small, k=10, rerank), C3 (cloud-large, k=5, direct), C4
  (cloud-large, k=10, rerank).
- Retrieval latency **measured** against `data/index/pilot_index.json`
  (200 queries / 450 chunks; smoke: hit@1 0.20, hit@5 0.60, hit@10 0.80):
  mean 6.35ms (4.21-12.58ms). Generator + cloud costs **estimated** (stub, no
  API calls, $0 spent). Local tiers **stubbed/preliminary**.
- Provenance: `git_sha=0e6ef6e…` (1 distinct), `config_hash` (1 distinct),
  prompt `v1`. Sweep is resumable/idempotent on `cache_key` (re-ran clean
  after the stub-gap fix; second run 140/140 inserted).
- Downstream targets proven on this DB: `train-router`
  (`router-v1.json`, 20 queries, cloud-label rate 1.0) and `eval`
  (baselines + router-v1 vs oracle; see `eval` section below).

## (b) Measured input/output tokens per query per route

`results/costpilot-06/tokens_by_query_route.csv` (140 rows). Per-route means:

| route | model | mean in | mean out | mean token_f1 |
|---|---|---|---|---|
| L0 | ollama-qwen2.5-3b | 123.4 | 30.4 | 0.533 |
| L1 | ollama-qwen2.5-3b | 141.3 | 24.7 | 0.950 |
| C0 | ollama-llama3.1-8b | 132.3 | 26.2 | 0.950 |
| C1 | gpt-4o-mini | 144.2 | 30.6 | 1.000 |
| C2 | gpt-4o-mini | 132.8 | 28.0 | 1.000 |
| C3 | gpt-4o | 128.1 | 24.9 | 1.000 |
| C4 | gpt-4o | 135.1 | 27.9 | 1.000 |

Token counts are hash-seeded stub draws (~80-200 in, ~8-48 out), so the
*rates* (verified 2026-09-19: gpt-4o-mini $0.15/$0.60, gpt-4o $2.50/$10.00
per 1M) are real but the *volumes* become measured only after live
generators land. Quality spread is stub-modelled (L0 closed-book misses
~1/3, local-augmented partials ~1/8, cloud exact) to restore the documented
oracle gap the normalizer had flattened.

## (c) Full-sweep extrapolation (Tier-A 200 x 7 routes = 1400 attempts)

`results/costpilot-06/extrapolation.json` + `per_route_summary.csv`.
Pilot is exactly 1/10th of the full sweep (140/1400 attempts), so
**full = pilot x 10** (per-route: route_sum / 20 x 200):

| route | pilot cloud $ | full-sweep cloud $ |
|---|---|---|
| L0 / L1 / C0 (local) | 0.000000 | 0.0000 |
| C1 (cloud-small) | 0.000800 | 0.0080 |
| C2 (cloud-small) | 0.000734 | 0.0073 |
| C3 (cloud-large) | 0.011375 | 0.1138 |
| C4 (cloud-large) | 0.012325 | 0.1233 |
| **total** | **0.025234** | **~$0.25** |

- **Full exhaustive sweep estimate: ~$0.25 cloud spend** (max ~$0.30 with
  headroom). Oracle-best policy over the same 1400 attempts: $0.0065
  (oracle mean $0.0000325/query x 200; oracle mix 10x C1 / 10x C2).
- Local amortized projection (PRELIMINARY, stubbed gpu_seconds):
  pilot $0.013651 x 10 = **~$0.14**.
- **Recommendation: approve the full 1400-attempt sweep.** Cloud exposure is
  cents, not dollars; nothing bigger than this runs without a fresh approval.
  Assumptions: per-route token/cost distributions from the 20-query slice
  hold for the full 200; live-generator token volumes will shift absolute
  numbers (re-estimate after real generators land).

## Headline economics (stub Ham, read carefully)

At stub token volumes, **cloud-small (~$0.00004/attempt) undercuts amortized
T4 local time (~$0.0002/attempt) ~6x**, so the oracle picks C1/C2 on all 20
queries and router-v1 (trained on all-cloud labels) ties always-C1
(accuracy 0.50, zero quality gap). Local tiers can only win this back on
quality or with real Colab timing -- both require the Colab attach. Eval
numbers: always-C3/C4 regret ~$0.0005/query (17x oracle); always-L0 quality
gap 0.47; random accuracy 0.20.

## (d) Colab handoff

- `docs/colab-handoff.md`: exact 4-step handoff (serve pinned Qwen2.5
  1.5B/7B in Colab, expose URL, `export COSTSMART_COLAB_ENDPOINT`,
  `scripts/colab_local_tier.py check`, verify via `get_client`).
- `scripts/colab_local_tier.py`: `serve` (Colab side, torch/transformers) +
  `check` (stdlib-only endpoint probe).
- `ColabClient.generate()` now delegates to the injected endpoint instead
  of raising `NotImplementedError` (still a clear `RuntimeError` when unset;
  no local-daemon assumption).

## Integration patches (minimal, no sibling redesigns)

1. `cli.py`: stubs wired to real slice entrypoints; argparse fallback when
   typer is absent (this env is stdlib+PyYAML only, no pip).
2. `Makefile`: `sweep`/`train-router`/`eval`/`report` pass
   `LIMIT`/`DB`/`CONFIG` through (previously dropped).
3. `oracle_sweep.py`: 7-route map + specs; `tier-a` query source from the
   corpus slice; measured retrieval latency via the pilot index (with
   feature-contract check); stub oracle-gap fix (L0 misses, local partials).
4. `baselines.py`: route list follows the sweep set (always-* for all 7).
5. New: `routing/train.py`, `eval/evaluate.py`, `scripts/analyze_pilot.py`,
   `config/experiments/pilot-20.yaml`, Colab artifacts above.
6. `costs.yaml`: cloud prices verified + filled (2026-09-19); local seconds
   calibrated from pilot means, marked preliminary.

## Not verifiable here (toolchain gaps, no pip)

`make install` / `pytest` / `ruff` (no pip/pytest/ruff in this worktree env;
`make`/`python` binaries absent -- ran `python3` module equivalents
directly). No live cloud calls by design ($0 spent). Colab not connected
(captain connects on request) -- local tiers preliminary.
