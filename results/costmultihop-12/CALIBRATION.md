# costmultihop-12 calibration: cloud stubs + frozen mix (zero cloud spend)

Zero-cloud-spend contract (same as costsweep-08): cloud-tier routes (C1..C4)
NEVER execute live - no live cloud code path exists
(`execute_live_local` raises for cloud routes). Their telemetry rows are
deterministic stub estimates (`generator_mode='stub'`) whose token volumes
are hash-seeded per `(seed, query_id, route_id)` and priced at the pinned
per-token prices. Local-tier routes (L0/L1) execute for real through the
Colab session after attach (`generator_mode='measured'`, temperature 0,
seed 0); C0 stays stub without a 7B endpoint (see costsweep-08 SUMMARY.md).

## Query set (frozen)

`results/costmultihop-12/splits_freeze.json`: 200 Tier-A queries -
30 NQ + 20 TriviaQA + 60 HotpotQA + 50 2WikiMultihopQA + 40 MuSiQue
(25% single-hop / 75% multi-hop), offline synthetic source
(`SYNTHETIC_SEED=20260919`). Set SHA256
`71d0c4ce623e2d203ba437aa4bff08b0896d5d4dec259a2b6dff4f2d2f1142f0`
(ordered query dicts, `sort_keys=True`; recomputed from the loader by
`tests/test_multihop_mix.py`). Sweep config
`config/experiments/sweep-200-multihop.yaml` (seed 0, prompt `v1`,
`config_hash 013fc66b10e00347e8ba70ceb2321301bd449df62c64f0effa29b2a239a6eb5b`).

The legacy pilot set is untouched: its freeze hash
(`fe00c48866520a3ea7acfdbd2c6ee63b647744261de5650ea6d966e331ff348c`) still
recomputes from `load_pilot_subset`, so costsweep-08 rows stay comparable.

## Retrieval index (frozen input, gitignored artifact)

Rebuilt deterministically before the sweep:

```sh
python -m costsmart.corpus.build_index --mix multihop --source synthetic \
    --out data/index/multihop_index.json
```

200 queries, 590 passages, 590 chunks; `mix: multihop` recorded in the index
metadata; SHA256
`574cac526330415c96e319f995dd3043aa71d7afcd596c4524ddac4b3f5444cb`
(byte-identical on rebuild in the stdlib hash-embedding fallback).
Retrieval latency is MEASURED for every row against this index
(`retrieval_mode='measured'`).

## Cloud-stub extrapolation math

Full sweep = 200 queries x 7 routes = 1400 attempts. Cloud USD per attempt =
`tokens_in x price_in + tokens_out x price_out` (tokens from the
deterministic stub draw; PRICES PINNED - no live price fetch):

| model | input / 1M | output / 1M | price snapshot |
|---|---|---|---|
| gpt-4o-mini (cloud-small: C1, C2) | $0.15 | $0.60 | 2026-09-19 |
| gpt-4o (cloud-large: C3, C4) | $2.50 | $10.00 | 2026-09-19 |

Stub sums over the new mix (rehearsal DB; cloud rows are identical in the
measured sweep because the stub is deterministic per query+route):

| route | n | mean token_f1 | cloud $ (sum) | amortized $ (sum) |
|---|---|---|---|---|
| L0 (local-small, k=0, direct) | 200 | 0.648 (stub) | 0.000000 | 0.041374 (stub) |
| L1 (local-small, k=5, direct) | 200 | 0.973 (stub) | 0.000000 | 0.042732 (stub) |
| C0 (local-medium, k=5, CoT) | 200 | 0.964 (stub) | 0.000000 | 0.040841 (stub) |
| C1 (cloud-small, k=5, direct) | 200 | 1.000 | 0.007454 | 0.0 |
| C2 (cloud-small, k=10, rerank) | 200 | 1.000 | 0.007429 | 0.0 |
| C3 (cloud-large, k=5, direct) | 200 | 1.000 | 0.122540 | 0.0 |
| C4 (cloud-large, k=10, rerank) | 200 | 1.000 | 0.120015 | 0.0 |
| **total** | 1400 | - | **0.257438** | 0.124947 |

(The 20-query pilot calibration that fixed the price/model mapping lives in
`results/costsweep-08/CALIBRATION.md`; absolute $ here shifts only through
the per-query stub token draws, which are mix-dependent by construction.)

## Stub pipeline smoke test (NOT evidence)

A full all-stub rehearsal of the 1400-attempt matrix (scratch DB, $0, no
Colab; 1400 unique cache keys, all `retrieval_mode='measured'`) gives the
stub gate L0-vs-C4: routable 0.675, saving 94.6%, gap 0.325
(McNemar p = 2.1e-15) - **GO**. This is an artifact of the stub contract
(cloud stubs answer exactly; stub L0 misses ~1/3 of queries by hash), so it
cannot validate the mix hypothesis and is excluded from the verdict. The
verdict reads only the measured local-tier rows after the live sweep; the
stub cloud rows remain the C4 reference by construction (zero spend).

## Live sweep runbook (after Colab attach)

```sh
export COSTSMART_COLAB_ENDPOINT=<tunnel>          # docs/colab-handoff.md

# 1) base matrix: L0/L1 measured, C0..C4 stub ($0 cloud)
PYTHONPATH=src python -m costsmart.eval.oracle_sweep --no-limit \
  --db results/costmultihop-12/sweep.db \
  --config config/experiments/sweep-200-multihop.yaml \
  --index data/index/multihop_index.json \
  --live-local --live-routes L0,L1 \
  --export-csv results/costmultihop-12/sweep.csv

# 2) stability repeats: 3 draws per measured L0/L1 pair (separate DB)
PYTHONPATH=src python scripts/run_repeats.py --mode live \
  --sweep-db results/costmultihop-12/sweep.db \
  --db results/costmultihop-12/repeats.db \
  --config config/experiments/sweep-200-multihop.yaml \
  --index data/index/multihop_index.json --routes L0,L1 \
  --out results/costmultihop-12/repeats_summary.json

# 3) stable-oracle rebuild + gate recount (majority labels, ties incorrect)
PYTHONPATH=src python scripts/stable_oracle.py \
  --sweep-db results/costmultihop-12/sweep.db \
  --repeats-db results/costmultihop-12/repeats.db \
  --out-dir results/costmultihop-12
```

Steps 1-2 are resumable (cache keys); re-run to continue after any
interruption. The attached endpoint must serve the sweep's measured model
id per tier (`run_repeats` aborts on mismatch rather than mislabeling).

## Assumptions / caveats

1. Real cloud token volumes are unknown ($0 spent); only relative route
   ordering enters the oracle comparison. C4 stub correctness 1.0 makes the
   recount gate a (conservative) statement about L0: every L0-correct
   query is routable by construction.
2. Local amortized $ stays PRELIMINARY until measured `gpu_seconds` from the
   live rows replace the stub assumptions (see `config/costs.yaml`).
3. Synthetic questions are a smoke-test corpus, not a benchmark: the
   multi-hop templates keep the answer entity in the question text, so the
   measured effect size is a pipeline signal, not an estimate of real
   2Wiki/MuSiQue difficulty. The HF loader path exists but no live fetch
   happens in this task.
