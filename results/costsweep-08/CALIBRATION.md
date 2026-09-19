# costsweep-08 calibration: cloud-stub math from the pilot-20 numbers

Zero-cloud-spend contract: cloud-tier routes (C1..C4) NEVER execute live in
this sweep — no live cloud code path exists (`execute_live_local` raises for
cloud routes). Their telemetry rows are deterministic stub estimates
(`generator_mode='stub'`) scaled from the 20-query pilot
(`results/costpilot-06/`), with $0 actually spent. Local-tier routes
(L0/L1/C0) execute for real through the Colab session (`generator_mode=
'measured'`, temperature 0, seed 0).

## Query set (frozen)

`splits_freeze.json` (committed): full Tier-A 200 queries — 80 NQ + 70
TriviaQA + 50 HotpotQA, offline synthetic source (`SYNTHETIC_SEED=20260919`).
Set SHA256: `fe00c48866520a3ea7acfdbd2c6ee63b647744261de5650ea6d966e331ff348c`
(ordered query dicts, `sort_keys=True`). Sweep config:
`config/experiments/sweep-200-tier-a.yaml` (seed 0, prompt `v1`).

## Cloud-stub extrapolation math

Full sweep = 200 queries x 7 routes = 1400 attempts. Pilot = 20 x 7 = 140
attempts = exactly 1/10th, so:

```
full_sweep_sum = pilot_sum x (1400 / 140) = pilot_sum x 10.0
per_route_full = route_pilot_sum / 20 x 200
```

Cloud USD per attempt = `tokens_in x price_in + tokens_out x price_out`
(tokens measured from the stub draw, PRICES PINNED — no live price fetch):

| model | input / 1M | output / 1M | price snapshot |
|---|---|---|---|
| gpt-4o-mini (cloud-small: C1, C2) | $0.15 | $0.60 | 2026-09-19 |
| gpt-4o (cloud-large: C3, C4) | $2.50 | $10.00 | 2026-09-19 |

Pilot route sums (`results/costpilot-06/extrapolation.json`) → full-sweep
stub estimates:

| route | pilot cloud $ (n=20) | full-sweep cloud $ (n=200) |
|---|---|---|
| L0 / L1 / C0 (local) | 0.000000 | 0.0000 |
| C1 (cloud-small, k=5, direct) | 0.000800 | 0.0080 |
| C2 (cloud-small, k=10, rerank) | 0.000734 | 0.0073 |
| C3 (cloud-large, k=5, direct) | 0.011375 | 0.1138 |
| C4 (cloud-large, k=10, rerank) | 0.012325 | 0.1233 |
| **total** | **0.025234** | **~$0.25** |

Oracle-best policy over the same 1400 attempts (stub): $0.0065
(oracle mean $0.0000325/query x 200; mix 10x C1 / 10x C2 at stub volumes).

## Assumptions / caveats

1. Per-route stub token/cost distributions from the 20-query stratified
   slice hold for the full 200. The stub draws token volumes from
   `sha256(seed:query_id:route_id)` (~80-200 in, ~8-48 out); real cloud
   token volumes are unknown by design (no calls), so absolute $ shifts if
   live generators ever land. Relative route ordering is what the oracle
   comparison consumes.
2. Local amortized projection is PRELIMINARY: pilot $0.013651 x 10 = ~$0.14
   from stub `gpu_seconds` (T4 $0.35/h; L0 2.479s / L1 2.454s / C0 2.088s
   means). Replaced by measured `gpu_seconds` per attempt after the Colab
   attach; `config/costs.yaml :: local_amortized` then gets recalibrated
   (duplicate-key bug fixed this branch: the stray 0.0 entry no longer
   shadows the calibrated value).
3. Retrieval latency is MEASURED for every row (`retrieval_mode='measured'`,
   hybrid search timed against `data/index/pilot_index.json`, pilot mean
   6.35ms) — including L0, matching the pilot convention (closed-book still
   pays the pre-routing retrieval that produced the router features).

## Row provenance flags (new in this branch)

| column | `stub` | `measured` | `unflagged-legacy` |
|---|---|---|---|
| `generator_mode` | stub executor (cloud routes always; local routes until Colab) | real Colab generation, temp 0, fixed seed | rows predating flags |
| `retrieval_mode` | `--no-retrieval` runs | timed `hybrid_search` vs pilot index | rows predating flags |

Legacy mapping: frozen `results/costpilot-06/pilot.db` rows read as
`unflagged-legacy`; per `findings.md` they are generator=`stub` (estimated)
+ retrieval=`measured`. The store migrates legacy DBs via ALTER TABLE
(old rows keep defaults, never rewritten).

## Live-local row identity

Measured local rows carry the SERVED model id as `model_version`
(e.g. `Qwen/Qwen2.5-1.5B-Instruct` from `config/models.yaml`), not the
`ROUTE_MODELS` telemetry label — so `cache_key` (which hashes
`model_version`) never collides with a stub row for the same query+route.
Stub rows keep the `ROUTE_MODELS` labels for pilot comparability.
