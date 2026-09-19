# costsmart-rag Experiment Report (stub - Week-1 skeleton)

> Each experiment below measures its claim against the exhaustive oracle over
> the joint (model x retrieval depth x reasoning strategy) space. Sections are
> placeholders until sweep telemetry lands.

## Headroom gate (costheadroom-09): cheapest-local vs strongest-cloud

Preliminary gate on the 20-query pilot (`results/costpilot-06/pilot.db`,
140 attempts). Correctness = `token_f1 >= 0.5`; costs = metered cloud +
amortized local per attempt. Recompute with
`python scripts/make_plots.py headroom --db <db> --out-dir results/costheadroom-09`;
full payload in `results/costheadroom-09/headroom.json`, figure in
`results/costheadroom-09/headroom.svg`.

### 2x2 contingency (L0 cheapest-local vs C4 strongest-cloud, n=20 paired)

|  | C4 correct | C4 wrong |
|---|---|---|
| L0 correct | 11 | 0 |
| L0 wrong | 9 | 0 |

### Metrics (95% bootstrap CIs, 10k resamples)

- Routable fraction (L0 already correct): **0.55 [0.35, 0.75]**.
- Max saving at no quality loss (oracle cheapest-correct vs all-C4): **94.7%
  [94.0%, 95.3%]** (oracle $0.000650 vs all-C4 $0.012325; 20/20 queries
  routed off C4, 0 quality-loss queries).
- Paired accuracy gap (C4 − L0): **0.45 [0.25, 0.65]**; McNemar exact
  two-sided p = **0.0039** (b=9, c=0, 9 discordant).

### Decision gate

Rule (plan): routable fraction below 0.10 means the tiers sit too close
together -> NO-GO (recommend shrinking L0 or reweighting toward multi-hop).
Pilot verdict: **GO (preliminary)** - 0.55 clears the 0.10 gate and the
McNemar test confirms real tier separation (p = 0.0039).

**Small-n caveat:** n=20 pilot queries, so CIs are wide (routable CI spans
0.35-0.75); do not over-claim. Re-run on the full sweep matrix when the
sibling sweep slice merges and update this section + the PR.
(Done: see "Full-matrix recount on stable labels (costfinal-10)" below.)

## Full-matrix recount on stable labels (costfinal-10): noisy-label fix

The sweep exposed small-model label noise (L0 string flip rate 0.88 across
3 live repeats at requested temperature 0). Every live local-tier pair
(L0/L1 x 200 queries = 400 pairs; C0 excluded - still stub, no 7B endpoint)
was re-run 3x live through the Colab 1.5B endpoint (temperature 0, seed 0;
1200 draws, all `generator_mode='measured'`, stored in
`results/costfinal-10/repeats.db` + `repeats.csv`, separate from the frozen
sweep matrix) and graded by majority (ties count as incorrect -
`src/costsmart/eval/stability.py`). The oracle matrix was rebuilt on stable
labels (`stable_attempts.json`: representative repeat draw per pair) and
the L0-vs-C4 gate recounted with fresh 10k-resample bootstrap CIs
(`headroom_single.json` = single-run baseline, `headroom_stable.json` =
recount; both reproducible via `python scripts/make_plots.py headroom
--attempts-json results/costfinal-10/stable_attempts.json --out-dir <dir>`).

### Flip-rate-per-pair distribution (n=200 pairs per route, 3 draws each)

| route | string flip rate | pairs by #distinct predictions (1/2/3) | label flip rate (token_f1>=0.5) | vote splits (correct-incorrect) |
|---|---|---|---|---|
| L0 | 0.865 | 27 / 40 / 133 | 0.125 | 3-0: 4, 2-1: 8, 1-2: 17, 0-3: 171 |
| L1 | 0.985 | 3 / 21 / 176 | 0.085 | 3-0: 2, 2-1: 6, 1-2: 11, 0-3: 181 |

String noise reproduces the sweep finding (L0 0.865 vs 0.88 on the 100-query
slice) and is worse on L1 (longer prompts, more paraphrase room) - but it
rarely crosses the correctness threshold: 21/400 pair-labels changed
single-run -> stable, and unanimous-incorrect dominates (352/400 pairs).

### Gate: single-run labels vs stable labels (n=200 paired, 95% CIs)

| labels | contingency (both / L0-only / C4-only / neither) | routable fraction | max saving | gap (C4-L0) | McNemar | verdict |
|---|---|---|---|---|---|---|
| single-run | 16 / 0 / 184 / 0 | 0.08 [0.045, 0.12] | 95.1% [94.9%, 95.3%] | 0.92 [0.88, 0.955] | p = 1.8e-41 | **NO-GO** |
| stable (majority) | 12 / 0 / 188 / 0 | 0.06 [0.03, 0.095] | 95.0% [94.8%, 95.3%] | 0.94 [0.905, 0.97] | p = 2.4e-42 | **NO-GO** |

(Oracle $0.00614 vs all-C4 $0.12385; 200/200 routed off C4, 0 quality-loss
queries - C4 stub is exactly correct on every query.)

### Did single-run labels change any Week-2 conclusion? No.

Both labelings give **NO-GO**: the stable routable CI ([0.03, 0.095]) sits
entirely below the 0.10 gate, and the single-vs-stable delta (0.02) is far
from the boundary. Separately, both full-matrix results overturn the
pilot-20 preliminary GO (0.55) - but that overturn is driven by the
measured-vs-stub quality gap (measured L0 0.06-0.08 vs stub-modelled 0.55),
not by label noise.

## Multi-hop mix live recount (costmultihop-12 / costsweep-13): gate still NO-GO

This is the live recount the costmultihop-12 mix was built for. The
multi-hop-reweighted 200-query Tier-A set (`results/costmultihop-12/`,
75% multi-hop; `MIX.md`) ran all L0/L1 local-tier rows live (temperature 0,
seed 0) plus the full 3x-majority repeat protocol. The Colab free tier
exhausted mid-repeats (269/1200 draws), so the run finished on the Kaggle GPU
route (`kernels/costsweep-13-local-sweep/`, `docs/kaggle-handoff.md`):
Qwen2.5-1.5B served by the repo's own transformers server, the committed
runbook driven against `http://127.0.0.1:8000`, resuming from the preserved
partial DBs by cache key. Base matrix: 400 measured L0/L1 rows + 1000 cloud/C0
stub rows; repeats: **1200/1200 draws (400 pairs x 3), all
`generator_mode='measured'`**. No measured row was overwritten by a stub and
no cloud route executed ($0 spent). The stable-oracle rebuild + gate recount
is reproducible via `scripts/stable_oracle.py` (see the runbook).

### Flip-rate-per-pair distribution (n=200 pairs per route, 3 draws each)

| route | string flip rate | pairs by #distinct predictions (1/2/3) | label flip rate (token_f1>=0.5) | vote splits (correct-incorrect) |
|---|---|---|---|---|
| L0 | 0.220 | 156 / 8 / 36 | 0.035 | 3-0: 8, 2-1: 1, 1-2: 6, 0-3: 185 |
| L1 | 0.225 | 155 / 1 / 44 | 0.005 | 3-0: 7, 1-2: 1, 0-3: 192 |

The multi-hop mix cut string noise roughly in half versus the legacy mix
(L0 0.865, L1 0.985 in costfinal-10) - shorter factual answers leave less
paraphrase room - and label flips are near zero. 30/400 pair labels still
change single-run -> stable.

### Gate: single-run labels vs stable labels (n=200 paired, 95% CIs)

| labels | contingency (both / L0-only / C4-only / neither) | routable fraction | max saving | gap (C4-L0) | McNemar | verdict |
|---|---|---|---|---|---|---|
| single-run | 19 / 0 / 181 / 0 | 0.095 [0.055, 0.140] | 94.69% [94.45%, 94.93%] | 0.905 | p = 8.0e-41 | **NO-GO** |
| stable (majority) | 9 / 0 / 191 / 0 | 0.045 [0.020, 0.075] | 94.67% [94.42%, 94.89%] | 0.955 | p = 5.2e-43 | **NO-GO** |

(Oracle $0.006402 vs all-C4 $0.120015; 200/200 routed off C4, 0 quality-loss
queries - the C4 stub is exactly correct on every query by the frozen stub
contract.)

### Verdict: reweighting did not open the gate

The hypothesis in `MIX.md` was that collapsing closed-book L0 accuracy on
multi-hop questions would open the routing gap. It half-held: the measured
C4-L0 gap *widened* (0.905 -> 0.955) and the stable CI is tight. But the
gate reads **routable fraction** - the share of queries the cheap tier can
already answer - and that *fell* instead of rising: stable 0.045 [0.020,
0.075] versus 0.06 [0.03, 0.095] on the legacy mix (costfinal-10). Making the
cheap tier worse does not create a learnable cheap default; it just moves
more traffic to the cloud. The gate is **NO-GO on both labelings**, and the
stable CI sits entirely below the 0.10 threshold.

Recommendation (unchanged, per `MIX.md`): do not reweight further. The
next lever is the policy around L0 (shrink/retire the closed-book local
tier, or gate on retrieval) rather than query difficulty - and any cloud
correctness claim here stays bounded by the exact-correct cloud stubs
(real cloud accuracy is untested; $0 spent).

**Corpus caveat:** the source is the offline synthetic corpus
(`SYNTHETIC_SEED=20260919`), whose templates keep the answer entity in the
question text; this is a pipeline signal, not an estimate of real
2Wiki/MuSiQue difficulty.

## Exp 1: Closed-book baselines per model tier
_TODO: quality/cost of k=0 direct for each model._

## Exp 2: Retrieval-depth sweep (fixed model, direct reasoning)
_TODO: k in {0, 2, 5, 10} quality/cost curves._

## Exp 3: Reasoning-strategy sweep (fixed model + depth)
_TODO: direct vs chain-of-thought vs rerank-then-answer._

## Exp 4: Full cross-product oracle cost-quality frontier
_TODO: Pareto frontier of all routes; oracle upper envelope._

## Exp 5: Post-retrieval signal predictiveness
_TODO: score margin / coverage / agreement vs oracle-best-route._

## Exp 6: Query-only router baseline
_TODO: router without retrieval signals; gap to oracle._

## Exp 7: Retrieval-aware router (full feature set)
_TODO: main result - cost savings at fixed quality vs oracle._

## Exp 8: Retrieval-depth-as-action ablation
_TODO: router with model-only actions vs joint-space actions._

## Exp 9: Local-tier (Colab) quality/cost characterization
_TODO: amortized Colab cost vs cloud; latency profile._

## Exp 10: Escalation policy (confidence-gated cloud fallback)
_TODO: two-stage router with escalation threshold sweep._

## Exp 11: Out-of-distribution generalization
_TODO: router trained on split A, evaluated on split B._

## Exp 12: Calibration of router confidence
_TODO: predicted vs realized oracle-regret bins._

## Exp 13: Cost-model sensitivity
_TODO: re-score frontier under +/- pricing scenarios._

## Exp 14: End-to-end agent loop vs single-shot routing
_TODO: re-retrieve / clarify / escalate loop on top of Exp 7 router._
