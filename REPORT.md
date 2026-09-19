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
