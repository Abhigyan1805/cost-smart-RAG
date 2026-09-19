# costsmart-rag Experiment Report (stub - Week-1 skeleton)

> Each experiment below measures its claim against the exhaustive oracle over
> the joint (model x retrieval depth x reasoning strategy) space. Sections are
> placeholders until sweep telemetry lands.

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
