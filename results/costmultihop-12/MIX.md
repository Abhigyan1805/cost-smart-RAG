# costmultihop-12 eval mix: reweight toward multi-hop

## Why reweight

The stable-oracle recount (costfinal-10) left the headroom gate at NO-GO:
routable fraction 0.06 [0.03, 0.095] on stable labels, CI entirely below
the 0.10 gate. The legacy mix is 75% single-hop lookup (80 NQ + 70
TriviaQA, answers retrievable from one passage and often guessable
closed-book from parametric memory), which inflates the cheapest-local
(L0, k=0 direct) accuracy and compresses the tier separation a router
could exploit. This branch tests whether a multi-hop-majority mix opens
the routing gap.

## New mix (n=200, same size so CIs stay comparable)

| dataset | n | share | hops | role in the ladder |
|---|---|---|---|---|
| NQ | 30 | 15% | 1 | single-hop lookup baseline |
| TriviaQA | 20 | 10% | 1 | single-hop lookup baseline |
| HotpotQA | 60 | 30% | 2 | bridge/comparison join over 2 passages |
| 2WikiMultihopQA | 50 | 25% | 2-5 | compositional/inference chains via a bridge entity |
| MuSiQue | 40 | 20% | 2-4 | nested hops + 2 distractor passages that must be filtered |

Single-hop share: 150/200 (75%) -> 50/200 (25%).
Multi-hop share: 50/200 (25%) -> 150/200 (75%).

## Difficulty rationale

1. **Single-hop (NQ/TriviaQA):** the answer entity is named by, or directly
   entailed by, one passage. A closed-book small model can answer from
   parametric memory alone - this is the L0-inflating slice.
2. **HotpotQA:** bridge/comparison questions need two passages joined
   (topic_a record points at topic_b and back). L0 (k=0, direct) sees
   neither passage, so it must guess the join; retrieval-augmented and
   large-reasoning routes keep both hops in context.
3. **2WikiMultihopQA:** longer compositional chains (topic_a -> bridge ->
   topic_b); the answer is only reachable via the bridge passage. This
   slice is meant to separate cloud-small direct (C1) from cloud-large
   rerank-then-answer (C4) - the mid-tier routing gap the degenerate
   all-correct cloud stubs currently hide.
4. **MuSiQue (hardest):** nested multi-hop with two distractor passages per
   query (one elsewhere). Rewards rerank-then-answer reasoning (C2/C4) and
   punishes direct answering that latches onto a distractor.

## Freeze

- Query set: `splits_freeze.json` (ordered query dicts, `sort_keys=True`);
  hash recomputed from the loader by `tests/test_multihop_mix.py`
  (`PYTHONPATH=src python3 -m unittest tests.test_multihop_mix`).
- Set SHA256: `71d0c4ce623e2d203ba437aa4bff08b0896d5d4dec259a2b6dff4f2d2f1142f0`
- Sweep config: `config/experiments/sweep-200-multihop.yaml`
  (seed 0, prompt `v1`, `corpus_source: synthetic`,
  `SYNTHETIC_SEED=20260919`).
- Retrieval index: `data/index/multihop_index.json` (200 queries,
  590 passages / 590 chunks - extra gold + MuSiQue distractors).
- Loader: `src/costsmart/corpus/loaders.py :: load_multihop_subset`
  (legacy `load_pilot_subset` untouched; its set hash still verifies as
  `fe00c488…`, see CALIBRATION.md).

## What the mix change can and cannot move

- The stub recount in `CALIBRATION.md` is a **pipeline smoke test only**:
  cloud routes are exactly-correct by the frozen stub contract and stub L0
  misses ~1/3 by hash, independent of difficulty. It cannot validate the
  hypothesis.
- The hypothesis lives or dies on **measured** local-tier rows (Colab,
  temperature 0, seed 0, 3x majority per the repeat protocol): multi-hop
  should collapse measured L0/L1 accuracy while stub C4 stays correct,
  widening the C4-L0 gap and moving routable fraction. If the gate still
  reads below 0.10 on measured rows, the verdict stays NO-GO and the
  recommendation is to shrink L0 - not to reweight further.
