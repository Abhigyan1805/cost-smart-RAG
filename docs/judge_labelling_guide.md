# Tier-B judge labelling guide

Validates the LLM judge (`src/costsmart/eval/graders.py`) against human
judgment before `judge_score` may be trusted in oracle-sweep analysis.

## Why hand-label

Tier-A datasets (NQ, TriviaQA, HotpotQA) have gold answers, so deterministic
EM / lenient-EM / token-F1 grade them. Tier-B answers are **free-form with no
gold** — only the LLM judge grades them. The judge is a measurement instrument
with error bars: this kit checks its agreement with humans (Cohen's kappa,
gate **≥ 0.7**) and its agreement with a second judge (disagreement rate).

## Workflow

1. **Sample the batch** (100 Tier-B pairs, seeded and reproducible):

   ```sh
   PYTHONPATH=src python scripts/sample_label_batch.py --db telemetry.db \
       --questions data/questions.jsonl --n 100 --seed 7 --out tierb_batch.csv
   ```

   Tier-B = attempts with an empty `reference` (no gold). One pair per query
   by default. The `human_label` column ships **empty** — the labeller fills
   it. The worker never fabricates labels; the captain supplies the completed
   file separately.

2. **Label each pair** with one of three levels (same scale the judge uses):

   | Label | Meaning | Example |
   |---|---|---|
   | `1` (correct) | Directly and correctly answers the question; paraphrases and equivalent numeric rephrasings count | Q: capital of France? A: "Paris" |
   | `0.5` (partial) | On-topic and partly right but omits a key fact, adds an important falsehood, or answers a different question | Q: who wrote 1984 and when? A: "George Orwell" (right author, missing year) |
   | `0` (incorrect) | Wrong, irrelevant, refusal, or empty | Q: capital of France? A: "Lyon" |

   Rules of thumb:

   - Judge semantic equivalence, not wording ("meters per second" = "m/s").
   - Be strict about numbers, names, dates: a wrong value is `0`, not `0.5`.
   - When a `reference` cell is filled (rare in Tier-B), it is authoritative.
   - `judge_score` is shown for context only — label from the question and
     answer, not from the judge's opinion. Blind labelling avoids anchoring.
   - Unsure between adjacent levels: pick the lower one and note why in
     `notes`.

3. **Compute agreement** (captain's completed file, never committed by the
   worker):

   ```sh
   PYTHONPATH=src python scripts/compute_kappa.py captain_labels.csv
   ```

   Accepted formats: CSV with a judge-like column (`judge`, `judge_label`,
   `judge_score`, …) and a human-like column (`human`, `human_label`,
   `captain`, …), or JSONL with the same keys. Blank cells are skipped and
   reported. Pass `--threshold 0.5` to binarize continuous scores first.

## Acceptance gate

- **kappa ≥ 0.7** (`KAPPA_GATE`): judge scores may enter sweep analysis.
- **Below 0.7**: recalibrate the rubric or replace the judge model; do not
  backfill `judge_score` into reported results.
- Second-judge check: `judge_disagreement_rate` (primary
  `claude-3-5-sonnet-20241022` vs `gpt-4o-mini-2024-07-18`, tol 0.25) should
  stay low; a high rate with high human agreement on one side picks the
  surviving judge.

## Model-choice note

The C2 strong route runs `gpt-4o-2024-08-06`. The primary judge is a
different-family model at temperature 0 so agreement cannot be explained by
self-preference bias. Re-pin the judge to the current strong non-OpenAI
model before any reported run and record the pin in run metadata.
