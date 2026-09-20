# costsmart-rag Experiment Report

> Each experiment measures its claim against the exhaustive oracle over the
> joint (model x retrieval depth x reasoning strategy) space. The headroom
> gates below are the only completed experiments; Exp 1-14 further down are
> still placeholders.

## Scope and claim strength (read first)

**The NO-GO holds on REAL labelled Tier-A data, not only on the synthetic
smoke corpus.** The audit objection was that the synthetic templates leak
the answer entity into the question, so the synthetic NO-GO is only a
pipeline smoke signal. The `realdata-15` recount answers that directly:
the same 7-route sweep and 3x-majority stable gate, run live on **real NQ
open + HotpotQA + MuSiQue** (200 queries, 75% multi-hop, fetched through
the HF loader path and committed at `results/realdata-15/corpus.json`),
also reads **NO-GO** - routable fraction **0.095 [0.055, 0.14]** on
single-run labels and **0.095 [0.055, 0.135]** on stable labels, both
below the 0.10 gate. The synthetic result did not manufacture the verdict;
the low cheap-tier routable fraction is a property of the measured local
model on real questions. **The real-data verdict is nevertheless
*marginal*: the point estimate sits at 0.095 against a 0.10 gate, and the
CI upper bound (~0.14) crosses it, whereas the synthetic stable run read a
comfortable 0.045.** Full section: "Real-data recount (realdata-15)"
below.

**The synthetic corpora remain pipeline results, not benchmark
conclusions.** Both full-matrix *synthetic* recounts (legacy mix,
`costfinal-10`; multi-hop mix, `costmultihop-12` / `costsweep-13`) run on
the offline synthetic corpus (`SYNTHETIC_SEED=20260919`), whose templates
keep the answer entity in the question text (`results/costmultihop-12/MIX.md`,
`results/costmultihop-12/CALIBRATION.md` "Assumptions / caveats"). Their
measured effect shows the pipeline responds to query difficulty; they are
**not** an estimate of real 2Wiki/MuSiQue behaviour. The real-data recount
below supersedes them for the product question.

Three further limits are load-bearing:

- **"Dense" retrieval is the stdlib hash-embedding fallback in this
  environment.** `sentence-transformers` is unavailable here, so
  `src/costsmart/retrieval/dense.py` falls back to a 256-dim sha256
  bag-of-words vector while the index metadata still names
  `all-MiniLM-L6-v2`. Every committed row is `retrieval_mode='measured'`,
  but what was measured is hybrid search over that fallback, not MiniLM
  dense retrieval - so the retrieval quality and timings in these artifacts
  are not the real-model numbers a reader might assume. (Exception: the
  `realdata-15` Kaggle host has `sentence-transformers`, so its L1 retrieval
  used the pinned MiniLM model end-to-end; see
  `results/realdata-15/KAGGLE_PROVENANCE.md`.)
- **The cloud-stub cost columns are ESTIMATED, not spent.** C1..C4 never
  execute (there is no live cloud code path); their `cloud_spend_usd` is the
  deterministic stub's token draw priced at pinned rates. Every dollar figure
  quoted below (oracle `$0.006402`, all-C4 `$0.120015`, etc.) is a stub
  estimate; **actual cloud spend is $0**. The oracle/comparison cost figures
  inherit this.
- **Measured-model identity is now server-attested for new rows, and flagged
  where it is not.** Before `tiersweep-16`, `model_version` was the string
  the client sent, not a server-side attestation: an endpoint could serve a
  different checkpoint and still be recorded as the pinned id. The serving
  endpoint now reports the resolved revision + weight hash and the client
  records them (`model_revision` / `weights_sha256` / `attestation`); every
  measured row lacking server provenance is flagged `unattested` by
  `scripts/attestation_audit.py`. The `tiersweep-16` 3B/7B rows are
  `server-attested`; the committed `realdata-15` 1.5B rows (and all older
  artifacts) predate the columns and are flagged, never falsified.

These are the boundaries within which every number below should be read.

## Real-data recount (realdata-15): gate NO-GO on real Tier-A data

This is the decisive re-run the audit called for. It replaces the synthetic
corpus with **real labelled Tier-A data** and repeats the exact committed
gate protocol (7-route sweep, measured L0/L1, 3x-majority stable recount,
10k-resample bootstrap CIs, McNemar). All artifacts are in
`results/realdata-15/`; the run is pinned by
`results/realdata-15/KAGGLE_PROVENANCE.md` + `kaggle_provenance.json`.

### Corpus (real, committed)

200 real validation queries - **50 NQ (`nq_open`) + 80 HotpotQA
(`distractor`) + 70 MuSiQue** = 75% multi-hop, mirroring the synthetic
`costmultihop-12` 25/75 composition. Fetched through the repo's existing HF
loader path (`scripts/fetch_tier_a.py` -> `normalize_hf_row`) on the Kaggle
host and committed at `results/realdata-15/corpus.json`; per-dataset
licence, resolved revision, and question/passage checksums are in the
manifest and `results/realdata-15/CORPUS_MANIFEST.md`. No dataset was
substituted or fabricated. NQ (`nq_open`) ships question+answer only, so NQ
queries have no gold passage and no passage pool; the gate reads L0
(closed-book) vs the C4 stub, not L1.

### Execution

Kaggle GPU kernel `abhigyan1818/realdata15-local-sweep` (T4, branch
`fm/costsmart-realdata-15`), pinned model `Qwen/Qwen2.5-1.5B-Instruct`,
temperature 0, fixed seed. **400 measured L0/L1 base rows** (200 + 200) and
**1200/1200 measured repeat draws** (400 pairs x 3), all
`generator_mode='measured'`; C0..C4 stub (C4 exactly correct by the frozen
stub contract). Cloud spend **$0** - no live cloud path executed, no stub
row overwrote a measured row. The L1 retrieval index used the pinned MiniLM
model (384-dim, `sentence-transformers`), not the hash fallback.

### Gate: single-run labels vs stable labels (n=200 paired, 95% CIs)

| labels | contingency (both / L0-only / C4-only / neither) | routable fraction | max saving | gap (C4-L0) | McNemar | verdict |
|---|---|---|---|---|---|---|
| single-run | 19 / 0 / 181 / 0 | 0.095 [0.055, 0.140] | 94.77% [94.51%, 95.03%] | 0.905 | p = 8.0e-41 | **NO-GO** |
| stable (majority) | 19 / 0 / 181 / 0 | 0.095 [0.055, 0.135] | 94.77% [94.51%, 95.02%] | 0.905 | p = 8.0e-41 | **NO-GO** |

(Measured L0 = 19/200 correct closed-book, L1 = 25/200; C4 stub = 200/200
correct. Oracle $0.006162 vs all-C4 $0.117915 - both stub-ESTIMATED cloud
spend, not spent; actual cloud spend is $0. 200/200 routed off C4, 0
quality-loss queries. `comparison.json` records 0 single->stable label
changes.)

### Measured accuracy by source (single-run)

| source | L0 (closed-book) | L0 mean token-F1 | L1 (k=5 retrieval) | L1 mean token-F1 |
|---|---|---|---|---|
| NQ | 0/50 (0.00) | 0.094 | 2/50 (0.04) | 0.071 |
| HotpotQA | 15/80 (0.19) | 0.193 | 16/80 (0.20) | 0.260 |
| MuSiQue | 4/70 (0.06) | 0.076 | 7/70 (0.10) | 0.104 |
| **all** | **19/200 (0.095)** | 0.127 | **25/200 (0.125)** | 0.158 |

### Label stability on real data

String flip rate is **0.0** on both routes (all 3 draws byte-identical per
pair) versus 0.220 on the synthetic multi-hop mix and 0.865 on the legacy
mix: real short factual answers leave no paraphrase room, so the
single-run/stable split is empty here (0 label changes). The real-data
verdict therefore does not depend on the label-noise fix.

### Verdict: real-data NO-GO, same direction as synthetic but marginal

The real-data verdict is **NO-GO, and it does not differ from the synthetic
verdict** - the low routable fraction is real, not a template artifact. But
it is **marginal**, not comfortable: the real point estimate (0.095) sits
just below the 0.10 gate and its CI upper bound (~0.14) crosses the gate,
whereas the synthetic stable estimate was 0.045. The honest reading is that
the synthetic smoke corpus *understated* the cheap tier's real routable
fraction: on real data a closed-book 1.5B model routes roughly one query in
ten to itself (mostly HotpotQA yes/no), still short of the exploitable-
separation gate but close enough that a GO call would not be robust.
Recommendation (unchanged in direction, strengthened in caveat): do not
build a routing policy on the L0 default yet; the measured gap is real
(McNemar p ~ 8e-41) but the cheap tier's coverage is at the decision
boundary. Any cloud-correctness/savings claim stays bounded by the
exact-correct C4 stub (real cloud accuracy untested; $0 spent).

## Cheap-tier break-even (tiersweep-16): 3B clears the closed-book gate

The `realdata-15` NO-GO is driven by the **cheap tier**, not the router: the
1.5B closed-book route (L0) covers only 0.095 of queries against the 0.10
gate, while the oracle ceiling is ~94% saving at no quality loss. This sweep
asks where a stronger cheap tier starts to pay: the same three cheap-tier
routes - **L0** (k=0, direct), **L1** (k=5, direct), **C0** (k=5,
chain-of-thought) - on **Qwen2.5-1.5B / 3B / 7B**, over the same committed
real Tier-A corpus (NQ + HotpotQA + MuSiQue), temperature 0, seed 0, 3x
majority repeats. The 1.5B tier reuses the committed `realdata-15` run (not
re-measured); 3B/7B ran fresh on the Kaggle GPU route
(`kernels/tiersweep-16-local-sweep/`, git `e282989`, 1800/1800 repeat draws
per tier) and are **server-attested**. Artifacts: `results/tiersweep-16/`
(`tiers.json`, `TIERS.md`, `tiers.svg`, `kaggle_provenance.json`,
`MODEL_MANIFEST.md`).

### Tier-vs-coverage-vs-cost (n=200 paired per route; 95% bootstrap CIs)

| tier | route | coverage (routable fraction) | C4-gap | McNemar p | GPU-s/query | cost ratio r | saving @ gate | verdict |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | L0 | 0.095 [0.055, 0.140] | 0.905 | 8.0e-41 | 1.00 | 0.166 | 0.083 | **NO-GO** |
| Qwen2.5-1.5B | L1 | 0.125 [0.080, 0.170] | 0.875 | 1.6e-39 | 3.25 | 0.536 | 0.046 | GO (marginal) |
| Qwen2.5-3B | L0 | **0.155 [0.105, 0.205]** | 0.845 | 3.3e-38 | 3.27 | 0.540 | 0.046 | **GO** |
| Qwen2.5-3B | L1 | 0.085 [0.050, 0.125] | 0.915 | 2.9e-41 | 6.80 | 1.122 | -0.012 | NO-GO |
| Qwen2.5-3B | C0 | 0.170 [0.120, 0.225] | 0.830 | 1.5e-37 | 5.07 | 0.836 | 0.016 | GO |
| Qwen2.5-7B | L0 | 0.150 [0.100, 0.200] | 0.850 | 2.0e-38 | 1.86 | 0.306 | 0.069 | **GO** |
| Qwen2.5-7B | L1 | 0.065 [0.035, 0.100] | 0.935 | 3.9e-42 | 8.03 | 1.325 | -0.033 | NO-GO |
| Qwen2.5-7B | C0 | 0.075 [0.040, 0.115] | 0.925 | 1.1e-41 | 13.74 | 2.265 | -0.127 | NO-GO |

### Break-even statement

At the measured cheap/strong cost ratio `r`, the no-quality-loss saving from
routing a covered fraction `f` is `f * (1 - r)`; the plan's 0.10
exploitable-separation gate is the break-even coverage, so clearing it
asserts a saving of at least `0.10 * (1 - r)`. **The closed-book cheap route
L0 first clears the gate at Qwen2.5-3B**: coverage 0.155 [0.105, 0.205] (the
CI lower bound clears 0.10), cost ratio r = 0.540. Qwen2.5-7B is statistically
indistinguishable (0.150 [0.100, 0.200], r = 0.306) - so the break-even sits
between 1.5B and 3B, and adding parameters past 3B buys no further closed-book
coverage here. The 1.5B L1 route already clears the gate on the point estimate
(0.125) but not its CI lower bound, so the historical NO-GO is specific to the
closed-book L0 route.

### Caveats (load-bearing)

- **The strong route is still a stub.** C4 is exactly correct on every query
  by the frozen stub contract, so the gate's routable fraction equals the
  measured cheap-tier accuracy and the C4-vs-tier gap is `1 - accuracy`; the
  McNemar tests are therefore degenerate (c = 0) and are reported only for
  completeness. Real cloud accuracy remains untested ($0 spent).
- **The cost ratio mixes measured local GPU cost with a stub cloud
  estimate.** Cheap cost is measured amortized T4 GPU-seconds; strong cost is
  the deterministic stub's token draw priced at gpt-4o rates (tiny token
  counts), so `r` is an artifact of the stub as much as of the hardware. The
  measured GPU-seconds/query column is the robust cost axis.
- **L1/C0 degrade with model size - a prompt-adherence artifact, not a
  capability claim.** The larger checkpoints answer direct/CoT prompts
  verbosely and ramble past the 256-token cap (`extract_final_answer` then
  returns the whole text), so `token_f1` falls below 0.5 on routes whose
  reference answers are short; e.g. 7B L1 averages 104 output tokens and hits
  the cap on many queries. The closed-book L0 route, which yields short
  answers, is the clean cross-tier comparison and the basis of the
  break-even.
- **Licence.** The first tier to clear the gate (3B) is under the **Qwen
  Research License**; the 7B is Apache-2.0. A commercially deployable
  break-even therefore depends on the 7B, whose coverage is statistically
  indistinguishable from 3B's (`MODEL_MANIFEST.md`).
- **Retrieval-latency caveat.** The `realdata-15` run reloaded the
  MiniLM embedding model once per attempt, inflating its measured
  `latency_ms_retrieval` (avg 1380 ms vs 72 ms after the `tiersweep-16`
  module-level cache fix) and wall-clock. This does **not** affect correctness
  labels or the amortized GPU cost (`gpu_seconds` is generation-only, from the
  server), so the two tiers' cost ratios stay comparable; the 1.5B rows keep
  their original latency columns.

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
  [94.0%, 95.3%]** (oracle $0.000650 vs all-C4 $0.012325, both stub-estimated;
  20/20 queries routed off C4, 0 quality-loss queries).
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

(Oracle $0.00614 vs all-C4 $0.12385 - both are stub-ESTIMATED cloud spend,
not spent; actual cloud spend is $0. 200/200 routed off C4, 0 quality-loss
queries - C4 stub is exactly correct on every query by the frozen stub
contract.)

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

(Oracle $0.006402 vs all-C4 $0.120015 - both are stub-ESTIMATED cloud spend,
not spent; actual cloud spend is $0. 200/200 routed off C4, 0 quality-loss
queries - the C4 stub is exactly correct on every query by the frozen stub
contract.)

### Verdict: reweighting did not open the gate on the synthetic mix

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
(real cloud accuracy is untested; $0 spent). This recommendation is
**scoped to the synthetic smoke corpus**: re-run on a real dataset before
treating it as product guidance.

**Corpus caveat (load-bearing).** The source is the offline synthetic corpus
(`SYNTHETIC_SEED=20260919`), whose templates keep the answer entity in the
question text. The NO-GO is therefore a **pipeline smoke signal, not an
estimate of real 2Wiki/MuSiQue difficulty and not a benchmark conclusion**;
a real-dataset re-run is required before any product decision. See "Scope
and claim strength (read first)" at the top of this report.

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
