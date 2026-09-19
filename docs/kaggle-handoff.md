# Kaggle handoff: local-tier GPU route (costsweep-13)

When the Colab free-tier GPU quota is exhausted, the L0/L1 local-tier sweep
and its stability repeats run on a **Kaggle GPU kernel** instead. The pinned
local model is unchanged (`config/models.yaml`: local-small =
`Qwen/Qwen2.5-1.5B-Instruct`); only the compute host moves.

Zero cloud spend is unchanged: cloud routes (C1..C4) are never executed -
there is no live cloud code path - and C0 (7B) stays stub until a 7B server
exists. The Kaggle run only ever adds `generator_mode='measured'` rows.

## What the kernel does

`kernels/costsweep-13-local-sweep/kernel.py`:

1. clones the public repo at the task branch
   (`fm/costsweep-13`) - Kaggle `kernels push` does not upload sibling
   files, so the kernel fetches the committed code + telemetry DBs itself;
2. rebuilds the frozen multihop retrieval index deterministically
   (`python -m costsmart.corpus.build_index --mix multihop --source synthetic`);
3. starts the repo's own transformers server
   (`scripts/colab_local_tier.py serve --model Qwen/Qwen2.5-1.5B-Instruct`);
4. drives the committed runbook against `http://127.0.0.1:8000`, resuming
   from the committed `results/costmultihop-12/{sweep,repeats}.db` by cache
   key so already-measured work is never re-spent;
5. copies `sweep.db`, `repeats.db`, `sweep.csv`, `repeats_summary.json` to
   `/kaggle/working` for download.

The kernel is resumable end to end: both the base matrix (`cache_key`) and
the repeat draws (`repeat_cache_key`, 5-tuple) are `INSERT OR IGNORE`, so a
re-push after an interruption only executes the remainder.

## Commands

```sh
# 1) push (creates/updates the private kernel; GPU + internet in metadata)
kaggle kernels push -p kernels/costsweep-13-local-sweep

# 2) monitor
kaggle kernels status abhigyan1818/costsweep13-local-sweep

# 3) download outputs when COMPLETE
kaggle kernels output abhigyan1818/costsweep13-local-sweep -p /tmp/kout
```

`kaggle kernels push` returns immediately; it only queues the run. Poll (2)
until the status is `KernelWorkerStatus.COMPLETE`, then download (3). The
kernel log (`/tmp/kout/costsweep13-local-sweep.log`) carries the
`RUNBOOK SUMMARY:` line with the row counts to check against the ingest.

After downloading, persist the log hash and the runbook summary next to the
ingested DBs (the log itself is ~9.3 MB, so it is not committed). The
committed record for costsweep-13 is
`results/costmultihop-12/kaggle_provenance.json` +
`results/costmultihop-12/KAGGLE_PROVENANCE.md`, which also hash-pin the
ingested `sweep.db` / `repeats.db` / `sweep.csv`.

## Ingest + recount (cloud side, stdlib only, $0)

```sh
# copy the downloaded DBs/CSV over the repo artifacts, then rebuild the
# stable-oracle matrix and recount the L0-vs-C4 headroom gate
cp /tmp/kout/sweep.db   results/costmultihop-12/sweep.db
cp /tmp/kout/repeats.db results/costmultihop-12/repeats.db
cp /tmp/kout/sweep.csv  results/costmultihop-12/sweep.csv
cp /tmp/kout/repeats_summary.json results/costmultihop-12/repeats_summary.json

PYTHONPATH=src python scripts/stable_oracle.py \
  --sweep-db results/costmultihop-12/sweep.db \
  --repeats-db results/costmultihop-12/repeats.db \
  --out-dir results/costmultihop-12
```

`scripts/stable_oracle.py` grades each repeated pair by 3x majority (ties
count as incorrect), rebuilds the oracle matrix on stable labels, and
recomputes the gate with 10k-resample bootstrap CIs + the McNemar paired
test (`results/costmultihop-12/headroom_stable.{json,svg}` and
`comparison.json`). Its `--sweep-db`, `--repeats-db`, and `--out-dir` are
required (no legacy defaults): a bare invocation refuses, and a sweep/repeats
pair drawn from different mixes refuses before writing anything, so the
recount cannot silently fall back to the legacy pilot mix.

## Verification / caveats

- The preserved partial run is the resume base: it carries 400 measured
  L0/L1 base rows (live Colab) and 269/1200 repeat draws. The Kaggle run
  completes the missing repeat draws; it does not re-roll the Colab base
  rows (same model id, cache-key hit) and never overwrites a measured row
  with a stub row.
- Before trusting a resumed base, check that the new Kaggle repeat draws
  agree with the preserved Colab repeat draws on the pairs present in both
  (same majority side, similar token_f1 distribution); a gross mismatch is
  a provenance red flag, not a result. For costsweep-13 this check is
  effectively vacuous: the Colab epoch (269 draws) and Kaggle epoch (931
  draws) share only **1** pair, so the two hosts cannot be cross-validated
  from the committed data - the Kaggle draws are trusted via the persisted
  log hash + runbook summary (`KAGGLE_PROVENANCE.md`), not via overlap.
- Kaggle GPU kernels have a weekly quota (currently ~30h) and a per-run
  wall-clock cap; the repeat run is the long pole (~930 draws * a few
  seconds). The kernel self-terminates the server when done.
- `enable_internet: true` is required for the model weight download; the
  repo clone is public, so no token is embedded anywhere.

## realdata-15: decisive run on REAL Tier-A data

`kernels/realdata-15-local-sweep/` is the real-data counterpart of the
synthetic `costsweep-13` route. It clones branch `fm/costsmart-realdata-15`
and, before serving the model, fetches + normalizes the real corpus through
the repo's HuggingFace loader path (`scripts/fetch_tier_a.py`), then builds
the index from that committed corpus (`build_index --mix real`). The driver,
resumability, and zero-cloud-spend guarantees are identical to costsweep-13.

```sh
kaggle kernels push   -p kernels/realdata-15-local-sweep
kaggle kernels status abhigyan1818/realdata15-local-sweep
kaggle kernels output abhigyan1818/realdata15-local-sweep -p /tmp/kout
```

Artifacts to ingest (files land flat in `/tmp/kout`):

```sh
cp /tmp/kout/tier_a_real.json      data/real/tier_a_real.json
cp /tmp/kout/real_index.json       data/index/real_index.json
cp /tmp/kout/sweep.db              results/realdata-15/sweep.db
cp /tmp/kout/repeats.db            results/realdata-15/repeats.db
cp /tmp/kout/sweep.csv             results/realdata-15/sweep.csv
cp /tmp/kout/repeats_summary.json  results/realdata-15/repeats_summary.json

PYTHONPATH=src python scripts/stable_oracle.py \
  --sweep-db results/realdata-15/sweep.db \
  --repeats-db results/realdata-15/repeats.db \
  --out-dir results/realdata-15
```

The real mix is 50 NQ + 80 HotpotQA + 70 MuSiQue (75% multi-hop), mirroring
the synthetic `costmultihop-12` composition. `nq_open` ships question+answer
only, so NQ has no gold passage / passage pool and its L1 retrieval draws
from the shared multi-hop pool; the headroom gate reads L0 (closed-book) vs
the C4 stub, not L1. Licences + revisions + checksums are recorded per
dataset in `data/real/tier_a_real.json`'s manifest and in
`data/real/MANIFEST.md` (never inferred from memory).
