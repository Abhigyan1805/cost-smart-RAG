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
`comparison.json`).

## Verification / caveats

- The preserved partial run is the resume base: it carries 400 measured
  L0/L1 base rows (live Colab) and 269/1200 repeat draws. The Kaggle run
  completes the missing repeat draws; it does not re-roll the Colab base
  rows (same model id, cache-key hit) and never overwrites a measured row
  with a stub row.
- Before trusting a resumed base, check that the new Kaggle repeat draws
  agree with the preserved Colab repeat draws on the pairs present in both
  (same majority side, similar token_f1 distribution); a gross mismatch is
  a provenance red flag, not a result.
- Kaggle GPU kernels have a weekly quota (currently ~30h) and a per-run
  wall-clock cap; the repeat run is the long pole (~930 draws * a few
  seconds). The kernel self-terminates the server when done.
- `enable_internet: true` is required for the model weight download; the
  repo clone is public, so no token is embedded anywhere.
