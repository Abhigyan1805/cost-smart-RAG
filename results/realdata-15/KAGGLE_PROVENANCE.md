# Kaggle run provenance (realdata-15, real Tier-A)

Machine-readable record: [`kaggle_provenance.json`](kaggle_provenance.json).

The `sweep.db` / `repeats.db` / `sweep.csv` in this directory were ingested
from the Kaggle kernel `abhigyan1818/realdata15-local-sweep`
(`kernels/realdata-15-local-sweep/`, branch `fm/costsmart-realdata-15`). The
kernel log (~1.8 MB) is not committed; its SHA256 and the runbook summary
are. To audit the run:

```sh
kaggle kernels output abhigyan1818/realdata15-local-sweep -p /tmp/kout
sha256sum /tmp/kout/realdata15-local-sweep.log
# 714a178bf76caff8ed0ae031b66a2748ad2427636c9c226d8b62174cf542a2e2
cmp /tmp/kout/sweep.db   results/realdata-15/sweep.db
cmp /tmp/kout/repeats.db results/realdata-15/repeats.db
cmp /tmp/kout/sweep.csv  results/realdata-15/sweep.csv
cmp /tmp/kout/corpus.json results/realdata-15/corpus.json
```

The downloaded DBs were byte-identical to the committed artifacts at ingest
time. `kaggle_run_summary.json` reports 1400 sweep rows (400 measured L0/L1 +
1000 stub) and 1200 repeat rows (400 pairs x 3, all measured), matching the
kernel log's `RUNBOOK SUMMARY:` line. Every measured row is
`generator_mode='measured'`; no cloud route executed (there is no live cloud
code path) and no stub row overwrote a measured row, so actual cloud spend
is $0.

The real-data run used the committed corpus (byte-identical to
`results/realdata-15/corpus.json`) and the Kaggle host's
`sentence-transformers`, so the L1 retrieval used the pinned MiniLM model
(384-dim) rather than the hash fallback used in the synthetic worker env.
See `CORPUS_MANIFEST.md` for the dataset licences/revisions/checksums and
`docs/kaggle-handoff.md` for the runbook.
