# Kaggle run provenance (costsweep-13)

Machine-readable record: [`kaggle_provenance.json`](kaggle_provenance.json).

The `sweep.db` / `repeats.db` / `sweep.csv` in this directory were ingested
from the Kaggle kernel `abhigyan1818/costsweep13-local-sweep`
(`kernels/costsweep-13-local-sweep/`, branch `fm/costsweep-13`). The kernel
log is ~9.3 MB, so it is not committed here; its SHA256 and the runbook
summary are. To audit the resume provenance:

```sh
# re-download the kernel output, then verify the log hash and the DBs
kaggle kernels output abhigyan1818/costsweep13-local-sweep -p /tmp/kout
sha256sum /tmp/kout/costsweep13-local-sweep.log
# 2ee7ccac6842d5610ba80b189b9bdee272267653b1284dcc627fe403b890e64d
cmp /tmp/kout/sweep.db   results/costmultihop-12/sweep.db
cmp /tmp/kout/repeats.db results/costmultihop-12/repeats.db
cmp /tmp/kout/sweep.csv  results/costmultihop-12/sweep.csv
```

The downloaded DBs were byte-identical to the committed artifacts at ingest
time, and `kaggle_run_summary.json` reports the same counts as the kernel
log's `RUNBOOK SUMMARY:` line: 1400 sweep rows (400 measured L0/L1 + 1000
stub) and 1200 repeat rows (400 pairs x 3, all measured). The repeats DB
carries two provenance epochs - 269 Colab draws at `3bf1cf6` and 931 Kaggle
draws at `777aa01`. See `docs/kaggle-handoff.md` for the runbook and
"Assumptions / caveats" in `CALIBRATION.md` for the limits of this route.
