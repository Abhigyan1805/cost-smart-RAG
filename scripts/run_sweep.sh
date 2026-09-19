#!/usr/bin/env bash
# Retrieval/corpus sweep stub -- owned by the costretrieval-02 slice.
#
# NOTE: parameter sweeps (k, fusion weights, chunk sizes) and their telemetry
# logging live in the TELEMETRY slice (costtelemetry-03). Do not implement
# sweep logic here; this stub exists so pilot docs can reference one entry
# point. The telemetry owner will replace this file with:
#   scripts/run_sweep.sh <sweep-config>  ->  logs to data/telemetry/
set -euo pipefail

echo "run_sweep.sh is a stub: sweep harness is owned by the telemetry slice (costtelemetry-03)." >&2
echo "For retrieval validation now, run: PYTHONPATH=src python scripts/smoke_retrieval.py" >&2
exit 2
