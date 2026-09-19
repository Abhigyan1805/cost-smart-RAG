# costsmart-rag Makefile (skeleton targets + telemetry sweep params merged).
#
# NOTE on `make sweep --limit 20`: that literal spelling is not valid GNU
# make syntax -- make parses `--limit` as a make option (not a target) and
# errors before any recipe runs. Supported equivalents:
#   make sweep                 # via CLI: python -m costsmart.cli sweep
#   make sweep SWEEP_ARGS="--limit 50"   # extra args passed through to CLI
#   make sweep DB=/tmp/t.db CONFIG=config/experiments/E2.yaml SWEEP_ARGS="--limit 50"
# Direct oracle_sweep equivalent (uses LIMIT/DB/CONFIG below):
#   PYTHONPATH=src python3 -m costsmart.eval.oracle_sweep --limit 20
#   PYTHONPATH=src python3 -m costsmart.eval.oracle_sweep --limit $(LIMIT) --db $(DB) --config $(CONFIG) $(SWEEP_ARGS)
# The trailing catch-all lets `make sweep -- --limit 20` parse (extra goals
# become no-ops; the recipe still uses LIMIT/SWEEP_ARGS).

LIMIT ?= 20
DB ?= telemetry.db
CONFIG ?= config/experiments/E1.yaml
SWEEP_ARGS ?=

.PHONY: index sweep train-router eval report test lint install

install:
	python -m pip install -e ".[dev]"

index:
	python -m costsmart.cli index

sweep:
	python -m costsmart.cli sweep $(SWEEP_ARGS)

train-router:
	python -m costsmart.cli train-router

eval:
	python -m costsmart.cli eval

report:
	python -m costsmart.cli report

test:
	python -m pytest -q

lint:
	python -m ruff check src tests

%:
	@:
