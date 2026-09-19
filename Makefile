.PHONY: index sweep train-router eval report test lint install

install:
	python -m pip install -e ".[dev]"

index:
	python -m costsmart.cli index

sweep:
	python -m costsmart.cli sweep

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
