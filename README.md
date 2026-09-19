# costsmart-rag

Existing LLM routers treat the query as the only input and the model as the only action. In a RAG setting, retrieval produces strong difficulty signals for free, and retrieval depth is itself a cheap action - costsmart-rag routes over the joint space and measures the resulting cost-quality frontier against an exhaustive oracle.

## How it works

A routed RAG agent: retrieval runs **before** routing, so post-retrieval signals (score margins, coverage, agreement) become free router features. The router picks a route from the joint space of **model x retrieval depth x reasoning strategy**, and every claim is measured against an **exhaustive oracle** that evaluates all routes per query.

Local model tiers run in **Google Colab**, not on this machine. Local execution goes through the Colab-compatible client interface in `src/costsmart/models/colab_client.py` - never a local Ollama daemon assumption. The captain connects the Colab session on request; see that file's handoff notes.

## Make instructions

```sh
make install       # install package + dev extras
make index         # build the retrieval index (retrieval slice)
make sweep         # run the exhaustive route sweep (oracle telemetry)
make train-router  # train the router on sweep telemetry
make eval          # evaluate router vs exhaustive oracle
make report        # render REPORT.md figures
make test          # run pytest
make lint          # run ruff
```

Individual CLI entrypoints are also available via `python -m costsmart.cli <command>`.

## Week-1 status

Skeleton only. This slice owns: `pyproject.toml`, `Makefile`, `.gitignore`, `config/`, `src/costsmart/` model interfaces + CLI stubs, `README.md`, `REPORT.md` stub, `tests/` scaffolding.

| Area | Owner | Status |
|---|---|---|
| Skeleton / model interfaces / CLI | this slice | in progress (Week-1) |
| Corpus + retrieval | sibling slice | not started |
| Telemetry / sweep harness | sibling slice | not started |
| Eval + oracle | sibling slice | not started |
| Routing / router training | sibling slice | not started |
| Verification / agent loop / demo | sibling slice | not started |

Pinned versions live in `pyproject.toml` and `config/models.yaml`. Cost placeholders live in `config/costs.yaml` (fill from provider pricing before any reported run).
