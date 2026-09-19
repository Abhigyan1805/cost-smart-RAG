"""Attempts table schema (plan section 11).

Every sweep execution writes one row per (query, route) attempt. The table
is the queryable output of ``make sweep`` / ``oracle_sweep.py``.

Resumability + idempotency come from the ``UNIQUE(cache_key)`` constraint:
re-running a sweep (e.g. restarting an interrupted 18k-execution run)
INSERT OR IGNOREs rows whose cache_key already exists, so completed work is
never duplicated.

Reproducibility columns: ``git_sha`` + ``config_hash``.
Cost accounting columns (both modes, always populated): ``cloud_spend_usd``
actual metered cloud spend, ``amortized_usd`` Colab GPU-seconds share.
Latency breakdown columns: ``latency_ms_*`` per pipeline stage.
"""

ATTEMPTS_DDL = """
CREATE TABLE IF NOT EXISTS attempts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key           TEXT NOT NULL UNIQUE,
    query_id            TEXT NOT NULL,
    route_id            TEXT NOT NULL,
    prompt_version      TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    prediction          TEXT NOT NULL DEFAULT '',
    reference           TEXT NOT NULL DEFAULT '',
    exact_match         INTEGER NOT NULL DEFAULT 0,
    lenient_em          INTEGER NOT NULL DEFAULT 0,
    token_f1            REAL NOT NULL DEFAULT 0.0,
    judge_score         REAL,
    judge_model         TEXT NOT NULL DEFAULT '',
    tokens_in           INTEGER NOT NULL DEFAULT 0,
    tokens_out          INTEGER NOT NULL DEFAULT 0,
    gpu_seconds         REAL NOT NULL DEFAULT 0.0,
    concurrency         INTEGER NOT NULL DEFAULT 1,
    cloud_spend_usd     REAL NOT NULL DEFAULT 0.0,
    amortized_usd       REAL NOT NULL DEFAULT 0.0,
    latency_ms_total    REAL NOT NULL DEFAULT 0.0,
    latency_ms_retrieval REAL NOT NULL DEFAULT 0.0,
    latency_ms_llm      REAL NOT NULL DEFAULT 0.0,
    latency_ms_verify   REAL NOT NULL DEFAULT 0.0,
    git_sha             TEXT NOT NULL DEFAULT '',
    config_hash         TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_attempts_query ON attempts(query_id);
CREATE INDEX IF NOT EXISTS idx_attempts_route ON attempts(route_id);
CREATE INDEX IF NOT EXISTS idx_attempts_cache ON attempts(cache_key);
"""

# Canonical column order for inserts/exports.
ATTEMPT_COLUMNS = (
    "cache_key",
    "query_id",
    "route_id",
    "prompt_version",
    "model_version",
    "prediction",
    "reference",
    "exact_match",
    "lenient_em",
    "token_f1",
    "judge_score",
    "judge_model",
    "tokens_in",
    "tokens_out",
    "gpu_seconds",
    "concurrency",
    "cloud_spend_usd",
    "amortized_usd",
    "latency_ms_total",
    "latency_ms_retrieval",
    "latency_ms_llm",
    "latency_ms_verify",
    "git_sha",
    "config_hash",
    "created_at",
)
