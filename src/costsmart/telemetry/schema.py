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
Provenance columns (costsweep-08): ``generator_mode`` / ``retrieval_mode``
are each ``measured`` (ran for real here) or ``stub`` (deterministic
estimate, $0 spent); ``temperature`` + ``seed`` pin the generator sampling
contract (local tiers run at temperature 0, fixed seed). Legacy rows written
before these columns existed migrate to ``unflagged-legacy`` (documented
per-artifact: pilot.db = stub generator + measured retrieval).

Attestation columns (tiersweep-16, audit finding): a measured row must
prove *which* weights produced it, not self-declare it. The serving endpoint
reports the resolved model revision (``model_revision``) and, best-effort, a
sha256 over the weight shards (``weights_sha256``); the client copies both
into the row and derives ``attestation``:

* ``server-attested`` -- measured row carrying a server-reported revision or
  weight hash (provenance is verifiable, not client-declared);
* ``unattested``     -- measured row with neither (flagged, never falsified);
* ``stub``           -- deterministic stub row (no model ran);
* ``unflagged-legacy`` -- row written before these columns existed.

Absent attestation is a flag, not a falsification: legacy measured rows keep
their original labels and are counted by the attestation audit.
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
    created_at          TEXT NOT NULL DEFAULT '',
    generator_mode      TEXT NOT NULL DEFAULT 'unflagged-legacy',
    retrieval_mode      TEXT NOT NULL DEFAULT 'unflagged-legacy',
    temperature         REAL NOT NULL DEFAULT 0.0,
    seed                INTEGER NOT NULL DEFAULT 0,
    model_revision      TEXT NOT NULL DEFAULT '',
    weights_sha256      TEXT NOT NULL DEFAULT '',
    attestation         TEXT NOT NULL DEFAULT 'unflagged-legacy'
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
    "generator_mode",
    "retrieval_mode",
    "temperature",
    "seed",
    "model_revision",
    "weights_sha256",
    "attestation",
)

#: Columns added after the pilot (costsweep-08 + tiersweep-16). Legacy DBs
#: (e.g. the frozen pilot.db) predate them; TelemetryStore migrates them via
#: ALTER TABLE so old rows read back as 'unflagged-legacy' instead of failing
#: inserts.
MIGRATED_COLUMNS = (
    "generator_mode",
    "retrieval_mode",
    "temperature",
    "seed",
    "model_revision",
    "weights_sha256",
    "attestation",
)

#: Attestation statuses (tiersweep-16). See the module docstring.
ATTESTATION_ATTESTED = "server-attested"
ATTESTATION_UNATTESTED = "unattested"
ATTESTATION_STUB = "stub"
ATTESTATION_LEGACY = "unflagged-legacy"

# ---------------------------------------------------------------------------
# Stability repeats (costfinal-10): the sweep exposed small-model label noise
# (L0 flip rate 0.88 across 3 live repeats at requested temperature 0), so
# every live local-tier (query, route) pair is re-run N_REPEATS times and
# graded by majority (see costsmart.eval.stability; ties count as incorrect).
#
# Repeat rows live in their own table (never in ``attempts``): the frozen
# sweep matrix stays untouched, and repeat cache keys always hash the
# 5-tuple (query_id | route_id | prompt_version | model_version |
# repeat_idx), so they can never collide with a base single-run row even at
# repeat_idx=0. Base rows are single draws; repeats are fresh draws under
# the same sampling contract (temperature 0, fixed seed) - repeat_idx only
# keys storage and never changes sampling.
# ---------------------------------------------------------------------------

#: Repeat draws per live local-tier pair for stable majority labels.
N_REPEATS = 3

REPEATS_DDL = """
CREATE TABLE IF NOT EXISTS repeat_attempts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key           TEXT NOT NULL UNIQUE,
    repeat_idx          INTEGER NOT NULL,
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
    created_at          TEXT NOT NULL DEFAULT '',
    generator_mode      TEXT NOT NULL DEFAULT 'unflagged-legacy',
    retrieval_mode      TEXT NOT NULL DEFAULT 'unflagged-legacy',
    temperature         REAL NOT NULL DEFAULT 0.0,
    seed                INTEGER NOT NULL DEFAULT 0,
    model_revision      TEXT NOT NULL DEFAULT '',
    weights_sha256      TEXT NOT NULL DEFAULT '',
    attestation         TEXT NOT NULL DEFAULT 'unflagged-legacy'
);
CREATE INDEX IF NOT EXISTS idx_repeats_pair ON repeat_attempts(query_id, route_id);
CREATE INDEX IF NOT EXISTS idx_repeats_cache ON repeat_attempts(cache_key);
"""

#: Repeat-row column order: attempt columns with repeat_idx after cache_key.
REPEAT_COLUMNS = (
    "cache_key",
    "repeat_idx",
    *ATTEMPT_COLUMNS[1:],
)
