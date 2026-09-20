"""SQLite store for the attempts table + Parquet/CSV/JSONL export.

Idempotency: ``insert_attempt`` uses ``INSERT OR IGNORE`` on the UNIQUE
``cache_key``, so a resumed sweep skips already-recorded attempts and never
double-counts. Returns True when the row was newly inserted.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from .schema import (
    ATTEMPT_COLUMNS,
    ATTEMPTS_DDL,
    MIGRATED_COLUMNS,
    REPEAT_COLUMNS,
    REPEATS_DDL,
)


# DEFAULT clauses for migrated columns (mirror schema.py DDL).
_MIGRATION_DEFAULTS = {
    "generator_mode": "'unflagged-legacy'",
    "retrieval_mode": "'unflagged-legacy'",
    "temperature": "0.0",
    "seed": "0",
    "model_revision": "''",
    "weights_sha256": "''",
    "attestation": "'unflagged-legacy'",
}


def _migrate_legacy_db(conn: sqlite3.Connection) -> None:
    """ALTER TABLE in columns missing from pre-flag DBs (frozen pilot.db).

    CREATE TABLE IF NOT EXISTS never adds columns to an existing table, so
    without this, opening a legacy DB and inserting a new-column row raises
    OperationalError. Missing columns get schema defaults; resident legacy
    rows read back as 'unflagged-legacy' (provenance documented per artifact).
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(attempts)")}
    for col in MIGRATED_COLUMNS:
        if col not in existing and col in ATTEMPT_COLUMNS:
            default = _MIGRATION_DEFAULTS[col]
            try:
                conn.execute(
                    f"ALTER TABLE attempts ADD COLUMN {col} "
                    f"{'REAL' if col == 'temperature' else ('INTEGER' if col == 'seed' else 'TEXT')}"
                    f" NOT NULL DEFAULT {default}"
                )
            except sqlite3.OperationalError:
                # Column appeared concurrently; safe to ignore.
                pass
    conn.commit()


class TelemetryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(ATTEMPTS_DDL)
        _migrate_legacy_db(self.conn)
        # Stability repeats (costfinal-10) live in their own table so the
        # frozen sweep matrix in ``attempts`` is never touched by repeat
        # writes. Any DB file can host repeats (the repeat runner keeps them
        # in results/costfinal-10/repeats.db, separate from sweep.db).
        self.conn.executescript(REPEATS_DDL)

    def close(self):
        self.conn.commit()
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def insert_attempt(self, attempt: dict) -> bool:
        """Insert one attempt row; False when cache_key already present.

        Columns absent from ``attempt`` (or ``None``) are omitted from the
        INSERT so the table DDL default applies - new provenance columns are
        therefore backward-compatible with callers written before them.
        """
        cols = [k for k in ATTEMPT_COLUMNS if attempt.get(k) is not None]
        placeholders = ", ".join("?" for _ in cols)
        cur = self.conn.execute(
            f"INSERT OR IGNORE INTO attempts ({', '.join(cols)}) VALUES ({placeholders})",
            [attempt[k] for k in cols],
        )
        self.conn.commit()
        return cur.rowcount == 1

    def has(self, cache_key: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM attempts WHERE cache_key = ?", (cache_key,)
        )
        return cur.fetchone() is not None

    # -- stability repeats (costfinal-10) ---------------------------------

    def insert_repeat(self, repeat: dict) -> bool:
        """Insert one repeat-attempt row; False when cache_key exists.

        Repeat rows carry ``repeat_idx`` (0..N_REPEATS-1) and a 5-tuple
        cache key, so re-running an interrupted repeat sweep skips recorded
        draws and only executes the remainder (same resumability contract
        as :meth:`insert_attempt`, in a table the sweep matrix never reads).
        """
        cols = [k for k in REPEAT_COLUMNS if repeat.get(k) is not None]
        placeholders = ", ".join("?" for _ in cols)
        cur = self.conn.execute(
            f"INSERT OR IGNORE INTO repeat_attempts ({', '.join(cols)}) "
            f"VALUES ({placeholders})",
            [repeat[k] for k in cols],
        )
        self.conn.commit()
        return cur.rowcount == 1

    def has_repeat(self, cache_key: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM repeat_attempts WHERE cache_key = ?", (cache_key,)
        )
        return cur.fetchone() is not None

    def count_repeats(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM repeat_attempts").fetchone()[0]

    def fetch_repeats(self, route_id: str | None = None) -> list[dict]:
        if route_id is None:
            cur = self.conn.execute("SELECT * FROM repeat_attempts ORDER BY id")
        else:
            cur = self.conn.execute(
                "SELECT * FROM repeat_attempts WHERE route_id = ? ORDER BY id",
                (route_id,),
            )
        return [dict(r) for r in cur.fetchall()]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]

    def fetch_all(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM attempts ORDER BY id")
        return [dict(r) for r in cur.fetchall()]

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self.conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    # -- exports ---------------------------------------------------------

    def export_csv(self, path: str | Path) -> Path:
        path = Path(path)
        rows = self.fetch_all()
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["id", *ATTEMPT_COLUMNS])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def export_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        with open(path, "w") as fh:
            for row in self.fetch_all():
                fh.write(json.dumps(row) + "\n")
        return path

    def export_parquet(self, path: str | Path) -> Path:
        """Export attempts to Parquet (needs pyarrow or pandas; else error).

        The sweep environment is stdlib-only, so this tries optional deps
        and raises a clear error telling the caller what to install; the
        SQLite table remains the queryable source of truth.
        """
        path = Path(path)
        rows = self.fetch_all()
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist(rows) if rows else pa.Table.from_arrays(
                [[] for _ in ATTEMPT_COLUMNS], names=list(ATTEMPT_COLUMNS)
            )
            pq.write_table(table, path)
            return path
        except ImportError:
            pass
        try:
            import pandas as pd

            pd.DataFrame(rows).to_parquet(path, index=False)
            return path
        except ImportError:
            pass
        raise RuntimeError(
            f"cannot write Parquet to {path}: install pyarrow (preferred) or "
            "pandas; SQLite/CSV/JSONL exports remain available"
        )
