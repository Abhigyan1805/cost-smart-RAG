#!/usr/bin/env python3
"""Sample a hand-labelling batch of Tier-B (query, answer) pairs.

Tier-B = free-form answers with no gold reference (``reference = ''`` in the
attempts table), the exact population the LLM judge grades. Tier-A rows
(NQ/TriviaQA/HotpotQA with gold answers) are excluded: deterministic
EM/lenient-EM/F1 already cover them.

Reads the telemetry SQLite DB (or a CSV/JSONL export of it), seeded-shuffles
the eligible rows, keeps one pair per query by default, and writes a labelling
CSV whose ``human_label`` column is left EMPTY for the hand labeller. The
captain supplies the completed labels file separately; this script never
fabricates labels.

Usage:
    PYTHONPATH=src python scripts/sample_label_batch.py --db telemetry.db
    PYTHONPATH=src python scripts/sample_label_batch.py --input attempts.jsonl \\
        --questions data/questions.jsonl --n 100 --seed 7 --out batch.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
import sys
from pathlib import Path

BATCH_COLUMNS = (
    "pair_id",
    "query_id",
    "question",
    "prediction",
    "reference",
    "judge_score",
    "judge_model",
    "human_label",
    "notes",
)


def load_attempts_from_db(db_path: str | Path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT query_id, prediction, reference, judge_score, judge_model,"
            " route_id FROM attempts ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def load_attempts_from_export(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix.lower() == ".csv":
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))
    raise ValueError(f"unsupported export format {path.suffix!r}; use .csv or .jsonl")


def load_questions(path: str | Path | None) -> dict[str, str]:
    """Optional query_id -> question join (attempts rows carry no question text)."""
    if path is None:
        return {}
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return {str(r["query_id"]): str(r.get("question", "")) for r in rows}
    with open(path, newline="") as fh:
        return {r["query_id"]: r.get("question", "") for r in csv.DictReader(fh)}


def sample_batch(
    attempts: list[dict],
    n: int = 100,
    seed: int = 0,
    one_per_query: bool = True,
) -> list[dict]:
    """Seeded sample of Tier-B pairs (empty reference = no gold answer)."""
    eligible = [a for a in attempts if not (a.get("reference") or "").strip()]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    if one_per_query:
        seen: set[str] = set()
        deduped = []
        for row in eligible:
            qid = str(row.get("query_id", ""))
            if qid in seen:
                continue
            seen.add(qid)
            deduped.append(row)
        eligible = deduped
    return eligible[:n]


def write_batch(rows: list[dict], questions: dict[str, str], out: Path) -> Path:
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(BATCH_COLUMNS))
        writer.writeheader()
        for i, row in enumerate(rows):
            qid = str(row.get("query_id", ""))
            writer.writerow(
                {
                    "pair_id": f"tierb-{i:04d}",
                    "query_id": qid,
                    "question": questions.get(qid, ""),
                    "prediction": row.get("prediction", ""),
                    "reference": row.get("reference", ""),
                    "judge_score": row.get("judge_score", ""),
                    "judge_model": row.get("judge_model", ""),
                    "human_label": "",
                    "notes": "",
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample a Tier-B hand-labelling batch.")
    parser.add_argument("--db", default="telemetry.db", help="telemetry SQLite path")
    parser.add_argument("--input", default=None, help="CSV/JSONL export (instead of --db)")
    parser.add_argument("--questions", default=None, help="JSONL/CSV query_id->question join")
    parser.add_argument("--n", type=int, default=100, help="pairs to sample (default 100)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    parser.add_argument("--out", default="tierb_label_batch.csv", help="output CSV path")
    parser.add_argument(
        "--allow-repeat-queries",
        action="store_true",
        help="allow several pairs from the same query",
    )
    args = parser.parse_args(argv)

    attempts = (
        load_attempts_from_export(args.input) if args.input else load_attempts_from_db(args.db)
    )
    eligible_total = sum(1 for a in attempts if not (a.get("reference") or "").strip())
    batch = sample_batch(attempts, n=args.n, seed=args.seed,
                         one_per_query=not args.allow_repeat_queries)
    if len(batch) < args.n:
        print(
            f"error: only {eligible_total} Tier-B pairs available "
            f"({len(batch)} sampled), need {args.n}: run a larger sweep with "
            "free-form (gold-less) queries first",
            file=sys.stderr,
        )
        return 2
    questions = load_questions(args.questions)
    if not questions:
        print("warning: no --questions join; 'question' column left blank", file=sys.stderr)
    out = write_batch(batch, questions, Path(args.out))
    print(json.dumps({"n": len(batch), "seed": args.seed, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
