#!/usr/bin/env python3
"""20-query smoke run: builds (or loads) the pilot index, runs hybrid
retrieval + rerank + router features on 20 sampled queries, and prints hit@k.

Usage:
    PYTHONPATH=src python scripts/smoke_retrieval.py [--index data/index/pilot_index.json] [--k 1 5 10]

Exit code 0 when hit@1 > 0 (sanity: retrieval returns at least one gold);
prints hit@1 / hit@5 / hit@10 regardless.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from costsmart.retrieval.features import FEATURE_NAMES, extract_routing_features
from costsmart.retrieval.hybrid import hybrid_search
from costsmart.retrieval.rerank import rerank

SMOKE_N = 20
SMOKE_SEED = 7


def _load_or_build(index_path: Path) -> dict:
    if index_path.exists():
        return json.loads(index_path.read_text())
    from costsmart.corpus.build_index import build_index

    index = build_index()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index))
    return index


def hit_at_k(hits: list[dict], gold_ids: set[str], k: int) -> bool:
    return any(h["chunk_id"] in gold_ids or h["doc_id"] in gold_ids for h in hits[:k])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="20-query retrieval smoke run.")
    parser.add_argument("--index", default="data/index/pilot_index.json")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args(argv)

    index = _load_or_build(Path(args.index))
    chunks: list[dict] = index["chunks"]
    lookup = {c["chunk_id"]: c["text"] for c in chunks}
    # Map each query to its gold chunk ids: chunks whose passage is gold.
    queries = index["queries"]
    rng = random.Random(SMOKE_SEED)
    sample = rng.sample(queries, min(SMOKE_N, len(queries)))

    hits_count = {k: 0 for k in args.k}
    for q in sample:
        gold_passages = set(q["gold_passage_ids"])
        gold_chunks = {c["chunk_id"] for c in chunks if c["passage_id"] in gold_passages}
        gold = gold_chunks | gold_passages | {q["query_id"]}
        ranked = hybrid_search(q["question"], chunks, index["bm25"], index["dense_vectors"])
        reranked = rerank(q["question"], ranked, lookup)
        feats = extract_routing_features(q["question"], ranked, reranked, lookup)
        assert set(feats) == set(FEATURE_NAMES), "router feature contract changed"
        for k in args.k:
            if hit_at_k(ranked, gold, k):
                hits_count[k] += 1

    n = len(sample)
    for k in args.k:
        print(f"hit@{k}: {hits_count[k]}/{n} = {hits_count[k] / n:.3f}")
    ok = hits_count.get(1, n) > 0
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
