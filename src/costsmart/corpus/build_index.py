"""Single index-build command for the Week-1 pilot.

Usage:
    python -m costsmart.corpus.build_index [--out data/index/pilot_index.json] [--source synthetic|hf]

Builds the pilot corpus (200 queries + passages), applies the pinned
chunking strategy, and writes one JSON index containing chunks plus the
precomputed BM25 document statistics and dense hash vectors, so retrieval
works offline with zero extra build steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from costsmart.corpus.chunking import CHUNK_OVERLAP_WORDS, CHUNK_STRATEGY, CHUNK_WORDS, chunk_passages
from costsmart.corpus.loaders import PILOT_TOTAL, load_pilot_subset
from costsmart.retrieval.bm25 import build_bm25_stats
from costsmart.retrieval.dense import EMBEDDING_MODEL, embed_texts

DEFAULT_INDEX_PATH = "data/index/pilot_index.json"


def build_index(source: str = "synthetic") -> dict:
    """Build the in-memory pilot index and return it as a JSON-able dict."""
    queries, passages = load_pilot_subset(source=source)
    chunks = chunk_passages(passages)
    texts = [c["text"] for c in chunks]
    bm25_stats = build_bm25_stats(texts)
    vectors = embed_texts(texts)
    return {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_strategy": CHUNK_STRATEGY,
        "chunk_words": CHUNK_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "corpus_source": source,
        "num_queries": len(queries),
        "num_passages": len(passages),
        "num_chunks": len(chunks),
        "queries": queries,
        "chunks": chunks,
        "bm25": bm25_stats,
        "dense_vectors": vectors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Week-1 pilot index.")
    parser.add_argument("--out", default=DEFAULT_INDEX_PATH, help="Output JSON index path.")
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "hf"],
                        help="Corpus source: offline synthetic pilot or HuggingFace datasets.")
    args = parser.parse_args(argv)
    index = build_index(source=args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index))
    print(
        f"built pilot index: {index['num_queries']}/{PILOT_TOTAL} queries, "
        f"{index['num_passages']} passages, {index['num_chunks']} chunks "
        f"-> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
