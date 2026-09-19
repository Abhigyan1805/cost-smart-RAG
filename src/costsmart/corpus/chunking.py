"""Pinned chunking strategy for the costsmart-rag pilot.

Exactly one strategy is pinned for Week-1: fixed word-window chunking with
character-safe word splitting and a fixed overlap. All index builds and eval
runs must use :func:`chunk_text` with the defaults below so passage ids and
offsets stay comparable across slices (retrieval, routing, telemetry).
"""

from __future__ import annotations

# Pinned Week-1 strategy parameters. Do not change without a firstmate decision.
CHUNK_STRATEGY = "fixed-word-window"
CHUNK_WORDS = 200
CHUNK_OVERLAP_WORDS = 50


def chunk_text(text: str, doc_id: str) -> list[dict]:
    """Split ``text`` into overlapping fixed word windows.

    Returns a list of ``{chunk_id, doc_id, text, start_word, end_word}``
    dicts. Short texts yield a single chunk.
    """
    words = text.split()
    if not words:
        return []
    step = max(CHUNK_WORDS - CHUNK_OVERLAP_WORDS, 1)
    chunks: list[dict] = []
    idx = 0
    start = 0
    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        chunk_words = words[start:end]
        chunks.append(
            {
                "chunk_id": f"{doc_id}#c{idx}",
                "doc_id": doc_id,
                "text": " ".join(chunk_words),
                "start_word": start,
                "end_word": end,
            }
        )
        if end == len(words):
            break
        start += step
        idx += 1
    return chunks


def chunk_passages(passages: list[dict]) -> list[dict]:
    """Apply the pinned strategy to every passage.

    Each passage needs ``passage_id`` (used as ``doc_id``) and ``text``.
    Chunk records carry both ``chunk_id`` and the parent ``passage_id``.
    """
    out: list[dict] = []
    for passage in passages:
        pid = passage["passage_id"]
        for chunk in chunk_text(passage["text"], doc_id=pid):
            chunk["passage_id"] = pid
            out.append(chunk)
    return out
