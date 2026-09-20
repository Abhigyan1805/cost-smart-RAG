"""Dense retrieval with a single pinned embedding model.

Pinned model (used everywhere in the pilot):
    ``sentence-transformers/all-MiniLM-L6-v2``

When ``sentence-transformers`` (+ torch) is installed, real model vectors
are used. Otherwise :func:`embed_texts` falls back to a deterministic
hash-based unit vector (stdlib only) so the index build and smoke retrieval
run offline. The fallback lives behind the same API, so swapping to the
real model needs no call-site changes.
"""

from __future__ import annotations

import hashlib
import math

# The one embedding model used everywhere. Pinned; change needs a decision.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Dimension of the offline hash fallback vectors.
FALLBACK_DIM = 256


def _hash_embed(text: str, dim: int = FALLBACK_DIM) -> list[float]:
    tokens = text.lower().split()
    vec = [0.0] * dim
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


#: Cached sentence-transformers model. The constructor loads ~90 MB of
#: weights from disk, so loading it per call (once per query and per repeat)
#: dominated sweep latency and inflated the measured per-query GPU cost.
#: Load once, reuse; cache the load *failure* too so a broken install does
#: not retry the constructor on every call.
_ST_MODEL = None
_ST_LOAD_FAILED = False


def _get_st_model():
    """Return the process-wide SentenceTransformer, loading it at most once.

    Returns ``None`` (without retrying) when sentence-transformers is absent
    or the model fails to load, so callers fall back to the hash embedding.
    """
    global _ST_MODEL, _ST_LOAD_FAILED
    if _ST_MODEL is not None:
        return _ST_MODEL
    if _ST_LOAD_FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        _ST_LOAD_FAILED = True
        return None
    try:
        _ST_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    except Exception:
        _ST_LOAD_FAILED = True
        return None
    return _ST_MODEL


def _reset_st_cache() -> None:
    """Drop the cached model/load-failure (test isolation + reload support)."""
    global _ST_MODEL, _ST_LOAD_FAILED
    _ST_MODEL = None
    _ST_LOAD_FAILED = False


def _st_embed(texts: list[str]) -> list[list[float]] | None:
    model = _get_st_model()
    if model is None:
        return None
    try:
        return [list(map(float, v))
                for v in model.encode(texts, normalize_embeddings=True)]
    except Exception:
        return None


#: Which backend the last ``embed_texts`` call actually used. Index metadata
#: must record this: the pinned name ``EMBEDDING_MODEL`` is the intent, not
#: the truth (the hash fallback fires whenever sentence-transformers is
#: absent or its model fails to load).
_LAST_BACKEND: dict[str, str] = {"name": "unset"}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` with the pinned model (or offline fallback)."""
    vectors = _st_embed(texts)
    if vectors is not None:
        _LAST_BACKEND["name"] = "sentence-transformers"
        return vectors
    _LAST_BACKEND["name"] = "hash-fallback"
    return [_hash_embed(t) for t in texts]


def embedding_backend() -> str:
    """Name of the backend used by the last :func:`embed_texts` call."""
    return _LAST_BACKEND["name"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity for (already normalised) vectors."""
    return sum(x * y for x, y in zip(a, b))


def dense_search(
    query: str, chunks: list[dict], vectors: list[list[float]], top_k: int = 10
) -> list[dict]:
    """Rank ``chunks`` by cosine similarity to the embedded query."""
    qvec = embed_texts([query])[0]
    scored = [
        {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "score": cosine_similarity(qvec, v)}
        for c, v in zip(chunks, vectors)
    ]
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]
