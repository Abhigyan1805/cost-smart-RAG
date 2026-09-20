#!/usr/bin/env python3
"""Fetch + normalize the real Tier-A corpus for the realdata-15 recount.

Runs on a network-capable host (the Kaggle GPU kernel; the worker env is
stdlib-only). Fetches each dataset through the repo's existing HuggingFace
loader path (``costsmart.corpus.loaders._hf_queries`` +
:func:`normalize_hf_row`) and writes ONE committed corpus JSON with a
per-dataset manifest: HuggingFace repo id, config, split, resolved revision
sha, licence (read from the dataset card - never inferred), row counts, and
content checksums. Nothing is fabricated: if the card reports no licence the
manifest records ``null``, and if a dataset cannot be fetched the run aborts
loudly rather than substituting silently.

Usage (Kaggle / any host with ``datasets`` + network):
    PYTHONPATH=src python scripts/fetch_tier_a.py \
        --out results/realdata-15/corpus.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from costsmart.corpus.loaders import (  # noqa: E402
    HF_DATASET_SPECS,
    REAL_SIZES,
    load_dataset_queries,
)

DEFAULT_OUT = "results/realdata-15/corpus.json"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _license_for(repo: str) -> dict:
    """Read the licence + resolved revision from the HF dataset card."""
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError:
        return {"license": None, "license_source": "huggingface_hub unavailable",
                "revision": None}
    api = HfApi()
    info = api.dataset_info(repo)
    license_id = None
    card = getattr(info, "cardData", None) or {}
    if isinstance(card, dict):
        license_id = card.get("license") or card.get("license_name")
    if not license_id:
        for tag in (getattr(info, "tags", None) or []):
            if str(tag).startswith("license:"):
                license_id = str(tag).split(":", 1)[1]
                break
    return {
        "license": license_id,
        "license_source": "huggingface dataset card" if license_id else "not stated on card",
        "revision": getattr(info, "sha", None),
    }


def fetch(out_path: str) -> dict:
    queries: list[dict] = []
    passages: list[dict] = []
    datasets: dict[str, dict] = {}
    for dataset in ("nq", "hotpotqa", "musique"):
        n = REAL_SIZES[dataset]
        qs, ps = load_dataset_queries(dataset, n=n, source="hf")
        if len(qs) != n:
            raise SystemExit(
                f"{dataset}: fetched {len(qs)} queries, expected {n}; aborting "
                "rather than committing a short corpus")
        spec = HF_DATASET_SPECS[dataset]
        card = _license_for(spec["repo"])
        q_text = json.dumps(
            [{"query_id": q["query_id"], "question": q["question"],
              "answers": q["answers"]} for q in qs],
            sort_keys=True)
        datasets[dataset] = {
            "hf_repo": spec["repo"],
            "config": spec["config"],
            "split": spec["split"],
            "revision": card["revision"],
            "license": card["license"],
            "license_source": card["license_source"],
            "n_queries": len(qs),
            "n_passages": len(ps),
            "n_gold_passages": sum(len(q["gold_passage_ids"]) for q in qs),
            "questions_sha256": _sha256_text(q_text),
            "passages_sha256": _sha256_text(
                "\n".join(p["text"] for p in ps)),
        }
        queries.extend(qs)
        passages.extend(ps)
        print(f"{dataset}: {len(qs)} queries, {len(ps)} passages, "
              f"license={card['license']!r}, revision={card['revision']}",
              flush=True)

    body = {"queries": queries, "passages": passages}
    corpus_sha = _sha256_text(json.dumps(body, sort_keys=True))
    manifest = {
        "experiment": "realdata-15",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "query_id_scheme": "{dataset}:{index:04d}",
        "mixed_from": list(REAL_SIZES),
        "n_queries": len(queries),
        "n_passages": len(passages),
        "multi_hop_share": round(
            (REAL_SIZES["hotpotqa"] + REAL_SIZES["musique"]) / sum(REAL_SIZES.values()), 4),
        "datasets": datasets,
        "corpus_sha256": corpus_sha,
        "note": "NQ comes from nq_open, which ships question+answer only; NQ "
                "queries have no gold passage and no passage pool, so NQ L1 "
                "retrieval draws from the shared multi-hop pool. The headroom "
                "gate reads L0 (closed-book) vs the C4 stub, not L1.",
    }
    payload = {"manifest": manifest, **body}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"wrote {out} ({out.stat().st_size} bytes)", flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch real Tier-A corpus")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    fetch(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
