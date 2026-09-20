# Real Tier-A corpus manifest (realdata-15)

Committed normalized corpus: [`corpus.json`](corpus.json)
(1,362,362 bytes, file sha256 `37044a37622f442c7ff0704027a22e7cbb86165df1f6e7c8e7378c4af11be7b8`,
body sha256 `0cdd131bc982f7a7e3e058858203a5d2ae91c427946bb8e6e55047672a190a9f`).
Fetched through the repo's HuggingFace loader path by
`scripts/fetch_tier_a.py` (`_hf_queries` + `normalize_hf_row`); the
machine-readable manifest is the `manifest` key inside the JSON.

Composition: **50 NQ + 80 HotpotQA + 70 MuSiQue = 200 queries, 75% multi-hop**,
mirroring the synthetic `costmultihop-12` 25/75 single-hop/multi-hop split so
the real-vs-synthetic recount is like-for-like. 2,192 passages (all from the
two multi-hop datasets; see the NQ note).

| dataset | HF repo | config | split | revision | licence | queries | passages | gold passages |
|---|---|---|---|---|---|---|---|---|
| NQ | `google-research-datasets/nq_open` | - | validation | `5dd9790a` | CC-BY-SA-3.0 | 50 | 0 | 0 |
| HotpotQA | `hotpotqa/hotpot_qa` | distractor | validation | `1908d6af` | CC-BY-SA-4.0 | 80 | 792 | 160 |
| MuSiQue | `dgslibisey/MuSiQue` | default | validation | `c8f4f8c9` | not stated on card | 70 | 1400 | 140 |

## Notes / substitutions

- **No dataset was substituted or fabricated.** All three datasets were
  fetched from HuggingFace on the Kaggle run host.
- **NQ has no passages.** `nq_open` ships question + answer only, so NQ
  queries have no gold passage and contribute no passage pool. NQ's *L1*
  retrieval therefore draws from the shared HotpotQA/MuSiQue pool; the
  headroom gate reads L0 (closed-book) vs the C4 stub, not L1. This is a
  property of the source dataset, not a corpus defect.
- **MuSiQue licence.** The `dgslibisey/MuSiQue` mirror card states no
  licence, so the manifest records `license: null` (never inferred). The
  canonical MuSiQue release is CC BY 4.0
  (<https://github.com/StonyBrookNLP/musique>). The mirror is the
  HuggingFace-loadable copy of the official `musique_ans_v1.0_dev.jsonl`.
- **Selection is deterministic.** The first *n* answerable rows of each
  `validation` split, in dataset order; per-dataset question + passage
  checksums are pinned in `splits_freeze.json` (this directory).

## Rebuild

```sh
# refetch (network host with `datasets`) - yields byte-identical manifest body
PYTHONPATH=src python scripts/fetch_tier_a.py \
    --out results/realdata-15/corpus.json
# offline index build used by the sweep (dense backend recorded in the index)
PYTHONPATH=src python -m costsmart.corpus.build_index \
    --mix real --source real --out data/index/real_index.json
```
