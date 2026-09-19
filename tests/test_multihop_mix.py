"""costmultihop-12: multihop-reweighted Tier-A mix, composition + freeze.

Guards the two things the gate recount depends on: the mix really is
200 queries at the documented 25/75 single-hop/multi-hop split, and the
committed ``splits_freeze.json`` still hashes from the loader (no silent
drift between the frozen file and the code that will run the sweep).
"""

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from costsmart.corpus import loaders
from costsmart.corpus.build_index import build_index

ROOT = Path(__file__).resolve().parents[1]
MULTIHOP_FREEZE = ROOT / "results/costmultihop-12/splits_freeze.json"
MULTIHOP_CONFIG = ROOT / "config/experiments/sweep-200-multihop.yaml"
PILOT_FREEZE = ROOT / "results/costsweep-08/splits_freeze.json"


def _record(query: dict) -> dict:
    return {
        "query_id": query["query_id"],
        "question": query["question"],
        "reference": (query.get("answers") or [""])[0],
    }


def _set_hash(queries: list[dict]) -> str:
    raw = json.dumps([_record(q) for q in queries], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


class CompositionTest(unittest.TestCase):
    def test_counts_and_multihop_majority(self):
        queries, passages = loaders.load_multihop_subset(source="synthetic")
        self.assertEqual(len(queries), 200)
        self.assertEqual(len(passages), 590)
        counts = Counter(q["source"] for q in queries)
        self.assertEqual(dict(counts), loaders.MULTIHOP_SIZES)
        single_hop = counts["nq"] + counts["triviaqa"]
        self.assertEqual(single_hop, 50)  # legacy mix was 150/200
        self.assertEqual(200 - single_hop, 150)  # 75% multi-hop

    def test_new_dataset_defaults_use_multihop_sizes(self):
        for dataset in ("2wiki", "musique"):
            queries, _ = loaders.load_dataset_queries(dataset, source="synthetic")
            self.assertEqual(len(queries), loaders.MULTIHOP_SIZES[dataset])

    def test_unknown_dataset_rejected(self):
        with self.assertRaises(ValueError):
            loaders.load_dataset_queries("squad", source="synthetic")

    def test_multihop_subset_counts_override(self):
        queries, _ = loaders.load_multihop_subset(
            source="synthetic", counts={"nq": 3, "musique": 2})
        self.assertEqual(len(queries), 5)
        with self.assertRaises(ValueError):
            loaders.load_multihop_subset(
                source="synthetic", counts={"nq": 1, "squad": 1})

    def test_musique_two_distractors_one_each_elsewhere(self):
        # gold + distractor passages per query: single-hop 1+1, 2wiki/hotpot
        # 2+1, musique (the distractor-filtering slice) 2+2.
        passages_per_query = {"nq": 2, "2wiki": 3, "hotpotqa": 3, "musique": 4}
        for dataset, expected in passages_per_query.items():
            _, passages = loaders.load_dataset_queries(dataset, n=2,
                                                       source="synthetic")
            by_query = Counter(p["passage_id"].split("#", 1)[0]
                               for p in passages)
            self.assertEqual(set(by_query.values()), {expected},
                             f"{dataset}: {by_query}")


class FrozenSetTest(unittest.TestCase):
    def test_set_hash_recomputes_from_loader(self):
        frozen = json.loads(MULTIHOP_FREEZE.read_text())
        queries, _ = loaders.load_multihop_subset(source="synthetic")
        self.assertEqual([_record(q) for q in queries], frozen["queries"])
        self.assertEqual(_set_hash(queries), frozen["set_sha256"])

    def test_config_hash_and_counts_pinned(self):
        frozen = json.loads(MULTIHOP_FREEZE.read_text())
        text = MULTIHOP_CONFIG.read_text()
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                         frozen["config_hash"])
        self.assertEqual(frozen["counts"], loaders.MULTIHOP_SIZES)
        self.assertIn("tier_a_counts: {nq: 30, triviaqa: 20, hotpotqa: 60, "
                      "2wiki: 50, musique: 40}", text)

    def test_legacy_pilot_freeze_unchanged(self):
        frozen = json.loads(PILOT_FREEZE.read_text())
        queries, _ = loaders.load_pilot_subset(source="synthetic")
        self.assertEqual(_set_hash(queries), frozen["set_sha256"],
                         "legacy pilot mix drifted; costsweep-08 rows pin it")


class IndexMixTest(unittest.TestCase):
    def test_multihop_index_metadata(self):
        index = build_index(source="synthetic", mix="multihop")
        self.assertEqual(index["mix"], "multihop")
        self.assertEqual(index["num_queries"], 200)
        self.assertEqual(index["num_passages"], 590)
        self.assertEqual(len(index["chunks"]), index["num_chunks"])
        self.assertEqual(len(index["dense_vectors"]), index["num_chunks"])

    def test_pilot_mix_still_builds(self):
        index = build_index(source="synthetic", mix="pilot")
        self.assertEqual(index["mix"], "pilot")
        self.assertEqual(index["num_queries"], 200)

    def test_unknown_mix_rejected(self):
        with self.assertRaises(ValueError):
            build_index(source="synthetic", mix="nope")


if __name__ == "__main__":
    unittest.main()
