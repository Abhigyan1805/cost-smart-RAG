"""realdata-15: real Tier-A fetch/normalize + committed-corpus loader.

Pins the conversion from each HuggingFace dataset's raw schema into the repo
record shape (so the real sweep cannot silently grade the wrong field), and
the offline slice of the committed corpus JSON. No network / ``datasets``
dependency: these run in the stdlib-only worker env.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from costsmart.corpus import loaders


class NormalizeHfRowTest(unittest.TestCase):
    def test_nq_open_has_no_passages(self):
        rec = loaders.normalize_hf_row(
            "nq", {"question": "when was the moon last visited",
                   "answer": ["14 December 1972 UTC", "December 1972"]})
        self.assertEqual(rec["question"], "when was the moon last visited")
        self.assertEqual(rec["answers"], ["14 December 1972 UTC", "December 1972"])
        self.assertEqual(rec["passages"], [])

    def test_hotpotqa_marks_supporting_passages(self):
        row = {
            "question": "Were X and Y the same nationality?",
            "answer": "yes",
            "supporting_facts": {"title": ["X", "Y"], "sent_id": [0, 0]},
            "context": {
                "title": ["X", "Y", "Distractor"],
                "sentences": [["X is a person."], ["Y is a person."],
                              ["Z is unrelated."]],
            },
        }
        rec = loaders.normalize_hf_row("hotpotqa", row)
        self.assertEqual(rec["answers"], ["yes"])
        self.assertEqual(len(rec["passages"]), 3)
        gold = [p["title"] for p in rec["passages"] if p["is_gold"]]
        self.assertEqual(gold, ["X", "Y"])
        self.assertEqual(rec["passages"][0]["text"], "X is a person.")

    def test_musique_uses_is_supporting_and_aliases(self):
        row = {
            "question": "Who is the spouse of the performer?",
            "answer": "Miquette Giraudy",
            "answer_aliases": ["Miquette"],
            "paragraphs": [
                {"title": "A", "paragraph_text": "A text.", "is_supporting": True},
                {"title": "B", "paragraph_text": "B text.", "is_supporting": False},
            ],
        }
        rec = loaders.normalize_hf_row("musique", row)
        self.assertEqual(rec["answers"], ["Miquette Giraudy", "Miquette"])
        self.assertEqual([p["is_gold"] for p in rec["passages"]], [True, False])

    def test_unknown_dataset_and_blank_question_rejected(self):
        with self.assertRaises(ValueError):
            loaders.normalize_hf_row("squad", {"question": "q", "answer": "a"})
        with self.assertRaises(ValueError):
            loaders.normalize_hf_row("nq", {"question": "  ", "answer": ["a"]})


class CommittedCorpusLoaderTest(unittest.TestCase):
    def _write(self, path: Path):
        corpus = {
            "manifest": {"experiment": "realdata-15"},
            "queries": [
                {"query_id": "nq:0000", "source": "nq",
                 "question": "q-nq", "answers": ["a-nq"],
                 "gold_passage_ids": []},
                {"query_id": "hotpotqa:0000", "source": "hotpotqa",
                 "question": "q-hp", "answers": ["a-hp"],
                 "gold_passage_ids": ["hotpotqa:0000#p0"]},
                {"query_id": "musique:0000", "source": "musique",
                 "question": "q-mq", "answers": ["a-mq"],
                 "gold_passage_ids": ["musique:0000#p0"]},
            ],
            "passages": [
                {"passage_id": "hotpotqa:0000#p0", "doc_id": "T",
                 "text": "gold hp"},
                {"passage_id": "musique:0000#p0", "doc_id": "U",
                 "text": "gold mq"},
                {"passage_id": "musique:0000#d0", "doc_id": "V",
                 "text": "distractor mq"},
            ],
        }
        path.write_text(json.dumps(corpus))

    def test_slice_by_dataset_keeps_only_its_passages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            self._write(path)
            qs, ps = loaders.real_queries_for("musique", path=str(path))
            self.assertEqual([q["query_id"] for q in qs], ["musique:0000"])
            self.assertEqual([p["passage_id"] for p in ps],
                             ["musique:0000#p0", "musique:0000#d0"])
            qs, ps = loaders.real_queries_for("nq", path=str(path))
            self.assertEqual(len(qs), 1)
            self.assertEqual(ps, [])

    def test_full_subset_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            self._write(path)
            qs, ps = loaders.load_real_subset(path=str(path))
            self.assertEqual(len(qs), 3)
            self.assertEqual(len(ps), 3)
            # cache returns the same object on re-read
            self.assertIs(loaders.load_real_corpus(path=str(path)),
                          loaders.load_real_corpus(path=str(path)))

    def test_missing_corpus_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                loaders.load_real_corpus(path=str(Path(tmp) / "nope.json"))


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "results/realdata-15/splits_freeze.json"
CONFIG = ROOT / "config/experiments/sweep-200-realdata15.yaml"


class RealCorpusFreezeTest(unittest.TestCase):
    """The committed corpus + config still hash from the loader (no drift)."""

    def _corpus(self):
        corpus_path = ROOT / loaders.REAL_CORPUS_PATH
        self.assertTrue(corpus_path.is_file(),
                        "committed real corpus missing; run scripts/fetch_tier_a.py")
        return json.loads(corpus_path.read_text())

    def test_freeze_matches_committed_corpus(self):
        frozen = json.loads(FREEZE.read_text())
        corpus = self._corpus()
        qs, ps = loaders.load_real_subset(path=str(ROOT / loaders.REAL_CORPUS_PATH))
        records = [{"query_id": q["query_id"], "question": q["question"],
                    "reference": (q.get("answers") or [""])[0]} for q in qs]
        set_sha = hashlib.sha256(
            json.dumps(records, sort_keys=True).encode()).hexdigest()
        self.assertEqual(set_sha, frozen["set_sha256"])
        self.assertEqual(records, frozen["queries"])
        self.assertEqual(len(qs), frozen["n_queries"])
        self.assertEqual(len(ps), frozen["n_passages"])
        self.assertEqual(frozen["counts"], loaders.REAL_SIZES)

    def test_corpus_body_checksum_and_dataset_manifest(self):
        corpus = self._corpus()
        body = {"queries": corpus["queries"], "passages": corpus["passages"]}
        body_sha = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()).hexdigest()
        self.assertEqual(body_sha, corpus["manifest"]["corpus_sha256"])
        datasets = corpus["manifest"]["datasets"]
        self.assertEqual(set(datasets), {"nq", "hotpotqa", "musique"})
        for name, meta in datasets.items():
            self.assertEqual(meta["n_queries"], loaders.REAL_SIZES[name])
            self.assertTrue(meta["questions_sha256"])

    def test_config_hash_pinned(self):
        frozen = json.loads(FREEZE.read_text())
        self.assertEqual(
            hashlib.sha256(CONFIG.read_text().encode()).hexdigest(),
            frozen["config_hash"])


if __name__ == "__main__":
    unittest.main()
