"""Dense embedding model must be constructed at most once (tiersweep-16).

The sweep calls ``dense_search`` -> ``embed_texts([query])`` for every query
and every repeat. Loading ``SentenceTransformer`` per call reloaded the
weights (~90 MB) thousands of times, which dominated sweep latency and
inflated the measured per-query GPU cost. These tests pin the module-level
cache.
"""

import sys
import types
import unittest

from costsmart.retrieval import dense


class DenseCacheTest(unittest.TestCase):
    def setUp(self):
        dense._reset_st_cache()
        self.calls = 0
        self.encode_calls = 0
        fake = types.ModuleType("sentence_transformers")

        class FakeModel:
            def __init__(self, name):
                self.name = name
                type(self).constructed = getattr(type(self), "constructed", 0) + 1

            def encode(self, texts, normalize_embeddings=True):
                self.__class__.encode_calls = getattr(
                    self.__class__, "encode_calls", 0) + 1
                return [[0.0, 1.0] for _ in texts]

        FakeModel.constructed = 0
        FakeModel.encode_calls = 0
        self.FakeModel = FakeModel
        fake.SentenceTransformer = FakeModel
        self._saved = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = fake

    def tearDown(self):
        dense._reset_st_cache()
        if self._saved is not None:
            sys.modules["sentence_transformers"] = self._saved
        else:
            sys.modules.pop("sentence_transformers", None)

    def test_model_constructed_once_across_calls(self):
        for _ in range(5):
            vectors = dense.embed_texts(["a query"])
            self.assertEqual(len(vectors), 1)
        self.assertEqual(self.FakeModel.constructed, 1,
                         "SentenceTransformer must be constructed at most once")
        self.assertEqual(self.FakeModel.encode_calls, 5)
        self.assertEqual(dense.embedding_backend(), "sentence-transformers")

    def test_failed_load_is_not_retried(self):
        # A model that raises on construction is cached as failed, so the
        # constructor is not retried on the next call (hash fallback used).
        class Boom:
            constructed = 0

            def __init__(self, name):
                Boom.constructed += 1
                raise RuntimeError("no weights")

        sys.modules["sentence_transformers"].SentenceTransformer = Boom
        dense._reset_st_cache()
        vectors = dense.embed_texts(["a query"])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(dense.embedding_backend(), "hash-fallback")
        dense.embed_texts(["another query"])
        self.assertEqual(Boom.constructed, 1, "failed load must not be retried")


if __name__ == "__main__":
    unittest.main()
