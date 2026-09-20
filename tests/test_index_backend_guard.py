"""Index/query embedding-backend guard (tiersweep-16 pre-flight audit C3).

A query embedded with the 256-dim hash fallback against a 384-dim MiniLM
index (or vice versa) used to truncate silently via ``zip``. The sweep must
refuse a mismatched index up front.
"""

import unittest

from costsmart.eval.oracle_sweep import _verify_index_backend
from costsmart.retrieval import dense


class IndexBackendGuardTest(unittest.TestCase):
    def setUp(self):
        dense._reset_st_cache()

    def test_matching_backend_passes(self):
        probe = dense.embed_texts(["probe"])
        backend = dense.embedding_backend()
        _verify_index_backend(
            {"embedding_backend": backend, "embedding_dim": len(probe[0])})

    def test_backend_mismatch_refuses(self):
        probe = dense.embed_texts(["probe"])
        _verify_index_backend({"embedding_backend": dense.embedding_backend(),
                               "embedding_dim": len(probe[0])})
        with self.assertRaises(RuntimeError):
            _verify_index_backend(
                {"embedding_backend": "some-other-backend",
                 "embedding_dim": len(probe[0])})

    def test_dimension_mismatch_refuses(self):
        probe = dense.embed_texts(["probe"])
        _verify_index_backend({"embedding_backend": dense.embedding_backend(),
                               "embedding_dim": len(probe[0])})
        with self.assertRaises(RuntimeError):
            _verify_index_backend(
                {"embedding_backend": dense.embedding_backend(),
                 "embedding_dim": len(probe[0]) + 7})

    def test_index_without_metadata_is_skipped(self):
        _verify_index_backend({})


if __name__ == "__main__":
    unittest.main()
