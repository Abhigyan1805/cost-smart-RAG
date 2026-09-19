"""Tests for the dual-mode cost model (unittest, stdlib-only env)."""

import os
import unittest

from costsmart.telemetry.cost import (
    amortized_usd,
    cloud_spend_usd,
    cost_record,
    read_concurrency,
)


class CloudSpendTest(unittest.TestCase):
    def test_local_routes_cost_zero(self):
        self.assertEqual(cloud_spend_usd(1000, 500, "ollama-qwen2.5-3b"), 0.0)
        self.assertEqual(cloud_spend_usd(1000, 500, "colab-t4-local"), 0.0)

    def test_cloud_pricing_math(self):
        # gpt-4o-mini: $0.15/$0.60 per 1M tokens.
        got = cloud_spend_usd(1_000_000, 1_000_000, "gpt-4o-mini")
        self.assertAlmostEqual(got, 0.75)

    def test_unknown_model_prices_zero(self):
        self.assertEqual(cloud_spend_usd(100, 100, "no-such-model"), 0.0)

    def test_negative_tokens_clamped(self):
        self.assertEqual(cloud_spend_usd(-5, -5, "gpt-4o"), 0.0)


class AmortizedTest(unittest.TestCase):
    def test_gpu_seconds_path(self):
        # 1 T4-hour at $0.35/h, concurrency 1 -> $0.35.
        self.assertAlmostEqual(amortized_usd(3600.0, "T4", concurrency=1), 0.35)

    def test_concurrency_divides_share(self):
        solo = amortized_usd(3600.0, "T4", concurrency=1)
        shared = amortized_usd(3600.0, "T4", concurrency=4)
        self.assertAlmostEqual(shared, solo / 4)

    def test_zero_gpu_zero_cost(self):
        self.assertEqual(amortized_usd(0.0, "T4", concurrency=1), 0.0)

    def test_reads_ollama_num_parallel(self):
        os.environ["OLLAMA_NUM_PARALLEL"] = "4"
        try:
            self.assertEqual(read_concurrency(), 4)
        finally:
            del os.environ["OLLAMA_NUM_PARALLEL"]

    def test_bad_concurrency_falls_back(self):
        os.environ["OLLAMA_NUM_PARALLEL"] = "nonsense"
        try:
            self.assertEqual(read_concurrency(default=1), 1)
        finally:
            del os.environ["OLLAMA_NUM_PARALLEL"]


class CostRecordTest(unittest.TestCase):
    def test_both_modes_populated(self):
        rec = cost_record(1000, 500, "gpt-4o-mini", gpu_seconds=12.0, concurrency=2)
        self.assertIn("cloud_spend_usd", rec)
        self.assertIn("amortized_usd", rec)
        self.assertGreater(rec["cloud_spend_usd"], 0.0)
        self.assertGreater(rec["amortized_usd"], 0.0)
        self.assertEqual(rec["concurrency"], 2)

    def test_local_route_amortized_only(self):
        rec = cost_record(100, 50, "ollama-llama3.1-8b", gpu_seconds=5.0, concurrency=1)
        self.assertEqual(rec["cloud_spend_usd"], 0.0)
        self.assertGreater(rec["amortized_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
