import unittest

from src.aggregation import average_batch_metrics, pool_metrics


class AggregationTests(unittest.TestCase):
    def test_pooled_norm_statistics_use_token_counts(self):
        batches = [
            {"layer": {"count": 2, "sum": 2.0, "sumsq": 2.0, "mean": 1.0, "std": 0.0}},
            {"layer": {"count": 1, "sum": 10.0, "sumsq": 100.0, "mean": 10.0, "std": 0.0}},
        ]

        corrected = pool_metrics(batches, "norm")
        legacy = average_batch_metrics(batches, "norm")

        self.assertAlmostEqual(corrected["layer"]["mean"], 4.0)
        self.assertAlmostEqual(legacy["layer"]["mean"], 5.5)


if __name__ == "__main__":
    unittest.main()
