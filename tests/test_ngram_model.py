import unittest
import math
import json
import tempfile
from pathlib import Path
from raga_pipeline.language_model import NgramModel


class TestNgramModel(unittest.TestCase):

    def test_add_sequence_counts_ngrams(self):
        """Adding a token sequence populates unigram through n-gram counts."""
        model = NgramModel(order=3)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga", "ma", "Pa"])
        counts = model.get_counts("Yaman")
        self.assertEqual(counts[1][("Sa",)], 1)
        self.assertEqual(counts[1][("Re",)], 1)
        self.assertEqual(counts[1][("<BOS>",)], 1)
        self.assertEqual(counts[2][("<BOS>", "Sa")], 1)
        self.assertEqual(counts[2][("Sa", "Re")], 1)
        self.assertEqual(counts[2][("Re", "Ga")], 1)
        self.assertEqual(counts[3][("<BOS>", "Sa", "Re")], 1)
        self.assertEqual(counts[3][("Sa", "Re", "Ga")], 1)

    def test_add_multiple_sequences_accumulates(self):
        """Multiple sequences for the same raga accumulate counts."""
        model = NgramModel(order=2)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga"])
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "ma"])
        counts = model.get_counts("Yaman")
        self.assertEqual(counts[2][("Sa", "Re")], 2)
        self.assertEqual(counts[2][("Re", "Ga")], 1)
        self.assertEqual(counts[2][("Re", "ma")], 1)

    def test_log_prob_smoothed_nonzero(self):
        """Smoothed probability is nonzero even for unseen n-grams."""
        model = NgramModel(order=2, smoothing="add-k", smoothing_k=0.01)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga"])
        model.finalize()
        lp_seen = model.log_prob("Yaman", "Re", ("Sa",))
        lp_unseen = model.log_prob("Yaman", "Pa", ("Sa",))
        self.assertGreater(lp_seen, lp_unseen)
        self.assertTrue(math.isfinite(lp_unseen))
        self.assertLess(lp_unseen, 0)

    def test_score_sequence(self):
        """score_sequence returns average log-likelihood per token."""
        model = NgramModel(order=2, smoothing="add-k", smoothing_k=0.01)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga", "ma", "Pa"] * 20)
        model.add_sequence("Bhairav", ["<BOS>", "Sa", "re", "Ga", "ma", "Pa"] * 20)
        model.finalize()
        yaman_seq = ["<BOS>", "Sa", "Re", "Ga", "ma", "Pa"]
        bhairav_seq = ["<BOS>", "Sa", "re", "Ga", "ma", "Pa"]
        self.assertGreater(model.score_sequence("Yaman", yaman_seq), model.score_sequence("Bhairav", yaman_seq))
        self.assertGreater(model.score_sequence("Bhairav", bhairav_seq), model.score_sequence("Yaman", bhairav_seq))

    def test_ragas_list(self):
        """Model tracks which ragas have been added."""
        model = NgramModel(order=2)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re"])
        model.add_sequence("Bhairav", ["<BOS>", "Sa", "re"])
        self.assertEqual(sorted(model.ragas()), ["Bhairav", "Yaman"])

    def test_empty_sequence_ignored(self):
        """Adding an empty sequence does not crash or create a raga entry."""
        model = NgramModel(order=2)
        model.add_sequence("Yaman", [])
        self.assertEqual(model.ragas(), [])


class TestNgramModelSerialization(unittest.TestCase):

    def test_to_dict_and_back(self):
        """Model survives a to_dict -> from_dict round-trip."""
        model = NgramModel(order=3, smoothing="add-k", smoothing_k=0.05)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga", "ma", "Pa"])
        model.add_sequence("Bhairav", ["<BOS>", "Sa", "re", "Ga", "ma", "dha"])
        model.finalize()
        data = model.to_dict()
        restored = NgramModel.from_dict(data)
        self.assertEqual(sorted(restored.ragas()), sorted(model.ragas()))
        seq = ["<BOS>", "Sa", "Re", "Ga"]
        for raga in model.ragas():
            self.assertAlmostEqual(
                restored.score_sequence(raga, seq),
                model.score_sequence(raga, seq), places=10)

    def test_json_round_trip(self):
        """Model survives JSON serialization to disk and back."""
        model = NgramModel(order=2)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga"] * 10)
        model.finalize()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model.to_dict(), f)
            path = Path(f.name)
        try:
            with path.open() as f:
                loaded = NgramModel.from_dict(json.load(f))
            self.assertEqual(loaded.ragas(), model.ragas())
            seq = ["<BOS>", "Sa", "Re"]
            self.assertAlmostEqual(
                loaded.score_sequence("Yaman", seq),
                model.score_sequence("Yaman", seq), places=10)
        finally:
            path.unlink()

    def test_metadata_fields(self):
        """Serialized dict contains expected metadata."""
        model = NgramModel(order=5, smoothing="add-k", smoothing_k=0.02)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re"])
        data = model.to_dict()
        self.assertEqual(data["metadata"]["order"], 5)
        self.assertEqual(data["metadata"]["smoothing"], "add-k")
        self.assertEqual(data["metadata"]["smoothing_params"]["k"], 0.02)
        self.assertEqual(data["metadata"]["training_ragas"], 1)
        self.assertIn("Sa", data["metadata"]["vocabulary"])


if __name__ == "__main__":
    unittest.main()
