import csv
import json
import tempfile
import unittest
from pathlib import Path

from raga_pipeline.language_model import NgramModel, score_transcription


def _make_model_file(tmp_path: Path) -> Path:
    """Create a trained model JSON for testing."""
    model = NgramModel(order=3, smoothing="add-k", smoothing_k=0.01)
    model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga", "ma", "Pa"] * 50)
    model.add_sequence("Bhairav", ["<BOS>", "Sa", "re", "Ga", "ma", "dha"] * 50)
    model.finalize()
    model_path = tmp_path / "model.json"
    with model_path.open("w") as f:
        json.dump(model.to_dict(), f)
    return model_path


def _write_transcription(path: Path, notes: list[tuple[float, float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "end", "pitch_midi", "sargam"])
        writer.writeheader()
        for start, end, midi, sargam in notes:
            writer.writerow({
                "start": f"{start:.3f}", "end": f"{end:.3f}",
                "pitch_midi": f"{midi:.1f}", "sargam": sargam,
            })


class TestScoreTranscription(unittest.TestCase):

    def test_correct_raga_ranks_first(self):
        """A Yaman-like transcription should rank Yaman first."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = _make_model_file(tmp_path)
            yaman_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                           enumerate([(60, "Sa"), (62, "Re"), (64, "Ga"), (65, "ma"), (67, "Pa")] * 10)]
            trans_path = tmp_path / "transcription.csv"
            _write_transcription(trans_path, yaman_notes)
            result = score_transcription(
                model_path=str(model_path), transcription_path=str(trans_path), tonic="C")
            self.assertEqual(result["rankings"][0]["raga"], "Yaman")
            self.assertEqual(result["rankings"][0]["rank"], 1)
            self.assertGreater(result["rankings"][0]["score"], result["rankings"][1]["score"])

    def test_segment_level_scores(self):
        """With segments=True, result includes segment-level breakdown."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = _make_model_file(tmp_path)
            notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                     enumerate([(60, "Sa"), (62, "Re"), (64, "Ga")] * 100)]
            trans_path = tmp_path / "transcription.csv"
            _write_transcription(trans_path, notes)
            result = score_transcription(
                model_path=str(model_path), transcription_path=str(trans_path),
                tonic="C", segments=True, segment_window=50)
            self.assertIn("segments", result)
            self.assertGreater(len(result["segments"]), 0)
            seg = result["segments"][0]
            self.assertIn("start_time", seg)
            self.assertIn("end_time", seg)
            self.assertIn("top_ragas", seg)


if __name__ == "__main__":
    unittest.main()
