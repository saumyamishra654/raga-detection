import csv
import json
import tempfile
import unittest
from pathlib import Path

from raga_pipeline.language_model.__main__ import main


def _write_gt_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Filename", "Raga", "Tonic"])
        writer.writeheader()
        for filename, raga, tonic in rows:
            writer.writerow({"Filename": filename, "Raga": raga, "Tonic": tonic})


def _write_transcription_csv(path: Path, rows: list[tuple[float, float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "end", "pitch_midi", "sargam"])
        writer.writeheader()
        for start, end, midi, sargam in rows:
            writer.writerow({
                "start": f"{start:.3f}", "end": f"{end:.3f}",
                "pitch_midi": f"{midi:.1f}", "sargam": sargam,
            })


class TestLMCli(unittest.TestCase):

    def test_train_and_score_smoke(self):
        """CLI train then score runs without error."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_path = tmp_path / "gt.csv"
            _write_gt_csv(gt_path, [
                ("song_a.mp3", "Yaman", "C"),
                ("song_b.mp3", "Yaman", "C"),
                ("song_c.mp3", "Yaman", "C"),
            ])
            results = tmp_path / "results"
            notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                     enumerate([(60, "Sa"), (62, "Re"), (64, "Ga")] * 30)]
            for name in ["song_a", "song_b", "song_c"]:
                _write_transcription_csv(
                    results / "demucs" / "htdemucs" / name / "transcribed_notes.csv", notes)
            model_path = tmp_path / "model.json"
            rc = main([
                "train", "--gt", str(gt_path), "--results-dir", str(results),
                "--output", str(model_path), "--order", "2", "--min-recordings", "1", "--quiet"])
            self.assertEqual(rc, 0)
            self.assertTrue(model_path.exists())
            trans_path = results / "demucs" / "htdemucs" / "song_a" / "transcribed_notes.csv"
            rc = main([
                "score", "--model", str(model_path),
                "--transcription", str(trans_path), "--tonic", "C"])
            self.assertEqual(rc, 0)

    def test_evaluate_smoke(self):
        """CLI evaluate runs without error."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_path = tmp_path / "gt.csv"
            _write_gt_csv(gt_path, [
                ("s1.mp3", "Yaman", "C"), ("s2.mp3", "Yaman", "C"),
                ("s3.mp3", "Bhairav", "C"), ("s4.mp3", "Bhairav", "C"),
            ])
            results = tmp_path / "results"
            y_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                       enumerate([(60, "Sa"), (62, "Re"), (64, "Ga")] * 30)]
            b_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                       enumerate([(60, "Sa"), (61, "re"), (64, "Ga")] * 30)]
            for name in ["s1", "s2"]:
                _write_transcription_csv(
                    results / "demucs" / "htdemucs" / name / "transcribed_notes.csv", y_notes)
            for name in ["s3", "s4"]:
                _write_transcription_csv(
                    results / "demucs" / "htdemucs" / name / "transcribed_notes.csv", b_notes)
            eval_path = tmp_path / "eval.csv"
            rc = main([
                "evaluate", "--gt", str(gt_path), "--results-dir", str(results),
                "--output", str(eval_path), "--order", "2", "--min-recordings", "1", "--quiet"])
            self.assertEqual(rc, 0)
            self.assertTrue(eval_path.exists())


class TestLambdaParsing(unittest.TestCase):
    """Tests for lambda CLI parsing and order convention."""

    def test_lambda_parsing_reverses_order(self):
        """CLI lambdas are highest-order-first; model expects unigram-first."""
        from raga_pipeline.language_model.__main__ import _parse_lambdas
        result = _parse_lambdas("0.6,0.3,0.1", order=3)
        # 0.6 is highest-order (trigram), should become index 2
        # 0.1 is unigram, should become index 0
        self.assertEqual(result, [0.1, 0.3, 0.6])

    def test_lambda_none_returns_none(self):
        from raga_pipeline.language_model.__main__ import _parse_lambdas
        self.assertIsNone(_parse_lambdas(None, order=3))

    def test_lambda_wrong_count_raises(self):
        from raga_pipeline.language_model.__main__ import _parse_lambdas
        with self.assertRaises(ValueError):
            _parse_lambdas("0.5,0.5", order=3)


if __name__ == "__main__":
    unittest.main()
