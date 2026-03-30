import csv
import tempfile
import unittest
from pathlib import Path

from raga_pipeline.language_model import evaluate_model


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


class TestEvaluateModel(unittest.TestCase):

    def _build_corpus(self, tmp_path: Path):
        gt_path = tmp_path / "gt.csv"
        results = tmp_path / "results"
        gt_rows = []
        yaman_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                       enumerate([(60, "Sa"), (62, "Re"), (64, "Ga"), (65, "ma"), (67, "Pa")] * 30)]
        bhairav_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                         enumerate([(60, "Sa"), (61, "re"), (64, "Ga"), (65, "ma"), (68, "dha")] * 30)]
        for i in range(3):
            name = f"yaman_{i}.mp3"
            gt_rows.append((name, "Yaman", "C"))
            _write_transcription_csv(
                results / "demucs" / "htdemucs" / f"yaman_{i}" / "transcribed_notes.csv", yaman_notes)
        for i in range(3):
            name = f"bhairav_{i}.mp3"
            gt_rows.append((name, "Bhairav", "C"))
            _write_transcription_csv(
                results / "demucs" / "htdemucs" / f"bhairav_{i}" / "transcribed_notes.csv", bhairav_notes)
        _write_gt_csv(gt_path, gt_rows)
        return gt_path, results

    def test_basic_evaluation(self):
        """Evaluation produces results CSV with expected columns."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_path, results = self._build_corpus(tmp_path)
            output_path = tmp_path / "eval.csv"
            summary = evaluate_model(
                ground_truth=str(gt_path), results_dir=str(results),
                output=str(output_path), order=2, min_recordings=1)
            self.assertTrue(output_path.exists())
            self.assertIn("top1_accuracy", summary)
            self.assertIn("top3_accuracy", summary)
            self.assertIn("mrr", summary)
            self.assertGreater(summary["top1_accuracy"], 0.5)

    def test_sweep_orders(self):
        """Order sweep produces per-order results."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_path, results = self._build_corpus(tmp_path)
            output_path = tmp_path / "eval.csv"
            summary = evaluate_model(
                ground_truth=str(gt_path), results_dir=str(results),
                output=str(output_path), order=3, min_recordings=1,
                sweep_orders=[2, 3])
            self.assertIn("sweep_results", summary)
            self.assertEqual(len(summary["sweep_results"]), 2)
            self.assertIn(2, [s["order"] for s in summary["sweep_results"]])
            self.assertIn(3, [s["order"] for s in summary["sweep_results"]])


if __name__ == "__main__":
    unittest.main()
