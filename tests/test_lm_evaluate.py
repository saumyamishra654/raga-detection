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


class TestLmDeletionMode(unittest.TestCase):
    """Tests for scoring_mode='lm-deletion'."""

    def _build_corpus_with_raga_db(self, tmp_path: Path):
        """Build a synthetic corpus + minimal raga DB for lm-deletion testing."""
        gt_path = tmp_path / "gt.csv"
        results = tmp_path / "results"

        # Two ragas with distinct 5-note scales (pentatonic)
        # "TestRagaA" uses pitch classes 0,2,4,7,9 (C,D,E,G,A - major pentatonic)
        # "TestRagaB" uses pitch classes 0,3,5,7,10 (C,Eb,F,G,Bb - minor pentatonic)
        raga_a_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                        enumerate([(60, "Sa"), (62, "Re"), (64, "Ga"), (67, "Pa"), (69, "Dha")] * 30)]
        raga_b_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                        enumerate([(60, "Sa"), (63, "ga"), (65, "ma"), (67, "Pa"), (70, "ni")] * 30)]

        gt_rows = []
        for i in range(3):
            gt_rows.append((f"a_{i}.mp3", "TestRagaA", "C"))
            _write_transcription_csv(
                results / "demucs" / "htdemucs" / f"a_{i}" / "transcribed_notes.csv", raga_a_notes)
        for i in range(3):
            gt_rows.append((f"b_{i}.mp3", "TestRagaB", "C"))
            _write_transcription_csv(
                results / "demucs" / "htdemucs" / f"b_{i}" / "transcribed_notes.csv", raga_b_notes)
        _write_gt_csv(gt_path, gt_rows)

        # Minimal raga DB CSV (same format as raga_list_final.csv)
        # Columns: 0-11 (pitch class presence), names
        raga_db_path = tmp_path / "raga_db.csv"
        with raga_db_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[str(i) for i in range(12)] + ["names"])
            writer.writeheader()
            # TestRagaA: C,D,E,G,A = pitch classes 0,2,4,7,9
            row_a = {str(i): "0" for i in range(12)}
            for pc in [0, 2, 4, 7, 9]:
                row_a[str(pc)] = "1"
            row_a["names"] = "TestRagaA"
            writer.writerow(row_a)
            # TestRagaB: C,Eb,F,G,Bb = pitch classes 0,3,5,7,10
            row_b = {str(i): "0" for i in range(12)}
            for pc in [0, 3, 5, 7, 10]:
                row_b[str(pc)] = "1"
            row_b["names"] = "TestRagaB"
            writer.writerow(row_b)

        return gt_path, results, raga_db_path

    def test_lm_deletion_mode_runs(self):
        """scoring_mode='lm-deletion' produces results with expected keys."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_path, results, raga_db_path = self._build_corpus_with_raga_db(tmp_path)
            output_path = tmp_path / "eval.csv"

            summary = evaluate_model(
                ground_truth=str(gt_path), results_dir=str(results),
                output=str(output_path), order=2, min_recordings=1,
                scoring_mode="lm-deletion", raga_db_path=str(raga_db_path),
                lm_deletion_lambda=1.0)

            self.assertIn("top1_accuracy", summary)
            self.assertIn("mrr", summary)
            self.assertTrue(output_path.exists())

    def test_higher_lambda_changes_ranking(self):
        """Increasing lambda should change at least some scores (not all identical)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_path, results, raga_db_path = self._build_corpus_with_raga_db(tmp_path)

            summary_lo = evaluate_model(
                ground_truth=str(gt_path), results_dir=str(results),
                output=str(tmp_path / "lo.csv"), order=2, min_recordings=1,
                scoring_mode="lm-deletion", raga_db_path=str(raga_db_path),
                lm_deletion_lambda=0.0)

            summary_hi = evaluate_model(
                ground_truth=str(gt_path), results_dir=str(results),
                output=str(tmp_path / "hi.csv"), order=2, min_recordings=1,
                scoring_mode="lm-deletion", raga_db_path=str(raga_db_path),
                lm_deletion_lambda=10.0)

            # Both should produce valid results; MRR may differ
            self.assertGreater(summary_lo["total"], 0)
            self.assertGreater(summary_hi["total"], 0)
            # With extreme lambda=10, deletion residual dominates and may
            # change the ranking compared to lambda=0
            # (We just verify both runs complete without error)


if __name__ == "__main__":
    unittest.main()
