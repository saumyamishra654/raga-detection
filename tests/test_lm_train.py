import csv
import json
import tempfile
import unittest
from pathlib import Path

from raga_pipeline.language_model import train_model


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


class TestTrainModel(unittest.TestCase):

    def test_basic_training(self):
        """train_model builds a model from GT CSV and transcription CSVs."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_path = tmp_path / "gt.csv"
            _write_gt_csv(gt_path, [
                ("song_a.mp3", "Yaman", "C"),
                ("song_b.mp3", "Yaman", "C"),
                ("song_c.mp3", "Yaman", "C"),
                ("song_d.mp3", "Bhairav", "C"),
                ("song_e.mp3", "Bhairav", "C"),
                ("song_f.mp3", "Bhairav", "C"),
            ])
            results = tmp_path / "results"
            yaman_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                           enumerate([(60, "Sa"), (62, "Re"), (64, "Ga"), (65, "ma"), (67, "Pa")] * 20)]
            for name in ["song_a", "song_b", "song_c"]:
                _write_transcription_csv(
                    results / "demucs" / "htdemucs" / name / "transcribed_notes.csv", yaman_notes)
            bhairav_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                            enumerate([(60, "Sa"), (61, "re"), (64, "Ga"), (65, "ma"), (68, "dha")] * 20)]
            for name in ["song_d", "song_e", "song_f"]:
                _write_transcription_csv(
                    results / "demucs" / "htdemucs" / name / "transcribed_notes.csv", bhairav_notes)

            output_path = tmp_path / "model.json"
            result = train_model(
                ground_truth=str(gt_path), results_dir=str(results),
                output=str(output_path), order=3, min_recordings=1)

            self.assertTrue(output_path.exists())
            with output_path.open() as f:
                data = json.load(f)
            self.assertIn("Yaman", data["ragas"])
            self.assertIn("Bhairav", data["ragas"])
            self.assertEqual(data["metadata"]["order"], 3)

    def test_min_recordings_filter(self):
        """Ragas with fewer recordings than min_recordings are excluded."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_path = tmp_path / "gt.csv"
            _write_gt_csv(gt_path, [
                ("song_a.mp3", "Yaman", "C"),
                ("song_b.mp3", "Rare", "C"),
            ])
            results = tmp_path / "results"
            notes = [(i * 0.3, (i + 1) * 0.3, 60, "Sa") for i in range(10)]
            for name in ["song_a", "song_b"]:
                _write_transcription_csv(
                    results / "demucs" / "htdemucs" / name / "transcribed_notes.csv", notes)

            output_path = tmp_path / "model.json"
            train_model(
                ground_truth=str(gt_path), results_dir=str(results),
                output=str(output_path), order=2, min_recordings=2)

            with output_path.open() as f:
                data = json.load(f)
            self.assertEqual(len(data["ragas"]), 0)


if __name__ == "__main__":
    unittest.main()
