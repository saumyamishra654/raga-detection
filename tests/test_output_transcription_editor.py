from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from raga_pipeline.output import (
        AnalysisResults,
        AnalysisStats,
        _build_analysis_report_metadata,
        generate_analysis_report,
    )
    from raga_pipeline.sequence import Note, Phrase

    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


@unittest.skipUnless(IMPORT_OK, "output module unavailable in current environment")
class TranscriptionEditorMetadataTests(unittest.TestCase):
    def _make_results(self, root: Path) -> tuple[AnalysisResults, AnalysisStats]:
        audio_path = root / "demo_song.mp3"
        vocals_path = root / "vocals.mp3"
        accomp_path = root / "accompaniment.mp3"
        transition_path = root / "transition_matrix.png"
        pitch_plot_path = root / "pitch_sargam.png"

        for media_path in [audio_path, vocals_path, accomp_path]:
            media_path.write_bytes(b"audio")
        for img_path in [transition_path, pitch_plot_path]:
            img_path.write_bytes(b"png")

        config = SimpleNamespace(
            audio_path=str(audio_path),
            vocals_path=str(vocals_path),
            accompaniment_path=str(accomp_path),
            melody_source="separated",
            transcription_smoothing_ms=70.0,
            transcription_min_duration=0.04,
            transcription_derivative_threshold=2.0,
            energy_threshold=0.0,
            show_rms_overlay=True,
        )

        note_a = Note(
            start=0.10,
            end=0.35,
            pitch_midi=60.0,
            pitch_hz=261.63,
            confidence=0.95,
            energy=0.2,
            sargam="Sa",
            pitch_class=0,
        )
        note_b = Note(
            start=0.40,
            end=0.62,
            pitch_midi=62.0,
            pitch_hz=293.66,
            confidence=0.95,
            energy=0.3,
            sargam="Re",
            pitch_class=2,
        )
        note_a.raw_pitch_midi = 60.24
        note_a.snapped_pitch_midi = 60.0
        note_a.corrected_pitch_midi = 60.0
        note_a.rendered_pitch_midi = 60.0
        phrase = Phrase(notes=[note_a, note_b])

        results = AnalysisResults(config=config)
        results.notes = [note_a, note_b]
        results.phrases = [phrase]
        results.detected_tonic = 60
        results.detected_raga = "Bhairavi"

        stats = AnalysisStats(
            correction_summary={},
            pattern_analysis={},
            raga_name="Bhairavi",
            tonic="C",
            transition_matrix_path=str(transition_path),
            pitch_plot_path=str(pitch_plot_path),
        )
        return results, stats

    def test_metadata_includes_base_transcription_edit_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            results, stats = self._make_results(root)
            metadata = _build_analysis_report_metadata(
                results=results,
                stats=stats,
                output_dir=str(root),
                report_path=str(root / "analysis_report.html"),
            )

            self.assertIn("transcription_edit_payload", metadata)
            payload = metadata["transcription_edit_payload"]
            self.assertIsInstance(payload, dict)
            self.assertIn("notes", payload)
            self.assertIn("phrases", payload)
            self.assertIn("sargam_options", payload)
            self.assertEqual(len(payload["notes"]), 2)
            self.assertEqual(len(payload["phrases"]), 1)
            self.assertGreaterEqual(len(payload["sargam_options"]), 12)

    def test_generated_analysis_report_no_longer_embeds_editor_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            results, stats = self._make_results(root)
            report_path = generate_analysis_report(
                results=results,
                stats=stats,
                output_dir=str(root),
                report_filename="analysis_report.html",
            )
            html = Path(report_path).read_text(encoding="utf-8")
            self.assertNotIn("Transcription Editor (Experimental)", html)
            self.assertNotIn("/api/transcription-edits/", html)
            self.assertIn("Musical Transcription", html)

    def test_metadata_is_json_serializable_with_numpy_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            results, stats = self._make_results(root)
            stats.correction_summary = {
                "total": np.int64(4),
                "valid_pcs": [np.int64(0), np.int64(2), np.int64(7)],
                "confidence_mean": np.float64(0.91),
                "non_finite": np.float64(np.nan),
            }
            stats.pattern_analysis = {
                "counts": np.array([1, 2, 3], dtype=np.int64),
                "checker": {
                    "score": np.float64(0.75),
                    "matched": np.int64(3),
                },
            }

            metadata = _build_analysis_report_metadata(
                results=results,
                stats=stats,
                output_dir=str(root),
                report_path=str(root / "analysis_report.html"),
            )
            encoded = json.dumps(metadata, allow_nan=False)
            self.assertTrue(encoded)
            self.assertIn("transcription_edit_payload", metadata)


if __name__ == "__main__":
    unittest.main()
