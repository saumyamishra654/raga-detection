import tempfile
import types
import unittest
import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import driver
    from raga_pipeline.audio import PitchData
    from raga_pipeline.config import PipelineConfig
    from raga_pipeline.sequence import Note

    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


def _dummy_pitch_data(audio_path: str) -> PitchData:
    timestamps = np.array([0.00, 0.01, 0.02, 0.03], dtype=float)
    pitch_hz = np.array([220.0, 221.0, 222.0, 223.0], dtype=float)
    confidence = np.array([0.99, 0.98, 0.97, 0.96], dtype=float)
    voicing = np.array([True, True, True, True], dtype=bool)
    valid_freqs = pitch_hz.copy()
    midi_vals = np.array([57.0, 57.1, 57.2, 57.3], dtype=float)
    energy = np.array([0.4, 0.45, 0.05, 0.06], dtype=float)
    return PitchData(
        timestamps=timestamps,
        pitch_hz=pitch_hz,
        confidence=confidence,
        voicing=voicing,
        valid_freqs=valid_freqs,
        midi_vals=midi_vals,
        energy=energy,
        frame_period=0.01,
        audio_path=audio_path,
    )


def _dummy_histogram() -> object:
    return types.SimpleNamespace(
        high_res=np.array([0.0, 1.0], dtype=float),
        low_res=np.array([0.0, 1.0], dtype=float),
        smoothed_high=np.array([0.0, 1.0], dtype=float),
        smoothed_low=np.array([0.0, 1.0], dtype=float),
        bin_centers_high=np.array([0.0, 100.0], dtype=float),
        bin_centers_low=np.array([0.0, 100.0], dtype=float),
        high_res_norm=np.array([0.0, 1.0], dtype=float),
        smoothed_high_norm=np.array([0.0, 1.0], dtype=float),
    )


def _dummy_peaks() -> object:
    return types.SimpleNamespace(
        high_res_indices=np.array([1], dtype=int),
        high_res_cents=np.array([100.0], dtype=float),
        low_res_indices=np.array([1], dtype=int),
        low_res_cents=np.array([100.0], dtype=float),
        validated_indices=np.array([1], dtype=int),
        validated_cents=np.array([100.0], dtype=float),
        pitch_classes={1},
    )


@unittest.skipUnless(IMPORT_OK, "driver or pipeline imports unavailable in current environment")
class DriverAnalyzeEnergyFlowTests(unittest.TestCase):
    def test_analyze_keeps_low_energy_inflections_after_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "demo.wav"
            audio_path.write_bytes(b"RIFF")

            config = PipelineConfig(
                audio_path=str(audio_path),
                output_dir=tmpdir,
                mode="analyze",
                tonic_override="C",
                raga_override="Bhairavi",
                energy_threshold=0.1,
            )
            config.raga_db_path = None

            stem_dir = Path(config.stem_dir)
            stem_dir.mkdir(parents=True, exist_ok=True)
            (stem_dir / "melody_pitch_data.csv").write_text("pitch_hz,timestamps\n", encoding="utf-8")

            transcribed_notes = [
                Note(
                    start=0.0,
                    end=0.2,
                    pitch_midi=60.0,
                    pitch_hz=261.6,
                    confidence=1.0,
                    energy=0.25,
                    sargam="Sa",
                    pitch_class=0,
                ),
                Note(
                    start=0.25,
                    end=0.26,
                    pitch_midi=61.0,
                    pitch_hz=277.2,
                    confidence=0.8,
                    energy=0.05,
                    sargam="re",
                    pitch_class=1,
                ),
            ]

            with (
                patch("driver._safe_get_audio_duration_seconds", return_value=None),
                patch("driver.librosa", new=types.SimpleNamespace(note_to_hz=lambda _note: 440.0)),
                patch("driver.extract_pitch", return_value=_dummy_pitch_data(str(audio_path))),
                patch("driver.transcription.transcribe_to_notes", return_value=transcribed_notes),
                patch("driver.merge_consecutive_notes", side_effect=lambda notes, **_: notes),
                patch("driver.plot_note_duration_histogram"),
                patch("driver.save_notes_to_csv"),
                patch("driver.detect_phrases", return_value=[]),
                patch("driver.cluster_phrases", return_value={}),
                patch(
                    "driver.build_transition_matrix_corrected",
                    return_value=(np.zeros((1, 1), dtype=float), ["Sa"], {}),
                ),
                patch("driver.plot_transition_heatmap_v2"),
                patch(
                    "driver.analyze_raga_patterns",
                    return_value={
                        "common_motifs": [],
                        "total_aaroh_runs": 0,
                        "total_avroh_runs": 0,
                    },
                ),
                patch("driver.plot_pitch_with_sargam_lines"),
                patch("driver.plot_note_segments"),
                patch("driver.compute_cent_histograms_from_config", return_value=_dummy_histogram()),
                patch("driver.detect_peaks_from_config", return_value=_dummy_peaks()),
                patch("driver.fit_gmm_to_peaks", return_value=[]),
                patch("driver.compute_gmm_bias_cents", return_value=None),
                patch(
                    "driver.generate_analysis_report",
                    return_value=str(stem_dir / "analysis_report.html"),
                ),
            ):
                results = driver.run_pipeline(config)

            self.assertEqual(len(results.notes), 2)
            self.assertTrue(any(n.confidence < 0.99 for n in results.notes))
            self.assertTrue(any(n.energy < config.energy_threshold for n in results.notes))

    def test_analyze_strict_raga_filter_uses_configured_cents_and_disables_keep_impure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "demo.wav"
            audio_path.write_bytes(b"RIFF")
            fake_db_path = Path(tmpdir) / "fake_raga_db.csv"
            fake_db_path.write_text("names\nBhairavi\n", encoding="utf-8")

            config = PipelineConfig(
                audio_path=str(audio_path),
                output_dir=tmpdir,
                mode="analyze",
                tonic_override="C",
                raga_override="Bhairavi",
                keep_impure_notes=True,
                strict_raga_35c_filter=True,
                strict_raga_max_cents=42.0,
            )
            config.raga_db_path = str(fake_db_path)

            stem_dir = Path(config.stem_dir)
            stem_dir.mkdir(parents=True, exist_ok=True)
            (stem_dir / "melody_pitch_data.csv").write_text("pitch_hz,timestamps\n", encoding="utf-8")

            transcribed_notes = [
                Note(
                    start=0.0,
                    end=0.2,
                    pitch_midi=60.0,
                    pitch_hz=261.6,
                    confidence=1.0,
                    energy=0.25,
                    sargam="Sa",
                    pitch_class=0,
                ),
            ]
            apply_calls: list[dict] = []

            real_isfile = os.path.isfile

            def fake_isfile(path: str) -> bool:
                if str(path) == str(fake_db_path):
                    return True
                return real_isfile(path)

            def fake_apply(notes, raga_db, raga_name, tonic, max_distance=1.0, keep_impure=False):
                apply_calls.append(
                    {
                        "max_distance": max_distance,
                        "keep_impure": keep_impure,
                        "raga_name": raga_name,
                        "tonic": tonic,
                    }
                )
                return notes, {"discarded": 0}, []

            with ExitStack() as stack:
                stack.enter_context(patch("driver.os.path.isfile", side_effect=fake_isfile))
                stack.enter_context(patch("driver.RagaDatabase", return_value=object()))
                stack.enter_context(patch("driver.load_aaroh_avroh_patterns", return_value={}))
                stack.enter_context(patch("driver._safe_get_audio_duration_seconds", return_value=None))
                stack.enter_context(patch("driver.librosa", new=types.SimpleNamespace(note_to_hz=lambda _note: 440.0)))
                stack.enter_context(patch("driver.extract_pitch", return_value=_dummy_pitch_data(str(audio_path))))
                stack.enter_context(patch("driver.transcription.transcribe_to_notes", return_value=transcribed_notes))
                stack.enter_context(patch("driver.apply_raga_correction_to_notes", side_effect=fake_apply))
                stack.enter_context(patch("driver.merge_consecutive_notes", side_effect=lambda notes, **_: notes))
                stack.enter_context(patch("driver.plot_note_duration_histogram"))
                stack.enter_context(patch("driver.save_notes_to_csv"))
                stack.enter_context(patch("driver.detect_phrases", return_value=[]))
                stack.enter_context(patch("driver.cluster_phrases", return_value={}))
                stack.enter_context(
                    patch(
                        "driver.build_transition_matrix_corrected",
                        return_value=(np.zeros((1, 1), dtype=float), ["Sa"], {}),
                    )
                )
                stack.enter_context(patch("driver.plot_transition_heatmap_v2"))
                stack.enter_context(
                    patch(
                        "driver.analyze_raga_patterns",
                        return_value={
                            "common_motifs": [],
                            "total_aaroh_runs": 0,
                            "total_avroh_runs": 0,
                        },
                    )
                )
                stack.enter_context(patch("driver.plot_pitch_with_sargam_lines"))
                stack.enter_context(patch("driver.plot_note_segments"))
                stack.enter_context(patch("driver.compute_cent_histograms_from_config", return_value=_dummy_histogram()))
                stack.enter_context(patch("driver.detect_peaks_from_config", return_value=_dummy_peaks()))
                stack.enter_context(patch("driver.fit_gmm_to_peaks", return_value=[]))
                stack.enter_context(patch("driver.compute_gmm_bias_cents", return_value=None))
                stack.enter_context(
                    patch(
                        "driver.generate_analysis_report",
                        return_value=str(stem_dir / "analysis_report.html"),
                    )
                )
                driver.run_pipeline(config)

            self.assertEqual(len(apply_calls), 1)
            self.assertAlmostEqual(float(apply_calls[0]["max_distance"]), 0.42, places=9)
            self.assertFalse(bool(apply_calls[0]["keep_impure"]))


if __name__ == "__main__":
    unittest.main()
