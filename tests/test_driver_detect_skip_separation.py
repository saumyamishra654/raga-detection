import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import driver
    from raga_pipeline.audio import PitchData
    from raga_pipeline.config import PipelineConfig

    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


def _dummy_pitch_data(audio_path: str) -> PitchData:
    timestamps = np.array([0.00, 0.01, 0.02], dtype=float)
    pitch_hz = np.array([220.0, 221.0, 0.0], dtype=float)
    confidence = np.array([0.99, 0.98, 0.2], dtype=float)
    voicing = np.array([True, True, False], dtype=bool)
    valid_freqs = np.array([220.0, 221.0], dtype=float)
    midi_vals = np.array([57.0, 57.08], dtype=float)
    energy = np.array([0.4, 0.45, 0.05], dtype=float)
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
class DriverDetectSkipSeparationTests(unittest.TestCase):
    def _build_detect_config(self, tmpdir: str, **overrides: object) -> PipelineConfig:
        audio_path = Path(tmpdir) / "demo.wav"
        audio_path.write_bytes(b"RIFF")
        kwargs = {
            "audio_path": str(audio_path),
            "output_dir": tmpdir,
            "mode": "detect",
        }
        kwargs.update(overrides)
        config = PipelineConfig(**kwargs)
        # Keep tests focused on detect pipeline branching, not raga DB loading/scoring.
        config.raga_db_path = None
        return config

    def _run_detect_with_mocks(self, config: PipelineConfig, mock_separate: object) -> "driver.AnalysisResults":
        dummy_hist = _dummy_histogram()
        dummy_peaks = _dummy_peaks()

        def fake_extract_pitch(
            audio_path: str,
            output_dir: str,
            prefix: str,
            fmin: float,
            fmax: float,
            confidence_threshold: float = 0.9,
            force_recompute: bool = False,
            energy_metric: str = "rms",
        ) -> PitchData:
            _ = (output_dir, prefix, fmin, fmax, confidence_threshold, force_recompute, energy_metric)
            return _dummy_pitch_data(audio_path)

        with (
            patch("driver.extract_pitch", side_effect=fake_extract_pitch),
            patch("driver.separate_stems", mock_separate),
            patch("driver.compute_cent_histograms_from_config", return_value=dummy_hist),
            patch("driver.detect_peaks_from_config", return_value=dummy_peaks),
            patch("driver.fit_gmm_to_peaks", return_value=[]),
            patch("driver.compute_gmm_bias_cents", return_value=None),
            patch("driver.plot_histograms"),
            patch("driver.plot_absolute_note_histogram", return_value=pd.DataFrame([{"C": 1.0}])),
            patch(
                "driver.transcription.detect_stationary_events",
                return_value=[types.SimpleNamespace(snapped_midi=60, duration=0.2)],
            ),
            patch("raga_pipeline.output.generate_detection_report"),
        ):
            return driver.run_pipeline(config)

    def test_skip_separation_detect_does_not_call_stem_separation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_detect_config(
                tmpdir,
                skip_separation=True,
                tonic_override="C",
            )
            mock_separate = unittest.mock.Mock(return_value=("unused_vocals.mp3", "unused_accompaniment.mp3"))

            results = self._run_detect_with_mocks(config, mock_separate)

            mock_separate.assert_not_called()
            self.assertIsNotNone(results.pitch_data_composite)
            self.assertIs(results.pitch_data_stem, results.pitch_data_composite)
            self.assertIsNone(results.pitch_data_accomp)

    def test_detect_without_skip_calls_stem_separation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_vocals = str((Path(tmpdir) / "vocals.mp3").resolve())
            fake_accomp = str((Path(tmpdir) / "accompaniment.mp3").resolve())
            config = self._build_detect_config(tmpdir)
            mock_separate = unittest.mock.Mock(return_value=(fake_vocals, fake_accomp))

            results = self._run_detect_with_mocks(config, mock_separate)

            mock_separate.assert_called_once()
            self.assertIsNotNone(results.pitch_data_stem)
            self.assertIsNotNone(results.pitch_data_composite)


if __name__ == "__main__":
    unittest.main()
