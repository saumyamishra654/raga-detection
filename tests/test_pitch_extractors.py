"""Tests for multi-backend pitch extraction (swiftf0, pyin)."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from raga_pipeline.audio import (
    _extract_pyin,
    _pitch_csv_path,
    extract_pitch,
)
from raga_pipeline.config import PipelineConfig, parse_config_from_argv


class TestPitchCsvPath(unittest.TestCase):
    """Cache filename generation."""

    def test_swiftf0_uses_legacy_name(self):
        path = _pitch_csv_path("/out", "vocals", "swiftf0")
        self.assertEqual(path, "/out/vocals_pitch_data.csv")

    def test_pyin_includes_suffix(self):
        path = _pitch_csv_path("/out", "vocals", "pyin")
        self.assertEqual(path, "/out/vocals_pitch_data_pyin.csv")


class TestExtractPyin(unittest.TestCase):
    """pYIN backend produces correct output shape and handles NaN."""

    @patch("librosa.load")
    @patch("librosa.pyin")
    def test_basic_output(self, mock_pyin, mock_load):
        sr = 22050
        n_frames = 100
        mock_load.return_value = (np.zeros(sr * 2), sr)

        f0 = np.full(n_frames, 440.0)
        f0[50:60] = np.nan  # unvoiced region
        voiced = np.ones(n_frames, dtype=bool)
        voiced[50:60] = False
        probs = np.ones(n_frames) * 0.95
        probs[50:60] = 0.1
        mock_pyin.return_value = (f0, voiced, probs)

        ts, hz, conf, voicing, fp = _extract_pyin(
            "/fake/audio.mp3", fmin=50.0, fmax=2000.0,
            confidence_threshold=0.9,
        )

        self.assertEqual(len(ts), n_frames)
        self.assertEqual(len(hz), n_frames)
        self.assertEqual(len(conf), n_frames)
        self.assertEqual(len(voicing), n_frames)
        # NaN should be converted to 0.0
        self.assertEqual(hz[55], 0.0)
        self.assertFalse(voicing[55])
        # Voiced region should have pitch
        self.assertAlmostEqual(hz[0], 440.0)
        self.assertTrue(voicing[0])
        self.assertGreater(fp, 0)

    @patch("librosa.load")
    @patch("librosa.pyin")
    def test_custom_hop_ms(self, mock_pyin, mock_load):
        sr = 22050
        mock_load.return_value = (np.zeros(sr), sr)
        mock_pyin.return_value = (
            np.array([440.0]),
            np.array([True]),
            np.array([0.95]),
        )

        _extract_pyin(
            "/fake/audio.mp3", fmin=50.0, fmax=2000.0,
            confidence_threshold=0.9, hop_ms=5.0,
        )

        # Verify hop_length was computed from hop_ms
        call_kwargs = mock_pyin.call_args[1]
        expected_hop = max(1, int(round(sr * 5.0 / 1000.0)))  # ~110
        self.assertEqual(call_kwargs["hop_length"], expected_hop)


class TestExtractPitchDispatch(unittest.TestCase):
    """extract_pitch dispatches to the correct backend."""

    @patch("raga_pipeline.audio._extract_swiftf0")
    @patch("raga_pipeline.audio.load_pitch_from_csv")
    def test_default_uses_swiftf0(self, mock_load_csv, mock_swiftf0):
        mock_swiftf0.return_value = (
            np.arange(10) * 0.01,
            np.full(10, 440.0),
            np.ones(10),
            np.ones(10, dtype=bool),
            0.01,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("librosa.load", return_value=(np.zeros(16000), 16000)):
                with patch("librosa.hz_to_midi", return_value=np.array([69.0] * 10)):
                    result = extract_pitch(
                        audio_path="/fake/audio.mp3",
                        output_dir=tmpdir,
                        prefix="test",
                        fmin=50.0, fmax=2000.0,
                    )
            mock_swiftf0.assert_called_once()
            self.assertEqual(len(result.timestamps), 10)

    @patch("raga_pipeline.audio._extract_pyin")
    def test_pyin_dispatch(self, mock_pyin):
        mock_pyin.return_value = (
            np.arange(10) * 0.02,
            np.full(10, 440.0),
            np.ones(10),
            np.ones(10, dtype=bool),
            0.02,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("librosa.load", return_value=(np.zeros(22050), 22050)):
                with patch("librosa.hz_to_midi", return_value=np.array([69.0] * 10)):
                    result = extract_pitch(
                        audio_path="/fake/audio.mp3",
                        output_dir=tmpdir,
                        prefix="test",
                        fmin=50.0, fmax=2000.0,
                        extractor="pyin",
                        hop_ms=5.0,
                    )
            mock_pyin.assert_called_once()
            # Verify hop_ms was passed through
            self.assertEqual(mock_pyin.call_args[1]["hop_ms"], 5.0)


class TestCliParsing(unittest.TestCase):
    """CLI args for pitch extractor parse correctly."""

    def _make_argv(self, tmpdir, extra_args):
        audio = Path(tmpdir) / "demo.wav"
        audio.write_bytes(b"RIFF")
        return ["detect", "--audio", str(audio), "--output", tmpdir] + extra_args

    def test_pitch_extractor_pyin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = parse_config_from_argv(
                self._make_argv(tmpdir, ["--pitch-extractor", "pyin"])
            )
            self.assertEqual(config.pitch_extractor, "pyin")

    def test_pitch_hop_ms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = parse_config_from_argv(
                self._make_argv(tmpdir, ["--pitch-hop-ms", "5"])
            )
            self.assertAlmostEqual(config.pitch_hop_ms, 5.0)

    def test_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = parse_config_from_argv(self._make_argv(tmpdir, []))
            self.assertEqual(config.pitch_extractor, "swiftf0")
            self.assertAlmostEqual(config.pitch_hop_ms, 0.0)


class TestExtractorConfidenceDefaults(unittest.TestCase):
    """Confidence thresholds auto-resolve based on pitch extractor."""

    def _make_argv(self, tmpdir, extra_args):
        audio = Path(tmpdir) / "demo.wav"
        audio.write_bytes(b"RIFF")
        return ["detect", "--audio", str(audio), "--output", tmpdir] + extra_args

    def test_swiftf0_default_confidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = parse_config_from_argv(self._make_argv(tmpdir, []))
            self.assertAlmostEqual(config.vocal_confidence, 0.95)
            self.assertAlmostEqual(config.accomp_confidence, 0.80)

    def test_pyin_default_confidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = parse_config_from_argv(
                self._make_argv(tmpdir, ["--pitch-extractor", "pyin"])
            )
            self.assertAlmostEqual(config.vocal_confidence, 0.15)
            self.assertAlmostEqual(config.accomp_confidence, 0.05)

    def test_explicit_override_wins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = parse_config_from_argv(
                self._make_argv(tmpdir, [
                    "--pitch-extractor", "pyin",
                    "--vocal-confidence", "0.30",
                    "--accomp-confidence", "0.10",
                ])
            )
            self.assertAlmostEqual(config.vocal_confidence, 0.30)
            self.assertAlmostEqual(config.accomp_confidence, 0.10)


if __name__ == "__main__":
    unittest.main()
