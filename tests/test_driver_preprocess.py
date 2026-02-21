import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

try:
    import driver
    from raga_pipeline.config import PipelineConfig
    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


@unittest.skipUnless(IMPORT_OK, "driver or pipeline imports unavailable in current environment")
class DriverPreprocessTests(unittest.TestCase):
    def _base_preprocess_config(self, tmpdir: str, **overrides: object) -> "PipelineConfig":
        kwargs = {
            "audio_path": None,
            "output_dir": tmpdir,
            "mode": "preprocess",
            "audio_dir": tmpdir,
            "filename_override": "demo_song",
            "preprocess_ingest": "youtube",
            "yt_url": "https://example.com/watch?v=test",
        }
        kwargs.update(overrides)
        return PipelineConfig(**kwargs)

    @patch("driver.download_youtube_audio")
    def test_preprocess_youtube_suggests_detect_without_tonic(self, mock_download: "patch") -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str((Path(tmpdir) / "demo_song.mp3").resolve())
            mock_download.return_value = output_path
            config = self._base_preprocess_config(tmpdir)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                driver.run_pipeline(config)

            text = buffer.getvalue()
            self.assertIn('  --audio "' + output_path + '" \\', text)
            self.assertIn('  --output "' + config.output_dir + '"', text)
            self.assertNotIn("--tonic", text)
            mock_download.assert_called_once()

    @patch("driver.ingest_recorded_audio_file")
    def test_preprocess_record_song_suggests_detect_without_tonic(self, mock_ingest: "patch") -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorded_source = Path(tmpdir) / "recorded.wav"
            recorded_source.write_bytes(b"RIFF")
            output_path = str((Path(tmpdir) / "demo_song.mp3").resolve())
            mock_ingest.return_value = output_path

            config = self._base_preprocess_config(
                tmpdir,
                preprocess_ingest="record",
                preprocess_record_mode="song",
                preprocess_recorded_audio=str(recorded_source),
                yt_url=None,
            )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                driver.run_pipeline(config)

            text = buffer.getvalue()
            self.assertIn('  --audio "' + output_path + '" \\', text)
            self.assertNotIn("--tonic", text)
            mock_ingest.assert_called_once()

    @patch("driver.get_tonic_from_tanpura_key")
    @patch("driver.ingest_recorded_audio_file")
    def test_preprocess_tanpura_vocal_suggests_detect_with_tonic(
        self,
        mock_ingest: "patch",
        mock_get_tonic: "patch",
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorded_source = Path(tmpdir) / "recorded.wav"
            recorded_source.write_bytes(b"RIFF")
            output_path = str((Path(tmpdir) / "demo_song.mp3").resolve())
            mock_ingest.return_value = output_path
            mock_get_tonic.return_value = "A"

            config = self._base_preprocess_config(
                tmpdir,
                preprocess_ingest="record",
                preprocess_record_mode="tanpura_vocal",
                preprocess_tanpura_key="A",
                preprocess_recorded_audio=str(recorded_source),
                yt_url=None,
            )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                driver.run_pipeline(config)

            text = buffer.getvalue()
            self.assertIn('  --audio "' + output_path + '" \\', text)
            self.assertIn('  --output "' + config.output_dir + '" \\', text)
            self.assertIn('  --tonic "A" \\', text)
            self.assertIn("  --skip-separation", text)
            mock_ingest.assert_called_once()
            mock_get_tonic.assert_called_once_with("A")


if __name__ == "__main__":
    unittest.main()
