import json
import tempfile
import unittest
from pathlib import Path

from raga_pipeline.config import PipelineConfig
from raga_pipeline.output import (
    AnalysisResults,
    AnalysisStats,
    _write_analysis_report_metadata,
    write_detection_report_metadata,
)


class ReportMetadataVersioningTests(unittest.TestCase):
    def test_detection_report_metadata_includes_runtime_fingerprint_and_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "song.mp3"
            audio_path.write_bytes(b"audio")

            cfg = PipelineConfig(
                audio_path=str(audio_path),
                output_dir=str(root / "out"),
                mode="detect",
            )
            results = AnalysisResults(config=cfg)
            results.detected_tonic = 0
            results.detected_raga = "Bhairavi"

            report_path = root / "out" / "detection_report.html"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("<html></html>", encoding="utf-8")

            runtime_fp = {
                "fingerprint_version": 1,
                "stage_manifest_version": 1,
                "stage_hashes": {"detect": "detect-hash", "analyze": "analyze-hash"},
                "git_commit": "abc123",
                "git_dirty": False,
                "file_hash": "hash",
                "source": "git+files",
                "computed_at": "2026-03-01T00:00:00+00:00",
            }
            meta_path = write_detection_report_metadata(
                results,
                str(report_path),
                runtime_fingerprint=runtime_fp,
            )
            payload = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            self.assertEqual(payload.get("pipeline_mode"), "detect")
            self.assertEqual(payload.get("runtime_fingerprint", {}).get("file_hash"), "hash")
            self.assertEqual(payload.get("run_identity", {}).get("mode"), "detect")
            self.assertEqual(payload.get("run_identity", {}).get("audio_path"), str(audio_path.resolve()))
            self.assertEqual(payload.get("stage_fingerprint", {}).get("mode"), "detect")
            self.assertEqual(payload.get("stage_fingerprint", {}).get("hash"), "detect-hash")

    def test_analysis_report_metadata_includes_runtime_fingerprint_and_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "song.mp3"
            audio_path.write_bytes(b"audio")

            cfg = PipelineConfig(
                audio_path=str(audio_path),
                output_dir=str(root / "out"),
                mode="analyze",
                tonic_override="C",
                raga_override="Bhairavi",
            )
            results = AnalysisResults(config=cfg)
            results.detected_tonic = 0
            results.detected_raga = "Bhairavi"

            stats = AnalysisStats(
                correction_summary={},
                pattern_analysis={},
                raga_name="Bhairavi",
                tonic="C",
                transition_matrix_path=str(root / "out" / "transition_matrix.png"),
                pitch_plot_path=str(root / "out" / "pitch_sargam.png"),
            )
            report_path = root / "out" / "analysis_report.html"
            report_path.parent.mkdir(parents=True, exist_ok=True)

            runtime_fp = {
                "fingerprint_version": 1,
                "stage_manifest_version": 1,
                "stage_hashes": {"detect": "detect-hash", "analyze": "analyze-hash"},
                "git_commit": "def456",
                "git_dirty": True,
                "file_hash": "hash2",
                "source": "git+files",
                "computed_at": "2026-03-01T00:00:00+00:00",
            }
            _write_analysis_report_metadata(
                results,
                stats,
                str(report_path.parent),
                str(report_path),
                runtime_fingerprint=runtime_fp,
            )

            payload = json.loads(report_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
            self.assertEqual(payload.get("pipeline_mode"), "analyze")
            self.assertEqual(payload.get("runtime_fingerprint", {}).get("git_commit"), "def456")
            self.assertEqual(payload.get("run_identity", {}).get("mode"), "analyze")
            self.assertEqual(payload.get("run_identity", {}).get("raga"), "Bhairavi")
            self.assertEqual(payload.get("stage_fingerprint", {}).get("mode"), "analyze")
            self.assertEqual(payload.get("stage_fingerprint", {}).get("hash"), "analyze-hash")


if __name__ == "__main__":
    unittest.main()
