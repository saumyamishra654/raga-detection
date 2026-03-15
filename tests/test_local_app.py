import base64
import json
import os
import tempfile
import time
import types
import unittest
from pathlib import Path
import re
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False

from local_app.jobs import Artifact, JobManager
if FASTAPI_AVAILABLE:
    from local_app.server import create_app


class _FakeRecordingSession:
    def __init__(self, target_mp3_path: str, tanpura_key: str | None = None) -> None:
        self.target_mp3_path = target_mp3_path
        self.tanpura_key = tanpura_key
        self.cancelled = False
        self.stopped = False

    def stop(self) -> str:
        self.stopped = True
        target = Path(self.target_mp3_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"recorded-bytes")
        return str(target)

    def cancel(self) -> None:
        self.cancelled = True


def wait_for_status(manager: JobManager, job_id: str, target: set[str], timeout_s: float = 5.0) -> str:
    start = time.time()
    while time.time() - start < timeout_s:
        job = manager.get(job_id)
        if job and job.status in target:
            return job.status
        time.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not reach status {target} in time.")


class JobManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo_root = Path.cwd()
        self.manager = JobManager(repo_root=self.repo_root, data_dir=Path(self.tmp.name))

    def tearDown(self) -> None:
        self.manager.shutdown()
        self.tmp.cleanup()

    def test_job_completes_with_fake_runner(self) -> None:
        artifact_path = Path(self.tmp.name) / "artifact.html"
        artifact_path.write_text("<html></html>", encoding="utf-8")

        def fake_run_job(self_obj: JobManager, job) -> list[Artifact]:
            self_obj._append_log(job, "[STEP 1/1] Fake complete")
            return [Artifact(name="artifact.html", path=str(artifact_path))]

        self.manager._run_job = types.MethodType(fake_run_job, self.manager)  # type: ignore[method-assign]
        job = self.manager.submit(mode="detect", params={}, extra_args=[])

        status = wait_for_status(self.manager, job.job_id, {"completed", "failed"})
        self.assertEqual(status, "completed")
        done = self.manager.get(job.job_id)
        assert done is not None
        self.assertGreaterEqual(done.progress, 1.0)
        self.assertTrue(done.artifacts)

    def test_cancel_queued_job(self) -> None:
        def slow_run_job(self_obj: JobManager, job) -> list[Artifact]:
            time.sleep(0.5)
            return []

        self.manager._run_job = types.MethodType(slow_run_job, self.manager)  # type: ignore[method-assign]
        first = self.manager.submit(mode="detect", params={}, extra_args=[])
        second = self.manager.submit(mode="analyze", params={}, extra_args=[])

        wait_for_status(self.manager, first.job_id, {"running", "completed"})
        cancelled = self.manager.cancel(second.job_id)
        self.assertTrue(cancelled)
        status = wait_for_status(self.manager, second.job_id, {"cancelled"})
        self.assertEqual(status, "cancelled")


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo_root = Path.cwd()
        self.manager = JobManager(repo_root=self.repo_root, data_dir=Path(self.tmp.name))

        def fake_run_job(self_obj: JobManager, job) -> list[Artifact]:
            artifact_path = Path(self.tmp.name) / f"{job.job_id}.html"
            artifact_path.write_text("<html>ok</html>", encoding="utf-8")
            self_obj._append_log(job, "[STEP 1/1] done")
            return [Artifact(name=artifact_path.name, path=str(artifact_path))]

        self.manager._run_job = types.MethodType(fake_run_job, self.manager)  # type: ignore[method-assign]
        self.app = create_app(job_manager=self.manager)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.manager.shutdown()
        self.tmp.cleanup()

    def test_schema_endpoint(self) -> None:
        response = self.client.get("/api/schema/detect")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "detect")
        self.assertTrue(payload["fields"])

    def test_app_page_includes_analyze_workspace_and_editor_script(self) -> None:
        response = self.client.get("/app")
        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertIn("view-library-btn", html)
        self.assertIn("view-classic-btn", html)
        self.assertIn("audio-library-panel", html)
        self.assertIn("library-song-table-body", html)
        self.assertIn("library-ground-truth", html)
        self.assertIn("library-batch-mode", html)
        self.assertIn("library-clear-all-btn", html)
        self.assertIn("run-config-drawer", html)
        self.assertIn("analyze-workspace-panel", html)
        self.assertIn("analyze-editor-root", html)
        self.assertIn("/static/transcription_editor.js", html)

    def test_detect_schema_includes_skip_separation(self) -> None:
        response = self.client.get("/api/schema/detect")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        fields = payload.get("fields", [])
        names = {field.get("name") for field in fields}
        self.assertIn("skip_separation", names)

    def test_submit_and_fetch_job(self) -> None:
        response = self.client.post(
            "/api/jobs",
            json={"mode": "detect", "params": {}, "extra_args": []},
        )
        self.assertEqual(response.status_code, 200)
        job_id = response.json()["job_id"]
        wait_for_status(self.manager, job_id, {"completed", "failed"})

        job_response = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(job_response.status_code, 200)
        self.assertEqual(job_response.json()["status"], "completed")

        logs_response = self.client.get(f"/api/jobs/{job_id}/logs")
        self.assertEqual(logs_response.status_code, 200)
        self.assertTrue(logs_response.json()["logs"])

    def test_audio_upload_endpoint(self) -> None:
        response = self.client.post(
            "/api/upload-audio",
            files={"audio_file": ("demo.mp3", b"abc123", "audio/mpeg")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("path", payload)
        uploaded = Path(payload["path"])
        self.assertTrue(uploaded.exists())
        self.assertEqual(uploaded.read_bytes(), b"abc123")

    def test_audio_upload_endpoint_accepts_webm(self) -> None:
        response = self.client.post(
            "/api/upload-audio",
            files={"audio_file": ("take.webm", b"webm-bytes", "audio/webm")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        uploaded = Path(payload["path"])
        self.assertTrue(uploaded.exists())
        self.assertEqual(uploaded.suffix.lower(), ".webm")
        self.assertEqual(uploaded.read_bytes(), b"webm-bytes")

    def test_preprocess_record_start_stop_endpoints(self) -> None:
        target_path = Path(self.tmp.name) / "recordings" / "take.mp3"
        captured = {}

        def fake_start(audio_dir: str, filename_base: str, tanpura_key: str | None = None) -> _FakeRecordingSession:
            captured["audio_dir"] = audio_dir
            captured["filename_base"] = filename_base
            captured["tanpura_key"] = tanpura_key
            return _FakeRecordingSession(str(target_path), tanpura_key=tanpura_key)

        with patch("local_app.server.start_microphone_recording_session", side_effect=fake_start):
            start_resp = self.client.post(
                "/api/preprocess-record/start",
                json={
                    "audio_dir": str(Path(self.tmp.name) / "recordings"),
                    "filename": "take",
                    "ingest": "tanpura_recording",
                    "tanpura_key": "C",
                },
            )
            self.assertEqual(start_resp.status_code, 200)
            start_payload = start_resp.json()
            self.assertEqual(start_payload["status"], "recording")
            self.assertEqual(start_payload["tonic"], "C")
            self.assertEqual(captured["filename_base"], "take")
            self.assertEqual(captured["tanpura_key"], "C")

            stop_resp = self.client.post("/api/preprocess-record/stop")
            self.assertEqual(stop_resp.status_code, 200)
            stop_payload = stop_resp.json()
            self.assertEqual(stop_payload["status"], "ready")
            self.assertEqual(stop_payload["path"], str(target_path.resolve()))
            self.assertEqual(stop_payload["tonic"], "C")
            self.assertIn("/local-files/", stop_payload["url"])
            self.assertTrue(Path(stop_payload["path"]).exists())

    def test_preprocess_record_start_requires_tanpura_key_for_tanpura_mode(self) -> None:
        response = self.client.post(
            "/api/preprocess-record/start",
            json={
                "audio_dir": self.tmp.name,
                "filename": "take",
                "ingest": "tanpura_recording",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("tanpura_key", response.json().get("detail", ""))

    def test_preprocess_record_start_rejects_non_recording_ingest(self) -> None:
        response = self.client.post(
            "/api/preprocess-record/start",
            json={
                "audio_dir": self.tmp.name,
                "filename": "take",
                "ingest": "yt",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("recording", response.json().get("detail", ""))

    def test_preprocess_record_start_accepts_legacy_record_mode_alias(self) -> None:
        target_path = Path(self.tmp.name) / "recordings" / "legacy_take.mp3"
        captured = {}

        def fake_start(audio_dir: str, filename_base: str, tanpura_key: str | None = None) -> _FakeRecordingSession:
            captured["tanpura_key"] = tanpura_key
            return _FakeRecordingSession(str(target_path), tanpura_key=tanpura_key)

        with patch("local_app.server.start_microphone_recording_session", side_effect=fake_start):
            response = self.client.post(
                "/api/preprocess-record/start",
                json={
                    "audio_dir": self.tmp.name,
                    "filename": "legacy_take",
                    "record_mode": "tanpura_vocal",
                    "tanpura_key": "D",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(captured["tanpura_key"], "D")

    def test_audio_files_endpoint_lists_supported_audio(self) -> None:
        audio_dir = Path(self.tmp.name) / "audio_dir"
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "trackB.mp3").write_bytes(b"1")
        (audio_dir / "trackA.wav").write_bytes(b"2")
        (audio_dir / "notes.txt").write_text("ignore", encoding="utf-8")
        (audio_dir / "nested").mkdir()

        response = self.client.get("/api/audio-files", params={"audio_dir": str(audio_dir)})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["exists"])
        self.assertEqual(payload["directory"], str(audio_dir.resolve()))
        self.assertEqual(
            [item["name"] for item in payload["files"]],
            ["trackA.wav", "trackB.mp3"],
        )

    def test_audio_files_endpoint_handles_missing_dir(self) -> None:
        missing = Path(self.tmp.name) / "does_not_exist"
        response = self.client.get("/api/audio-files", params={"audio_dir": str(missing)})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["exists"])
        self.assertEqual(payload["files"], [])

    def test_runtime_fingerprint_endpoint_returns_expected_shape(self) -> None:
        response = self.client.get("/api/runtime-fingerprint")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("fingerprint_version", payload)
        self.assertIn("stage_manifest_version", payload)
        self.assertIn("stage_hashes", payload)
        self.assertIn("detect", payload["stage_hashes"])
        self.assertIn("analyze", payload["stage_hashes"])
        self.assertIn("git_commit", payload)
        self.assertIn("git_dirty", payload)
        self.assertIn("file_hash", payload)
        self.assertIn("source", payload)
        self.assertIn("computed_at", payload)

    def test_library_endpoint_reports_current_unknown_stale_and_missing(self) -> None:
        audio_dir = Path(self.tmp.name) / "audio_lib"
        output_root = Path(self.tmp.name) / "batch_results"
        audio_dir.mkdir(parents=True, exist_ok=True)
        for name in ["song_current.mp3", "song_unknown.mp3", "song_stale.mp3", "song_missing.mp3"]:
            (audio_dir / name).write_bytes(b"audio")

        runtime_fp_resp = self.client.get("/api/runtime-fingerprint")
        self.assertEqual(runtime_fp_resp.status_code, 200)
        runtime_fp = runtime_fp_resp.json()

        current_dir = output_root / "htdemucs" / "song_current"
        current_dir.mkdir(parents=True, exist_ok=True)
        current_report = current_dir / "detection_report.html"
        current_report.write_text("<html>detect-current</html>", encoding="utf-8")
        current_report.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "pipeline_mode": "detect",
                    "generated_at": "2026-03-01T00:00:00+00:00",
                    "runtime_fingerprint": runtime_fp,
                    "stage_fingerprint": {
                        "mode": "detect",
                        "hash": runtime_fp.get("stage_hashes", {}).get("detect"),
                        "stage_manifest_version": runtime_fp.get("stage_manifest_version", 1),
                    },
                    "run_identity": {"audio_path": str((audio_dir / "song_current.mp3").resolve())},
                }
            ),
            encoding="utf-8",
        )

        unknown_dir = output_root / "htdemucs" / "song_unknown"
        unknown_dir.mkdir(parents=True, exist_ok=True)
        (unknown_dir / "analysis_report.html").write_text("<html>analyze-unknown</html>", encoding="utf-8")

        stale_dir = output_root / "htdemucs" / "song_stale"
        stale_dir.mkdir(parents=True, exist_ok=True)
        stale_report = stale_dir / "detection_report.html"
        stale_report.write_text("<html>detect-stale</html>", encoding="utf-8")
        stale_hash = "deadbeef" + str(runtime_fp.get("stage_hashes", {}).get("detect", ""))[:8]
        stale_report.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "pipeline_mode": "detect",
                    "generated_at": "2026-03-01T00:00:00+00:00",
                    "runtime_fingerprint": runtime_fp,
                    "stage_fingerprint": {
                        "mode": "detect",
                        "hash": stale_hash,
                        "stage_manifest_version": runtime_fp.get("stage_manifest_version", 1),
                    },
                    "run_identity": {"audio_path": str((audio_dir / "song_stale.mp3").resolve())},
                }
            ),
            encoding="utf-8",
        )

        response = self.client.get(
            "/api/library",
            params={
                "audio_dir": str(audio_dir),
                "output_dir": str(output_root),
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        songs = payload.get("songs", [])
        self.assertEqual(len(songs), 4)
        by_name = {item["audio_name"]: item for item in songs}
        self.assertEqual(by_name["song_current.mp3"]["detect"]["status"], "current")
        self.assertEqual(by_name["song_unknown.mp3"]["analyze"]["status"], "unknown")
        self.assertEqual(by_name["song_stale.mp3"]["detect"]["status"], "stale")
        self.assertEqual(by_name["song_missing.mp3"]["detect"]["status"], "missing")

    def test_library_variants_endpoint_returns_variant_rows(self) -> None:
        audio_dir = Path(self.tmp.name) / "audio_variants"
        output_root = Path(self.tmp.name) / "batch_results"
        audio_dir.mkdir(parents=True, exist_ok=True)
        song_path = audio_dir / "raag_demo.mp3"
        song_path.write_bytes(b"audio")

        runtime_fp_resp = self.client.get("/api/runtime-fingerprint")
        self.assertEqual(runtime_fp_resp.status_code, 200)
        runtime_fp = runtime_fp_resp.json()

        stem_a = output_root / "htdemucs" / "raag_demo"
        stem_b = output_root / "mdx" / "raag_demo"
        stem_a.mkdir(parents=True, exist_ok=True)
        stem_b.mkdir(parents=True, exist_ok=True)

        rep_a = stem_a / "detection_report.html"
        rep_a.write_text("<html>a</html>", encoding="utf-8")
        rep_a.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "pipeline_mode": "detect",
                    "generated_at": "2026-03-01T00:00:00+00:00",
                    "runtime_fingerprint": runtime_fp,
                    "stage_fingerprint": {
                        "mode": "detect",
                        "hash": runtime_fp.get("stage_hashes", {}).get("detect"),
                        "stage_manifest_version": runtime_fp.get("stage_manifest_version", 1),
                    },
                    "run_identity": {"separator": "demucs", "demucs_model": "htdemucs"},
                }
            ),
            encoding="utf-8",
        )

        rep_b = stem_b / "analysis_report.html"
        rep_b.write_text("<html>b</html>", encoding="utf-8")
        rep_b.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "pipeline_mode": "analyze",
                    "generated_at": "2026-03-01T00:01:00+00:00",
                    "runtime_fingerprint": runtime_fp,
                    "stage_fingerprint": {
                        "mode": "analyze",
                        "hash": runtime_fp.get("stage_hashes", {}).get("analyze"),
                        "stage_manifest_version": runtime_fp.get("stage_manifest_version", 1),
                    },
                    "run_identity": {"separator": "demucs", "demucs_model": "mdx", "tonic": "C", "raga": "Yaman"},
                }
            ),
            encoding="utf-8",
        )

        lib_resp = self.client.get(
            "/api/library",
            params={"audio_dir": str(audio_dir), "output_dir": str(output_root)},
        )
        self.assertEqual(lib_resp.status_code, 200, lib_resp.text)
        songs = lib_resp.json().get("songs", [])
        self.assertEqual(len(songs), 1)
        song_id = songs[0]["song_id"]

        variant_resp = self.client.get(
            f"/api/library/{song_id}/variants",
            params={"audio_dir": str(audio_dir), "output_dir": str(output_root)},
        )
        self.assertEqual(variant_resp.status_code, 200, variant_resp.text)
        variants = variant_resp.json().get("variants", [])
        self.assertEqual(len(variants), 2)
        self.assertTrue(any(v.get("demucs_model") == "htdemucs" for v in variants))
        self.assertTrue(any(v.get("demucs_model") == "mdx" for v in variants))

    def test_library_endpoint_stage_specific_current_and_stale_are_independent(self) -> None:
        audio_dir = Path(self.tmp.name) / "audio_stage_specific"
        output_root = Path(self.tmp.name) / "batch_results"
        audio_dir.mkdir(parents=True, exist_ok=True)
        song_a = audio_dir / "song_detect_current_analyze_stale.mp3"
        song_b = audio_dir / "song_detect_stale_analyze_current.mp3"
        song_a.write_bytes(b"a")
        song_b.write_bytes(b"b")

        runtime_fp_resp = self.client.get("/api/runtime-fingerprint")
        self.assertEqual(runtime_fp_resp.status_code, 200)
        runtime_fp = runtime_fp_resp.json()
        detect_hash = str(runtime_fp.get("stage_hashes", {}).get("detect") or "")
        analyze_hash = str(runtime_fp.get("stage_hashes", {}).get("analyze") or "")
        self.assertTrue(detect_hash)
        self.assertTrue(analyze_hash)

        stale_detect_hash = "stale-" + detect_hash[:12]
        stale_analyze_hash = "stale-" + analyze_hash[:12]

        stem_a = output_root / "htdemucs" / "song_detect_current_analyze_stale"
        stem_a.mkdir(parents=True, exist_ok=True)
        (stem_a / "detection_report.html").write_text("<html>detect-a</html>", encoding="utf-8")
        (stem_a / "analysis_report.html").write_text("<html>analyze-a</html>", encoding="utf-8")
        (stem_a / "detection_report.meta.json").write_text(
            json.dumps(
                {
                    "pipeline_mode": "detect",
                    "generated_at": "2026-03-01T00:00:00+00:00",
                    "runtime_fingerprint": runtime_fp,
                    "stage_fingerprint": {
                        "mode": "detect",
                        "hash": detect_hash,
                        "stage_manifest_version": runtime_fp.get("stage_manifest_version", 1),
                    },
                    "run_identity": {"audio_path": str(song_a.resolve())},
                }
            ),
            encoding="utf-8",
        )
        (stem_a / "analysis_report.meta.json").write_text(
            json.dumps(
                {
                    "pipeline_mode": "analyze",
                    "generated_at": "2026-03-01T00:00:01+00:00",
                    "runtime_fingerprint": runtime_fp,
                    "stage_fingerprint": {
                        "mode": "analyze",
                        "hash": stale_analyze_hash,
                        "stage_manifest_version": runtime_fp.get("stage_manifest_version", 1),
                    },
                    "run_identity": {"audio_path": str(song_a.resolve())},
                }
            ),
            encoding="utf-8",
        )

        stem_b = output_root / "htdemucs" / "song_detect_stale_analyze_current"
        stem_b.mkdir(parents=True, exist_ok=True)
        (stem_b / "detection_report.html").write_text("<html>detect-b</html>", encoding="utf-8")
        (stem_b / "analysis_report.html").write_text("<html>analyze-b</html>", encoding="utf-8")
        (stem_b / "detection_report.meta.json").write_text(
            json.dumps(
                {
                    "pipeline_mode": "detect",
                    "generated_at": "2026-03-01T00:00:02+00:00",
                    "runtime_fingerprint": runtime_fp,
                    "stage_fingerprint": {
                        "mode": "detect",
                        "hash": stale_detect_hash,
                        "stage_manifest_version": runtime_fp.get("stage_manifest_version", 1),
                    },
                    "run_identity": {"audio_path": str(song_b.resolve())},
                }
            ),
            encoding="utf-8",
        )
        (stem_b / "analysis_report.meta.json").write_text(
            json.dumps(
                {
                    "pipeline_mode": "analyze",
                    "generated_at": "2026-03-01T00:00:03+00:00",
                    "runtime_fingerprint": runtime_fp,
                    "stage_fingerprint": {
                        "mode": "analyze",
                        "hash": analyze_hash,
                        "stage_manifest_version": runtime_fp.get("stage_manifest_version", 1),
                    },
                    "run_identity": {"audio_path": str(song_b.resolve())},
                }
            ),
            encoding="utf-8",
        )

        response = self.client.get(
            "/api/library",
            params={"audio_dir": str(audio_dir), "output_dir": str(output_root)},
        )
        self.assertEqual(response.status_code, 200, response.text)
        songs = response.json().get("songs", [])
        self.assertEqual(len(songs), 2)
        by_name = {item["audio_name"]: item for item in songs}
        self.assertEqual(
            by_name["song_detect_current_analyze_stale.mp3"]["detect"]["status"],
            "current",
        )
        self.assertEqual(
            by_name["song_detect_current_analyze_stale.mp3"]["analyze"]["status"],
            "stale",
        )
        self.assertEqual(
            by_name["song_detect_stale_analyze_current.mp3"]["detect"]["status"],
            "stale",
        )
        self.assertEqual(
            by_name["song_detect_stale_analyze_current.mp3"]["analyze"]["status"],
            "current",
        )

    def test_library_clear_song_outputs_endpoint_removes_song_variants_and_logs(self) -> None:
        audio_dir = Path(self.tmp.name) / "audio_clear_song"
        output_root = Path(self.tmp.name) / "batch_results"
        audio_dir.mkdir(parents=True, exist_ok=True)
        song_a = audio_dir / "song_to_clear.mp3"
        song_b = audio_dir / "song_to_keep.mp3"
        song_a.write_bytes(b"a")
        song_b.write_bytes(b"b")

        stem_a_ht = output_root / "htdemucs" / "song_to_clear"
        stem_a_mdx = output_root / "mdx" / "song_to_clear"
        stem_b_ht = output_root / "htdemucs" / "song_to_keep"
        stem_a_ht.mkdir(parents=True, exist_ok=True)
        stem_a_mdx.mkdir(parents=True, exist_ok=True)
        stem_b_ht.mkdir(parents=True, exist_ok=True)
        (stem_a_ht / "analysis_report.html").write_text("<html>a</html>", encoding="utf-8")
        (stem_a_mdx / "detection_report.html").write_text("<html>b</html>", encoding="utf-8")
        (stem_b_ht / "analysis_report.html").write_text("<html>keep</html>", encoding="utf-8")

        logs_dir = output_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_a = logs_dir / "song_to_clear.mp3.log"
        log_b = logs_dir / "song_to_keep.mp3.log"
        log_a.write_text("clear me", encoding="utf-8")
        log_b.write_text("keep me", encoding="utf-8")

        lib_resp = self.client.get(
            "/api/library",
            params={"audio_dir": str(audio_dir), "output_dir": str(output_root)},
        )
        self.assertEqual(lib_resp.status_code, 200, lib_resp.text)
        songs = lib_resp.json().get("songs", [])
        by_name = {item["audio_name"]: item for item in songs}
        song_id = by_name["song_to_clear.mp3"]["song_id"]

        clear_resp = self.client.post(
            f"/api/library/{song_id}/clear-outputs",
            params={"audio_dir": str(audio_dir), "output_dir": str(output_root)},
        )
        self.assertEqual(clear_resp.status_code, 200, clear_resp.text)
        payload = clear_resp.json()
        self.assertTrue(payload.get("ok", False))
        self.assertGreaterEqual(int(payload.get("deleted_files", 0)), 3)
        self.assertGreaterEqual(int(payload.get("deleted_dirs", 0)), 2)

        self.assertFalse(stem_a_ht.exists())
        self.assertFalse(stem_a_mdx.exists())
        self.assertTrue(stem_b_ht.exists())
        self.assertFalse(log_a.exists())
        self.assertTrue(log_b.exists())

    def test_library_clear_song_outputs_endpoint_returns_404_for_unknown_song(self) -> None:
        audio_dir = Path(self.tmp.name) / "audio_missing_song"
        output_root = Path(self.tmp.name) / "batch_results"
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "known.mp3").write_bytes(b"k")
        output_root.mkdir(parents=True, exist_ok=True)

        response = self.client.post(
            "/api/library/not-a-real-song/clear-outputs",
            params={"audio_dir": str(audio_dir), "output_dir": str(output_root)},
        )
        self.assertEqual(response.status_code, 404)

    def test_library_clear_all_outputs_preserves_stems_and_pitch_csvs_only(self) -> None:
        output_root = Path(self.tmp.name) / "batch_results"
        stem_dir = output_root / "htdemucs" / "yaman_demo"
        logs_dir = output_root / "logs"
        stem_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        keep_vocals = stem_dir / "vocals.mp3"
        keep_pitch = stem_dir / "composite_pitch_data.csv"
        keep_root_melody = output_root / "melody.mp3"
        drop_report = stem_dir / "analysis_report.html"
        drop_candidates = stem_dir / "candidates.csv"
        drop_log = logs_dir / "yaman_demo.mp3.log"

        keep_vocals.write_bytes(b"vocals")
        keep_pitch.write_text("ts,pitch\n0.0,220.0\n", encoding="utf-8")
        keep_root_melody.write_bytes(b"melody")
        drop_report.write_text("<html>drop</html>", encoding="utf-8")
        drop_candidates.write_text("raga,score\nYaman,0.9\n", encoding="utf-8")
        drop_log.write_text("drop log", encoding="utf-8")

        response = self.client.post(
            "/api/library/clear-all-outputs",
            params={"output_dir": str(output_root)},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload.get("ok", False))
        self.assertGreaterEqual(int(payload.get("deleted_files", 0)), 3)
        self.assertGreaterEqual(int(payload.get("preserved_files", 0)), 3)

        self.assertTrue(output_root.exists())
        self.assertTrue(keep_vocals.exists())
        self.assertTrue(keep_pitch.exists())
        self.assertTrue(keep_root_melody.exists())
        self.assertFalse(drop_report.exists())
        self.assertFalse(drop_candidates.exists())
        self.assertFalse(drop_log.exists())

    def test_tanpura_tracks_endpoint_returns_catalog(self) -> None:
        response = self.client.get("/api/tanpura-tracks")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        tracks = payload.get("tracks", [])
        self.assertEqual(len(tracks), 12)
        keys = [item.get("key") for item in tracks]
        self.assertEqual(keys, ["A", "Bb", "B", "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab"])
        for item in tracks:
            self.assertTrue(item.get("url"))
            self.assertIn("/local-files/", item["url"])

    def test_preprocess_job_accepts_recorded_audio_payload(self) -> None:
        recorded_path = Path(self.tmp.name) / "recorded.webm"
        recorded_path.write_bytes(b"abc123")
        response = self.client.post(
            "/api/jobs",
            json={
                "mode": "preprocess",
                "params": {
                    "ingest": "recording",
                    "recorded_audio": str(recorded_path),
                    "audio_dir": self.tmp.name,
                    "filename": "my_take",
                    "output": "batch_results",
                },
                "extra_args": [],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "preprocess")
        self.assertIn(payload["status"], {"queued", "running", "completed"})

    def test_preprocess_job_accepts_legacy_record_mode_payload(self) -> None:
        recorded_path = Path(self.tmp.name) / "recorded_legacy.webm"
        recorded_path.write_bytes(b"legacy")
        response = self.client.post(
            "/api/jobs",
            json={
                "mode": "preprocess",
                "params": {
                    "ingest": "record",
                    "record_mode": "song",
                    "recorded_audio": str(recorded_path),
                    "audio_dir": self.tmp.name,
                    "filename": "legacy_take",
                    "output": "batch_results",
                },
                "extra_args": [],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "preprocess")
        self.assertIn(payload["status"], {"queued", "running", "completed"})

    def test_audio_artifacts_endpoint_finds_reports_by_audio_stem(self) -> None:
        output_root = Path(self.tmp.name) / "batch_results"
        stem_dir = output_root / "htdemucs" / "bhimpalasi"
        stem_dir.mkdir(parents=True, exist_ok=True)
        detect_path = stem_dir / "detection_report.html"
        analyze_path = stem_dir / "analysis_report.html"
        detect_path.write_text("<html>detect</html>", encoding="utf-8")
        analyze_path.write_text("<html>analyze</html>", encoding="utf-8")

        response = self.client.get(
            "/api/audio-artifacts",
            params={
                "audio_path": "/tmp/audio_test_files/bhimpalasi.mp3",
                "output_dir": str(output_root),
                "separator": "demucs",
                "demucs_model": "htdemucs",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["found"])
        self.assertEqual(Path(payload["stem_dir"]), stem_dir.resolve())
        self.assertTrue(payload["detect_report_url"])
        self.assertTrue(payload["analyze_report_url"])
        self.assertIsInstance(payload["analyze_report_context"], dict)
        self.assertEqual(payload["analyze_report_context"]["report_name"], "analysis_report.html")
        self.assertTrue(payload["analyze_report_context"]["dir_token"])
        self.assertTrue(payload["analyze_report_context"]["url"])

        detect_resp = self.client.get(payload["detect_report_url"])
        self.assertEqual(detect_resp.status_code, 200)
        self.assertIn("detect", detect_resp.text)
        self.assertIn("/local-report/", payload["detect_report_url"])

    def test_local_report_rewrites_relative_audio_sources(self) -> None:
        output_root = Path(self.tmp.name) / "batch_results"
        stem_dir = output_root / "htdemucs" / "aahatein"
        stem_dir.mkdir(parents=True, exist_ok=True)

        # First source is missing in stem dir; second source exists outside stem dir.
        fallback_audio = Path(self.tmp.name) / "audio_test_files" / "aahatein.mp3"
        fallback_audio.parent.mkdir(parents=True, exist_ok=True)
        fallback_audio.write_bytes(b"abc")
        img_path = stem_dir / "histogram_melody.png"
        img_path.write_bytes(b"png")

        report_path = stem_dir / "detection_report.html"
        report_path.write_text(
            """
            <html><body>
              <audio controls>
                <source src="aahatein.mp3" type="audio/mpeg">
                <source src="../../../audio_test_files/aahatein.mp3" type="audio/mpeg">
              </audio>
              <img src="histogram_melody.png">
            </body></html>
            """,
            encoding="utf-8",
        )

        response = self.client.get(
            "/api/audio-artifacts",
            params={
                "audio_path": str(fallback_audio),
                "output_dir": str(output_root),
                "separator": "demucs",
                "demucs_model": "htdemucs",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["found"])

        report_url = payload["detect_report_url"]
        self.assertTrue(report_url)
        self.assertIn("/local-report/", report_url)

        html_resp = self.client.get(report_url)
        self.assertEqual(html_resp.status_code, 200)
        html = html_resp.text
        self.assertIn("/local-files/", html)
        self.assertNotIn('src="aahatein.mp3"', html)
        self.assertNotIn("../../../audio_test_files/aahatein.mp3", html)

        # At least one rewritten source should be fetchable (the fallback audio).
        rewritten_links = re.findall(r'(/local-files/[^"\'> ]+)', html)
        self.assertTrue(rewritten_links)
        fetched = False
        for link in rewritten_links:
            r = self.client.get(link)
            if r.status_code == 200 and r.content == b"abc":
                fetched = True
                break
        self.assertTrue(fetched)

    def test_local_report_skips_ambiguous_basename_audio_fallback(self) -> None:
        output_root = Path(self.tmp.name) / "batch_results"
        stem_dir = output_root / "htdemucs" / "dup_song"
        stem_dir.mkdir(parents=True, exist_ok=True)

        audio_a = Path(self.tmp.name) / "audio_a" / "dup_song.mp3"
        audio_b = Path(self.tmp.name) / "audio_b" / "dup_song.mp3"
        audio_a.parent.mkdir(parents=True, exist_ok=True)
        audio_b.parent.mkdir(parents=True, exist_ok=True)
        audio_a.write_bytes(b"aaa")
        audio_b.write_bytes(b"bbb")

        report_path = stem_dir / "detection_report.html"
        report_path.write_text(
            """
            <html><body>
              <audio controls>
                <source src="dup_song.mp3" type="audio/mpeg">
                <source src="../../../audio_a/dup_song.mp3" type="audio/mpeg">
                <source src="../../../audio_b/dup_song.mp3" type="audio/mpeg">
              </audio>
            </body></html>
            """,
            encoding="utf-8",
        )

        response = self.client.get(
            "/api/audio-artifacts",
            params={
                "audio_path": str(audio_a),
                "output_dir": str(output_root),
                "separator": "demucs",
                "demucs_model": "htdemucs",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["found"])

        report_url = payload["detect_report_url"]
        self.assertTrue(report_url)

        html_resp = self.client.get(report_url)
        self.assertEqual(html_resp.status_code, 200)
        html = html_resp.text

        # Ambiguous basename fallback should not rewrite the unresolved sibling path.
        self.assertIn('src="dup_song.mp3"', html)

        rewritten_links = re.findall(r'(/local-files/[^"\'> ]+)', html)
        self.assertGreaterEqual(len(rewritten_links), 2)
        resolved_payloads = {self.client.get(link).content for link in rewritten_links}
        self.assertIn(b"aaa", resolved_payloads)
        self.assertIn(b"bbb", resolved_payloads)

    def test_create_batch_job_endpoint_accepts_detect_and_analyze_modes(self) -> None:
        ground_truth_csv = Path(self.tmp.name) / "ground_truth_v6.csv"
        ground_truth_csv.write_text("filename,raga,tonic\n", encoding="utf-8")

        detect_response = self.client.post(
            "/api/batch-jobs",
            json={
                "input_dir": self.tmp.name,
                "output_dir": "batch_results",
                "mode": "detect",
                "silent": True,
            },
        )
        self.assertEqual(detect_response.status_code, 200)
        detect_payload = detect_response.json()
        self.assertEqual(detect_payload["mode"], "batch")
        self.assertIn(detect_payload["status"], {"queued", "running", "completed"})
        job = self.manager.get(detect_payload["job_id"])
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job.params.get("batch_mode"), "detect")

        analyze_response = self.client.post(
            "/api/batch-jobs",
            json={
                "input_dir": self.tmp.name,
                "output_dir": "batch_results",
                "mode": "analyze",
                "ground_truth": str(ground_truth_csv),
                "silent": True,
            },
        )
        self.assertEqual(analyze_response.status_code, 200)
        analyze_payload = analyze_response.json()
        self.assertEqual(analyze_payload["mode"], "batch")
        self.assertIn(analyze_payload["status"], {"queued", "running", "completed"})
        job = self.manager.get(analyze_payload["job_id"])
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job.params.get("batch_mode"), "analyze")
        self.assertEqual(job.params.get("ground_truth"), str(ground_truth_csv))

    def test_create_batch_job_endpoint_rejects_auto_mode(self) -> None:
        response = self.client.post(
            "/api/batch-jobs",
            json={
                "input_dir": self.tmp.name,
                "output_dir": "batch_results",
                "mode": "auto",
                "silent": True,
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("detect", response.json().get("detail", ""))

    def test_create_batch_job_endpoint_requires_ground_truth_for_analyze(self) -> None:
        response = self.client.post(
            "/api/batch-jobs",
            json={
                "input_dir": self.tmp.name,
                "output_dir": "batch_results",
                "mode": "analyze",
                "silent": True,
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("ground_truth", response.json().get("detail", ""))

    def test_raga_list_endpoint_reads_names_column(self) -> None:
        csv_path = Path(self.tmp.name) / "ragas.csv"
        csv_path.write_text(
            "0,1,names\n1,0,\"Bhairavi, Sindhu Bhairavi\"\n0,1,Yaman\n1,1,bhairavi\n",
            encoding="utf-8",
        )
        response = self.client.get("/api/ragas", params={"raga_db": str(csv_path)})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source"], str(csv_path.resolve()))
        self.assertEqual(payload["ragas"], ["Bhairavi", "Sindhu Bhairavi", "Yaman"])

    def test_raga_list_endpoint_splits_unquoted_extra_columns(self) -> None:
        csv_path = Path(self.tmp.name) / "ragas_extras.csv"
        csv_path.write_text(
            "0,1,names\n1,0,Bageshri,Shudh Bageshri\n",
            encoding="utf-8",
        )
        response = self.client.get("/api/ragas", params={"raga_db": str(csv_path)})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ragas"], ["Bageshri", "Shudh Bageshri"])

    def test_regenerate_without_metadata_recovers_audio_and_header_from_report_html(self) -> None:
        stem_dir = Path(self.tmp.name) / "batch_results" / "htdemucs" / "demo_song"
        stem_dir.mkdir(parents=True, exist_ok=True)
        external_audio_dir = Path(self.tmp.name) / "audio_test_files"
        external_audio_dir.mkdir(parents=True, exist_ok=True)

        original_audio = external_audio_dir / "demo_song.mp3"
        vocals_audio = stem_dir / "vocals.mp3"
        accomp_audio = stem_dir / "accompaniment.mp3"
        original_audio.write_bytes(b"orig")
        vocals_audio.write_bytes(b"voc")
        accomp_audio.write_bytes(b"acc")

        original_rel = os.path.relpath(str(original_audio.resolve()), str(stem_dir.resolve()))
        report_html = f"""
        <html>
            <head><title>Raga Analysis Report: Yaman</title></head>
            <body>
                <p class="subtitle">Yaman (Tonic: C#)</p>
                <audio id="original-player" controls>
                    <source src="{original_rel}" type="audio/mpeg">
                </audio>
                <audio id="vocals-player" controls>
                    <source src="vocals.mp3" type="audio/mpeg">
                </audio>
                <audio id="accomp-player" controls>
                    <source src="accompaniment.mp3" type="audio/mpeg">
                </audio>
            </body>
        </html>
        """
        report_path = stem_dir / "analysis_report.html"
        report_path.write_text(report_html, encoding="utf-8")

        for img_name in ["transition_matrix.png", "pitch_sargam.png"]:
            (stem_dir / img_name).write_bytes(b"png")

        pitch_csv = (
            "timestamp,pitch_hz,confidence,voicing,energy\n"
            "0.0,261.63,0.95,1,0.10\n"
            "0.1,261.63,0.95,1,0.20\n"
            "0.2,293.66,0.95,1,0.25\n"
            "0.3,293.66,0.95,1,0.22\n"
        )
        (stem_dir / "composite_pitch_data.csv").write_text(pitch_csv, encoding="utf-8")
        (stem_dir / "melody_pitch_data.csv").write_text(pitch_csv, encoding="utf-8")
        (stem_dir / "accompaniment_pitch_data.csv").write_text(pitch_csv, encoding="utf-8")

        token = base64.urlsafe_b64encode(str(stem_dir.resolve()).encode("utf-8")).decode("ascii").rstrip("=")
        report_name = "analysis_report.html"
        save_payload = {
            "notes": [
                {
                    "id": "n00001",
                    "start": 0.10,
                    "end": 0.35,
                    "pitch_midi": 60.0,
                    "pitch_hz": 261.63,
                    "confidence": 0.95,
                    "energy": 0.2,
                    "sargam": "Sa",
                    "pitch_class": 0,
                },
                {
                    "id": "n00002",
                    "start": 0.40,
                    "end": 0.62,
                    "pitch_midi": 62.0,
                    "pitch_hz": 293.66,
                    "confidence": 0.95,
                    "energy": 0.3,
                    "sargam": "Re",
                    "pitch_class": 2,
                },
            ],
            "phrases": [
                {
                    "id": "p0001",
                    "start": 0.10,
                    "end": 0.62,
                    "note_ids": ["n00001", "n00002"],
                }
            ],
        }

        save_resp = self.client.post(
            f"/api/transcription-edits/{token}/{report_name}/save",
            json=save_payload,
        )
        self.assertEqual(save_resp.status_code, 200, save_resp.text)
        save_data = save_resp.json()
        self.assertEqual(save_data["version"]["version_id"], "edited")

        regenerate_resp = self.client.post(
            f"/api/transcription-edits/{token}/{report_name}/version/v0001/regenerate"
        )
        self.assertEqual(regenerate_resp.status_code, 200, regenerate_resp.text)
        regenerate_data = regenerate_resp.json()
        self.assertEqual(regenerate_data["version"]["version_id"], "edited")
        report_resp = self.client.get(regenerate_data["version"]["report_url"])
        self.assertEqual(report_resp.status_code, 200)
        self.assertIn("Yaman (Tonic: C#)", report_resp.text)

        original_audio_block = re.search(
            r'<audio id="original-player"[^>]*>(.*?)</audio>',
            report_resp.text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(original_audio_block)
        self.assertIn("/local-files/", original_audio_block.group(1))

    def test_transcription_edit_endpoints_save_and_load_versions(self) -> None:
        stem_dir = Path(self.tmp.name) / "batch_results" / "htdemucs" / "demo_song"
        stem_dir.mkdir(parents=True, exist_ok=True)

        report_path = stem_dir / "analysis_report.html"
        report_path.write_text("<html><body>base report</body></html>", encoding="utf-8")

        original_audio = stem_dir / "demo_song.mp3"
        vocals_audio = stem_dir / "vocals.mp3"
        accomp_audio = stem_dir / "accompaniment.mp3"
        original_audio.write_bytes(b"orig")
        vocals_audio.write_bytes(b"voc")
        accomp_audio.write_bytes(b"acc")
        original_audio_rel = os.path.relpath(str(original_audio.resolve()), str(self.repo_root.resolve()))
        vocals_audio_rel = os.path.relpath(str(vocals_audio.resolve()), str(self.repo_root.resolve()))
        accomp_audio_rel = os.path.relpath(str(accomp_audio.resolve()), str(self.repo_root.resolve()))

        for img_name in ["transition_matrix.png", "pitch_sargam.png", "note_duration_histogram.png"]:
            (stem_dir / img_name).write_bytes(b"png")

        pitch_csv = (
            "timestamp,pitch_hz,confidence,voicing,energy\n"
            "0.0,261.63,0.95,1,0.10\n"
            "0.1,261.63,0.95,1,0.20\n"
            "0.2,293.66,0.95,1,0.25\n"
            "0.3,293.66,0.95,1,0.22\n"
        )
        (stem_dir / "composite_pitch_data.csv").write_text(pitch_csv, encoding="utf-8")
        (stem_dir / "melody_pitch_data.csv").write_text(pitch_csv, encoding="utf-8")
        (stem_dir / "accompaniment_pitch_data.csv").write_text(pitch_csv, encoding="utf-8")

        metadata = {
            "schema_version": 1,
            "report_filename": report_path.name,
            "config": {
                "audio_path": original_audio_rel,
                "vocals_path": vocals_audio_rel,
                "accompaniment_path": accomp_audio_rel,
                "melody_source": "separated",
                "transcription_smoothing_ms": 70.0,
                "transcription_min_duration": 0.04,
                "transcription_derivative_threshold": 2.0,
                "energy_threshold": 0.0,
                "show_rms_overlay": True,
            },
            "detected": {
                "tonic": 0,
                "raga": "Bhairavi",
            },
            "stats": {
                "correction_summary": {},
                "pattern_analysis": {},
                "raga_name": "Bhairavi",
                "tonic": "C",
                "transition_matrix_path": "transition_matrix.png",
                "pitch_plot_path": "pitch_sargam.png",
            },
            "plot_paths": {
                "note_duration_histogram": "note_duration_histogram.png",
            },
            "pitch_csv_paths": {
                "original": ["composite_pitch_data.csv"],
                "vocals": ["melody_pitch_data.csv"],
                "accompaniment": ["accompaniment_pitch_data.csv"],
            },
            "transcription_edit_payload": {
                "tonic": 0,
                "notes": [
                    {
                        "id": "n00001",
                        "start": 0.10,
                        "end": 0.35,
                        "pitch_midi": 60.0,
                        "pitch_hz": 261.63,
                        "confidence": 0.95,
                        "energy": 0.2,
                        "sargam": "Sa",
                        "pitch_class": 0,
                    },
                    {
                        "id": "n00002",
                        "start": 0.40,
                        "end": 0.62,
                        "pitch_midi": 62.0,
                        "pitch_hz": 293.66,
                        "confidence": 0.95,
                        "energy": 0.3,
                        "sargam": "Re",
                        "pitch_class": 2,
                    },
                ],
                "phrases": [
                    {
                        "id": "p0001",
                        "start": 0.10,
                        "end": 0.62,
                        "note_ids": ["n00001", "n00002"],
                    }
                ],
                "sargam_options": [
                    {"offset": 0, "label": "Sa", "midi": 60},
                    {"offset": 2, "label": "Re", "midi": 62},
                ],
            },
        }
        report_path.with_suffix(".meta.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        token = base64.urlsafe_b64encode(str(stem_dir.resolve()).encode("utf-8")).decode("ascii").rstrip("=")
        report_name = "analysis_report.html"
        save_payload = {
            "notes": [
                {
                    "id": "n00001",
                    "start": 0.10,
                    "end": 0.35,
                    "pitch_midi": 60.0,
                    "pitch_hz": 261.63,
                    "confidence": 0.95,
                    "energy": 0.2,
                    "sargam": "Sa",
                    "pitch_class": 0,
                },
                {
                    "id": "n00002",
                    "start": 0.40,
                    "end": 0.62,
                    "pitch_midi": 62.0,
                    "pitch_hz": 293.66,
                    "confidence": 0.95,
                    "energy": 0.3,
                    "sargam": "Re",
                    "pitch_class": 2,
                },
            ],
            "phrases": [
                {
                    "id": "p0001",
                    "start": 0.10,
                    "end": 0.62,
                    "note_ids": ["n00001", "n00002"],
                }
            ],
        }

        base_payload_resp = self.client.get(
            f"/api/transcription-edits/{token}/{report_name}/base",
        )
        self.assertEqual(base_payload_resp.status_code, 200, base_payload_resp.text)
        base_payload_data = base_payload_resp.json()
        self.assertTrue(base_payload_data["ready"])
        self.assertFalse(base_payload_data["requires_rerun"])
        self.assertEqual(base_payload_data["tonic"], 0)
        self.assertEqual(len(base_payload_data["payload"]["notes"]), 2)

        save_resp = self.client.post(
            f"/api/transcription-edits/{token}/{report_name}/save",
            json=save_payload,
        )
        self.assertEqual(save_resp.status_code, 200, save_resp.text)
        save_data = save_resp.json()
        self.assertEqual(save_data["save_mode"], "created")
        self.assertEqual(save_data["version"]["version_id"], "edited")
        self.assertEqual(save_data["version"]["note_count"], 2)
        self.assertEqual(save_data["version"]["phrase_count"], 1)
        self.assertTrue(save_data["version"]["report_url"])
        self.assertEqual(save_data["default_selection"], "original")
        self.assertTrue(save_data["default_report_url"])

        edit_root = stem_dir / "transcription_edits"
        self.assertTrue((edit_root / "transcription_edited.json").exists())
        self.assertTrue((edit_root / "transcription_edited.csv").exists())
        self.assertTrue((stem_dir / "analysis_report_edited.html").exists())

        versions_resp = self.client.get(f"/api/transcription-edits/{token}/{report_name}/versions")
        self.assertEqual(versions_resp.status_code, 200)
        versions_data = versions_resp.json()
        self.assertEqual(versions_data["latest_version_id"], "edited")
        self.assertEqual(len(versions_data["versions"]), 1)
        self.assertEqual(versions_data["default_selection"], "original")
        self.assertTrue(versions_data["default_report_url"])
        self.assertEqual(versions_data["versions"][0]["version_id"], "edited")
        self.assertTrue(versions_data["versions"][0]["report_url"])

        set_default_v1_resp = self.client.post(
            f"/api/transcription-edits/{token}/{report_name}/default",
            params={"default_selection": "v0001"},
        )
        self.assertEqual(set_default_v1_resp.status_code, 200, set_default_v1_resp.text)
        set_default_v1_data = set_default_v1_resp.json()
        self.assertEqual(set_default_v1_data["default_selection"], "edited")
        self.assertTrue(set_default_v1_data["default_report_url"])
        self.assertIn("/local-report/", set_default_v1_data["default_report_url"])

        defaulted_source_report_resp = self.client.get(f"/local-report/{token}/{report_name}")
        self.assertEqual(defaulted_source_report_resp.status_code, 200)
        self.assertNotIn("base report", defaulted_source_report_resp.text)

        set_default_original_resp = self.client.post(
            f"/api/transcription-edits/{token}/{report_name}/default",
            params={"default_selection": "original"},
        )
        self.assertEqual(set_default_original_resp.status_code, 200, set_default_original_resp.text)
        set_default_original_data = set_default_original_resp.json()
        self.assertEqual(set_default_original_data["default_selection"], "original")
        source_report_resp = self.client.get(f"/local-report/{token}/{report_name}")
        self.assertEqual(source_report_resp.status_code, 200)
        self.assertIn("base report", source_report_resp.text)

        updated_payload = json.loads(json.dumps(save_payload))
        updated_payload["notes"][0]["end"] = 0.48
        update_resp = self.client.post(
            f"/api/transcription-edits/{token}/{report_name}/save",
            params={"target_version_id": "v0001"},
            json=updated_payload,
        )
        self.assertEqual(update_resp.status_code, 200, update_resp.text)
        update_data = update_resp.json()
        self.assertEqual(update_data["save_mode"], "updated")
        self.assertEqual(update_data["version"]["version_id"], "edited")
        self.assertEqual(len(update_data["versions"]), 1)

        version_after_update_resp = self.client.get(
            f"/api/transcription-edits/{token}/{report_name}/version/v0001"
        )
        self.assertEqual(version_after_update_resp.status_code, 200)
        version_after_update = version_after_update_resp.json()
        self.assertEqual(version_after_update["payload"]["notes"][0]["end"], 0.48)

        regenerate_resp = self.client.post(
            f"/api/transcription-edits/{token}/{report_name}/version/v0001/regenerate"
        )
        self.assertEqual(regenerate_resp.status_code, 200, regenerate_resp.text)
        regenerate_data = regenerate_resp.json()
        self.assertTrue(regenerate_data["has_version"])
        self.assertEqual(regenerate_data["version"]["version_id"], "edited")
        self.assertTrue(regenerate_data["version"]["report_url"])
        self.assertEqual(regenerate_data["payload"]["notes"][0]["end"], 0.48)
        edited_report_url = regenerate_data["version"]["report_url"]
        regenerated_report_resp = self.client.get(edited_report_url)
        self.assertEqual(regenerated_report_resp.status_code, 200)
        original_audio_block = re.search(
            r'<audio id="original-player"[^>]*>(.*?)</audio>',
            regenerated_report_resp.text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(original_audio_block)
        self.assertIn("/local-files/", original_audio_block.group(1))

        latest_resp = self.client.get(f"/api/transcription-edits/{token}/{report_name}/latest")
        self.assertEqual(latest_resp.status_code, 200)
        latest_data = latest_resp.json()
        self.assertTrue(latest_data["has_version"])
        self.assertEqual(latest_data["version"]["version_id"], "edited")
        self.assertEqual(len(latest_data["payload"]["notes"]), 2)

        version_resp = self.client.get(
            f"/api/transcription-edits/{token}/{report_name}/version/v0001"
        )
        self.assertEqual(version_resp.status_code, 200)
        version_data = version_resp.json()
        self.assertTrue(version_data["has_version"])
        self.assertEqual(version_data["version"]["version_id"], "edited")

        self.assertNotIn("Transcription Editor (Experimental)", regenerated_report_resp.text)
        self.assertNotIn("/api/transcription-edits/", regenerated_report_resp.text)

        delete_v1_resp = self.client.delete(
            f"/api/transcription-edits/{token}/{report_name}/version/v0001"
        )
        self.assertEqual(delete_v1_resp.status_code, 200, delete_v1_resp.text)
        delete_v1_data = delete_v1_resp.json()
        self.assertIsNone(delete_v1_data["latest_version_id"])
        self.assertEqual(len(delete_v1_data["versions"]), 0)
        self.assertEqual(delete_v1_data["default_selection"], "original")
        self.assertFalse((edit_root / "transcription_edited.json").exists())
        self.assertFalse((stem_dir / "analysis_report_edited.html").exists())

    def test_transcription_edit_base_requires_rerun_for_legacy_report(self) -> None:
        stem_dir = Path(self.tmp.name) / "batch_results" / "htdemucs" / "legacy_song"
        stem_dir.mkdir(parents=True, exist_ok=True)
        report_path = stem_dir / "analysis_report.html"
        report_path.write_text("<html><body>legacy report</body></html>", encoding="utf-8")

        token = base64.urlsafe_b64encode(str(stem_dir.resolve()).encode("utf-8")).decode("ascii").rstrip("=")
        report_name = "analysis_report.html"
        response = self.client.get(f"/api/transcription-edits/{token}/{report_name}/base")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["ready"])
        self.assertTrue(payload["requires_rerun"])
        self.assertIsNone(payload["payload"])
        self.assertIn("Rerun analyze", payload["detail"])

    def test_transcription_edit_base_does_not_fallback_to_transcribed_notes_csv(self) -> None:
        stem_dir = Path(self.tmp.name) / "batch_results" / "htdemucs" / "legacy_song_csv_only"
        stem_dir.mkdir(parents=True, exist_ok=True)
        report_path = stem_dir / "analysis_report.html"
        report_path.write_text("<html><body>legacy report</body></html>", encoding="utf-8")
        (stem_dir / "transcribed_notes.csv").write_text(
            "id,start,end,pitch_midi,pitch_hz,confidence,energy,sargam,pitch_class\n"
            "n00001,0.10,0.35,60.0,261.63,0.95,0.20,Sa,0\n",
            encoding="utf-8",
        )

        token = base64.urlsafe_b64encode(str(stem_dir.resolve()).encode("utf-8")).decode("ascii").rstrip("=")
        report_name = "analysis_report.html"
        response = self.client.get(f"/api/transcription-edits/{token}/{report_name}/base")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["ready"])
        self.assertTrue(payload["requires_rerun"])
        self.assertIsNone(payload["payload"])
        self.assertIn("Rerun analyze", payload["detail"])

    def test_local_report_keeps_edited_when_default_edit_selected(self) -> None:
        stem_dir = Path(self.tmp.name) / "batch_results" / "htdemucs" / "stale_case"
        stem_dir.mkdir(parents=True, exist_ok=True)

        report_path = stem_dir / "analysis_report.html"
        report_path.write_text("<html><body>source-v1</body></html>", encoding="utf-8")

        edited_report = stem_dir / "analysis_report_edited.html"
        edited_report.write_text("<html><body>edited-v1</body></html>", encoding="utf-8")

        edit_root = stem_dir / "transcription_edits"
        edit_root.mkdir(parents=True, exist_ok=True)
        (edit_root / "transcription_edited.json").write_text(
            json.dumps({"notes": [], "phrases": []}),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "source_report": "analysis_report.html",
            "default_selection": "v0001",
            "versions": [
                {
                    "version_id": "v0001",
                    "created_at": "2026-02-23T00:00:00+00:00",
                    "updated_at": "2026-02-23T00:00:00+00:00",
                    "json_file": "transcription_edited.json",
                    "csv_file": "transcription_edited.csv",
                    "report_file": edited_report.name,
                    "note_count": 1,
                    "phrase_count": 1,
                }
            ],
        }
        (edit_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        token = base64.urlsafe_b64encode(str(stem_dir.resolve()).encode("utf-8")).decode("ascii").rstrip("=")
        report_name = "analysis_report.html"

        before_refresh = self.client.get(f"/local-report/{token}/{report_name}")
        self.assertEqual(before_refresh.status_code, 200)
        self.assertIn("edited-v1", before_refresh.text)

        time.sleep(1.1)
        report_path.write_text("<html><body>source-v2</body></html>", encoding="utf-8")

        after_refresh = self.client.get(f"/local-report/{token}/{report_name}")
        self.assertEqual(after_refresh.status_code, 200)
        self.assertIn("edited-v1", after_refresh.text)

        versions_resp = self.client.get(f"/api/transcription-edits/{token}/{report_name}/versions")
        self.assertEqual(versions_resp.status_code, 200, versions_resp.text)
        versions_data = versions_resp.json()
        self.assertEqual(versions_data["default_selection"], "edited")
        self.assertEqual(versions_data["latest_version_id"], "edited")
        self.assertEqual(versions_data["versions"][0]["version_id"], "edited")


if __name__ == "__main__":
    unittest.main()
