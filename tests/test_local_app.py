import tempfile
import time
import types
import unittest
from pathlib import Path
import re

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False

from local_app.jobs import Artifact, JobManager
if FASTAPI_AVAILABLE:
    from local_app.server import create_app


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
        self.assertNotIn("../../../audio_test_files/aahatein.mp3", html)

        # At least one rewritten source should be fetchable (the fallback audio).
        rewritten_links = re.findall(r'(/local-files/[^"\\\']+aahatein\\.mp3)', html)
        self.assertTrue(rewritten_links)
        fetched = False
        for link in rewritten_links:
            r = self.client.get(link)
            if r.status_code == 200:
                fetched = True
                break
        self.assertTrue(fetched)

    def test_create_batch_job_endpoint(self) -> None:
        response = self.client.post(
            "/api/batch-jobs",
            json={
                "input_dir": self.tmp.name,
                "output_dir": "batch_results",
                "mode": "auto",
                "silent": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "batch")
        self.assertIn(payload["status"], {"queued", "running", "completed"})

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


if __name__ == "__main__":
    unittest.main()
