import tempfile
import time
import types
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
