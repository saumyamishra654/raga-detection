from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from raga_pipeline.cli_args import params_to_argv
from raga_pipeline.config import parse_config_from_argv


STEP_RE = re.compile(r"\[STEP\s+(\d+)\s*/\s*(\d+)\]")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Artifact:
    name: str
    path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "exists": os.path.exists(self.path),
        }


@dataclass
class JobRecord:
    job_id: str
    mode: str
    params: Dict[str, Any]
    extra_args: List[str]
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    cancel_requested: bool = False
    created_at: str = field(default_factory=_utc_now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    argv: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "mode": self.mode,
            "params": self.params,
            "extra_args": self.extra_args,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "cancel_requested": self.cancel_requested,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "argv": self.argv,
            "logs": self.logs,
            "artifacts": [item.to_dict() for item in self.artifacts],
        }


class _LogWriter:
    """Stream-like adapter used for redirect_stdout/redirect_stderr."""

    def __init__(self, on_line: Callable[[str], None]) -> None:
        self._on_line = on_line
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._on_line(line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._on_line(self._buffer.rstrip())
        self._buffer = ""


class JobManager:
    """
    Single-worker local job queue for pipeline runs.

    Jobs are executed serially in a background thread.
    """

    def __init__(self, repo_root: Path, data_dir: Optional[Path] = None) -> None:
        self.repo_root = repo_root
        self.data_dir = data_dir or (repo_root / ".local_app_data")
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._queue: Deque[str] = deque()
        self._jobs: Dict[str, JobRecord] = {}
        self._stop = False

        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="local-app-job-worker")
        self._worker.start()

    def submit(self, mode: str, params: Dict[str, Any], extra_args: List[str]) -> JobRecord:
        job_id = uuid.uuid4().hex
        job = JobRecord(
            job_id=job_id,
            mode=mode,
            params=dict(params),
            extra_args=list(extra_args),
        )
        with self._cv:
            self._jobs[job_id] = job
            self._queue.append(job_id)
            self._persist_locked(job)
            self._cv.notify()
        return job

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_logs(self, job_id: str) -> List[str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return []
            return list(job.logs)

    def cancel(self, job_id: str) -> bool:
        with self._cv:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if job.status == "queued":
                job.status = "cancelled"
                job.message = "Cancelled while queued."
                job.finished_at = _utc_now_iso()
                # Remove from queue if still present.
                self._queue = deque(item for item in self._queue if item != job_id)
                self._persist_locked(job)
                return True

            if job.status == "running":
                job.cancel_requested = True
                job.message = "Cancellation requested. Running task will finish current execution."
                self._persist_locked(job)
                return True

            return False

    def shutdown(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._worker.join(timeout=2.0)

    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stop:
                    self._cv.wait(timeout=0.5)
                if self._stop:
                    return
                job_id = self._queue.popleft()
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                if job.status == "cancelled":
                    continue
                job.status = "running"
                job.started_at = _utc_now_iso()
                job.message = "Running"
                job.progress = max(job.progress, 0.01)
                self._persist_locked(job)

            try:
                artifacts = self._run_job(job)
                with self._lock:
                    if job.cancel_requested:
                        job.message = "Cancellation requested while running; job completed current execution."
                    else:
                        job.message = "Completed"
                    job.status = "completed"
                    job.progress = 1.0
                    job.artifacts = artifacts
                    job.finished_at = _utc_now_iso()
                    self._persist_locked(job)
            except Exception as exc:
                with self._lock:
                    job.status = "failed"
                    job.message = "Failed"
                    job.error = str(exc)
                    job.finished_at = _utc_now_iso()
                    self._persist_locked(job)

    def _append_log(self, job: JobRecord, line: str) -> None:
        clean_line = line.rstrip()
        if not clean_line:
            return
        with self._lock:
            job.logs.append(clean_line)
            match = STEP_RE.search(clean_line)
            if match:
                step = int(match.group(1))
                total = max(int(match.group(2)), 1)
                # Keep a little headroom before completion marker.
                job.progress = max(job.progress, min(0.95, step / total))
            self._persist_locked(job)

    def _run_job(self, job: JobRecord) -> List[Artifact]:
        from driver import run_pipeline

        argv = params_to_argv(job.mode, job.params, job.extra_args)
        with self._lock:
            job.argv = list(argv)
            self._persist_locked(job)

        config = parse_config_from_argv(argv)

        logger = _LogWriter(lambda line: self._append_log(job, line))
        with redirect_stdout(logger), redirect_stderr(logger):
            results = run_pipeline(config)
        logger.flush()

        return self._collect_artifacts(config.stem_dir, results.plot_paths)

    def _collect_artifacts(self, stem_dir: str, plot_paths: Dict[str, str]) -> List[Artifact]:
        candidates: Dict[str, str] = {}

        fixed = [
            "detection_report.html",
            "analysis_report.html",
            "report.html",
            "candidates.csv",
            "transcribed_notes.csv",
            "transition_matrix.png",
            "pitch_sargam.png",
        ]
        for name in fixed:
            path = os.path.join(stem_dir, name)
            candidates[name] = path

        for key, path in plot_paths.items():
            candidates[f"{key}:{os.path.basename(path)}"] = path

        artifacts: List[Artifact] = []
        for name, path in candidates.items():
            if os.path.exists(path):
                artifacts.append(Artifact(name=name, path=os.path.abspath(path)))
        return artifacts

    def _persist_locked(self, job: JobRecord) -> None:
        path = self.jobs_dir / f"{job.job_id}.json"
        payload = job.to_dict()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
