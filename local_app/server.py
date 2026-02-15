from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from local_app.jobs import JobManager, JobRecord
from local_app.schemas import ArtifactInfo, JobCreateRequest, JobLogsResponse, JobStatusResponse
from raga_pipeline.cli_schema import get_mode_schema, list_modes


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _health_checks() -> List[str]:
    warnings: List[str] = []
    if shutil.which("ffmpeg") is None:
        warnings.append("ffmpeg not found in PATH.")
    if shutil.which("ffprobe") is None:
        warnings.append("ffprobe not found in PATH.")
    try:
        import driver  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive startup check
        warnings.append(f"Failed to import pipeline driver: {exc}")
    return warnings


def _artifact_to_response(path: str) -> ArtifactInfo:
    abs_path = Path(path).resolve()
    exists = abs_path.exists()
    url = None
    try:
        rel = abs_path.relative_to(REPO_ROOT.resolve())
        rel_posix = str(rel).replace(os.sep, "/")
        url = f"/artifacts/{rel_posix}"
    except Exception:
        url = None
    return ArtifactInfo(name=abs_path.name, path=str(abs_path), exists=exists, url=url)


def _job_to_response(job: JobRecord) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        mode=job.mode,
        status=job.status,
        progress=job.progress,
        message=job.message,
        cancel_requested=job.cancel_requested,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        argv=list(job.argv),
        artifacts=[_artifact_to_response(item.path) for item in job.artifacts],
    )


def create_app(job_manager: JobManager | None = None) -> FastAPI:
    app = FastAPI(title="Raga Local App", version="0.1.0")
    manager = job_manager or JobManager(repo_root=REPO_ROOT)
    app.state.job_manager = manager
    app.state.warnings = _health_checks()

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/artifacts", StaticFiles(directory=str(REPO_ROOT)), name="artifacts")

    @app.get("/", response_class=HTMLResponse)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/app")

    @app.get("/app", response_class=HTMLResponse)
    def app_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "modes": list_modes(),
                "warnings": list(app.state.warnings),
            },
        )

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "warnings": list(app.state.warnings),
        }

    @app.get("/api/modes")
    def api_modes() -> Dict[str, List[str]]:
        return {"modes": list_modes()}

    @app.get("/api/schema/{mode}")
    def api_schema(mode: str) -> Dict[str, Any]:
        try:
            return get_mode_schema(mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs", response_model=JobStatusResponse)
    def api_create_job(payload: JobCreateRequest) -> JobStatusResponse:
        if payload.mode not in list_modes():
            raise HTTPException(status_code=400, detail=f"Unsupported mode: {payload.mode}")
        job = app.state.job_manager.submit(
            mode=payload.mode,
            params=payload.params,
            extra_args=payload.extra_args,
        )
        return _job_to_response(job)

    @app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
    def api_job(job_id: str) -> JobStatusResponse:
        job = app.state.job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_to_response(job)

    @app.get("/api/jobs/{job_id}/logs", response_model=JobLogsResponse)
    def api_job_logs(job_id: str) -> JobLogsResponse:
        job = app.state.job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobLogsResponse(job_id=job_id, logs=app.state.job_manager.get_logs(job_id))

    @app.get("/api/jobs/{job_id}/artifacts", response_model=List[ArtifactInfo])
    def api_job_artifacts(job_id: str) -> List[ArtifactInfo]:
        job = app.state.job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return [_artifact_to_response(item.path) for item in job.artifacts]

    @app.post("/api/jobs/{job_id}/cancel", response_model=JobStatusResponse)
    def api_job_cancel(job_id: str) -> JobStatusResponse:
        ok = app.state.job_manager.cancel(job_id)
        if not ok:
            raise HTTPException(status_code=409, detail="Job cannot be cancelled")
        job = app.state.job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_to_response(job)

    return app


app = create_app()
