from __future__ import annotations

import csv
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from local_app.jobs import JobManager, JobRecord
from local_app.schemas import ArtifactInfo, JobCreateRequest, JobLogsResponse, JobStatusResponse
from raga_pipeline.cli_schema import get_mode_schema, list_modes
from raga_pipeline.config import find_default_raga_db_path


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".mp4", ".aac", ".ogg"}
DEFAULT_AUDIO_DIR_REL = "../audio_test_files"
RAGA_NAME_COLUMNS = ["names", "raga", "raga_name", "name", "Raga", "RagaName"]


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


def _load_raga_names_from_csv(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        name_col: Optional[str] = None
        for candidate in RAGA_NAME_COLUMNS:
            if candidate in reader.fieldnames:
                name_col = candidate
                break

        if name_col is None:
            # Fallback: choose the first non-numeric-like column.
            for col in reader.fieldnames:
                if not col.isdigit():
                    name_col = col
                    break

        if name_col is None:
            return []

        seen = set()
        ragas: List[str] = []
        for row in reader:
            raw_values: List[str] = []
            primary = row.get(name_col)
            if isinstance(primary, str):
                raw_values.append(primary)

            # Handle unquoted comma-separated aliases that spill into extras.
            extras = row.get(None)
            if isinstance(extras, list):
                raw_values.extend([item for item in extras if isinstance(item, str)])

            for raw in raw_values:
                for part in raw.split(","):
                    name = part.strip()
                    if not name:
                        continue
                    key = name.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    ragas.append(name)
        ragas.sort(key=lambda x: x.lower())
        return ragas


def _resolve_audio_dir(audio_dir: Optional[str]) -> Path:
    raw = (audio_dir or "").strip() or DEFAULT_AUDIO_DIR_REL
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


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
                "default_audio_dir": DEFAULT_AUDIO_DIR_REL,
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

    @app.get("/api/ragas")
    def api_ragas(raga_db: Optional[str] = Query(default=None, description="Optional custom raga DB CSV path")) -> Dict[str, Any]:
        db_path = Path(raga_db).expanduser().resolve() if raga_db else None
        if db_path is None:
            default_path = find_default_raga_db_path()
            if default_path:
                db_path = Path(default_path)

        if db_path is None:
            return {"ragas": [], "source": None}

        ragas = _load_raga_names_from_csv(db_path)
        return {"ragas": ragas, "source": str(db_path)}

    @app.get("/api/audio-files")
    def api_audio_files(audio_dir: Optional[str] = Query(default=None, description="Audio directory to list files from")) -> Dict[str, Any]:
        directory = _resolve_audio_dir(audio_dir)
        if not directory.exists() or not directory.is_dir():
            return {
                "directory": str(directory),
                "exists": False,
                "files": [],
                "default_directory": DEFAULT_AUDIO_DIR_REL,
            }

        files = []
        for entry in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
                continue
            files.append({"name": entry.name, "path": str(entry.resolve())})

        return {
            "directory": str(directory),
            "exists": True,
            "files": files,
            "default_directory": DEFAULT_AUDIO_DIR_REL,
        }

    @app.post("/api/upload-audio")
    async def api_upload_audio(audio_file: UploadFile = File(...)) -> Dict[str, str]:
        filename = Path(audio_file.filename or "").name
        if not filename:
            raise HTTPException(status_code=400, detail="Missing upload filename.")

        suffix = Path(filename).suffix.lower()
        if suffix and suffix not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio extension '{suffix}'.",
            )

        uploads_dir = app.state.job_manager.data_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dest_path = uploads_dir / f"{uuid.uuid4().hex}_{filename}"

        try:
            with dest_path.open("wb") as f:
                while True:
                    chunk = await audio_file.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        finally:
            await audio_file.close()

        return {
            "filename": filename,
            "path": str(dest_path.resolve()),
        }

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
