from __future__ import annotations

import base64
import csv
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote, urlparse, urlunparse

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from local_app.jobs import JobManager, JobRecord
from local_app.schemas import ArtifactInfo, BatchJobRequest, JobCreateRequest, JobLogsResponse, JobStatusResponse
from raga_pipeline.cli_schema import get_mode_schema, list_modes
from raga_pipeline.config import find_default_raga_db_path


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".mp4", ".aac", ".ogg"}
DEFAULT_AUDIO_DIR_REL = "../audio_test_files"
DEFAULT_OUTPUT_DIR_REL = "batch_results"
DEFAULT_UI_MODE = "detect"
STATIC_VERSION = str(int((STATIC_DIR / "app.js").stat().st_mtime)) if (STATIC_DIR / "app.js").exists() else "1"
RAGA_NAME_COLUMNS = ["names", "raga", "raga_name", "name", "Raga", "RagaName"]
KNOWN_ARTIFACT_FILES = [
    "detection_report.html",
    "analysis_report.html",
    "report.html",
    "candidates.csv",
    "transcribed_notes.csv",
    "transition_matrix.png",
    "pitch_sargam.png",
]
ASSET_ATTR_RE = re.compile(r'(\b(?:src|href)\s*=\s*)(["\'])([^"\']+)(\2)', re.IGNORECASE)


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


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _encode_dir_token(path: Path) -> str:
    payload = str(path.resolve()).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_dir_token(token: str) -> Path:
    padded = token + "=" * ((4 - len(token) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid file token.") from exc
    return Path(raw).resolve()


def _local_file_url(path: Path) -> Optional[str]:
    abs_path = path.resolve()
    if not abs_path.exists() or not abs_path.is_file():
        return None
    token = _encode_dir_token(abs_path.parent)
    return f"/local-files/{token}/{quote(abs_path.name, safe='')}"


def _local_report_url(path: Path) -> Optional[str]:
    abs_path = path.resolve()
    if not abs_path.exists() or not abs_path.is_file():
        return None
    token = _encode_dir_token(abs_path.parent)
    return f"/local-report/{token}/{quote(abs_path.name, safe='')}"


def _rewrite_report_asset_urls(html: str, report_path: Path) -> str:
    report_dir = report_path.parent.resolve()

    def _replace(match: re.Match[str]) -> str:
        prefix, q1, raw_value, q2 = match.groups()
        raw_value_stripped = raw_value.strip()
        raw_value_lower = raw_value_stripped.lower()

        # Fast-path skip for absolute/external/data URIs.
        # Analysis reports embed very large base64 image URIs; avoid urlparse on them.
        if raw_value_lower.startswith(("http://", "https://", "data:", "javascript:", "mailto:")):
            return match.group(0)
        if raw_value_stripped.startswith("#"):
            return match.group(0)
        if raw_value_stripped.startswith(("/local-files/", "/local-report/", "/static/")):
            return match.group(0)

        parsed = urlparse(raw_value)
        candidate = parsed.path.strip()
        if not candidate:
            return match.group(0)

        lower = candidate.lower()
        if lower.startswith(("http://", "https://", "data:", "javascript:", "mailto:")):
            return match.group(0)
        if candidate.startswith("#"):
            return match.group(0)
        if candidate.startswith("/local-files/") or candidate.startswith("/local-report/") or candidate.startswith("/static/"):
            return match.group(0)

        decoded = unquote(candidate)
        if os.path.isabs(decoded):
            abs_target = Path(decoded).resolve()
        else:
            abs_target = (report_dir / decoded).resolve()

        new_url = _local_file_url(abs_target)
        if not new_url:
            return match.group(0)

        rebuilt = urlunparse(parsed._replace(path=new_url))
        return f"{prefix}{q1}{rebuilt}{q2}"

    return ASSET_ATTR_RE.sub(_replace, html)


def _artifact_to_response(path: str) -> ArtifactInfo:
    abs_path = Path(path).resolve()
    exists = abs_path.exists()
    url = None
    if exists:
        if abs_path.suffix.lower() == ".html":
            url = _local_report_url(abs_path)
        else:
            url = _local_file_url(abs_path)
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


def _resolve_output_dir(output_dir: Optional[str]) -> Path:
    raw = (output_dir or "").strip() or DEFAULT_OUTPUT_DIR_REL
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def _discover_stem_dirs(output_dir: Path, stem_name: str, separator: Optional[str], demucs_model: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    if separator == "spleeter":
        candidates.append(output_dir / "spleeter" / stem_name)
    else:
        candidates.append(output_dir / (demucs_model or "htdemucs") / stem_name)

    for subdir in ["htdemucs", "htdemucs_ft", "mdx", "mdx_extra", "spleeter"]:
        candidates.append(output_dir / subdir / stem_name)

    if output_dir.exists():
        for child in sorted(output_dir.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir():
                candidates.append(child / stem_name)

    deduped: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _collect_stem_artifacts(stem_dir: Path) -> List[ArtifactInfo]:
    artifacts: List[ArtifactInfo] = []
    seen: set[str] = set()

    for name in KNOWN_ARTIFACT_FILES:
        path = stem_dir / name
        if path.exists() and path.is_file():
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            artifacts.append(_artifact_to_response(str(path)))

    if stem_dir.exists():
        for path in sorted(stem_dir.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".html", ".csv", ".png", ".jpg", ".jpeg", ".svg"}:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            artifacts.append(_artifact_to_response(str(path)))

    return artifacts


def _artifact_to_dict(item: ArtifactInfo) -> Dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()  # type: ignore[no-any-return]
    return item.dict()  # type: ignore[no-any-return]


def create_app(job_manager: JobManager | None = None) -> FastAPI:
    app = FastAPI(title="Raga Local App", version="0.1.0")
    manager = job_manager or JobManager(repo_root=REPO_ROOT)
    app.state.job_manager = manager
    app.state.warnings = _health_checks()

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/app")

    @app.get("/local-files/{dir_token}/{relative_path:path}")
    def local_files(dir_token: str, relative_path: str) -> FileResponse:
        base_dir = _decode_dir_token(dir_token)
        target = (base_dir / relative_path).resolve()
        if not _is_relative_to(target, base_dir):
            raise HTTPException(status_code=403, detail="Path traversal blocked.")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found.")
        return FileResponse(path=str(target))

    @app.get("/local-report/{dir_token}/{relative_path:path}")
    def local_report(dir_token: str, relative_path: str) -> HTMLResponse:
        base_dir = _decode_dir_token(dir_token)
        report_path = (base_dir / relative_path).resolve()
        if not _is_relative_to(report_path, base_dir):
            raise HTTPException(status_code=403, detail="Path traversal blocked.")
        if not report_path.exists() or not report_path.is_file():
            raise HTTPException(status_code=404, detail="File not found.")
        if report_path.suffix.lower() != ".html":
            raise HTTPException(status_code=400, detail="Only HTML reports are supported by this route.")

        raw_html = report_path.read_text(encoding="utf-8")
        rewritten_html = _rewrite_report_asset_urls(raw_html, report_path)
        return HTMLResponse(content=rewritten_html)

    @app.get("/app", response_class=HTMLResponse)
    def app_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "modes": list_modes(),
                "default_mode": DEFAULT_UI_MODE,
                "warnings": list(app.state.warnings),
                "default_audio_dir": DEFAULT_AUDIO_DIR_REL,
                "static_version": STATIC_VERSION,
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

    @app.get("/api/audio-artifacts")
    def api_audio_artifacts(
        audio_path: str = Query(..., description="Selected audio file path"),
        output_dir: Optional[str] = Query(default=None, description="Pipeline output directory"),
        separator: Optional[str] = Query(default=None, description="Separator engine (demucs|spleeter)"),
        demucs_model: Optional[str] = Query(default=None, description="Demucs model name"),
    ) -> Dict[str, Any]:
        stem_name = Path(audio_path).stem.strip()
        if not stem_name:
            raise HTTPException(status_code=400, detail="audio_path must include a file name.")

        resolved_output = _resolve_output_dir(output_dir)
        stem_dirs = _discover_stem_dirs(resolved_output, stem_name, separator, demucs_model)
        existing_dirs = [d.resolve() for d in stem_dirs if d.exists() and d.is_dir()]

        if not existing_dirs:
            return {
                "found": False,
                "audio_path": audio_path,
                "audio_stem": stem_name,
                "output_dir": str(resolved_output),
                "searched_dirs": [str(p.resolve()) for p in stem_dirs],
                "stem_dir": None,
                "artifacts": [],
                "detect_report_url": None,
                "analyze_report_url": None,
            }

        stem_dir = existing_dirs[0]
        artifacts = _collect_stem_artifacts(stem_dir)
        detect_url = None
        analyze_url = None
        for item in artifacts:
            lower_name = item.name.lower()
            if detect_url is None and lower_name == "detection_report.html":
                detect_url = item.url
            if analyze_url is None and lower_name in {"analysis_report.html", "report.html"}:
                analyze_url = item.url

        return {
            "found": True,
            "audio_path": audio_path,
            "audio_stem": stem_name,
            "output_dir": str(resolved_output),
            "searched_dirs": [str(p.resolve()) for p in stem_dirs],
            "stem_dir": str(stem_dir),
            "artifacts": [_artifact_to_dict(item) for item in artifacts],
            "detect_report_url": detect_url,
            "analyze_report_url": analyze_url,
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

    @app.post("/api/batch-jobs", response_model=JobStatusResponse)
    def api_create_batch_job(payload: BatchJobRequest) -> JobStatusResponse:
        mode = payload.mode.strip() if payload.mode else "auto"
        if mode not in {"auto", "detect"}:
            raise HTTPException(status_code=400, detail="Batch mode must be 'auto' or 'detect'.")
        job = app.state.job_manager.submit(
            mode="batch",
            params={
                "input_dir": payload.input_dir,
                "output_dir": payload.output_dir or DEFAULT_OUTPUT_DIR_REL,
                "batch_mode": mode,
                "ground_truth": payload.ground_truth,
                "silent": payload.silent,
            },
            extra_args=[],
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
