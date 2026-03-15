from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from html import unescape
from math import isfinite
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote, urlparse, urlunparse

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from local_app.jobs import JobManager, JobRecord
from local_app.schemas import (
    ArtifactInfo,
    BatchJobRequest,
    CleanupResponse,
    EditableNote,
    EditablePhrase,
    JobCreateRequest,
    JobLogsResponse,
    JobStatusResponse,
    LibraryResponse,
    LibrarySongRow,
    LibraryVariantRow,
    ReportStatus,
    RuntimeFingerprint,
    TranscriptionEditBaseResponse,
    TranscriptionEditPayload,
    TranscriptionEditSaveResponse,
    TranscriptionEditVersionInfo,
    TranscriptionEditVersionResponse,
    TranscriptionEditVersionsResponse,
)
from raga_pipeline.cli_schema import get_mode_schema, list_modes
from raga_pipeline.config import find_default_raga_db_path
from raga_pipeline.audio import (
    get_tonic_from_tanpura_key,
    load_pitch_from_csv,
    list_tanpura_tracks,
    start_microphone_recording_session,
)
from raga_pipeline.runtime_fingerprint import (
    get_runtime_fingerprint as _shared_get_runtime_fingerprint,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".mp4", ".aac", ".ogg", ".webm", ".weba"}
DEFAULT_AUDIO_DIR_REL = "../audio_test_files"
DEFAULT_OUTPUT_DIR_REL = "batch_results"
DEFAULT_UI_MODE = "detect"
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
TRANSCRIPTION_EDIT_DIRNAME = "transcription_edits"
TRANSCRIPTION_EDIT_MANIFEST = "manifest.json"
TRANSCRIPTION_EDIT_SCHEMA_VERSION = 1
TRANSCRIPTION_EDIT_SINGLE_VERSION_ID = "edited"
RUNTIME_FINGERPRINT_TTL_SECONDS = 5.0
CLEAR_ALL_PRESERVE_BASENAMES = {
    "vocals.mp3",
    "melody.mp3",
    "accompaniment.mp3",
    "composite_pitch_data.csv",
    "melody_pitch_data.csv",
    "vocals_pitch_data.csv",
    "accompaniment_pitch_data.csv",
}


def _compute_static_version() -> str:
    candidates = [
        STATIC_DIR / "app.js",
        STATIC_DIR / "style.css",
        STATIC_DIR / "transcription_editor.js",
    ]
    mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
    if not mtimes:
        return "1"
    return str(int(max(mtimes)))


STATIC_VERSION = _compute_static_version()


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
    basename_fallbacks: Dict[str, List[Path]] = {}

    # Build a lookup of any resolvable local assets referenced in the report.
    # This lets us rewrite missing sibling paths (for example "song.mp3")
    # to an equivalent known path found elsewhere in the same document
    # (for example "../../../audio_test_files/song.mp3").
    for match in ASSET_ATTR_RE.finditer(html):
        raw_value = (match.group(3) or "").strip()
        if not raw_value:
            continue
        raw_lower = raw_value.lower()
        if raw_lower.startswith(("http://", "https://", "data:", "javascript:", "mailto:")):
            continue
        if raw_value.startswith(("#", "/local-files/", "/local-report/", "/static/")):
            continue
        parsed = urlparse(raw_value)
        candidate = (parsed.path or "").strip()
        if not candidate:
            continue
        decoded = unquote(candidate)
        if os.path.isabs(decoded):
            resolved = Path(decoded).resolve()
        else:
            resolved = (report_dir / decoded).resolve()
        if resolved.exists() and resolved.is_file():
            key = resolved.name.lower()
            paths = basename_fallbacks.setdefault(key, [])
            if all(str(existing) != str(resolved) for existing in paths):
                paths.append(resolved)

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
            fallback_targets = basename_fallbacks.get(Path(decoded).name.lower(), [])
            if len(fallback_targets) == 1:
                new_url = _local_file_url(fallback_targets[0])
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


def _get_runtime_fingerprint(force_refresh: bool = False) -> RuntimeFingerprint:
    payload = _shared_get_runtime_fingerprint(
        REPO_ROOT,
        cache_ttl_seconds=RUNTIME_FINGERPRINT_TTL_SECONDS,
        force_refresh=force_refresh,
    )
    return _model_validate(RuntimeFingerprint, payload)


def _song_id_for_audio_path(audio_path: Path) -> str:
    return hashlib.sha1(str(audio_path.resolve()).encode("utf-8")).hexdigest()


def _scan_audio_library(audio_dir: Path) -> List[Path]:
    if not audio_dir.exists() or not audio_dir.is_dir():
        return []

    files: List[Path] = []
    for entry in sorted(audio_dir.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
            continue
        files.append(entry.resolve())
    return files


def _parse_datetime_maybe(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _meta_generated_at(meta: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    generated = meta.get("generated_at")
    if generated is None:
        return None
    text = str(generated).strip()
    return text or None


def _infer_pipeline_mode_from_path(report_path: Path) -> Optional[str]:
    name = report_path.name.lower()
    if name == "detection_report.html":
        return "detect"
    if name in {"analysis_report.html", "report.html"}:
        return "analyze"
    return None


def _resolve_report_pipeline_mode(meta: Optional[Dict[str, Any]], report_path: Path) -> Optional[str]:
    if isinstance(meta, dict):
        raw_mode = str(meta.get("pipeline_mode") or "").strip().lower()
        if raw_mode in {"detect", "analyze"}:
            return raw_mode
    return _infer_pipeline_mode_from_path(report_path)


def _match_stage_fingerprint(
    meta: Dict[str, Any],
    current_fp: RuntimeFingerprint,
    mode: str,
) -> tuple[Optional[bool], Optional[str]]:
    stage_meta = meta.get("stage_fingerprint")
    if not isinstance(stage_meta, dict):
        return None, "legacy metadata: missing stage_fingerprint"

    stage_mode = str(stage_meta.get("mode") or "").strip().lower()
    if stage_mode and stage_mode != mode:
        return None, f"invalid stage_fingerprint mode: {stage_mode}"

    report_stage_hash = stage_meta.get("hash")
    if report_stage_hash is None or str(report_stage_hash).strip() == "":
        return None, f"missing stage_fingerprint hash for {mode}"
    report_stage_hash_text = str(report_stage_hash).strip()

    current_stage_hashes = current_fp.stage_hashes if isinstance(current_fp.stage_hashes, dict) else {}
    current_stage_hash = current_stage_hashes.get(mode)
    if current_stage_hash is None or str(current_stage_hash).strip() == "":
        return None, f"runtime fingerprint missing stage hash for {mode}"
    current_stage_hash_text = str(current_stage_hash).strip()

    if report_stage_hash_text == current_stage_hash_text:
        return True, None
    return False, f"stage hash mismatch for {mode}"


def _read_report_meta(
    report_path: Path,
) -> tuple[Optional[Dict[str, Any]], Optional[Path], Optional[str]]:
    meta_path = report_path.with_suffix(".meta.json")
    if not meta_path.exists() or not meta_path.is_file():
        return None, None, "missing metadata sidecar"
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None, meta_path, "invalid metadata json"
    if not isinstance(payload, dict):
        return None, meta_path, "invalid metadata format"
    return payload, meta_path, None


def _infer_variant_identity(stem_dir: Path, output_dir: Path, audio_path: Path) -> Dict[str, Any]:
    separator = "demucs"
    demucs_model: Optional[str] = None
    try:
        rel_parts = stem_dir.resolve().relative_to(output_dir.resolve()).parts
    except Exception:
        rel_parts = ()

    if rel_parts:
        first = str(rel_parts[0])
        if first == "spleeter":
            separator = "spleeter"
            demucs_model = None
        else:
            separator = "demucs"
            demucs_model = first

    return {
        "audio_path": str(audio_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "separator": separator,
        "demucs_model": demucs_model or "htdemucs",
    }


def _build_report_status(
    report_path: Path,
    runtime_fingerprint: RuntimeFingerprint,
    variant_id: str,
) -> ReportStatus:
    abs_report = report_path.resolve()
    if not abs_report.exists() or not abs_report.is_file():
        return ReportStatus(
            exists=False,
            status="missing",
            report_path=str(abs_report),
            report_url=None,
            meta_path=None,
            generated_at=None,
            variant_id=variant_id,
            reason="report file not found",
        )

    meta, meta_path, meta_error = _read_report_meta(abs_report)
    generated_at = _meta_generated_at(meta)

    if meta is None:
        return ReportStatus(
            exists=True,
            status="unknown",
            report_path=str(abs_report),
            report_url=_local_report_url(abs_report),
            meta_path=str(meta_path.resolve()) if meta_path else None,
            generated_at=generated_at,
            variant_id=variant_id,
            reason=meta_error or "metadata unavailable",
        )

    mode = _resolve_report_pipeline_mode(meta, abs_report)
    if mode not in {"detect", "analyze"}:
        return ReportStatus(
            exists=True,
            status="unknown",
            report_path=str(abs_report),
            report_url=_local_report_url(abs_report),
            meta_path=str(meta_path.resolve()) if meta_path else None,
            generated_at=generated_at,
            variant_id=variant_id,
            reason="invalid or missing pipeline_mode",
        )

    matched, reason = _match_stage_fingerprint(meta, runtime_fingerprint, mode)
    if matched is None:
        return ReportStatus(
            exists=True,
            status="unknown",
            report_path=str(abs_report),
            report_url=_local_report_url(abs_report),
            meta_path=str(meta_path.resolve()) if meta_path else None,
            generated_at=generated_at,
            variant_id=variant_id,
            reason=reason or "stage fingerprint unavailable",
        )
    return ReportStatus(
        exists=True,
        status="current" if matched else "stale",
        report_path=str(abs_report),
        report_url=_local_report_url(abs_report),
        meta_path=str(meta_path.resolve()) if meta_path else None,
        generated_at=generated_at,
        variant_id=variant_id,
        reason=reason if matched is False else None,
    )


def _resolve_analyze_report_path(stem_dir: Path) -> Optional[Path]:
    primary = stem_dir / "analysis_report.html"
    legacy = stem_dir / "report.html"
    if primary.exists() and primary.is_file():
        return primary
    if legacy.exists() and legacy.is_file():
        return legacy
    return None


def _report_activity_ts(status: ReportStatus) -> float:
    dt = _parse_datetime_maybe(status.generated_at)
    if dt is not None:
        return dt.timestamp()
    if status.report_path:
        try:
            return Path(status.report_path).stat().st_mtime
        except Exception:
            return 0.0
    return 0.0


def _missing_report_status(kind: str) -> ReportStatus:
    return ReportStatus(
        exists=False,
        status="missing",
        report_path=None,
        report_url=None,
        meta_path=None,
        generated_at=None,
        variant_id=None,
        reason=f"{kind} report missing",
    )


def _latest_status(candidates: List[ReportStatus], *, kind: str) -> ReportStatus:
    existing = [item for item in candidates if item.exists]
    if not existing:
        return _missing_report_status(kind)
    existing.sort(key=_report_activity_ts, reverse=True)
    return existing[0]


def _scan_variants_for_song(audio_path: Path, output_dir: Path) -> List[LibraryVariantRow]:
    stem_name = audio_path.stem.strip()
    if not stem_name:
        return []

    runtime_fp = _get_runtime_fingerprint()
    stem_dirs = _discover_stem_dirs(output_dir, stem_name, None, None)
    existing_dirs = [d.resolve() for d in stem_dirs if d.exists() and d.is_dir()]
    variants: List[LibraryVariantRow] = []

    for stem_dir in existing_dirs:
        variant_id = hashlib.sha1(str(stem_dir).encode("utf-8")).hexdigest()
        detect_report = stem_dir / "detection_report.html"
        analyze_report = _resolve_analyze_report_path(stem_dir)

        detect_status = _build_report_status(detect_report, runtime_fp, variant_id)
        if analyze_report is None:
            analyze_status = _missing_report_status("analyze")
            analyze_status.variant_id = variant_id
            analyze_status.report_path = str((stem_dir / "analysis_report.html").resolve())
        else:
            analyze_status = _build_report_status(analyze_report, runtime_fp, variant_id)

        infer_identity = _infer_variant_identity(stem_dir, output_dir, audio_path)
        run_identity = dict(infer_identity)
        separator = infer_identity.get("separator")
        demucs_model = infer_identity.get("demucs_model")

        analyze_meta, _analyze_meta_path, _analyze_meta_err = (
            _read_report_meta(analyze_report) if analyze_report is not None else (None, None, None)
        )
        detect_meta, _detect_meta_path, _detect_meta_err = _read_report_meta(detect_report)
        source_meta = analyze_meta if isinstance(analyze_meta, dict) else detect_meta
        if isinstance(source_meta, dict):
            source_identity = source_meta.get("run_identity")
            if isinstance(source_identity, dict):
                run_identity.update(source_identity)
            separator = str(run_identity.get("separator") or separator or "demucs")
            demucs_model = run_identity.get("demucs_model") or demucs_model

        variants.append(
            LibraryVariantRow(
                variant_id=variant_id,
                stem_dir=str(stem_dir),
                separator=separator,
                demucs_model=str(demucs_model) if demucs_model is not None else None,
                detect=detect_status,
                analyze=analyze_status,
                run_identity=run_identity,
            )
        )

    variants.sort(
        key=lambda item: max(_report_activity_ts(item.detect), _report_activity_ts(item.analyze)),
        reverse=True,
    )
    return variants


def _build_song_row(audio_path: Path, variants: List[LibraryVariantRow]) -> LibrarySongRow:
    detect_status = _latest_status([item.detect for item in variants], kind="detect")
    analyze_status = _latest_status([item.analyze for item in variants], kind="analyze")

    latest_activity_ts = max(_report_activity_ts(detect_status), _report_activity_ts(analyze_status))
    latest_activity_at = (
        datetime.fromtimestamp(latest_activity_ts, tz=timezone.utc).isoformat()
        if latest_activity_ts > 0
        else None
    )

    return LibrarySongRow(
        song_id=_song_id_for_audio_path(audio_path),
        audio_name=audio_path.name,
        audio_path=str(audio_path.resolve()),
        detect=detect_status,
        analyze=analyze_status,
        variant_count=len(variants),
        latest_activity_at=latest_activity_at,
    )


def _song_matches_filter(song: LibrarySongRow, status_filter: Optional[str], query_text: Optional[str]) -> bool:
    if status_filter:
        target = status_filter.strip().lower()
        if target and song.detect.status != target and song.analyze.status != target:
            return False
    if query_text:
        q = query_text.strip().lower()
        if q and q not in song.audio_name.lower():
            return False
    return True


def _build_library_rows(
    audio_dir: Path,
    output_dir: Path,
    status_filter: Optional[str] = None,
    query_text: Optional[str] = None,
) -> List[LibrarySongRow]:
    rows: List[LibrarySongRow] = []
    for audio_path in _scan_audio_library(audio_dir):
        variants = _scan_variants_for_song(audio_path, output_dir)
        row = _build_song_row(audio_path, variants)
        if _song_matches_filter(row, status_filter, query_text):
            rows.append(row)

    rows.sort(key=lambda item: item.audio_name.lower())
    return rows


def _build_library_counts(rows: List[LibrarySongRow]) -> Dict[str, int]:
    counts: Dict[str, int] = {
        "songs_total": len(rows),
        "detect_current": 0,
        "detect_stale": 0,
        "detect_unknown": 0,
        "detect_missing": 0,
        "analyze_current": 0,
        "analyze_stale": 0,
        "analyze_unknown": 0,
        "analyze_missing": 0,
    }
    for row in rows:
        detect_key = f"detect_{row.detect.status}"
        analyze_key = f"analyze_{row.analyze.status}"
        if detect_key in counts:
            counts[detect_key] += 1
        if analyze_key in counts:
            counts[analyze_key] += 1
    return counts


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


def _resolve_song_for_id(song_id: str, audio_dir: Path) -> Optional[Path]:
    for audio_path in _scan_audio_library(audio_dir):
        if _song_id_for_audio_path(audio_path) == song_id:
            return audio_path
    return None


def _safe_unlink(path: Path, result: Dict[str, Any]) -> None:
    try:
        path.unlink()
        result["deleted_files"] += 1
    except Exception as exc:
        result["warnings"].append(f"Failed to delete file '{path}': {exc}")


def _safe_rmtree(path: Path, result: Dict[str, Any]) -> None:
    file_count = 0
    dir_count = 0
    try:
        for _root, dirs, files in os.walk(path, topdown=False):
            file_count += len(files)
            dir_count += len(dirs)
        dir_count += 1  # include the directory itself
    except Exception as exc:
        result["warnings"].append(f"Failed to inspect directory '{path}': {exc}")

    try:
        shutil.rmtree(path)
        result["deleted_files"] += file_count
        result["deleted_dirs"] += dir_count
    except Exception as exc:
        result["warnings"].append(f"Failed to delete directory '{path}': {exc}")


def _clear_song_outputs(audio_path: Path, output_dir: Path) -> CleanupResponse:
    resolved_output = output_dir.resolve()
    result: Dict[str, Any] = {
        "ok": True,
        "deleted_files": 0,
        "deleted_dirs": 0,
        "preserved_files": 0,
        "warnings": [],
    }

    stem_name = audio_path.stem.strip()
    if not stem_name:
        result["warnings"].append("Audio file has an empty stem; nothing to clear.")
        return _model_validate(CleanupResponse, result)

    variants = _discover_stem_dirs(resolved_output, stem_name, None, None)
    seen_dirs: set[str] = set()
    for candidate in variants:
        abs_candidate = candidate.resolve()
        key = str(abs_candidate)
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        if not abs_candidate.exists():
            continue
        if not _is_relative_to(abs_candidate, resolved_output):
            result["warnings"].append(f"Skipped path outside output directory: {abs_candidate}")
            continue
        if abs_candidate.is_symlink() or abs_candidate.is_file():
            _safe_unlink(abs_candidate, result)
            continue
        if abs_candidate.is_dir():
            _safe_rmtree(abs_candidate, result)

    logs_dir = (resolved_output / "logs").resolve()
    if logs_dir.exists() and logs_dir.is_dir():
        audio_name_lower = audio_path.name.lower()
        stem_lower = audio_path.stem.lower()
        for log_path in sorted(logs_dir.iterdir(), key=lambda p: p.name.lower()):
            if not log_path.exists() or not log_path.is_file():
                continue
            log_name_lower = log_path.name.lower()
            log_stem_lower = log_path.stem.lower()
            should_delete = (
                log_name_lower == f"{audio_name_lower}.log"
                or log_name_lower == f"{stem_lower}.log"
                or log_stem_lower == audio_name_lower
                or log_stem_lower == stem_lower
                or log_stem_lower.startswith(f"{audio_name_lower}.")
                or log_stem_lower.startswith(f"{stem_lower}.")
            )
            if should_delete:
                _safe_unlink(log_path, result)

        try:
            if not any(logs_dir.iterdir()):
                logs_dir.rmdir()
                result["deleted_dirs"] += 1
        except Exception as exc:
            result["warnings"].append(f"Failed to prune logs directory '{logs_dir}': {exc}")

    return _model_validate(CleanupResponse, result)


def _clear_all_outputs(output_dir: Path) -> CleanupResponse:
    resolved_output = output_dir.resolve()
    result: Dict[str, Any] = {
        "ok": True,
        "deleted_files": 0,
        "deleted_dirs": 0,
        "preserved_files": 0,
        "warnings": [],
    }

    if not resolved_output.exists():
        result["warnings"].append(f"Output directory not found: {resolved_output}")
        return _model_validate(CleanupResponse, result)
    if not resolved_output.is_dir():
        raise HTTPException(status_code=400, detail="Selected output directory is not a directory.")

    preserve = {name.lower() for name in CLEAR_ALL_PRESERVE_BASENAMES}
    for root, dirs, files in os.walk(resolved_output, topdown=False):
        root_path = Path(root).resolve()
        if not _is_relative_to(root_path, resolved_output):
            result["warnings"].append(f"Skipped path outside output directory: {root_path}")
            continue

        for filename in files:
            file_path = (root_path / filename).resolve()
            if not _is_relative_to(file_path, resolved_output):
                result["warnings"].append(f"Skipped file outside output directory: {file_path}")
                continue
            if filename.lower() in preserve:
                result["preserved_files"] += 1
                continue
            _safe_unlink(file_path, result)

        for dirname in dirs:
            dir_path = (root_path / dirname).resolve()
            if not _is_relative_to(dir_path, resolved_output):
                result["warnings"].append(f"Skipped directory outside output directory: {dir_path}")
                continue
            if dir_path.is_symlink():
                _safe_unlink(dir_path, result)
                continue
            if not dir_path.exists() or not dir_path.is_dir():
                continue
            try:
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    result["deleted_dirs"] += 1
            except Exception as exc:
                result["warnings"].append(f"Failed to remove directory '{dir_path}': {exc}")

    return _model_validate(CleanupResponse, result)


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


def _model_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[no-any-return]
    return model.dict()  # type: ignore[no-any-return]


def _model_validate(model_cls: Any, payload: Dict[str, Any]) -> Any:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _report_metadata_path(report_path: Path) -> Path:
    return report_path.with_suffix(".meta.json")


def _edit_root_for_report(report_path: Path) -> Path:
    return report_path.parent / TRANSCRIPTION_EDIT_DIRNAME


def _manifest_path_for_report(report_path: Path) -> Path:
    return _edit_root_for_report(report_path) / TRANSCRIPTION_EDIT_MANIFEST


def _default_edit_manifest(report_path: Path) -> Dict[str, Any]:
    return {
        "schema_version": TRANSCRIPTION_EDIT_SCHEMA_VERSION,
        "source_report": report_path.name,
        "default_selection": "original",
        "versions": [],
    }


def _normalize_single_edit_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep transcription edits to a single editable copy alongside original.

    Legacy manifests may contain multiple historical versions; preserve only the
    latest entry and normalize its id to a stable single-copy id.
    """
    versions_raw = manifest.get("versions")
    if not isinstance(versions_raw, list):
        versions_raw = []

    entries = [item for item in versions_raw if isinstance(item, dict)]
    if not entries:
        manifest["versions"] = []
        manifest["default_selection"] = "original"
        return manifest

    entries.sort(
        key=lambda item: (
            _parse_version_number(str(item.get("version_id", ""))),
            str(item.get("created_at", "")),
        )
    )
    latest_entry = dict(entries[-1])
    latest_entry["version_id"] = TRANSCRIPTION_EDIT_SINGLE_VERSION_ID
    manifest["versions"] = [latest_entry]

    requested_default = _canonical_version_id(
        manifest.get("default_selection", "original"),
        allow_original=True,
    )
    manifest["default_selection"] = (
        TRANSCRIPTION_EDIT_SINGLE_VERSION_ID
        if requested_default == TRANSCRIPTION_EDIT_SINGLE_VERSION_ID
        else "original"
    )
    return manifest


def _load_edit_manifest(report_path: Path) -> Dict[str, Any]:
    manifest_path = _manifest_path_for_report(report_path)
    if not manifest_path.exists():
        return _default_edit_manifest(report_path)

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return _default_edit_manifest(report_path)

    if not isinstance(data, dict):
        return _default_edit_manifest(report_path)
    versions = data.get("versions")
    if not isinstance(versions, list):
        data["versions"] = []
    data.setdefault("source_report", report_path.name)
    data.setdefault("schema_version", TRANSCRIPTION_EDIT_SCHEMA_VERSION)
    data.setdefault("default_selection", "original")
    return _normalize_single_edit_manifest(data)


def _save_edit_manifest(report_path: Path, manifest: Dict[str, Any]) -> None:
    edit_root = _edit_root_for_report(report_path)
    edit_root.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path_for_report(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _effective_default_selection(manifest: Dict[str, Any]) -> str:
    raw = _canonical_version_id(manifest.get("default_selection", "original"), allow_original=True)
    if raw == "original":
        return "original"
    if raw != TRANSCRIPTION_EDIT_SINGLE_VERSION_ID:
        return "original"
    if _find_version_entry(manifest, TRANSCRIPTION_EDIT_SINGLE_VERSION_ID) is None:
        return "original"
    return TRANSCRIPTION_EDIT_SINGLE_VERSION_ID


def _default_report_path(report_path: Path, manifest: Dict[str, Any]) -> Path:
    selection = _effective_default_selection(manifest)
    if selection == "original":
        return report_path

    entry = _find_version_entry(manifest, selection)
    if entry is None:
        return report_path
    report_file = str(entry.get("report_file", "")).strip()
    if not report_file:
        return report_path
    candidate = (report_path.parent / report_file).resolve()
    parent = report_path.parent.resolve()
    if not _is_relative_to(candidate, parent):
        return report_path
    if not candidate.exists() or not candidate.is_file():
        return report_path
    return candidate


def _single_edited_report_filename(report_path: Path) -> str:
    return f"{report_path.stem}_edited.html"


def _parse_version_number(version_id: str) -> int:
    raw = str(version_id or "").strip().lower()
    if raw.startswith("v"):
        raw = raw[1:]
    if raw.isdigit():
        return int(raw)
    return -1


def _canonical_version_id(version_id: Any, *, allow_original: bool = False) -> str:
    raw = str(version_id or "").strip()
    if not raw:
        return "original" if allow_original else ""
    lowered = raw.lower()
    if allow_original and lowered == "original":
        return "original"
    if lowered == TRANSCRIPTION_EDIT_SINGLE_VERSION_ID:
        return TRANSCRIPTION_EDIT_SINGLE_VERSION_ID
    if re.fullmatch(r"v\d+", raw, flags=re.IGNORECASE):
        return TRANSCRIPTION_EDIT_SINGLE_VERSION_ID
    return raw


def _next_version_id(manifest: Dict[str, Any]) -> str:
    versions = manifest.get("versions", [])
    max_num = 0
    for item in versions:
        if not isinstance(item, dict):
            continue
        max_num = max(max_num, _parse_version_number(str(item.get("version_id", ""))))
    return f"v{max_num + 1:04d}"


def _find_version_entry(manifest: Dict[str, Any], version_id: str) -> Optional[Dict[str, Any]]:
    requested = _canonical_version_id(version_id, allow_original=False)
    if not requested:
        return None
    versions_raw = manifest.get("versions", [])
    if not isinstance(versions_raw, list):
        return None
    for item in versions_raw:
        if not isinstance(item, dict):
            continue
        item_id = _canonical_version_id(item.get("version_id", ""), allow_original=False)
        if item_id == requested:
            return item
    return None


def _version_file_path(report_path: Path, filename: Any) -> Optional[Path]:
    if filename is None:
        return None
    text = str(filename).strip()
    if not text:
        return None
    if text.lower() in {"none", "null"}:
        return None
    return (_edit_root_for_report(report_path) / text).resolve()


def _resolve_report_relative_path(base_dir: Path, report_name: str) -> Path:
    decoded_name = unquote(str(report_name or "")).strip()
    if not decoded_name:
        raise HTTPException(status_code=400, detail="report_name is required.")
    target = (base_dir / decoded_name).resolve()
    if not _is_relative_to(target, base_dir):
        raise HTTPException(status_code=403, detail="Path traversal blocked.")
    return target


def _coerce_float(value: Any, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not isfinite(parsed):
        return fallback
    return parsed


def _coerce_optional_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return parsed


def _midi_to_hz(midi: float) -> float:
    return float(440.0 * (2.0 ** ((midi - 69.0) / 12.0)))


def _normalize_edit_payload(payload: TranscriptionEditPayload) -> TranscriptionEditPayload:
    note_items: List[EditableNote] = []
    note_lookup: Dict[str, EditableNote] = {}
    for idx, note in enumerate(payload.notes):
        note_id = str(note.id or "").strip() or f"n{idx + 1}"
        if note_id in note_lookup:
            continue

        start = _coerce_float(note.start, 0.0)
        end = _coerce_float(note.end, start)
        if end < start:
            start, end = end, start
        if end <= start:
            end = start + 1e-3

        pitch_midi = _coerce_float(note.pitch_midi, 60.0)
        pitch_hz = _coerce_float(note.pitch_hz, _midi_to_hz(pitch_midi))
        if pitch_hz <= 0:
            pitch_hz = _midi_to_hz(pitch_midi)
        raw_pitch_midi = _coerce_optional_float(note.raw_pitch_midi)
        snapped_pitch_midi = _coerce_optional_float(note.snapped_pitch_midi)
        corrected_pitch_midi = _coerce_optional_float(note.corrected_pitch_midi)
        rendered_pitch_midi = _coerce_optional_float(note.rendered_pitch_midi)

        confidence = _coerce_float(note.confidence, 0.8)
        energy = _coerce_float(note.energy, 0.0)
        sargam = str(note.sargam or "").strip()

        pitch_class_raw = note.pitch_class
        if pitch_class_raw is None:
            pitch_class = int(round(pitch_midi)) % 12
        else:
            try:
                pitch_class = int(pitch_class_raw) % 12
            except Exception:
                pitch_class = int(round(pitch_midi)) % 12

        normalized_note = EditableNote(
            id=note_id,
            start=start,
            end=end,
            pitch_midi=pitch_midi,
            pitch_hz=pitch_hz,
            raw_pitch_midi=raw_pitch_midi,
            snapped_pitch_midi=snapped_pitch_midi,
            corrected_pitch_midi=corrected_pitch_midi,
            rendered_pitch_midi=rendered_pitch_midi,
            confidence=confidence,
            energy=energy,
            sargam=sargam,
            pitch_class=pitch_class,
        )
        note_lookup[note_id] = normalized_note
        note_items.append(normalized_note)

    note_items.sort(key=lambda n: (n.start, n.end, n.pitch_midi, n.id))
    note_lookup = {note.id: note for note in note_items}

    phrase_items: List[EditablePhrase] = []
    seen_phrase_ids: set[str] = set()
    assigned_note_ids: set[str] = set()
    auto_phrase_idx = 1
    for idx, phrase in enumerate(payload.phrases):
        phrase_id = str(phrase.id or "").strip() or f"p{idx + 1}"
        if phrase_id in seen_phrase_ids:
            continue
        seen_phrase_ids.add(phrase_id)

        local_seen: set[str] = set()
        phrase_note_ids: List[str] = []
        for raw_note_id in phrase.note_ids:
            note_id = str(raw_note_id or "").strip()
            if not note_id or note_id not in note_lookup:
                continue
            if note_id in local_seen:
                continue
            if note_id in assigned_note_ids:
                continue
            local_seen.add(note_id)
            assigned_note_ids.add(note_id)
            phrase_note_ids.append(note_id)

        if not phrase_note_ids:
            continue

        phrase_notes = [note_lookup[nid] for nid in phrase_note_ids]
        phrase_notes.sort(key=lambda n: (n.start, n.end))
        phrase_note_ids = [n.id for n in phrase_notes]
        phrase_start = min(n.start for n in phrase_notes)
        phrase_end = max(n.end for n in phrase_notes)
        phrase_items.append(
            EditablePhrase(
                id=phrase_id,
                start=phrase_start,
                end=phrase_end,
                note_ids=phrase_note_ids,
            )
        )

    for note in note_items:
        if note.id in assigned_note_ids:
            continue
        phrase_id = f"auto_p{auto_phrase_idx:04d}"
        auto_phrase_idx += 1
        phrase_items.append(
            EditablePhrase(
                id=phrase_id,
                start=note.start,
                end=note.end,
                note_ids=[note.id],
            )
        )

    phrase_items.sort(key=lambda p: (p.start, p.end, p.id))
    return TranscriptionEditPayload(notes=note_items, phrases=phrase_items)


def _editable_note_to_sequence_note(note: EditableNote) -> Any:
    from raga_pipeline.sequence import Note

    pitch_class = note.pitch_class if note.pitch_class is not None else int(round(note.pitch_midi)) % 12
    pitch_hz = note.pitch_hz if note.pitch_hz is not None else _midi_to_hz(note.pitch_midi)
    seq_note = Note(
        start=float(note.start),
        end=float(note.end),
        pitch_midi=float(note.pitch_midi),
        pitch_hz=float(pitch_hz),
        confidence=float(note.confidence),
        energy=float(note.energy),
        sargam=str(note.sargam or ""),
        pitch_class=int(pitch_class) % 12,
    )
    raw_pitch = _coerce_optional_float(note.raw_pitch_midi)
    if raw_pitch is not None:
        seq_note.raw_pitch_midi = raw_pitch
        seq_note.raw_pitch_hz = _midi_to_hz(raw_pitch)
    snapped_pitch = _coerce_optional_float(note.snapped_pitch_midi)
    if snapped_pitch is not None:
        seq_note.snapped_pitch_midi = snapped_pitch
    corrected_pitch = _coerce_optional_float(note.corrected_pitch_midi)
    if corrected_pitch is not None:
        seq_note.corrected_pitch_midi = corrected_pitch
    rendered_pitch = _coerce_optional_float(note.rendered_pitch_midi)
    if rendered_pitch is not None:
        seq_note.rendered_pitch_midi = rendered_pitch
    return seq_note


def _payload_to_phrase_objects(payload: TranscriptionEditPayload) -> tuple[List[Any], List[Any]]:
    from raga_pipeline.sequence import Phrase

    note_lookup = {note.id: _editable_note_to_sequence_note(note) for note in payload.notes}
    phrases: List[Any] = []

    for phrase in payload.phrases:
        phrase_notes = [note_lookup[nid] for nid in phrase.note_ids if nid in note_lookup]
        if not phrase_notes:
            continue
        phrase_notes.sort(key=lambda n: (n.start, n.end))
        phrases.append(Phrase(notes=phrase_notes))

    phrases.sort(key=lambda p: (p.start, p.end))
    flat_notes = [note for phrase in phrases for note in phrase.notes]
    return phrases, flat_notes


def _write_transcription_csv(notes: List[Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "start",
        "end",
        "duration",
        "pitch_midi",
        "pitch_hz",
        "confidence",
        "pitch_class",
        "sargam",
        "energy",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for note in notes:
            duration = max(0.0, float(getattr(note, "end", 0.0)) - float(getattr(note, "start", 0.0)))
            writer.writerow(
                {
                    "start": f"{float(getattr(note, 'start', 0.0)):.3f}",
                    "end": f"{float(getattr(note, 'end', 0.0)):.3f}",
                    "duration": f"{duration:.3f}",
                    "pitch_midi": f"{float(getattr(note, 'pitch_midi', 0.0)):.2f}",
                    "pitch_hz": f"{float(getattr(note, 'pitch_hz', 0.0)):.1f}",
                    "confidence": f"{float(getattr(note, 'confidence', 0.0)):.2f}",
                    "pitch_class": int(getattr(note, "pitch_class", 0)),
                    "sargam": str(getattr(note, "sargam", "")),
                    "energy": f"{float(getattr(note, 'energy', 0.0)):.4f}",
                }
            )


def _resolve_context_path(base_dir: Path, raw_path: Any) -> Optional[Path]:
    if raw_path is None:
        return None
    path_text = str(raw_path).strip()
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_media_path(
    base_dir: Path,
    raw_path: Any,
    fallbacks: List[Path],
) -> Path:
    """
    Resolve report media paths robustly.

    Metadata may contain absolute paths, stem-dir-relative paths, or repo/cwd-relative paths.
    Prefer an existing file among those interpretations, then fall back to known stem assets.
    """
    candidates: List[Path] = []
    resolved = _resolve_context_path(base_dir, raw_path)
    if resolved is not None:
        candidates.append(resolved)

    raw_text = str(raw_path or "").strip()
    if raw_text:
        raw_candidate = Path(raw_text).expanduser()
        if raw_candidate.is_absolute():
            candidates.append(raw_candidate.resolve())
        else:
            candidates.append((REPO_ROOT / raw_candidate).resolve())
            candidates.append((Path.cwd() / raw_candidate).resolve())

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    for fallback in fallbacks:
        try:
            candidate = fallback.resolve()
        except Exception:
            candidate = fallback
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    if resolved is not None:
        return resolved
    if fallbacks:
        return fallbacks[0].resolve()
    return base_dir


def _parse_tonic_pitch_class(tonic_label: str) -> Optional[int]:
    text = str(tonic_label or "").strip().replace("♯", "#").replace("♭", "b")
    if not text:
        return None
    normalized = text.replace(" ", "")
    lookup = {
        "C": 0,
        "B#": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "FB": 4,
        "F": 5,
        "E#": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
        "CB": 11,
    }
    return lookup.get(normalized.upper())


def _extract_report_subtitle_context(report_html: str) -> tuple[Optional[str], Optional[str], Optional[int]]:
    raga_name: Optional[str] = None
    tonic_label: Optional[str] = None

    subtitle_match = re.search(
        r"<p[^>]*class=[\"'][^\"']*subtitle[^\"']*[\"'][^>]*>(.*?)</p>",
        report_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if subtitle_match:
        subtitle_text = unescape(re.sub(r"<[^>]+>", "", subtitle_match.group(1))).strip()
        subtitle_parts = re.match(r"(.+?)\s*\(Tonic:\s*([^)]+)\)\s*$", subtitle_text)
        if subtitle_parts:
            candidate_raga = subtitle_parts.group(1).strip()
            candidate_tonic = subtitle_parts.group(2).strip()
            if candidate_raga:
                raga_name = candidate_raga
            if candidate_tonic:
                tonic_label = candidate_tonic

    if not raga_name:
        title_match = re.search(
            r"<title>\s*Raga Analysis Report:\s*([^<]+)\s*</title>",
            report_html,
            flags=re.IGNORECASE,
        )
        if title_match:
            candidate_raga = unescape(title_match.group(1)).strip()
            if candidate_raga:
                raga_name = candidate_raga

    if not tonic_label:
        tonic_match = re.search(r"\(Tonic:\s*([^)]+)\)", report_html, flags=re.IGNORECASE)
        if tonic_match:
            candidate_tonic = unescape(tonic_match.group(1)).strip()
            if candidate_tonic:
                tonic_label = candidate_tonic

    tonic_pc = _parse_tonic_pitch_class(tonic_label or "") if tonic_label else None
    return raga_name, tonic_label, tonic_pc


def _extract_audio_candidates_from_report(
    report_path: Path,
    report_html: str,
    audio_id: str,
) -> List[Path]:
    block_match = re.search(
        rf"<audio[^>]*id=[\"']{re.escape(audio_id)}[\"'][^>]*>(.*?)</audio>",
        report_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not block_match:
        return []

    candidates: List[Path] = []
    source_block = block_match.group(1)
    source_matches = re.findall(
        r"<source[^>]*src=[\"']([^\"']+)[\"'][^>]*>",
        source_block,
        flags=re.IGNORECASE,
    )
    for source in source_matches:
        raw = unquote(str(source or "").strip())
        if not raw:
            continue
        lower = raw.lower()
        if lower.startswith(("http://", "https://", "data:", "javascript:", "mailto:")):
            continue
        if raw.startswith("#"):
            continue
        path_obj = Path(raw).expanduser()
        if path_obj.is_absolute():
            candidates.append(path_obj.resolve())
        else:
            candidates.append((report_path.parent / path_obj).resolve())
    return candidates


def _is_unknown_text(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"", "unknown", "n/a", "none", "null"}


def _enrich_context_from_report_html(report_path: Path, context: Dict[str, Any]) -> Dict[str, Any]:
    if not report_path.exists() or not report_path.is_file():
        return context

    try:
        report_html = report_path.read_text(encoding="utf-8")
    except Exception:
        return context

    cfg = context.get("config")
    if not isinstance(cfg, dict):
        cfg = {}
    context["config"] = cfg

    detected = context.get("detected")
    if not isinstance(detected, dict):
        detected = {}
    context["detected"] = detected

    stats = context.get("stats")
    if not isinstance(stats, dict):
        stats = {}
    context["stats"] = stats

    base_dir = report_path.parent.resolve()

    audio_specs: List[tuple[str, str]] = [
        ("audio_path", "original-player"),
        ("vocals_path", "vocals-player"),
        ("accompaniment_path", "accomp-player"),
    ]
    for config_key, audio_id in audio_specs:
        existing_raw = cfg.get(config_key)
        has_existing = False
        if existing_raw:
            existing_path = _resolve_media_path(base_dir, existing_raw, [])
            has_existing = existing_path.exists() and existing_path.is_file()
        if has_existing:
            continue

        for candidate in _extract_audio_candidates_from_report(report_path, report_html, audio_id):
            if candidate.exists() and candidate.is_file():
                cfg[config_key] = str(candidate)
                break

    raga_name, tonic_label, tonic_pc = _extract_report_subtitle_context(report_html)
    if raga_name:
        if _is_unknown_text(detected.get("raga")):
            detected["raga"] = raga_name
        if _is_unknown_text(stats.get("raga_name")):
            stats["raga_name"] = raga_name

    if tonic_label and _is_unknown_text(stats.get("tonic")):
        stats["tonic"] = tonic_label
    if tonic_pc is not None and detected.get("tonic") is None:
        detected["tonic"] = tonic_pc

    return context


def _fallback_report_context(report_path: Path) -> Dict[str, Any]:
    stem_dir = report_path.parent
    stem_name = stem_dir.name

    def _first_existing(candidates: List[Path], fallback: Path) -> str:
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return str(fallback.resolve())

    original_audio_fallback = stem_dir / f"{stem_name}.mp3"
    original_audio = _first_existing(
        [stem_dir / f"{stem_name}.mp3", stem_dir / f"{stem_name}.wav"],
        original_audio_fallback,
    )
    vocals_audio = _first_existing(
        [stem_dir / "vocals.mp3", stem_dir / "vocals.wav"],
        stem_dir / "vocals.mp3",
    )
    accomp_audio = _first_existing(
        [stem_dir / "accompaniment.mp3", stem_dir / "accompaniment.wav"],
        stem_dir / "accompaniment.mp3",
    )

    plot_paths: Dict[str, str] = {}
    note_duration_path = stem_dir / "note_duration_histogram.png"
    gmm_path = stem_dir / "gmm_overlay.png"
    if note_duration_path.exists():
        plot_paths["note_duration_histogram"] = note_duration_path.name
    if gmm_path.exists():
        plot_paths["gmm_overlay"] = gmm_path.name

    context = {
        "schema_version": 1,
        "config": {
            "audio_path": original_audio,
            "vocals_path": vocals_audio,
            "accompaniment_path": accomp_audio,
            "melody_source": "separated",
            "transcription_smoothing_ms": 70.0,
            "transcription_min_duration": 0.04,
            "transcription_derivative_threshold": 2.0,
            "energy_threshold": 0.0,
            "show_rms_overlay": True,
        },
        "detected": {"tonic": None, "raga": "Unknown"},
        "stats": {
            "correction_summary": {},
            "pattern_analysis": {},
            "raga_name": "Unknown",
            "tonic": "Unknown",
            "transition_matrix_path": "transition_matrix.png",
            "pitch_plot_path": "pitch_sargam.png",
        },
        "plot_paths": plot_paths,
        "pitch_csv_paths": {
            "original": ["composite_pitch_data.csv"],
            "vocals": ["melody_pitch_data.csv", "vocals_pitch_data.csv"],
            "accompaniment": ["accompaniment_pitch_data.csv"],
        },
    }
    return _enrich_context_from_report_html(report_path, context)


def _load_report_context(report_path: Path) -> Dict[str, Any]:
    meta_path = _report_metadata_path(report_path)
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return _enrich_context_from_report_html(report_path, loaded)
        except Exception:
            pass
    return _fallback_report_context(report_path)


def _load_pitch_from_context(base_dir: Path, context: Dict[str, Any], key: str) -> Any:
    candidate_map = context.get("pitch_csv_paths", {})
    if not isinstance(candidate_map, dict):
        candidate_map = {}
    defaults: Dict[str, List[str]] = {
        "original": ["composite_pitch_data.csv"],
        "vocals": ["melody_pitch_data.csv", "vocals_pitch_data.csv"],
        "accompaniment": ["accompaniment_pitch_data.csv"],
    }
    candidates_raw = candidate_map.get(key, defaults.get(key, []))
    if not isinstance(candidates_raw, list):
        candidates_raw = defaults.get(key, [])

    for candidate in candidates_raw:
        resolved = _resolve_context_path(base_dir, candidate)
        if resolved is None or not resolved.exists() or not resolved.is_file():
            continue
        try:
            return load_pitch_from_csv(str(resolved))
        except Exception:
            continue
    return None


def _resolve_plot_path(base_dir: Path, raw_path: Any, fallback_name: str) -> str:
    resolved = _resolve_context_path(base_dir, raw_path)
    if resolved is None:
        return str((base_dir / fallback_name).resolve())
    return str(resolved)


def _render_edited_report(
    report_path: Path,
    context: Dict[str, Any],
    payload: TranscriptionEditPayload,
    edited_report_filename: str,
) -> Path:
    from raga_pipeline.output import AnalysisResults, AnalysisStats, generate_analysis_report

    base_dir = report_path.parent.resolve()
    cfg = context.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    detected = context.get("detected", {})
    if not isinstance(detected, dict):
        detected = {}
    stats_blob = context.get("stats", {})
    if not isinstance(stats_blob, dict):
        stats_blob = {}

    audio_path = _resolve_media_path(
        base_dir,
        cfg.get("audio_path"),
        [
            base_dir / f"{base_dir.name}.mp3",
            base_dir / f"{base_dir.name}.wav",
        ],
    )
    vocals_path = _resolve_media_path(
        base_dir,
        cfg.get("vocals_path"),
        [
            base_dir / "vocals.mp3",
            base_dir / "vocals.wav",
        ],
    )
    accomp_path = _resolve_media_path(
        base_dir,
        cfg.get("accompaniment_path"),
        [
            base_dir / "accompaniment.mp3",
            base_dir / "accompaniment.wav",
        ],
    )

    config_obj = SimpleNamespace(
        audio_path=str(audio_path),
        vocals_path=str(vocals_path),
        accompaniment_path=str(accomp_path),
        melody_source=str(cfg.get("melody_source", "separated")),
        transcription_smoothing_ms=_coerce_float(cfg.get("transcription_smoothing_ms"), 70.0),
        transcription_min_duration=_coerce_float(cfg.get("transcription_min_duration"), 0.04),
        transcription_derivative_threshold=_coerce_float(cfg.get("transcription_derivative_threshold"), 2.0),
        energy_threshold=_coerce_float(cfg.get("energy_threshold"), 0.0),
        show_rms_overlay=bool(cfg.get("show_rms_overlay", True)),
    )

    phrases, flat_notes = _payload_to_phrase_objects(payload)
    results = AnalysisResults(config=config_obj)
    results.notes = flat_notes
    results.phrases = phrases

    tonic_value = detected.get("tonic")
    try:
        results.detected_tonic = int(tonic_value) if tonic_value is not None else None
    except Exception:
        results.detected_tonic = None

    raga_value = detected.get("raga")
    results.detected_raga = str(raga_value) if raga_value else None

    results.pitch_data_composite = _load_pitch_from_context(base_dir, context, "original")
    results.pitch_data_stem = _load_pitch_from_context(base_dir, context, "vocals")
    results.pitch_data_vocals = results.pitch_data_stem
    results.pitch_data_accomp = _load_pitch_from_context(base_dir, context, "accompaniment")

    derivative_blob = context.get("transcription_derivative_profile", {})
    if isinstance(derivative_blob, dict):
        ts_vals = derivative_blob.get("timestamps")
        d_vals = derivative_blob.get("values")
        voiced_vals = derivative_blob.get("voiced_mask")
        if isinstance(ts_vals, list):
            results.transcription_derivative_timestamps = ts_vals
        if isinstance(d_vals, list):
            results.transcription_derivative_values = d_vals
        if isinstance(voiced_vals, list):
            results.transcription_derivative_voiced_mask = voiced_vals

    plot_paths_blob = context.get("plot_paths", {})
    plot_paths: Dict[str, str] = {}
    if isinstance(plot_paths_blob, dict):
        for key, value in plot_paths_blob.items():
            resolved = _resolve_context_path(base_dir, value)
            if resolved is not None:
                plot_paths[str(key)] = str(resolved)
    results.plot_paths = plot_paths

    stats_obj = AnalysisStats(
        correction_summary=stats_blob.get("correction_summary", {}),
        pattern_analysis=stats_blob.get("pattern_analysis", {}),
        raga_name=str(stats_blob.get("raga_name", results.detected_raga or "Unknown")),
        tonic=str(stats_blob.get("tonic", "Unknown")),
        transition_matrix_path=_resolve_plot_path(base_dir, stats_blob.get("transition_matrix_path"), "transition_matrix.png"),
        pitch_plot_path=_resolve_plot_path(base_dir, stats_blob.get("pitch_plot_path"), "pitch_sargam.png"),
    )

    out_path = generate_analysis_report(
        results,
        stats_obj,
        str(base_dir),
        report_filename=edited_report_filename,
    )
    return Path(out_path).resolve()


def _version_entry_to_info(report_path: Path, entry: Dict[str, Any]) -> TranscriptionEditVersionInfo:
    edit_root = _edit_root_for_report(report_path)

    def _resolve_name(name: Any) -> Optional[Path]:
        if name is None:
            return None
        text = str(name).strip()
        if not text:
            return None
        return (edit_root / text).resolve()

    json_path = _resolve_name(entry.get("json_file"))
    csv_path = _resolve_name(entry.get("csv_file"))
    report_file = entry.get("report_file")
    report_version_path = None
    if report_file:
        report_version_path = (report_path.parent / str(report_file)).resolve()

    return TranscriptionEditVersionInfo(
        version_id=_canonical_version_id(entry.get("version_id", ""), allow_original=False),
        created_at=str(entry.get("created_at", "")),
        note_count=int(entry.get("note_count", 0)),
        phrase_count=int(entry.get("phrase_count", 0)),
        json_url=_local_file_url(json_path) if json_path else None,
        csv_url=_local_file_url(csv_path) if csv_path else None,
        report_url=_local_report_url(report_version_path) if report_version_path else None,
        source_report_url=_local_report_url(report_path),
    )


def _list_version_infos(report_path: Path, manifest: Dict[str, Any]) -> List[TranscriptionEditVersionInfo]:
    versions_raw = manifest.get("versions", [])
    if not isinstance(versions_raw, list):
        return []

    version_entries = [item for item in versions_raw if isinstance(item, dict)]
    version_entries.sort(
        key=lambda item: (
            _parse_version_number(str(item.get("version_id", ""))),
            str(item.get("created_at", "")),
        )
    )
    return [_version_entry_to_info(report_path, item) for item in version_entries]


def _build_versions_response(
    report_path: Path,
    manifest: Dict[str, Any],
) -> TranscriptionEditVersionsResponse:
    versions = _list_version_infos(report_path, manifest)
    latest_version_id = versions[-1].version_id if versions else None
    default_selection = _effective_default_selection(manifest)
    default_report_path = _default_report_path(report_path, manifest)
    default_report_url = _local_report_url(default_report_path)
    return TranscriptionEditVersionsResponse(
        versions=versions,
        latest_version_id=latest_version_id,
        default_selection=default_selection,
        default_report_url=default_report_url,
    )


def _load_version_payload(report_path: Path, entry: Dict[str, Any]) -> TranscriptionEditPayload:
    json_file = str(entry.get("json_file", "")).strip()
    if not json_file:
        raise HTTPException(status_code=404, detail="Version payload file is missing.")
    payload_path = (_edit_root_for_report(report_path) / json_file).resolve()
    if not payload_path.exists() or not payload_path.is_file():
        raise HTTPException(status_code=404, detail="Version payload file not found.")

    try:
        loaded = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read version payload: {exc}") from exc

    if not isinstance(loaded, dict):
        raise HTTPException(status_code=500, detail="Version payload has invalid format.")
    return _model_validate(TranscriptionEditPayload, loaded)


def _load_base_edit_payload(report_path: Path) -> Optional[Dict[str, Any]]:
    def _build_sargam_options(tonic_pc: int) -> List[Dict[str, Any]]:
        from raga_pipeline.sequence import OFFSET_TO_SARGAM

        tonic_pc = int(tonic_pc) % 12
        base_sa_midi = tonic_pc + 60
        options: List[Dict[str, Any]] = []
        for offset in range(12):
            options.append(
                {
                    "offset": offset,
                    "label": OFFSET_TO_SARGAM.get(offset, f"?{offset}"),
                    "midi": int(base_sa_midi + offset),
                }
            )
        return options

    def _tonic_from_context(context_blob: Dict[str, Any]) -> Optional[int]:
        detected_blob = context_blob.get("detected", {})
        if not isinstance(detected_blob, dict):
            return None
        tonic_raw = detected_blob.get("tonic")
        if tonic_raw is None:
            return None
        try:
            return int(round(float(tonic_raw))) % 12
        except Exception:
            return None

    def _from_metadata_payload(context_blob: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        payload_raw = context_blob.get("transcription_edit_payload")
        if not isinstance(payload_raw, dict):
            return None

        payload_blob = {
            "notes": payload_raw.get("notes", []),
            "phrases": payload_raw.get("phrases", []),
        }
        try:
            payload = _model_validate(TranscriptionEditPayload, payload_blob)
        except Exception:
            return None

        normalized = _normalize_edit_payload(payload)
        if not normalized.notes:
            return None

        tonic_raw = payload_raw.get("tonic")
        tonic_value = None
        if tonic_raw is not None:
            try:
                tonic_value = int(round(float(tonic_raw))) % 12
            except Exception:
                tonic_value = None
        if tonic_value is None:
            tonic_value = _tonic_from_context(context_blob)

        sargam_options_raw = payload_raw.get("sargam_options", [])
        sargam_options: List[Dict[str, Any]] = []
        if isinstance(sargam_options_raw, list):
            for item in sargam_options_raw:
                if isinstance(item, dict):
                    sargam_options.append(dict(item))
        if not sargam_options:
            sargam_options = _build_sargam_options(tonic_value or 0)

        return {
            "payload": normalized,
            "tonic": tonic_value,
            "sargam_options": sargam_options,
        }

    context = _load_report_context(report_path)
    from_metadata = _from_metadata_payload(context)
    if from_metadata is not None:
        return from_metadata
    return None


def create_app(job_manager: JobManager | None = None) -> FastAPI:
    app = FastAPI(title="Raga Local App", version="0.1.0")
    manager = job_manager or JobManager(repo_root=REPO_ROOT)
    app.state.job_manager = manager
    app.state.warnings = _health_checks()
    app.state.preprocess_recording_session = None
    app.state.preprocess_recording_lock = threading.Lock()

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
        requested_report_path = (base_dir / relative_path).resolve()
        if not _is_relative_to(requested_report_path, base_dir):
            raise HTTPException(status_code=403, detail="Path traversal blocked.")
        if not requested_report_path.exists() or not requested_report_path.is_file():
            raise HTTPException(status_code=404, detail="File not found.")
        if requested_report_path.suffix.lower() != ".html":
            raise HTTPException(status_code=400, detail="Only HTML reports are supported by this route.")

        report_path = requested_report_path
        try:
            manifest = _load_edit_manifest(requested_report_path)
            source_report_name = str(manifest.get("source_report", requested_report_path.name)).strip()
            if requested_report_path.name == source_report_name:
                report_path = _default_report_path(requested_report_path, manifest)
        except Exception:
            report_path = requested_report_path

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

    @app.get("/api/runtime-fingerprint", response_model=RuntimeFingerprint)
    def api_runtime_fingerprint() -> RuntimeFingerprint:
        return _get_runtime_fingerprint(force_refresh=False)

    @app.get("/api/library", response_model=LibraryResponse)
    def api_library(
        audio_dir: Optional[str] = Query(default=None, description="Audio library directory"),
        output_dir: Optional[str] = Query(default=None, description="Pipeline output directory"),
        status_filter: Optional[str] = Query(
            default=None,
            description="Optional filter: stale|missing|unknown|current",
        ),
        q: Optional[str] = Query(default=None, description="Optional case-insensitive song name substring filter"),
    ) -> LibraryResponse:
        resolved_audio = _resolve_audio_dir(audio_dir)
        resolved_output = _resolve_output_dir(output_dir)

        rows = _build_library_rows(
            resolved_audio,
            resolved_output,
            status_filter=status_filter,
            query_text=q,
        )

        return LibraryResponse(
            runtime_fingerprint=_get_runtime_fingerprint(force_refresh=False),
            audio_dir=str(resolved_audio),
            output_dir=str(resolved_output),
            songs=rows,
            counts=_build_library_counts(rows),
        )

    @app.get("/api/library/{song_id}/variants")
    def api_library_variants(
        song_id: str,
        audio_dir: Optional[str] = Query(default=None, description="Audio library directory"),
        output_dir: Optional[str] = Query(default=None, description="Pipeline output directory"),
    ) -> Dict[str, Any]:
        resolved_audio = _resolve_audio_dir(audio_dir)
        resolved_output = _resolve_output_dir(output_dir)
        audio_files = _scan_audio_library(resolved_audio)

        target_audio: Optional[Path] = None
        for audio_path in audio_files:
            if _song_id_for_audio_path(audio_path) == song_id:
                target_audio = audio_path
                break
        if target_audio is None:
            raise HTTPException(status_code=404, detail="Song not found in selected audio directory.")

        variants = _scan_variants_for_song(target_audio, resolved_output)
        return {
            "song_id": song_id,
            "audio_name": target_audio.name,
            "audio_path": str(target_audio.resolve()),
            "variants": [_model_dump(item) for item in variants],
        }

    @app.post("/api/library/{song_id}/clear-outputs", response_model=CleanupResponse)
    def api_library_clear_song_outputs(
        song_id: str,
        audio_dir: Optional[str] = Query(default=None, description="Audio library directory"),
        output_dir: Optional[str] = Query(default=None, description="Pipeline output directory"),
    ) -> CleanupResponse:
        resolved_audio = _resolve_audio_dir(audio_dir)
        resolved_output = _resolve_output_dir(output_dir)
        target_audio = _resolve_song_for_id(song_id, resolved_audio)
        if target_audio is None:
            raise HTTPException(status_code=404, detail="Song not found in selected audio directory.")
        return _clear_song_outputs(target_audio, resolved_output)

    @app.post("/api/library/clear-all-outputs", response_model=CleanupResponse)
    def api_library_clear_all_outputs(
        output_dir: Optional[str] = Query(default=None, description="Pipeline output directory"),
    ) -> CleanupResponse:
        resolved_output = _resolve_output_dir(output_dir)
        return _clear_all_outputs(resolved_output)

    @app.get("/api/tanpura-tracks")
    def api_tanpura_tracks() -> Dict[str, Any]:
        tracks = []
        for track in list_tanpura_tracks(require_exists=True):
            track_path = Path(track["path"]).resolve()
            track_url = _local_file_url(track_path)
            tracks.append(
                {
                    "key": track["key"],
                    "label": track["label"],
                    "filename": track["filename"],
                    "path": str(track_path),
                    "url": track_url,
                }
            )
        return {"tracks": tracks}

    @app.get("/api/audio-artifacts")
    def api_audio_artifacts(
        audio_path: str = Query(..., description="Selected audio file path"),
        output_dir: Optional[str] = Query(default=None, description="Pipeline output directory"),
        separator: Optional[str] = Query(default=None, description="Separator engine (demucs|spleeter)"),
        demucs_model: Optional[str] = Query(default=None, description="Demucs model name"),
    ) -> Dict[str, Any]:
        resolved_audio = Path(audio_path).expanduser().resolve()
        stem_name = resolved_audio.stem.strip()
        if not stem_name:
            raise HTTPException(status_code=400, detail="audio_path must include a file name.")

        resolved_output = _resolve_output_dir(output_dir)
        variants = _scan_variants_for_song(resolved_audio, resolved_output)
        if separator:
            variants = [item for item in variants if (item.separator or "").strip() == separator]
        if demucs_model:
            variants = [item for item in variants if (item.demucs_model or "").strip() == demucs_model]

        if not variants:
            searched_dirs = _discover_stem_dirs(resolved_output, stem_name, separator, demucs_model)
            return {
                "found": False,
                "audio_path": audio_path,
                "audio_stem": stem_name,
                "output_dir": str(resolved_output),
                "searched_dirs": [str(p.resolve()) for p in searched_dirs],
                "stem_dir": None,
                "artifacts": [],
                "detect_report_url": None,
                "analyze_report_url": None,
                "analyze_report_context": None,
            }

        variants.sort(
            key=lambda item: max(_report_activity_ts(item.detect), _report_activity_ts(item.analyze)),
            reverse=True,
        )
        selected_variant = variants[0]
        stem_dir = Path(selected_variant.stem_dir).resolve()
        artifacts = _collect_stem_artifacts(stem_dir)
        detect_url = selected_variant.detect.report_url
        analyze_url = selected_variant.analyze.report_url
        analyze_report_name = Path(selected_variant.analyze.report_path).name if selected_variant.analyze.report_path else None

        analyze_report_context = None
        if analyze_url and analyze_report_name:
            analyze_report_context = {
                "url": analyze_url,
                "dir_token": _encode_dir_token(stem_dir),
                "report_name": analyze_report_name,
            }

        return {
            "found": True,
            "audio_path": audio_path,
            "audio_stem": stem_name,
            "output_dir": str(resolved_output),
            "searched_dirs": [item.stem_dir for item in variants],
            "stem_dir": str(stem_dir),
            "artifacts": [_artifact_to_dict(item) for item in artifacts],
            "detect_report_url": detect_url,
            "analyze_report_url": analyze_url,
            "analyze_report_context": analyze_report_context,
        }

    @app.get(
        "/api/transcription-edits/{dir_token}/{report_name}/base",
        response_model=TranscriptionEditBaseResponse,
    )
    def api_transcription_edit_base(dir_token: str, report_name: str) -> TranscriptionEditBaseResponse:
        base_dir = _decode_dir_token(dir_token)
        report_path = _resolve_report_relative_path(base_dir, report_name)
        if not report_path.exists() or not report_path.is_file():
            raise HTTPException(status_code=404, detail="Report file not found.")
        if report_path.suffix.lower() != ".html":
            raise HTTPException(status_code=400, detail="Only HTML reports support transcription edits.")

        payload_bundle = _load_base_edit_payload(report_path)
        if payload_bundle is None:
            return TranscriptionEditBaseResponse(
                ready=False,
                requires_rerun=True,
                detail=(
                    "Base transcription payload is unavailable for this report. "
                    "Rerun analyze to enable in-app transcription editing."
                ),
                payload=None,
                tonic=None,
                sargam_options=[],
            )

        return TranscriptionEditBaseResponse(
            ready=True,
            requires_rerun=False,
            detail=None,
            payload=payload_bundle["payload"],
            tonic=payload_bundle["tonic"],
            sargam_options=payload_bundle["sargam_options"],
        )

    @app.get(
        "/api/transcription-edits/{dir_token}/{report_name}/versions",
        response_model=TranscriptionEditVersionsResponse,
    )
    def api_transcription_edit_versions(dir_token: str, report_name: str) -> TranscriptionEditVersionsResponse:
        base_dir = _decode_dir_token(dir_token)
        report_path = _resolve_report_relative_path(base_dir, report_name)
        if not report_path.exists() or not report_path.is_file():
            raise HTTPException(status_code=404, detail="Report file not found.")
        if report_path.suffix.lower() != ".html":
            raise HTTPException(status_code=400, detail="Only HTML reports support transcription edits.")

        manifest = _load_edit_manifest(report_path)
        return _build_versions_response(report_path, manifest)

    @app.post(
        "/api/transcription-edits/{dir_token}/{report_name}/default",
        response_model=TranscriptionEditVersionsResponse,
    )
    def api_transcription_edit_set_default(
        dir_token: str,
        report_name: str,
        default_selection: str = Query(
            ...,
            description="'original' or a saved version id.",
        ),
    ) -> TranscriptionEditVersionsResponse:
        base_dir = _decode_dir_token(dir_token)
        report_path = _resolve_report_relative_path(base_dir, report_name)
        if not report_path.exists() or not report_path.is_file():
            raise HTTPException(status_code=404, detail="Report file not found.")
        if report_path.suffix.lower() != ".html":
            raise HTTPException(status_code=400, detail="Only HTML reports support transcription edits.")

        manifest = _normalize_single_edit_manifest(_load_edit_manifest(report_path))
        requested = _canonical_version_id(default_selection, allow_original=True)
        if not requested:
            raise HTTPException(status_code=400, detail="default_selection is required.")

        if requested != "original":
            entry = _find_version_entry(manifest, requested)
            if entry is None:
                raise HTTPException(status_code=404, detail="Requested transcription edit version was not found.")

            report_file = str(entry.get("report_file", "")).strip()
            report_candidate = (report_path.parent / report_file).resolve() if report_file else None
            if report_candidate is None or not report_candidate.exists() or not report_candidate.is_file():
                payload = _load_version_payload(report_path, entry)
                normalized_payload = _normalize_edit_payload(payload)
                edited_report_filename = _single_edited_report_filename(report_path)
                context = _load_report_context(report_path)
                try:
                    edited_report_path = _render_edited_report(
                        report_path=report_path,
                        context=context,
                        payload=normalized_payload,
                        edited_report_filename=edited_report_filename,
                    )
                except HTTPException:
                    raise
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Failed to generate edited report: {exc}") from exc

                entry["report_file"] = edited_report_path.name
                entry["note_count"] = len(normalized_payload.notes)
                entry["phrase_count"] = len(normalized_payload.phrases)
                entry["updated_at"] = _utc_now_iso()

        manifest["default_selection"] = requested
        _save_edit_manifest(report_path, manifest)
        return _build_versions_response(report_path, manifest)

    @app.get(
        "/api/transcription-edits/{dir_token}/{report_name}/latest",
        response_model=TranscriptionEditVersionResponse,
    )
    def api_transcription_edit_latest(dir_token: str, report_name: str) -> TranscriptionEditVersionResponse:
        base_dir = _decode_dir_token(dir_token)
        report_path = _resolve_report_relative_path(base_dir, report_name)
        if not report_path.exists() or not report_path.is_file():
            raise HTTPException(status_code=404, detail="Report file not found.")
        if report_path.suffix.lower() != ".html":
            raise HTTPException(status_code=400, detail="Only HTML reports support transcription edits.")

        manifest = _load_edit_manifest(report_path)
        versions_raw = manifest.get("versions", [])
        if not isinstance(versions_raw, list):
            versions_raw = []
        dict_versions = [item for item in versions_raw if isinstance(item, dict)]
        if not dict_versions:
            return TranscriptionEditVersionResponse(has_version=False, version=None, payload=None)

        dict_versions.sort(
            key=lambda item: (
                _parse_version_number(str(item.get("version_id", ""))),
                str(item.get("created_at", "")),
            )
        )
        latest_entry = dict_versions[-1]
        payload = _load_version_payload(report_path, latest_entry)
        version_info = _version_entry_to_info(report_path, latest_entry)
        return TranscriptionEditVersionResponse(
            has_version=True,
            version=version_info,
            payload=payload,
        )

    @app.get(
        "/api/transcription-edits/{dir_token}/{report_name}/version/{version_id}",
        response_model=TranscriptionEditVersionResponse,
    )
    def api_transcription_edit_version(
        dir_token: str,
        report_name: str,
        version_id: str,
    ) -> TranscriptionEditVersionResponse:
        base_dir = _decode_dir_token(dir_token)
        report_path = _resolve_report_relative_path(base_dir, report_name)
        if not report_path.exists() or not report_path.is_file():
            raise HTTPException(status_code=404, detail="Report file not found.")
        if report_path.suffix.lower() != ".html":
            raise HTTPException(status_code=400, detail="Only HTML reports support transcription edits.")

        manifest = _load_edit_manifest(report_path)
        versions_raw = manifest.get("versions", [])
        if not isinstance(versions_raw, list):
            versions_raw = []
        canonical_version = _canonical_version_id(version_id, allow_original=False)
        requested = None
        for item in versions_raw:
            if not isinstance(item, dict):
                continue
            item_version = _canonical_version_id(item.get("version_id", ""), allow_original=False)
            if item_version == canonical_version:
                requested = item
                break
        if requested is None:
            raise HTTPException(status_code=404, detail="Requested transcription edit version was not found.")

        payload = _load_version_payload(report_path, requested)
        version_info = _version_entry_to_info(report_path, requested)
        return TranscriptionEditVersionResponse(
            has_version=True,
            version=version_info,
            payload=payload,
        )

    @app.post(
        "/api/transcription-edits/{dir_token}/{report_name}/save",
        response_model=TranscriptionEditSaveResponse,
    )
    def api_transcription_edit_save(
        dir_token: str,
        report_name: str,
        payload: TranscriptionEditPayload,
        target_version_id: Optional[str] = Query(
            default=None,
            description="When set, update this existing version in-place.",
        ),
        create_new_version: bool = Query(
            default=False,
            description="Force creation of a new version instead of updating the selected/latest version.",
        ),
    ) -> TranscriptionEditSaveResponse:
        base_dir = _decode_dir_token(dir_token)
        report_path = _resolve_report_relative_path(base_dir, report_name)
        if not report_path.exists() or not report_path.is_file():
            raise HTTPException(status_code=404, detail="Report file not found.")
        if report_path.suffix.lower() != ".html":
            raise HTTPException(status_code=400, detail="Only HTML reports support transcription edits.")

        normalized_payload = _normalize_edit_payload(payload)
        if not normalized_payload.notes:
            raise HTTPException(status_code=400, detail="Cannot save an empty transcription payload.")

        phrases, flat_notes = _payload_to_phrase_objects(normalized_payload)
        if not phrases or not flat_notes:
            raise HTTPException(status_code=400, detail="Payload must include at least one phrase with notes.")

        edit_root = _edit_root_for_report(report_path)
        edit_root.mkdir(parents=True, exist_ok=True)

        manifest = _normalize_single_edit_manifest(_load_edit_manifest(report_path))
        versions_raw = manifest.get("versions", [])
        if not isinstance(versions_raw, list):
            versions_raw = []
        manifest["versions"] = versions_raw

        # The API keeps a single editable copy now; legacy versioning query args
        # are accepted for backward compatibility but intentionally ignored.
        _ = target_version_id
        _ = create_new_version

        selected_entry = _find_version_entry(manifest, TRANSCRIPTION_EDIT_SINGLE_VERSION_ID)
        save_mode = "updated" if selected_entry is not None else "created"
        version_id = TRANSCRIPTION_EDIT_SINGLE_VERSION_ID
        if selected_entry is not None:
            json_filename = str(selected_entry.get("json_file", "")).strip() or "transcription_edited.json"
            csv_filename = str(selected_entry.get("csv_file", "")).strip() or "transcription_edited.csv"
            edited_report_filename = str(selected_entry.get("report_file", "")).strip()
            created_at = str(selected_entry.get("created_at", "")).strip()
        else:
            json_filename = "transcription_edited.json"
            csv_filename = "transcription_edited.csv"
            edited_report_filename = ""
            created_at = ""

        timestamp = _utc_now_iso()

        json_path = (edit_root / json_filename).resolve()
        csv_path = (edit_root / csv_filename).resolve()
        json_path.write_text(json.dumps(_model_dump(normalized_payload), indent=2), encoding="utf-8")
        _write_transcription_csv(flat_notes, csv_path)

        edited_report_filename = edited_report_filename or _single_edited_report_filename(report_path)
        context = _load_report_context(report_path)
        try:
            edited_report_path = _render_edited_report(
                report_path=report_path,
                context=context,
                payload=normalized_payload,
                edited_report_filename=edited_report_filename,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to generate edited report: {exc}") from exc

        entry = {
            "version_id": version_id,
            "created_at": created_at or timestamp,
            "updated_at": timestamp,
            "json_file": json_filename,
            "csv_file": csv_filename,
            "report_file": edited_report_path.name,
            "note_count": len(flat_notes),
            "phrase_count": len(phrases),
        }
        if selected_entry is not None:
            selected_entry.clear()
            selected_entry.update(entry)
            manifest["versions"] = [selected_entry]
        else:
            manifest["versions"] = [entry]
        _save_edit_manifest(report_path, manifest)

        version_infos = _list_version_infos(report_path, manifest)
        current_info = None
        for info in version_infos:
            if info.version_id == version_id:
                current_info = info
                break
        if current_info is None:
            current_info = _version_entry_to_info(report_path, entry)

        return TranscriptionEditSaveResponse(
            save_mode=save_mode,
            version=current_info,
            versions=version_infos,
            default_selection=_effective_default_selection(manifest),
            default_report_url=_local_report_url(_default_report_path(report_path, manifest)),
            payload=normalized_payload,
        )

    @app.post(
        "/api/transcription-edits/{dir_token}/{report_name}/version/{version_id}/regenerate",
        response_model=TranscriptionEditVersionResponse,
    )
    def api_transcription_edit_regenerate_version(
        dir_token: str,
        report_name: str,
        version_id: str,
    ) -> TranscriptionEditVersionResponse:
        base_dir = _decode_dir_token(dir_token)
        report_path = _resolve_report_relative_path(base_dir, report_name)
        if not report_path.exists() or not report_path.is_file():
            raise HTTPException(status_code=404, detail="Report file not found.")
        if report_path.suffix.lower() != ".html":
            raise HTTPException(status_code=400, detail="Only HTML reports support transcription edits.")

        manifest = _normalize_single_edit_manifest(_load_edit_manifest(report_path))
        canonical_version = _canonical_version_id(version_id, allow_original=False)
        entry = _find_version_entry(manifest, canonical_version)
        if entry is None:
            raise HTTPException(status_code=404, detail="Requested transcription edit version was not found.")

        payload = _load_version_payload(report_path, entry)
        normalized_payload = _normalize_edit_payload(payload)
        phrases, flat_notes = _payload_to_phrase_objects(normalized_payload)
        if not phrases or not flat_notes:
            raise HTTPException(status_code=400, detail="Stored payload is empty and cannot be rendered.")

        edited_report_filename = str(entry.get("report_file", "")).strip() or _single_edited_report_filename(report_path)
        context = _load_report_context(report_path)
        try:
            edited_report_path = _render_edited_report(
                report_path=report_path,
                context=context,
                payload=normalized_payload,
                edited_report_filename=edited_report_filename,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to regenerate edited report: {exc}") from exc

        entry["report_file"] = edited_report_path.name
        entry["note_count"] = len(flat_notes)
        entry["phrase_count"] = len(phrases)
        entry["updated_at"] = _utc_now_iso()
        _save_edit_manifest(report_path, manifest)

        version_info = _version_entry_to_info(report_path, entry)
        return TranscriptionEditVersionResponse(
            has_version=True,
            version=version_info,
            payload=normalized_payload,
        )

    @app.delete(
        "/api/transcription-edits/{dir_token}/{report_name}/version/{version_id}",
        response_model=TranscriptionEditVersionsResponse,
    )
    def api_transcription_edit_delete_version(
        dir_token: str,
        report_name: str,
        version_id: str,
    ) -> TranscriptionEditVersionsResponse:
        base_dir = _decode_dir_token(dir_token)
        report_path = _resolve_report_relative_path(base_dir, report_name)
        if not report_path.exists() or not report_path.is_file():
            raise HTTPException(status_code=404, detail="Report file not found.")
        if report_path.suffix.lower() != ".html":
            raise HTTPException(status_code=400, detail="Only HTML reports support transcription edits.")

        manifest = _load_edit_manifest(report_path)
        versions_raw = manifest.get("versions", [])
        if not isinstance(versions_raw, list):
            versions_raw = []
            manifest["versions"] = versions_raw

        canonical_version = _canonical_version_id(version_id, allow_original=False)
        idx_to_remove = None
        entry_to_remove: Optional[Dict[str, Any]] = None
        for idx, item in enumerate(versions_raw):
            if not isinstance(item, dict):
                continue
            item_version = _canonical_version_id(item.get("version_id", ""), allow_original=False)
            if item_version == canonical_version:
                idx_to_remove = idx
                entry_to_remove = item
                break
        if idx_to_remove is None or entry_to_remove is None:
            raise HTTPException(status_code=404, detail="Requested transcription edit version was not found.")

        del versions_raw[idx_to_remove]
        if _canonical_version_id(manifest.get("default_selection", ""), allow_original=True) == canonical_version:
            manifest["default_selection"] = "original"
        _save_edit_manifest(report_path, manifest)

        json_path = _version_file_path(report_path, entry_to_remove.get("json_file"))
        csv_path = _version_file_path(report_path, entry_to_remove.get("csv_file"))
        report_file = entry_to_remove.get("report_file")
        report_version_path = None
        if report_file is not None and str(report_file).strip():
            report_version_path = (report_path.parent / str(report_file).strip()).resolve()

        for candidate in [json_path, csv_path, report_version_path]:
            if candidate is None:
                continue
            try:
                if candidate.exists() and candidate.is_file():
                    candidate.unlink()
            except Exception:
                # Best-effort cleanup; keep manifest as source of truth.
                pass

        return _build_versions_response(report_path, manifest)

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

    @app.post("/api/preprocess-record/start")
    def api_preprocess_record_start(payload: Dict[str, Any]) -> Dict[str, Any]:
        audio_dir = str(payload.get("audio_dir") or DEFAULT_AUDIO_DIR_REL).strip() or DEFAULT_AUDIO_DIR_REL
        filename = str(payload.get("filename") or "").strip()
        ingest_raw = str(payload.get("ingest") or "").strip()
        if not ingest_raw:
            legacy_record_mode = str(payload.get("record_mode") or "").strip()
            if legacy_record_mode == "tanpura_vocal":
                ingest_raw = "tanpura_recording"
            elif legacy_record_mode == "song":
                ingest_raw = "recording"
        ingest_aliases = {
            "youtube": "yt",
            "record": "recording",
        }
        ingest = ingest_aliases.get(ingest_raw, ingest_raw)
        tanpura_key_raw = str(payload.get("tanpura_key") or "").strip()
        tanpura_key = tanpura_key_raw or None

        if not filename:
            raise HTTPException(status_code=400, detail="filename is required.")
        if not ingest:
            raise HTTPException(status_code=400, detail="ingest is required.")
        if ingest not in {"recording", "tanpura_recording"}:
            raise HTTPException(
                status_code=400,
                detail="ingest must be 'recording' or 'tanpura_recording' for this endpoint.",
            )
        if ingest == "tanpura_recording" and not tanpura_key:
            raise HTTPException(status_code=400, detail="tanpura_key is required for tanpura_recording ingest.")
        if ingest != "tanpura_recording":
            tanpura_key = None

        with app.state.preprocess_recording_lock:
            if app.state.preprocess_recording_session is not None:
                raise HTTPException(status_code=409, detail="A recording is already in progress.")

            try:
                session = start_microphone_recording_session(
                    audio_dir=audio_dir,
                    filename_base=filename,
                    tanpura_key=tanpura_key,
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            app.state.preprocess_recording_session = session

        tonic = get_tonic_from_tanpura_key(session.tanpura_key) if session.tanpura_key else None
        return {
            "status": "recording",
            "path": session.target_mp3_path,
            "filename": Path(session.target_mp3_path).name,
            "tonic": tonic,
            "audio_input_device": getattr(session, "audio_input_device", None),
        }

    @app.post("/api/preprocess-record/stop")
    def api_preprocess_record_stop() -> Dict[str, Any]:
        with app.state.preprocess_recording_lock:
            session = app.state.preprocess_recording_session
            if session is None:
                raise HTTPException(status_code=409, detail="No active recording session.")
            app.state.preprocess_recording_session = None

        try:
            output_path = Path(session.stop()).resolve()
        except Exception as exc:
            try:
                session.cancel()
            except Exception:
                pass
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        tonic = get_tonic_from_tanpura_key(session.tanpura_key) if session.tanpura_key else None
        return {
            "status": "ready",
            "path": str(output_path),
            "filename": output_path.name,
            "url": _local_file_url(output_path),
            "tonic": tonic,
        }

    @app.post("/api/preprocess-record/cancel")
    def api_preprocess_record_cancel() -> Dict[str, Any]:
        with app.state.preprocess_recording_lock:
            session = app.state.preprocess_recording_session
            app.state.preprocess_recording_session = None

        cancelled = False
        if session is not None:
            cancelled = True
            try:
                session.cancel()
            except Exception:
                pass
        return {"cancelled": cancelled}

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
        mode = payload.mode.strip() if payload.mode else "detect"
        if mode not in {"detect", "analyze"}:
            raise HTTPException(status_code=400, detail="Batch mode must be 'detect' or 'analyze'.")
        if mode == "analyze":
            gt = str(payload.ground_truth or "").strip()
            if not gt:
                raise HTTPException(status_code=400, detail="ground_truth is required for batch mode 'analyze'.")
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
