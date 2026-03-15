"""Runtime fingerprint helpers for report reproducibility and freshness checks."""

from __future__ import annotations

import ast
import hashlib
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TypedDict

FINGERPRINT_VERSION = 2
STAGE_MANIFEST_VERSION = 1
DEFAULT_CACHE_TTL_SECONDS = 5.0

_CACHE_VALUE: Optional[Dict[str, Any]] = None
_CACHE_EXPIRES_AT: float = 0.0


class StageSelector(TypedDict):
    path: str
    type: str
    name: str


def _selector(path: str, kind: str, name: str) -> StageSelector:
    return {"path": path, "type": kind, "name": name}


STAGE_SELECTORS: Dict[str, List[StageSelector]] = {
    "detect": [
        _selector(
            "driver.py",
            "region",
            "# PHASE 1: DETECTION (or load cache)|# PHASE 2: SEQUENCE ANALYSIS",
        ),
        _selector("raga_pipeline/config.py", "function", "build_cli_parser"),
        _selector("raga_pipeline/config.py", "function", "_config_from_parsed_args"),
        _selector("raga_pipeline/config.py", "function", "parse_config_from_argv"),
        _selector("raga_pipeline/audio.py", "function", "separate_stems"),
        _selector("raga_pipeline/audio.py", "function", "_separate_demucs"),
        _selector("raga_pipeline/audio.py", "function", "_separate_spleeter"),
        _selector("raga_pipeline/audio.py", "function", "extract_pitch"),
        _selector("raga_pipeline/audio.py", "function", "_resolve_demucs_device"),
        _selector("raga_pipeline/analysis.py", "function", "compute_cent_histograms_from_config"),
        _selector("raga_pipeline/analysis.py", "function", "detect_peaks_from_config"),
        _selector("raga_pipeline/analysis.py", "function", "fit_gmm_to_peaks"),
        _selector("raga_pipeline/analysis.py", "function", "compute_gmm_bias_cents"),
        _selector("raga_pipeline/raga.py", "method", "RagaScorer.score"),
        _selector("raga_pipeline/raga.py", "function", "score_candidates_full"),
        _selector("raga_pipeline/raga.py", "function", "get_tonic_candidates"),
        _selector("raga_pipeline/raga.py", "function", "_parse_tonic_list"),
        _selector("raga_pipeline/output.py", "function", "generate_detection_report"),
        _selector("raga_pipeline/output.py", "function", "_build_detection_report_metadata"),
        _selector("raga_pipeline/output.py", "function", "write_detection_report_metadata"),
        _selector("raga_pipeline/output.py", "function", "_generate_peak_section"),
        _selector("raga_pipeline/output.py", "function", "_generate_ranking_section"),
        _selector("raga_pipeline/output.py", "function", "_generate_metadata_section"),
        _selector("raga_pipeline/output.py", "function", "_generate_audio_players_section"),
    ],
    "analyze": [
        _selector(
            "driver.py",
            "region",
            "# PHASE 2: SEQUENCE ANALYSIS (Analyze & Full Mode)|def _tonic_name(tonic):",
        ),
        _selector("raga_pipeline/config.py", "function", "build_cli_parser"),
        _selector("raga_pipeline/config.py", "function", "_config_from_parsed_args"),
        _selector("raga_pipeline/config.py", "function", "parse_config_from_argv"),
        _selector("raga_pipeline/sequence.py", "function", "merge_consecutive_notes"),
        _selector("raga_pipeline/sequence.py", "function", "split_phrases_by_silence"),
        _selector("raga_pipeline/sequence.py", "function", "detect_phrases"),
        _selector("raga_pipeline/sequence.py", "function", "cluster_phrases"),
        _selector("raga_pipeline/sequence.py", "function", "build_transition_matrix_corrected"),
        _selector("raga_pipeline/sequence.py", "function", "analyze_raga_patterns"),
        _selector("raga_pipeline/transcription.py", "function", "detect_stationary_events"),
        _selector("raga_pipeline/transcription.py", "function", "detect_pitch_inflection_points"),
        _selector("raga_pipeline/transcription.py", "function", "transcribe_to_notes"),
        _selector("raga_pipeline/raga.py", "function", "apply_raga_correction_to_notes"),
        _selector("raga_pipeline/raga.py", "function", "get_raga_notes"),
        _selector("raga_pipeline/raga.py", "function", "snap_to_raga_notes"),
        _selector("raga_pipeline/raga.py", "function", "_parse_tonic"),
        _selector("raga_pipeline/output.py", "function", "generate_analysis_report"),
        _selector("raga_pipeline/output.py", "function", "_write_analysis_report_metadata"),
        _selector("raga_pipeline/output.py", "function", "_build_analysis_report_metadata"),
        _selector("raga_pipeline/output.py", "function", "_build_transcription_editor_payload"),
        _selector("raga_pipeline/output.py", "function", "_generate_transcription_section"),
        _selector("raga_pipeline/output.py", "function", "create_scrollable_pitch_plot_html"),
        _selector("raga_pipeline/output.py", "function", "_generate_karaoke_section"),
    ],
}


def get_repo_root(start_path: Optional[Path] = None) -> Path:
    """Resolve repository root from an optional starting path."""
    if start_path is None:
        return Path(__file__).resolve().parent.parent
    return Path(start_path).resolve()


def get_critical_paths_for_fingerprint(repo_root: Optional[Path] = None) -> List[Path]:
    """Return critical files that define pipeline/local-app behavior."""
    root = get_repo_root(repo_root)
    paths: List[Path] = [root / "driver.py"]
    paths.extend(sorted((root / "raga_pipeline").glob("*.py"), key=lambda p: p.name.lower()))
    paths.extend(
        [
            root / "local_app" / "server.py",
            root / "local_app" / "schemas.py",
            root / "local_app" / "jobs.py",
            root / "local_app" / "static" / "app.js",
            root / "local_app" / "static" / "transcription_editor.js",
        ]
    )

    deduped: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _normalize_source_text(source: str) -> str:
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines)


def _extract_region_source(file_path: Path, name: str) -> str:
    text = file_path.read_text(encoding="utf-8")
    parts = name.split("|", 1)
    start_marker = parts[0].strip() if parts else ""
    end_marker = parts[1].strip() if len(parts) > 1 else ""

    start_idx = text.find(start_marker) if start_marker else 0
    if start_idx < 0:
        return f"<missing-selector:region:{name}>"
    if end_marker:
        end_idx = text.find(end_marker, start_idx + len(start_marker))
        if end_idx < 0:
            end_idx = len(text)
    else:
        end_idx = len(text)
    return text[start_idx:end_idx]


def _extract_ast_node_source(tree: ast.AST, source: str, selector: StageSelector) -> str:
    kind = selector["type"]
    name = selector["name"]
    target_node: Optional[ast.AST] = None

    if kind == "function":
        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                target_node = node
                break
    elif kind == "method":
        class_name, sep, method_name = name.partition(".")
        if not sep:
            return f"<missing-selector:method:{name}>"
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                        target_node = child
                        break
                break

    if target_node is None:
        return f"<missing-selector:{kind}:{name}>"
    segment = ast.get_source_segment(source, target_node)
    if segment is None:
        return f"<missing-selector:{kind}:{name}>"
    return segment


def _extract_selector_source(repo_root: Path, selector: StageSelector) -> str:
    rel_path = selector["path"]
    file_path = (repo_root / rel_path).resolve()
    if not file_path.exists() or not file_path.is_file():
        return f"<missing-selector:file:{rel_path}>"

    if selector["type"] == "region":
        return _extract_region_source(file_path, selector["name"])

    source = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(file_path))
    except Exception:
        return f"<missing-selector:parse:{rel_path}:{selector['type']}:{selector['name']}>"
    return _extract_ast_node_source(tree, source, selector)


def build_stage_hashes(repo_root: Optional[Path] = None) -> Dict[str, str]:
    root = get_repo_root(repo_root)
    result: Dict[str, str] = {}

    for stage, selectors in STAGE_SELECTORS.items():
        digest = hashlib.sha256()
        digest.update(f"manifest_version:{STAGE_MANIFEST_VERSION}".encode("utf-8"))
        digest.update(b"\0")

        for selector in selectors:
            selector_id = f"{selector['path']}:{selector['type']}:{selector['name']}"
            digest.update(selector_id.encode("utf-8"))
            digest.update(b"\0")
            source = _extract_selector_source(root, selector)
            normalized = _normalize_source_text(source)
            digest.update(normalized.encode("utf-8"))
            digest.update(b"\0")

        result[stage] = digest.hexdigest()
    return result


def compute_file_hash(paths: Iterable[Path], repo_root: Optional[Path] = None) -> str:
    """Compute deterministic SHA256 across files (path + content)."""
    root = get_repo_root(repo_root)
    digest = hashlib.sha256()
    for path in sorted([Path(p).resolve() for p in paths], key=lambda p: str(p).lower()):
        try:
            rel = path.relative_to(root)
            rel_txt = str(rel)
        except Exception:
            rel_txt = str(path)
        digest.update(rel_txt.encode("utf-8"))
        digest.update(b"\0")

        if path.exists() and path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _run_git(args: List[str], repo_root: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    value = proc.stdout.strip()
    return value or None


def _git_commit_and_dirty(repo_root: Path) -> tuple[Optional[str], bool]:
    commit = _run_git(["rev-parse", "HEAD"], repo_root)
    if commit is None:
        return None, False

    dirty_output = _run_git(["status", "--porcelain"], repo_root)
    dirty = bool(dirty_output)
    return commit, dirty


def build_runtime_fingerprint(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Build an uncached runtime fingerprint payload."""
    root = get_repo_root(repo_root)
    critical_paths = get_critical_paths_for_fingerprint(root)
    file_hash = compute_file_hash(critical_paths, root)
    stage_hashes = build_stage_hashes(root)
    git_commit, git_dirty = _git_commit_and_dirty(root)

    source = "git+files" if git_commit else "files-only"
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "stage_manifest_version": STAGE_MANIFEST_VERSION,
        "stage_hashes": stage_hashes,
        "git_commit": git_commit,
        "git_dirty": bool(git_dirty),
        "file_hash": file_hash,
        "source": source,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def get_runtime_fingerprint(
    repo_root: Optional[Path] = None,
    *,
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Return runtime fingerprint, cached briefly for repeated API calls."""
    global _CACHE_EXPIRES_AT, _CACHE_VALUE

    now = time.time()
    if (
        not force_refresh
        and _CACHE_VALUE is not None
        and cache_ttl_seconds > 0
        and now < _CACHE_EXPIRES_AT
    ):
        return dict(_CACHE_VALUE)

    fingerprint = build_runtime_fingerprint(repo_root)
    _CACHE_VALUE = dict(fingerprint)
    _CACHE_EXPIRES_AT = now + max(float(cache_ttl_seconds), 0.0)
    return fingerprint
