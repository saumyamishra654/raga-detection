"""
Configuration module for the raga detection pipeline.

Provides:
- PipelineConfig: Dataclass with all pipeline parameters
- load_config_from_cli: CLI argument parser
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


VALID_MODES = {"preprocess", "detect", "analyze"}
VALID_PREPROCESS_INGESTS = {"yt", "recording", "tanpura_recording"}
LEGACY_PREPROCESS_INGEST_ALIASES = {
    "youtube": "yt",
    "record": "recording",
}
LEGACY_RECORD_MODE_TO_INGEST = {
    "song": "recording",
    "tanpura_vocal": "tanpura_recording",
}
TANPURA_KEYS = ["A", "Bb", "B", "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab"]

# Per-extractor default confidence thresholds.
# pYIN voiced_probs are much lower than SwiftF0 (melody median ~0.18, accomp max ~0.76).
# These defaults are applied when the user does not explicitly pass
# --vocal-confidence / --accomp-confidence on the CLI.
EXTRACTOR_CONFIDENCE_DEFAULTS = {
    "swiftf0": {"vocal": 0.95, "accomp": 0.80},
    "pyin":    {"vocal": 0.15, "accomp": 0.05},
}


def _clean_optional_str(value: Optional[str]) -> Optional[str]:
    """Trim optional CLI strings and normalize empty strings to None."""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


@dataclass
class PipelineConfig:
    """config for raga detection pipeline"""

    # mandatory paths
    audio_path: Optional[str]
    output_dir: str

    # Optional preprocess inputs (YouTube ingest)
    yt_url: Optional[str] = None
    audio_dir: Optional[str] = None
    filename_override: Optional[str] = None
    preprocess_start_time: Optional[str] = None
    preprocess_end_time: Optional[str] = None
    preprocess_ingest: Optional[str] = None  # "yt", "recording", or "tanpura_recording"
    preprocess_tanpura_key: Optional[str] = None
    preprocess_recorded_audio: Optional[str] = None

    # separator settings
    separator_engine: str = "demucs"  # 'demucs' or 'spleeter'
    demucs_model: str = "htdemucs"    # htdemucs, htdemucs_ft, mdx, mdx_extra

    # source type - determines stem separation behavior
    source_type: str = "mixed"  # "mixed" (stem separation), "instrumental", "vocal"

    # Melody source - determines which audio to use for melody analysis
    melody_source: str = "separated" # "separated" (vocals stem), "composite" (original mix)

    # vocalist gender - only used when source_type="vocal" (affects tonic bias)
    vocalist_gender: Optional[str] = None  # "male", "female", or None (auto)

    # instrument type - only used when source_type="instrumental" (affects tonic bias)
    instrument_type: str = "autodetect"  # "sitar", "sarod", "bansuri", "slide_guitar", "autodetect"

    # skip stem separation (detect mode): use original audio directly for melody analysis
    skip_separation: bool = False
    skip_separation_forced_composite: bool = False

    # pitch extraction confidence thresholds
    vocal_confidence: float = 0.95
    accomp_confidence: float = 0.80

    # Pitch range (MIDI notes)
    fmin_note: str = "G1"   # ~49 Hz (notebook default)
    fmax_note: str = "D6"   # Extended upper range for taans/small movements

    # Histogram parameters
    histogram_bins_high: int = 100   # High-res: 100 bins
    histogram_bins_low: int = 33    # Low-res: 33 bins
    smoothing_sigma: float = 0.8     # Gaussian smoothing kernel width
    use_confidence_weights: bool = True  # Notebook uses unweighted histograms

    # Peak detection parameters
    tolerance_cents: float = 35.0         # ±35¢ for note mapping
    peak_tolerance_cents: float = 45.0    # Cross-resolution validation window
    prominence_high_factor: float = 0.01  # min prominence = factor * max (lowered to catch weak peaks)
    prominence_low_factor: float = 0.03  # (lowered to catch weak peaks)

    # Note detection parameters (will evolve with new stationary point methods)
    note_min_duration: float = 0.1        # Minimum note duration (seconds)
    pitch_change_threshold: float = 0.3   # Semitone threshold for note boundary
    derivative_threshold: float = 0.15    # For stationary point detection
    smoothing_method: str = "gaussian"    # 'gaussian' or 'median'
    smoothing_note_sigma: float = 1.5     # Smoothing kernel for note detection
    snap_to_semitones: bool = True
    transcription_smoothing_ms: float = 0.0 # Transcription pre-smoothing sigma (ms)
    transcription_min_duration: float = 0.02 # Min duration for stationary notes (20ms)
    transcription_derivative_threshold: float = 4.0 # Stability threshold (semitones/sec)
    energy_threshold: float = 0.0 # Per-track normalized energy threshold (0-1) for note gating
    energy_metric: str = "rms"  # 'rms' (peak-normalised) or 'log_amp' (dBFS, percentile-normalised)

    # Pitch extractor selection
    pitch_extractor: str = "swiftf0"  # "swiftf0" or "pyin"
    pitch_hop_ms: float = 0.0         # 0 = extractor default; applies to pyin only
    compare_extractors: bool = False  # Analyze: run both SwiftF0 and pYIN, show toggled report

    # Phrase detection parameters
    phrase_max_gap: float = 1.0           # Max silence between notes in phrase
    phrase_min_length: int = 1            # Minimum notes per phrase
    phrase_min_duration: float = 0.2      # Minimum phrase duration in seconds

    # Silence-based phrase splitting (RMS energy)
    # When > 0, phrases are additionally split at points where vocal RMS
    # drops below this fraction of the track's peak energy for at least
    # silence_min_duration seconds.
    silence_threshold: float = 0.10       # 0 = disabled; 0.10 = 10% of peak energy
    silence_min_duration: float = 0.25    # Min consecutive low-energy seconds to count as silence

    # RMS overlay on pitch plots
    show_rms_overlay: bool = True         # Show energy trace on pitch analysis plots

    # ml params (currently unused)
    use_ml_model: bool = False  # Disabled per migration plan
    model_path: Optional[str] = None      # Path to trained model (unused)

    # LM scoring (detect mode)
    use_lm_scoring: bool = False
    lm_model_path: Optional[str] = None
    lm_skip_correction: bool = False
    lm_deletion_lambda: float = 2.0
    lm_deletion_slope: float = -0.0684
    lm_deletion_intercept: float = 0.6640

    # db paths
    raga_db_path: Optional[str] = None    # Auto-locates if None

    # processing options
    force_recompute: bool = False         # Force pitch extraction only (stems reused if present)
    force_stem_recompute: bool = False    # Detect only: force stem separation recomputation
    save_intermediates: bool = True       # Save CSVs, plots, etc.
    skip_report: bool = False             # Skip HTML report generation (candidates.csv still saved)
    pitch_only: bool = False              # Detect: exit after pitch extraction (steps 1-2 only); skip histogram, scoring, report
    transcription_only: bool = False      # Analyze: produce only transcribed_notes.csv; skip plots, GMM, report

    # GMM parameters
    gmm_window_cents: float = 150.0
    gmm_components: int = 1
    bias_rotation: bool = True

    # execution mode
    mode: str = "detect"  # "preprocess", "detect", or "analyze"
    tonic_override: Optional[str] = None
    raga_override: Optional[str] = None
    keep_impure_notes: bool = False
    strict_raga_35c_filter: bool = False
    strict_raga_max_cents: float = 35.0
    skip_raga_correction: bool = False

    def __post_init__(self):
        """Validate and normalize paths."""
        self.audio_path = _clean_optional_str(self.audio_path)
        self.yt_url = _clean_optional_str(self.yt_url)
        self.audio_dir = _clean_optional_str(self.audio_dir)
        self.filename_override = _clean_optional_str(self.filename_override)
        self.preprocess_ingest = _clean_optional_str(self.preprocess_ingest)
        self.preprocess_tanpura_key = _clean_optional_str(self.preprocess_tanpura_key)
        self.preprocess_recorded_audio = _clean_optional_str(self.preprocess_recorded_audio)
        self.tonic_override = _clean_optional_str(self.tonic_override)
        self.raga_override = _clean_optional_str(self.raga_override)

        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        if self.mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{self.mode}'. Expected one of: {sorted(VALID_MODES)}")

        if self.mode == "preprocess":
            if not self.audio_dir:
                raise ValueError("Preprocess mode requires --audio-dir")
            if not self.filename_override:
                raise ValueError("Preprocess mode requires --filename")
            if not self.preprocess_ingest:
                raise ValueError("Preprocess mode requires --ingest")
            if self.preprocess_ingest not in VALID_PREPROCESS_INGESTS:
                raise ValueError(
                    f"Invalid preprocess ingest '{self.preprocess_ingest}'. "
                    f"Expected one of: {sorted(VALID_PREPROCESS_INGESTS)}"
                )

            self.audio_dir = os.path.abspath(self.audio_dir)
            os.makedirs(self.audio_dir, exist_ok=True)
            self.audio_path = os.path.join(self.audio_dir, f"{self.filename_override}.mp3")

            if self.preprocess_ingest == "yt":
                if not self.yt_url:
                    raise ValueError("Preprocess mode with --ingest yt requires --yt")
            else:
                if self.preprocess_start_time or self.preprocess_end_time:
                    raise ValueError("--start-time and --end-time are only valid when --ingest yt.")
                if (
                    self.preprocess_ingest == "tanpura_recording"
                    and not self.preprocess_tanpura_key
                ):
                    raise ValueError(
                        "Preprocess mode with --ingest tanpura_recording requires --tanpura-key."
                    )
                if self.preprocess_tanpura_key and self.preprocess_tanpura_key not in TANPURA_KEYS:
                    raise ValueError(
                        f"Invalid --tanpura-key '{self.preprocess_tanpura_key}'. "
                        f"Expected one of: {TANPURA_KEYS}"
                    )
                if self.preprocess_recorded_audio:
                    self.preprocess_recorded_audio = os.path.abspath(self.preprocess_recorded_audio)
                    if not os.path.isfile(self.preprocess_recorded_audio):
                        raise FileNotFoundError(
                            f"Recorded audio file not found: {self.preprocess_recorded_audio}"
                        )
        else:
            if not self.audio_path:
                raise ValueError(f"{self.mode.capitalize()} mode requires --audio/-a")
            self.audio_path = os.path.abspath(self.audio_path)

            if not os.path.isfile(self.audio_path):
                raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

        if self.mode == "detect" and self.force_stem_recompute and not self.force_recompute:
            raise ValueError("Detect mode with --force-stems requires --force.")

        if self.use_lm_scoring and not self.lm_model_path:
            self.lm_model_path = self._find_lm_model_path()
        if self.use_lm_scoring and not self.lm_model_path:
            raise ValueError("--use-lm-scoring requires --lm-model (path to trained n-gram model JSON)")
        if self.use_lm_scoring and self.lm_model_path and not os.path.exists(self.lm_model_path):
            raise ValueError(f"--lm-model path does not exist: {self.lm_model_path}")

        if self.mode == "detect" and self.skip_separation:
            tonic_tokens = [
                token.strip()
                for token in str(self.tonic_override or "").split(",")
                if token.strip()
            ]
            if not tonic_tokens:
                raise ValueError("Detect mode with --skip-separation requires --tonic.")
            if self.melody_source != "composite":
                self.melody_source = "composite"
                self.skip_separation_forced_composite = True

        if not math.isfinite(float(self.strict_raga_max_cents)) or float(self.strict_raga_max_cents) < 0.0:
            raise ValueError("--strict-raga-max-cents must be a finite non-negative value.")

        # auto-locate model and database if not specified
        if self.model_path is None:
            self.model_path = self._find_model_path()
        if self.raga_db_path is None:
            self.raga_db_path = self._find_raga_db_path()

    def _find_model_path(self) -> Optional[str]:
        """Find trained model in standard locations."""
        package_dir = Path(__file__).parent.parent
        candidates = [
            package_dir / "models" / "raga_mlp_model.pkl",
            package_dir / "raga_mlp_model.pkl",
            package_dir.parent / "raga_mlp_model.pkl",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    def _find_lm_model_path(self) -> Optional[str]:
        """Find trained n-gram LM in standard locations."""
        project_root = Path(__file__).parent.parent
        candidates = [
            project_root / "compmusic_ngram_model_uncorrected.json",
            project_root / "raga_pipeline" / "models" / "compmusic_ngram_model_uncorrected.json",
            project_root / "compmusic_ngram_model.json",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    def _find_raga_db_path(self) -> Optional[str]:
        """Find raga database in standard locations."""
        return find_default_raga_db_path()

    @property
    def filename(self) -> str:
        """Extract filename without extension from audio path."""
        if self.audio_path:
            return os.path.splitext(os.path.basename(self.audio_path))[0]
        if self.filename_override:
            return self.filename_override
        return "unknown_audio"

    @property
    def stem_dir(self) -> str:
        """Directory where stems are saved."""
        # different subdirectory for different separators
        if self.separator_engine == "spleeter":
            subdir = "spleeter"
        else:
            subdir = self.demucs_model  # e.g., htdemucs
        return os.path.join(self.output_dir, subdir, self.filename)

    @property
    def vocals_path(self) -> str:
        """Path to separated vocals file."""
        return self._resolve_stem_path("vocals")

    @property
    def accompaniment_path(self) -> str:
        """Path to separated accompaniment file."""
        return self._resolve_stem_path("accompaniment")

    def _resolve_stem_path(self, stem_name: str) -> str:
        """Prefer MP3 stems, but fall back to legacy WAV if present."""
        mp3_path = os.path.join(self.stem_dir, f"{stem_name}.mp3")
        wav_path = os.path.join(self.stem_dir, f"{stem_name}.wav")
        if os.path.exists(mp3_path):
            return mp3_path
        if os.path.exists(wav_path):
            return wav_path
        return mp3_path


def _add_common_args(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument("--audio", "-a", required=required, help="Input audio file (relative or absolute path)")
    parser.add_argument("--output", "-o", default="batch_results", help="Parent directory for output results")
    parser.add_argument("--separator", choices=["demucs", "spleeter"], default="demucs", help="Stem separation engine to use")
    parser.add_argument("--demucs-model", default="htdemucs", help="Specific Demucs model (e.g., htdemucs, mdx_extra)")

    # Source and Melody settings
    parser.add_argument(
        "--source-type",
        choices=["mixed", "instrumental", "vocal"],
        default="mixed",
        help="Audio source type: 'vocal' enables gender tonic bias, 'instrumental' enables instrument tonic bias",
    )
    parser.add_argument(
        "--melody-source",
        choices=["separated", "composite"],
        default="separated",
        help="Use 'separated' (stem) or 'composite' (full mix) for melody pitch extraction",
    )
    parser.add_argument("--vocalist-gender", choices=["male", "female"], help="Vocalist gender for tonic biasing (only used for source-type=vocal)")
    parser.add_argument(
        "--instrument-type",
        choices=["autodetect", "sitar", "sarod", "bansuri", "slide_guitar"],
        default="autodetect",
        help="Instrument type for tonic biasing (only used for source-type=instrumental)",
    )

    # Pitch Extraction settings
    parser.add_argument(
        "--fmin-note",
        default="G1",
        help=(
            "Minimum note for pitch extraction (e.g., G1). "
            "For taan-heavy recordings, raise to A1/B1 to reduce low-frequency noise."
        ),
    )
    parser.add_argument(
        "--fmax-note",
        default="D6",
        help=(
            "Maximum note for pitch extraction (e.g., D6). "
            "Raise for fast high taans; lower if octave-jump false positives appear."
        ),
    )
    parser.add_argument(
        "--vocal-confidence",
        type=float,
        default=None,
        help=(
            "Confidence threshold (0-1) for melody pitch data. "
            "Default depends on --pitch-extractor: swiftf0=0.95, pyin=0.15. "
            "Lower to retain subtle taans; raise to suppress noisy transients."
        ),
    )
    parser.add_argument(
        "--accomp-confidence",
        type=float,
        default=None,
        help=(
            "Confidence threshold (0-1) for accompaniment pitch data. "
            "Default depends on --pitch-extractor: swiftf0=0.80, pyin=0.05."
        ),
    )

    # Pitch extractor selection
    parser.add_argument(
        "--pitch-extractor",
        choices=["swiftf0", "pyin"],
        default="swiftf0",
        help="Pitch extraction backend. swiftf0 (default, 16ms hop fixed), "
             "pyin (librosa, configurable hop).",
    )
    parser.add_argument(
        "--pitch-hop-ms",
        type=float,
        default=0.0,
        help="Pitch frame hop in milliseconds (pyin only). "
             "0 = extractor default (~23ms). Lower for drut passages (e.g. 5).",
    )
    parser.add_argument(
        "--compare-extractors",
        action="store_true",
        help="Analyze mode: run both SwiftF0 and pYIN, calibrate confidence "
             "thresholds so both produce the same number of raw notes, and show "
             "both transcriptions in the report with a toggle.",
    )

    # Peak Detection settings
    parser.add_argument("--prominence-high", type=float, default=0.01, help="Prominence threshold factor for high-res peak detection")
    parser.add_argument("--prominence-low", type=float, default=0.03, help="Prominence threshold factor for low-res peak detection")
    parser.add_argument(
        "--bias-rotation",
        action="store_false",
        dest="bias_rotation",
        help="Disable histogram bias rotation (enabled by default)",
    )

    # Miscellaneous
    parser.add_argument("--force", "-f", action="store_true", help="Force recompute pitch extraction (reuses existing stems if found)")
    parser.add_argument(
        "--skip-separation",
        action="store_true",
        help="Detect mode only: skip stem separation (fast path). Uncheck this in UI to enable denoising via stem separation. Requires --tonic and forces --melody-source composite",
    )
    parser.add_argument("--raga-db", help="Override path to raga database CSV")
    parser.add_argument("--skip-report", action="store_true", help="Skip HTML report generation (candidates.csv and pitch caches are still saved)")
    parser.add_argument("--pitch-only", action="store_true", help="Detect: exit after stem separation + pitch extraction (steps 1-2 only). Skips histogram, scoring, and report. Use for batch runs where only analyze/motif mining is needed downstream")
    parser.add_argument("--transcription-only", action="store_true", help="Analyze: produce only transcribed_notes.csv. Skips all plots, GMM analysis, and HTML report. Use for motif mining batch runs")


def _normalize_preprocess_argv_aliases(argv: Sequence[str]) -> List[str]:
    """Normalize legacy preprocess ingest flags/values to canonical tokens."""
    normalized: List[str] = []
    pending_record_mode_ingest: Optional[str] = None
    ingest_value_index: Optional[int] = None
    idx = 0

    while idx < len(argv):
        token = str(argv[idx])

        if token == "--record-mode":
            if idx + 1 < len(argv):
                record_mode_value = str(argv[idx + 1]).strip()
                pending_record_mode_ingest = LEGACY_RECORD_MODE_TO_INGEST.get(
                    record_mode_value,
                    record_mode_value,
                )
                idx += 2
            else:
                idx += 1
            continue

        if token.startswith("--record-mode="):
            record_mode_value = token.split("=", 1)[1].strip()
            pending_record_mode_ingest = LEGACY_RECORD_MODE_TO_INGEST.get(
                record_mode_value,
                record_mode_value,
            )
            idx += 1
            continue

        if token == "--ingest":
            normalized.append(token)
            if idx + 1 < len(argv):
                raw_value = str(argv[idx + 1]).strip()
                mapped_value = LEGACY_PREPROCESS_INGEST_ALIASES.get(raw_value, raw_value)
                normalized.append(mapped_value)
                ingest_value_index = len(normalized) - 1
                idx += 2
            else:
                idx += 1
            continue

        if token.startswith("--ingest="):
            raw_value = token.split("=", 1)[1].strip()
            mapped_value = LEGACY_PREPROCESS_INGEST_ALIASES.get(raw_value, raw_value)
            normalized.append(f"--ingest={mapped_value}")
            ingest_value_index = len(normalized) - 1
            idx += 1
            continue

        normalized.append(token)
        idx += 1

    is_preprocess = any(token == "preprocess" for token in normalized)
    if pending_record_mode_ingest and is_preprocess:
        if ingest_value_index is not None:
            ingest_token = normalized[ingest_value_index]
            if ingest_token.startswith("--ingest="):
                normalized[ingest_value_index] = f"--ingest={pending_record_mode_ingest}"
            else:
                normalized[ingest_value_index] = pending_record_mode_ingest
        else:
            normalized.extend(["--ingest", pending_record_mode_ingest])

    return normalized


def build_cli_parser() -> argparse.ArgumentParser:
    """Create and return the canonical CLI parser for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Raga Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # create subparsers for modes: preprocess, detect, analyze
    subparsers = parser.add_subparsers(dest="command", help="Pipeline mode")

    # --- Detect Mode ---
    detect_parser = subparsers.add_parser("detect", help="Phase 1: Detection only")
    _add_common_args(detect_parser, required=True)
    detect_parser.add_argument("--tonic", help="Force tonic (comma-separated allowed, e.g. C,D#)")
    detect_parser.add_argument("--raga", help="Force raga name")
    detect_parser.add_argument(
        "--force-stems",
        dest="force_stem_recompute",
        action="store_true",
        help="Detect mode only: requires --force and also forces stem-separation recomputation.",
    )
    detect_parser.add_argument("--use-lm-scoring", action="store_true",
                               help="Re-rank candidates using n-gram language model (writes lm_candidates.csv)")
    detect_parser.add_argument("--lm-model", dest="lm_model_path", default=None,
                               help="Path to trained n-gram model JSON (auto-discovered if omitted)")
    detect_parser.add_argument("--lm-skip-correction", action="store_true",
                               help="Score uncorrected chromatic transcription (no per-raga correction). "
                                    "Use with a model trained on uncorrected transcriptions.")
    detect_parser.add_argument("--lm-deletion-lambda", type=float, default=2.0,
                               help="Weight for deletion residual in combined LM scoring (default: 2.0)")
    detect_parser.add_argument("--lm-deletion-slope", type=float, default=-0.0684,
                               help="Regression slope for expected deletion vs scale size (default: -0.0684)")
    detect_parser.add_argument("--lm-deletion-intercept", type=float, default=0.6640,
                               help="Regression intercept for expected deletion vs scale size (default: 0.6640)")

    # --- Analyze Mode ---
    analyze_parser = subparsers.add_parser("analyze", help="Phase 2: Analysis only")
    _add_common_args(analyze_parser, required=True)
    analyze_parser.add_argument("--tonic", required=True, help="Tonic (e.g. C, D#)")
    analyze_parser.add_argument("--raga", required=True, help="Raga name")
    analyze_parser.add_argument("--skip-raga-correction", action="store_true",
                                   help="Skip post-transcription raga correction (keep chromatic transcription as-is)")
    analyze_parser.add_argument("--keep-impure-notes", action="store_true", help="Keep notes not in raga (default: remove)")
    analyze_parser.add_argument(
        "--strict-raga-35c-filter",
        action="store_true",
        help=(
            "Discard notes farther than 35 cents from the nearest valid raga note "
            "(disables impure-note keeping while enabled)."
        ),
    )
    analyze_parser.add_argument(
        "--strict-raga-max-cents",
        type=float,
        default=35.0,
        help=(
            "Maximum allowed distance in cents from nearest valid raga note when "
            "--strict-raga-35c-filter is enabled."
        ),
    )
    analyze_parser.add_argument(
        "--transcription-smoothing-ms",
        type=float,
        default=0.0,
        help=(
            "Smoothing sigma (ms) for transcription. Set to 0 to disable. "
            "Keep low (0-20ms) for taans; higher values can blur small movements."
        ),
    )
    analyze_parser.add_argument(
        "--transcription-min-duration",
        type=float,
        default=0.02,
        help=(
            "Minimum duration (s) for a transcribed note. "
            "Use 0.01-0.03 for short taan fragments."
        ),
    )
    analyze_parser.add_argument(
        "--transcription-stability-threshold",
        type=float,
        default=4.0,
        help=(
            "Max pitch change (semitones/sec) to be considered stable. "
            "Increase to capture faster note motions in taans."
        ),
    )
    analyze_parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.0,
        help=(
            "Per-track normalized energy threshold (0.0-1.0) for transcription note gating. "
            "This is relative to the selected melody track (not absolute loudness); "
            "typical RMS range is ~0.03 to 0.12. "
            "For soft taans, keep this low (0.0-0.05)."
        ),
    )
    analyze_parser.add_argument(
        "--energy-metric",
        choices=["rms", "log_amp"],
        default="rms",
        help="Energy metric: 'rms' (peak-normalised) or 'log_amp' (dBFS, percentile-scaled). "
        "log_amp gives better dynamic range for quiet passages.",
    )
    analyze_parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.10,
        help="RMS energy threshold (0.0-1.0) for silence-based phrase splitting. 0 disables.",
    )
    analyze_parser.add_argument("--silence-min-duration", type=float, default=0.25, help="Minimum silence duration (seconds) to trigger a phrase break.")
    analyze_parser.add_argument("--phrase-min-duration", type=float, default=0.2, help="Exclude phrases shorter than this duration (seconds).")
    analyze_parser.add_argument("--phrase-min-notes", type=int, default=1, help="Exclude phrases with fewer notes than this count.")
    analyze_parser.add_argument("--no-rms-overlay", action="store_true", help="Disable RMS energy overlay on pitch analysis plots.")
    analyze_parser.add_argument("--no-smoothing", action="store_true", help="Disable transcription smoothing")

    # --- Preprocess Mode ---
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Ingest YouTube or recorded audio to a local MP3 and exit",
    )
    preprocess_parser.add_argument(
        "--ingest",
        choices=["yt", "recording", "tanpura_recording"],
        required=True,
        help="Preprocess ingest path: yt, recording, or tanpura_recording",
    )
    preprocess_parser.add_argument("--yt", help="YouTube URL to download (required when --ingest yt)")
    preprocess_parser.add_argument("--audio-dir", default="../audio_test_files", help="Directory where downloaded MP3 should be saved")
    preprocess_parser.add_argument(
        "--filename",
        required=True,
        help="Output filename base (without extension); saved as <filename>.mp3",
    )
    preprocess_parser.add_argument(
        "--tanpura-key",
        choices=TANPURA_KEYS,
        help="Tanpura key/tonic for tanpura recording ingest",
    )
    preprocess_parser.add_argument(
        "--recorded-audio",
        help="Existing recorded audio path (used by local app uploads). "
        "If omitted with recording/tanpura_recording ingest, CLI uses live microphone capture.",
    )
    preprocess_parser.add_argument("--start-time", help="Optional trim start time (SS, MM:SS, or HH:MM:SS)")
    preprocess_parser.add_argument("--end-time", help="Optional trim end time (SS, MM:SS, or HH:MM:SS)")
    preprocess_parser.add_argument(
        "--output",
        "-o",
        default="batch_results",
        help="Default detect output directory suggestion for next-step command",
    )

    # --- Root Parser (Legacy support - defaults to detect) ---
    _add_common_args(parser, required=False)
    parser.add_argument("--tonic", help="Force tonic (comma-separated allowed, e.g. C,D#)")
    parser.add_argument("--raga", help="Force raga")
    parser.add_argument(
        "--force-stems",
        dest="force_stem_recompute",
        action="store_true",
        help="Detect mode only: requires --force and also forces stem-separation recomputation.",
    )

    return parser


def _config_from_parsed_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> PipelineConfig:
    # Validation logic
    if args.command is None:
        if not args.audio:
            parser.error("the following arguments are required: --audio/-a")

    if args.vocalist_gender and args.source_type != "vocal":
        args.source_type = "vocal"

    # Determine effective mode (default to detect)
    mode = args.command if args.command in VALID_MODES else "detect"

    # Validate --compare-extractors
    compare_extractors = getattr(args, 'compare_extractors', False)
    if compare_extractors and mode != "analyze":
        parser.error("--compare-extractors is only supported in analyze mode.")

    # Resolve extractor-specific confidence defaults when user did not override
    extractor = getattr(args, 'pitch_extractor', 'swiftf0')
    ext_defaults = EXTRACTOR_CONFIDENCE_DEFAULTS.get(
        extractor, EXTRACTOR_CONFIDENCE_DEFAULTS["swiftf0"]
    )
    raw_vocal_conf = getattr(args, 'vocal_confidence', None)
    raw_accomp_conf = getattr(args, 'accomp_confidence', None)
    resolved_vocal_conf = raw_vocal_conf if raw_vocal_conf is not None else ext_defaults["vocal"]
    resolved_accomp_conf = raw_accomp_conf if raw_accomp_conf is not None else ext_defaults["accomp"]

    return PipelineConfig(
        audio_path=getattr(args, 'audio', None),
        output_dir=getattr(args, 'output', 'batch_results'),
        yt_url=getattr(args, 'yt', None),
        audio_dir=getattr(args, 'audio_dir', None),
        filename_override=getattr(args, 'filename', None),
        preprocess_start_time=getattr(args, 'start_time', None),
        preprocess_end_time=getattr(args, 'end_time', None),
        preprocess_ingest=getattr(args, 'ingest', None),
        preprocess_tanpura_key=getattr(args, 'tanpura_key', None),
        preprocess_recorded_audio=getattr(args, 'recorded_audio', None),
        separator_engine=getattr(args, 'separator', 'demucs'),
        demucs_model=getattr(args, 'demucs_model', 'htdemucs'),
        source_type=getattr(args, 'source_type', 'mixed'),
        vocalist_gender=getattr(args, 'vocalist_gender', None),
        instrument_type=getattr(args, 'instrument_type', 'autodetect'),
        skip_separation=getattr(args, 'skip_separation', False),
        vocal_confidence=resolved_vocal_conf,
        accomp_confidence=resolved_accomp_conf,
        force_recompute=getattr(args, 'force', False),
        force_stem_recompute=getattr(args, 'force_stem_recompute', False),
        skip_report=getattr(args, 'skip_report', False),
        pitch_only=getattr(args, 'pitch_only', False),
        transcription_only=getattr(args, 'transcription_only', False),
        raga_db_path=getattr(args, 'raga_db', None),
        use_lm_scoring=getattr(args, 'use_lm_scoring', False),
        lm_model_path=getattr(args, 'lm_model_path', None),
        lm_skip_correction=getattr(args, 'lm_skip_correction', False),
        lm_deletion_lambda=getattr(args, 'lm_deletion_lambda', 2.0),
        lm_deletion_slope=getattr(args, 'lm_deletion_slope', -0.0684),
        lm_deletion_intercept=getattr(args, 'lm_deletion_intercept', 0.6640),
        mode=mode,
        tonic_override=getattr(args, 'tonic', None),
        raga_override=getattr(args, 'raga', None),
        skip_raga_correction=getattr(args, 'skip_raga_correction', False),
        keep_impure_notes=getattr(args, 'keep_impure_notes', False),
        strict_raga_35c_filter=getattr(args, 'strict_raga_35c_filter', False),
        strict_raga_max_cents=getattr(args, 'strict_raga_max_cents', 35.0),
        transcription_smoothing_ms=0.0 if getattr(args, 'no_smoothing', False) else getattr(args, 'transcription_smoothing_ms', 0.0),
        transcription_min_duration=getattr(args, 'transcription_min_duration', 0.02),
        transcription_derivative_threshold=getattr(args, 'transcription_stability_threshold', 4.0),
        energy_threshold=getattr(args, 'energy_threshold', 0.0),
        energy_metric=getattr(args, 'energy_metric', 'rms'),
        pitch_extractor=getattr(args, 'pitch_extractor', 'swiftf0'),
        pitch_hop_ms=getattr(args, 'pitch_hop_ms', 0.0),
        compare_extractors=compare_extractors,
        silence_threshold=getattr(args, 'silence_threshold', 0.0),
        silence_min_duration=getattr(args, 'silence_min_duration', 0.25),
        phrase_min_duration=getattr(args, 'phrase_min_duration', 0.2),
        phrase_min_length=getattr(args, 'phrase_min_notes', 1),
        show_rms_overlay=not getattr(args, 'no_rms_overlay', False),
        melody_source=getattr(args, 'melody_source', "separated"),
        fmin_note=getattr(args, 'fmin_note', "G1"),
        fmax_note=getattr(args, 'fmax_note', "D6"),
        prominence_high_factor=getattr(args, 'prominence_high', 0.01),
        prominence_low_factor=getattr(args, 'prominence_low', 0.03),
        bias_rotation=getattr(args, 'bias_rotation', True),
    )


def parse_config_from_argv(argv: Sequence[str]) -> PipelineConfig:
    """
    Parse PipelineConfig from an explicit argv list.

    Args:
        argv: Command-line style arguments, excluding executable name.
    """
    parser = build_cli_parser()
    normalized_argv = _normalize_preprocess_argv_aliases(list(argv))
    args = parser.parse_args(normalized_argv)
    return _config_from_parsed_args(args, parser)


def load_config_from_cli() -> PipelineConfig:
    """Parse command-line arguments from sys.argv and return configuration."""
    parser = build_cli_parser()
    normalized_argv = _normalize_preprocess_argv_aliases(sys.argv[1:])
    args = parser.parse_args(normalized_argv)
    return _config_from_parsed_args(args, parser)


def create_config(audio_path: str, output_dir: str, **kwargs) -> PipelineConfig:
    """
    Convenience function to create configuration programmatically.

    Args:
        audio_path: Path to input audio file
        output_dir: Output directory for results
        **kwargs: Override any default configuration values

    Returns:
        PipelineConfig instance
    """
    return PipelineConfig(audio_path=audio_path, output_dir=output_dir, **kwargs)


def get_default_raga_db_candidates() -> List[Path]:
    """Return candidate paths for the default raga database CSV."""
    package_dir = Path(__file__).parent.parent
    return [
        package_dir / "data" / "raga_list_final.csv",
        package_dir / "main notebooks" / "raga_list_final.csv",
        package_dir.parent / "db" / "raga_list_final.csv",
    ]


def find_default_raga_db_path() -> Optional[str]:
    """Find raga database in standard locations."""
    for path in get_default_raga_db_candidates():
        if path.exists():
            return str(path)
    return None
