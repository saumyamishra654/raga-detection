"""
Configuration module for the raga detection pipeline.

Provides:
- PipelineConfig: Dataclass with all pipeline parameters
- load_config_from_cli: CLI argument parser
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import argparse
import os


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

    # skip stem separation - only effective when source_type="instrumental"
    # when true, uses full audio as melody (for solo instrument recordings)
    skip_separation: bool = False

    # pitch extraction confidence thresholds
    vocal_confidence: float = 0.98
    accomp_confidence: float = 0.80

    # Pitch range (MIDI notes)
    fmin_note: str = "G1"   # ~49 Hz (notebook default)
    fmax_note: str = "C6"   # Increased to C5 to capture C#4, D4, E4 which are > C4 (261Hz)

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
    energy_threshold: float = 0.0 # Normalized energy threshold (0-1) for filtering
    energy_metric: str = "rms"  # 'rms' (peak-normalised) or 'log_amp' (dBFS, percentile-normalised)

    # Phrase detection parameters
    phrase_max_gap: float = 1.0           # Max silence between notes in phrase
    phrase_min_length: int = 13            # Minimum notes per phrase

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

    # db paths
    raga_db_path: Optional[str] = None    # Auto-locates if None

    # processing options
    force_recompute: bool = False         # Force pitch extraction only (stems reused if present)
    save_intermediates: bool = True       # Save CSVs, plots, etc.

    # GMM parameters
    gmm_window_cents: float = 150.0
    gmm_components: int = 1
    bias_rotation: bool = True

    # execution mode
    mode: str = "detect"  # "preprocess", "detect", or "analyze"
    tonic_override: Optional[str] = None
    raga_override: Optional[str] = None
    keep_impure_notes: bool = False

    def __post_init__(self):
        """Validate and normalize paths."""
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        if self.mode == "preprocess":
            if not self.yt_url:
                raise ValueError("Preprocess mode requires --yt")
            if not self.audio_dir:
                raise ValueError("Preprocess mode requires --audio-dir")
            if not self.filename_override:
                raise ValueError("Preprocess mode requires --filename")

            self.audio_dir = os.path.abspath(self.audio_dir)
            os.makedirs(self.audio_dir, exist_ok=True)
            self.audio_path = os.path.join(self.audio_dir, f"{self.filename_override}.mp3")
        else:
            if not self.audio_path:
                raise ValueError(f"{self.mode.capitalize()} mode requires --audio/-a")
            self.audio_path = os.path.abspath(self.audio_path)

            if not os.path.isfile(self.audio_path):
                raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

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

    def _find_raga_db_path(self) -> Optional[str]:
        """Find raga database in standard locations."""
        package_dir = Path(__file__).parent.parent
        candidates = [
            package_dir / "data" / "raga_list_final.csv",
            package_dir / "main notebooks" / "raga_list_final.csv",
            package_dir.parent / "db" / "raga_list_final.csv",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

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


def load_config_from_cli() -> PipelineConfig:
    """Parse command-line arguments and return configuration."""
    parser = argparse.ArgumentParser(
        description="Raga Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # create subparsers for modes: preprocess, detect, analyze
    subparsers = parser.add_subparsers(dest="command", help="Pipeline mode")

    # --- Common arguments function ---
    def add_common_args(p, required=True):
        p.add_argument("--audio", "-a", required=required, help="Input audio file (relative or absolute path)")
        p.add_argument("--output", "-o", default="batch_results", help="Parent directory for output results")
        p.add_argument("--separator", choices=["demucs", "spleeter"], default="demucs", help="Stem separation engine to use")
        p.add_argument("--demucs-model", default="htdemucs", help="Specific Demucs model (e.g., htdemucs, mdx_extra)")

        # Source and Melody settings
        p.add_argument("--source-type", choices=["mixed", "instrumental", "vocal"],
                       default="mixed", help="Audio source type: 'vocal' enables gender tonic bias, 'instrumental' enables instrument tonic bias")
        p.add_argument("--melody-source", choices=["separated", "composite"],
                       default="separated", help="Use 'separated' (stem) or 'composite' (full mix) for melody pitch extraction")
        p.add_argument("--vocalist-gender", choices=["male", "female"],
                       help="Vocalist gender for tonic biasing (only used for source-type=vocal)")
        p.add_argument("--instrument-type",
                       choices=["autodetect", "sitar", "sarod", "bansuri", "slide_guitar"],
                       default="autodetect",
                       help="Instrument type for tonic biasing (only used for source-type=instrumental)")

        # Pitch Extraction settings
        p.add_argument("--fmin-note", default="G1", help="Minimum note for pitch extraction (e.g., G1)")
        p.add_argument("--fmax-note", default="C6", help="Maximum note for pitch extraction (e.g., C6)")
        p.add_argument("--vocal-confidence", type=float, default=0.98, help="Confidence threshold (0-1) for melody pitch data")
        p.add_argument("--accomp-confidence", type=float, default=0.80, help="Confidence threshold (0-1) for accompaniment pitch data")

        # Peak Detection settings
        p.add_argument("--prominence-high", type=float, default=0.01, help="Prominence threshold factor for high-res peak detection")
        p.add_argument("--prominence-low", type=float, default=0.03, help="Prominence threshold factor for low-res peak detection")
        p.add_argument(
            "--bias-rotation",
            action="store_false",
            dest="bias_rotation",
            help="Disable histogram bias rotation (enabled by default)",
        )

        # Miscellaneous
        p.add_argument("--force", "-f", action="store_true", help="Force recompute pitch extraction (reuses existing stems if found)")
        p.add_argument("--skip-separation", action="store_true",
                       help="Skip stem separation entirely (only valid for instrument-mode)")
        p.add_argument("--raga-db", help="Override path to raga database CSV")

    # --- Detect Mode ---
    detect_parser = subparsers.add_parser("detect", help="Phase 1: Detection only")
    add_common_args(detect_parser, required=True)
    detect_parser.add_argument("--tonic", help="Force tonic (comma-separated allowed, e.g. C,D#)")
    detect_parser.add_argument("--raga", help="Force raga name")

    # --- Analyze Mode ---
    analyze_parser = subparsers.add_parser("analyze", help="Phase 2: Analysis only")
    add_common_args(analyze_parser, required=True)
    analyze_parser.add_argument("--tonic", required=True, help="Tonic (e.g. C, D#)")
    analyze_parser.add_argument("--raga", required=True, help="Raga name")
    analyze_parser.add_argument("--keep-impure-notes", action="store_true",
                                help="Keep notes not in raga (default: remove)")
    analyze_parser.add_argument("--transcription-smoothing-ms", type=float, default=0.0,
                                help="Smoothing sigma (ms) for transcription. Set to 0 to disable.")
    analyze_parser.add_argument("--transcription-min-duration", type=float, default=0.02,
                                help="Minimum duration (s) for a transcribed note.")
    analyze_parser.add_argument("--transcription-stability-threshold", type=float, default=4.0,
                                help="Max pitch change (semitones/sec) to be considered stable.")
    analyze_parser.add_argument("--energy-threshold", type=float, default=0.0,
                                help="Energy threshold (0.0-1.0) for filtering unvoiced segments.")
    analyze_parser.add_argument("--energy-metric", choices=["rms", "log_amp"],
                                default="rms",
                                help="Energy metric: 'rms' (peak-normalised) or 'log_amp' (dBFS, percentile-scaled). "
                                     "log_amp gives better dynamic range for quiet passages.")
    analyze_parser.add_argument("--silence-threshold", type=float, default=0.10,
                                help="RMS energy threshold (0.0-1.0) for silence-based phrase splitting. 0 disables.")
    analyze_parser.add_argument("--silence-min-duration", type=float, default=0.25,
                                help="Minimum silence duration (seconds) to trigger a phrase break.")
    analyze_parser.add_argument("--no-rms-overlay", action="store_true",
                                help="Disable RMS energy overlay on pitch analysis plots.")
    analyze_parser.add_argument("--no-smoothing", action="store_true", help="Disable transcription smoothing")

    # --- Preprocess Mode ---
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Download YouTube audio to a local MP3 and exit"
    )
    preprocess_parser.add_argument("--yt", required=True, help="YouTube URL to download")
    preprocess_parser.add_argument("--audio-dir", default="../audio_test_files",
                                   help="Directory where downloaded MP3 should be saved")
    preprocess_parser.add_argument("--filename", required=True,
                                   help="Output filename base (without extension); saved as <filename>.mp3")
    preprocess_parser.add_argument("--start-time",
                                   help="Optional trim start time (SS, MM:SS, or HH:MM:SS)")
    preprocess_parser.add_argument("--end-time",
                                   help="Optional trim end time (SS, MM:SS, or HH:MM:SS)")
    preprocess_parser.add_argument("--output", "-o", default="batch_results",
                                   help="Default detect output directory suggestion for next-step command")

    # --- Root Parser (Legacy support - defaults to detect) ---
    add_common_args(parser, required=False)
    parser.add_argument("--tonic", help="Force tonic (comma-separated allowed, e.g. C,D#)")
    parser.add_argument("--raga", help="Force raga")

    args = parser.parse_args()

    # Validation logic
    if args.command is None:
        if not args.audio:
            parser.error("the following arguments are required: --audio/-a")

    if args.vocalist_gender and args.source_type != "vocal":
        args.source_type = "vocal"

    # Determine effective mode (default to detect)
    mode = args.command if args.command in ["preprocess", "detect", "analyze"] else "detect"

    return PipelineConfig(
        audio_path=getattr(args, 'audio', None),
        output_dir=getattr(args, 'output', 'batch_results'),
        yt_url=getattr(args, 'yt', None),
        audio_dir=getattr(args, 'audio_dir', None),
        filename_override=getattr(args, 'filename', None),
        preprocess_start_time=getattr(args, 'start_time', None),
        preprocess_end_time=getattr(args, 'end_time', None),
        separator_engine=getattr(args, 'separator', 'demucs'),
        demucs_model=getattr(args, 'demucs_model', 'htdemucs'),
        source_type=getattr(args, 'source_type', 'mixed'),
        vocalist_gender=getattr(args, 'vocalist_gender', None),
        instrument_type=getattr(args, 'instrument_type', 'autodetect'),
        skip_separation=getattr(args, 'skip_separation', False),
        vocal_confidence=getattr(args, 'vocal_confidence', 0.98),
        accomp_confidence=getattr(args, 'accomp_confidence', 0.80),
        force_recompute=getattr(args, 'force', False),
        raga_db_path=getattr(args, 'raga_db', None),
        mode=mode,
        tonic_override=getattr(args, 'tonic', None),
        raga_override=getattr(args, 'raga', None),
        keep_impure_notes=getattr(args, 'keep_impure_notes', False),
        transcription_smoothing_ms=0.0 if getattr(args, 'no_smoothing', False) else getattr(args, 'transcription_smoothing_ms', 0.0),
        transcription_min_duration=getattr(args, 'transcription_min_duration', 0.02),
        transcription_derivative_threshold=getattr(args, 'transcription_stability_threshold', 4.0),
        energy_threshold=getattr(args, 'energy_threshold', 0.0),
        energy_metric=getattr(args, 'energy_metric', 'rms'),
        silence_threshold=getattr(args, 'silence_threshold', 0.0),
        silence_min_duration=getattr(args, 'silence_min_duration', 0.25),
        show_rms_overlay=not getattr(args, 'no_rms_overlay', False),
        melody_source=getattr(args, 'melody_source', "separated"),
        fmin_note=getattr(args, 'fmin_note', "G1"),
        fmax_note=getattr(args, 'fmax_note', "C6"),
        prominence_high_factor=getattr(args, 'prominence_high', 0.01),
        prominence_low_factor=getattr(args, 'prominence_low', 0.03),
        bias_rotation=getattr(args, 'bias_rotation', False),
    )


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
