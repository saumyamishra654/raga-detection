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
    audio_path: str
    output_dir: str
    
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
    fmax_note: str = "C5"   # Increased to C5 to capture C#4, D4, E4 which are > C4 (261Hz)
    
    # Histogram parameters
    histogram_bins_high: int = 100   # High-res: 12 cents per bin
    histogram_bins_low: int = 33     # Low-res: 33 bins (matches notebook ssje_tweaked)
    smoothing_sigma: float = 0.8     # Gaussian smoothing kernel width
    use_confidence_weights: bool = False  # Notebook uses unweighted histograms
    
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
    transcription_smoothing_ms: float = 70.0 # Transcription pre-smoothing sigma (ms)
    transcription_min_duration: float = 0.04 # Min duration for stationary notes (40ms)
    transcription_min_duration: float = 0.04 # Min duration for stationary notes (40ms)
    transcription_derivative_threshold: float = 2.0 # Stability threshold (semitones/sec)
    energy_threshold: float = 0.0 # Normalized energy threshold (0-1) for filtering
    
    # Phrase detection parameters
    phrase_max_gap: float = 2.0           # Max silence between notes in phrase
    phrase_min_length: int = 3            # Minimum notes per phrase
    
    # Scoring parameters (ML disabled - manual weights only)
    use_ml_model: bool = False  # Disabled per migration plan
    model_path: Optional[str] = None      # Path to trained model (unused)
    
    # Database paths (relative to package or absolute)
    raga_db_path: Optional[str] = None    # Auto-locates if None
    
    # Processing options
    force_recompute: bool = False         # Force pitch extraction only (stems reused if present)
    save_intermediates: bool = True       # Save CSVs, plots, etc.
    
    # GMM parameters
    gmm_window_cents: float = 150.0
    gmm_components: int = 1
    
    # Execution Mode (only detect and analyze supported)
    mode: str = "detect"  # "detect" or "analyze" only
    tonic_override: Optional[str] = None
    raga_override: Optional[str] = None
    keep_impure_notes: bool = False
    
    def __post_init__(self):
        """Validate and normalize paths."""
        self.audio_path = os.path.abspath(self.audio_path)
        self.output_dir = os.path.abspath(self.output_dir)
        
        if not os.path.isfile(self.audio_path):
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
        
        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Auto-locate model and database if not specified
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
        return os.path.splitext(os.path.basename(self.audio_path))[0]
    
    @property
    def stem_dir(self) -> str:
        """Directory where stems are saved."""
        # Use different subdirectory for different separators
        if self.separator_engine == "spleeter":
            subdir = "spleeter"
        else:
            subdir = self.demucs_model  # e.g., htdemucs
        return os.path.join(self.output_dir, subdir, self.filename)
    
    @property
    def vocals_path(self) -> str:
        """Path to separated vocals file."""
        return os.path.join(self.stem_dir, "vocals.wav")
    
    @property
    def accompaniment_path(self) -> str:
        """Path to separated accompaniment file."""
        return os.path.join(self.stem_dir, "accompaniment.wav")


def load_config_from_cli() -> PipelineConfig:
    """Parse command-line arguments and return configuration."""
    parser = argparse.ArgumentParser(
        description="Raga Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for modes: detect, analyze, full (default)
    subparsers = parser.add_subparsers(dest="command", help="Pipeline mode")
    
    # --- Common arguments function ---
    def add_common_args(p, required=True):
        p.add_argument("--audio", "-a", required=required, help="Input audio file")
        p.add_argument("--output", "-o", default="results", help="Output directory")
        p.add_argument("--separator", choices=["demucs", "spleeter"], default="demucs")
        p.add_argument("--demucs-model", default="htdemucs")
        # New source type arguments
        p.add_argument("--source-type", choices=["mixed", "instrumental", "vocal"], 
                       default="mixed", help="Audio source type (mixed requires stem separation)")
        p.add_argument("--melody-source", choices=["separated", "composite"],
                       default="separated", help="Source for melody pitch extraction")
        p.add_argument("--vocalist-gender", choices=["male", "female"], 
                       help="Vocalist gender for tonic bias (only for source-type=vocal)")
        p.add_argument("--instrument-type", 
                       choices=["autodetect", "sitar", "sarod", "bansuri", "slide_guitar"],
                       default="autodetect",
                       help="Instrument type for tonic bias (only for source-type=instrumental)")
        p.add_argument("--vocal-confidence", type=float, default=0.98)
        p.add_argument("--accomp-confidence", type=float, default=0.80)
        p.add_argument("--force", "-f", action="store_true", help="Force recompute")
        p.add_argument("--skip-separation", action="store_true",
                       help="Skip stem separation (only for source-type=instrumental)")
        p.add_argument("--raga-db", help="Path to raga database CSV")

    # --- Detect Mode ---
    detect_parser = subparsers.add_parser("detect", help="Phase 1: Detection only")
    add_common_args(detect_parser, required=True)
    
    # --- Analyze Mode ---
    analyze_parser = subparsers.add_parser("analyze", help="Phase 2: Analysis only")
    add_common_args(analyze_parser, required=True)
    analyze_parser.add_argument("--tonic", required=True, help="Tonic (e.g. C, D#)")
    analyze_parser.add_argument("--raga", required=True, help="Raga name")
    analyze_parser.add_argument("--keep-impure-notes", action="store_true", 
                                help="Keep notes not in raga (default: remove)")
    analyze_parser.add_argument("--transcription-smoothing-ms", type=float, default=70.0,
                                help="Smoothing sigma (ms) for transcription. Set to 0 to disable.")
    analyze_parser.add_argument("--transcription-min-duration", type=float, default=0.04,
                                help="Minimum duration (s) for a transcribed note.")
    analyze_parser.add_argument("--transcription-stability-threshold", type=float, default=2.0,
                                help="Max pitch change (semitones/sec) to be considered stable.")
    analyze_parser.add_argument("--energy-threshold", type=float, default=0.0,
                                help="Energy threshold (0.0-1.0) for filtering unvoiced segments.")
    analyze_parser.add_argument("--no-smoothing", action="store_true", help="Disable transcription smoothing")
    
    # --- Root Parser (Legacy support - defaults to detect) ---
    add_common_args(parser, required=False)
    parser.add_argument("--tonic", help="Force tonic")
    parser.add_argument("--raga", help="Force raga")
    
    args = parser.parse_args()
    
    # Validation logic
    if args.command is None:
        if not args.audio:
            parser.error("the following arguments are required: --audio/-a")
            
    # Determine effective mode (default to detect, no full mode)
    mode = args.command if args.command in ["detect", "analyze"] else "detect"
    
    return PipelineConfig(
        audio_path=args.audio,
        output_dir=args.output,
        separator_engine=args.separator,
        demucs_model=args.demucs_model,
        source_type=args.source_type,
        vocalist_gender=args.vocalist_gender,
        instrument_type=args.instrument_type,
        skip_separation=args.skip_separation,
        vocal_confidence=args.vocal_confidence,
        accomp_confidence=args.accomp_confidence,
        force_recompute=args.force,
        raga_db_path=args.raga_db,
        mode=mode,
        tonic_override=getattr(args, 'tonic', None),
        raga_override=getattr(args, 'raga', None),
        keep_impure_notes=getattr(args, 'keep_impure_notes', False),
        transcription_smoothing_ms=0.0 if getattr(args, 'no_smoothing', False) else getattr(args, 'transcription_smoothing_ms', 70.0),
        transcription_min_duration=getattr(args, 'transcription_min_duration', 0.04),
        transcription_derivative_threshold=getattr(args, 'transcription_stability_threshold', 2.0),

        energy_threshold=getattr(args, 'energy_threshold', 0.0),
        melody_source=getattr(args, 'melody_source', "separated"),
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
