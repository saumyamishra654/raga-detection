# Raga Pipeline Package
"""
Modular raga detection pipeline for Hindustani classical music.

Modules:
- config: Configuration and CLI argument handling
- audio: Stem separation and pitch extraction
- analysis: Histogram construction, peak detection, GMM fitting
- raga: Raga database, candidate matching, scoring
- sequence: Note detection, sargam conversion, phrase analysis
- output: Visualization and HTML report generation
"""

from .config import PipelineConfig, build_cli_parser, create_config, load_config_from_cli, parse_config_from_argv
from .cli_schema import get_mode_schema, list_modes
from .cli_args import params_to_argv

__version__ = "0.1.0"

__all__ = [
    # Config
    "PipelineConfig",
    "build_cli_parser",
    "create_config",
    "load_config_from_cli",
    "parse_config_from_argv",
    # Local-app helpers
    "get_mode_schema",
    "list_modes",
    "params_to_argv",
]

# Optional heavy imports (audio/analysis/plot stack). This keeps lightweight
# tooling usable even when DSP dependencies are not installed in the active interpreter.
try:
    from .audio import PitchData, extract_pitch, separate_stems

    __all__.extend(["PitchData", "extract_pitch", "separate_stems"])
except Exception:
    pass

try:
    from .analysis import GMMResult, HistogramData, PeakData, compute_cent_histograms, detect_peaks, fit_gmm_to_peaks

    __all__.extend(
        [
            "GMMResult",
            "HistogramData",
            "PeakData",
            "compute_cent_histograms",
            "detect_peaks",
            "fit_gmm_to_peaks",
        ]
    )
except Exception:
    pass

try:
    from .raga import Candidate, RagaDatabase, RagaScorer, ScoringParams, generate_candidates, score_candidates_full

    __all__.extend(
        [
            "Candidate",
            "RagaDatabase",
            "RagaScorer",
            "ScoringParams",
            "generate_candidates",
            "score_candidates_full",
        ]
    )
except Exception:
    pass

try:
    from .sequence import Note, Phrase, cluster_phrases, compute_transition_matrix, detect_notes, detect_phrases

    __all__.extend(
        [
            "Note",
            "Phrase",
            "cluster_phrases",
            "compute_transition_matrix",
            "detect_notes",
            "detect_phrases",
        ]
    )
except Exception:
    pass

try:
    from .output import AnalysisResults, generate_html_report

    __all__.extend(["AnalysisResults", "generate_html_report"])
except Exception:
    pass

