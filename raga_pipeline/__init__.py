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

from .config import PipelineConfig, load_config_from_cli, create_config
from .audio import PitchData, separate_stems, extract_pitch
from .analysis import HistogramData, PeakData, GMMResult, compute_cent_histograms, detect_peaks, fit_gmm_to_peaks
from .raga import RagaDatabase, Candidate, RagaScorer, ScoringParams, generate_candidates, score_candidates_full
from .sequence import Note, Phrase, detect_notes, detect_phrases, cluster_phrases, compute_transition_matrix
from .output import AnalysisResults, generate_html_report

__version__ = "0.1.0"
__all__ = [
    # Config
    "PipelineConfig", "load_config_from_cli", "create_config",
    # Audio
    "PitchData", "separate_stems", "extract_pitch",
    # Analysis
    "HistogramData", "PeakData", "GMMResult", 
    "compute_cent_histograms", "detect_peaks", "fit_gmm_to_peaks",
    # Raga
    "RagaDatabase", "Candidate", "RagaScorer", "ScoringParams",
    "generate_candidates", "score_candidates_full",
    # Sequence
    "Note", "Phrase", "detect_notes", "detect_phrases", 
    "cluster_phrases", "compute_transition_matrix",
    # Output
    "AnalysisResults", "generate_html_report",
]

