# Raga Pipeline Documentation

A modular Python package for raga detection and analysis of Hindustani classical music.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [CLI Reference](#cli-reference)
6. [Module Reference](#module-reference)
   - [config.py](#configpy)
   - [audio.py](#audiopy)
   - [analysis.py](#analysispy)
   - [raga.py](#ragapy)
   - [sequence.py](#sequencepy)
   - [output.py](#outputpy)
7. [Data Flow](#data-flow)
8. [Output Files](#output-files)

---

## Overview

The raga_pipeline package refactors the functionality from two Jupyter notebooks (`ssje_tweaked_wit_peaks.ipynb` and `note_sequence_playground.ipynb`) into a modular, maintainable Python package. It provides:

- **Histogram-based raga detection**: Analyzes pitch class distributions to identify raga and tonic
- **Sequential note analysis**: Detects discrete notes, phrases, and transition patterns
- **Interactive HTML reports**: Generates comprehensive reports with synchronized audio visualization

---

## Directory Structure

```
raga-detection/
├── raga_pipeline/                    # Main Python package
│   ├── __init__.py                   # Package initialization & public API
│   ├── config.py                     # Configuration & CLI argument handling
│   ├── audio.py                      # Stem separation + pitch extraction
│   ├── analysis.py                   # Histogram + peak detection + GMM
│   ├── raga.py                       # Database + matching + scoring
│   ├── sequence.py                   # Note detection + transcription
│   └── output.py                     # Visualization + HTML report
├── driver.py                         # Main CLI entry point
├── requirements.txt                  # Python dependencies
├── models/                           # Trained ML models (optional)
│   ├── raga_mlp_model.pkl
│   └── joint_score_model.pkl
└── data/                             # Reference data
    └── raga_list_final.csv           # Raga database
```

---

## Installation

```bash
# Navigate to raga-detection directory
cd raga-detection

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from raga_pipeline import PipelineConfig; print('OK')"
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `librosa` | Audio loading, MIDI/Hz conversion |
| `demucs` | High-quality stem separation |
| `spleeter` | Alternative stem separation |
| `swift-f0` | Pitch (f0) extraction |
| `numpy`, `pandas` | Data manipulation |
| `scipy` | Signal processing (peaks, filters) |
| `scikit-learn` | ML models, GMM fitting |
| `matplotlib` | Static plot generation |
| `plotly` | Interactive visualizations |
| `torch` | Deep learning (for Demucs) |

---

## Quick Start

### Command Line

```bash
# Run detection phase
./run_pipeline.sh detect --audio /path/to/song.mp3 --output /path/to/results

# Run analysis phase (after detection)
./run_pipeline.sh analyze --audio /path/to/song.mp3 --output /path/to/results --tonic "C#" --raga "Bhairavi"
```

### Python API

```python
from raga_pipeline import (
    create_config,
    separate_stems,
    extract_pitch,
    compute_cent_histograms,
    detect_peaks,
    generate_candidates,
    RagaDatabase,
    RagaScorer,
)

# Create configuration
config = create_config(
    audio_path="song.mp3",
    output_dir="./results",
    vocal_confidence=0.95,
)

# Run separation
vocals_path, accomp_path = separate_stems(
    config.audio_path,
    config.output_dir,
)

# Extract pitch
pitch_data = extract_pitch(
    vocals_path,
    config.stem_dir,
    prefix="vocals",
    fmin=61.7,
    fmax=1046.5,
    confidence_threshold=0.98,
)

# Analyze
histogram = compute_cent_histograms(pitch_data)
peaks = detect_peaks(histogram)

# Match ragas
raga_db = RagaDatabase("raga_list_final.csv")
candidates = generate_candidates(peaks.pitch_classes, raga_db)
```

---

## CLI Reference

### Basic Usage

```bash
python driver.py --audio <input_file> --output <output_dir> [options]
```

### Required Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--audio` | `-a` | Path to input audio file (MP3, WAV, FLAC, etc.) |
| `--output` | `-o` | Output directory for results |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `full` | Analysis mode: `full`, `histogram`, or `sequence` |
| `--separator` | `demucs` | Stem separation engine: `demucs` or `spleeter` |
| `--demucs-model` | `htdemucs` | Demucs model variant |
| `--instrument` | `autodetect` | Instrument: `autodetect`, `sitar`, `sarod`, `vocal` |
| `--vocal-confidence` | `0.98` | Pitch confidence threshold for vocals |
| `--force` / `-f` | `false` | Force recompute, ignore cached results |
| `--no-ml` | `false` | Use manual weights instead of ML model |
| `--raga-db` | auto | Path to raga database CSV |
| `--model-path` | auto | Path to trained ML model |

### Mode Descriptions

| Mode | Description | Speed |
|------|-------------|-------|
| `full` | Run both histogram and sequence analysis | Slowest |
| `histogram` | Only histogram-based raga detection | Fast |
| `sequence` | Only note sequence and phrase analysis | Medium |

### Examples

```bash
# Use spleeter instead of demucs
python driver.py -a song.mp3 -o ./out --separator spleeter

# Lower confidence threshold for noisy recordings
python driver.py -a song.mp3 -o ./out --vocal-confidence 0.90

# Use manual scoring weights (no ML model)
python driver.py -a song.mp3 -o ./out --no-ml

# Specify custom raga database
python driver.py -a song.mp3 -o ./out --raga-db /path/to/my_ragas.csv
```

---

## Module Reference

### config.py

Configuration management and CLI argument handling.

#### Classes

##### `PipelineConfig`

Dataclass containing all pipeline parameters.

```python
@dataclass
class PipelineConfig:
    # Required
    audio_path: str              # Input audio file path
    output_dir: str              # Output directory
    
    # Separator settings
    separator_engine: str        # 'demucs' or 'spleeter'
    demucs_model: str            # Demucs model name
    
    # Pitch extraction
    vocal_confidence: float      # Confidence threshold for vocals
    accomp_confidence: float     # Confidence for accompaniment
    fmin_note: str               # Minimum pitch (e.g., "B1")
    fmax_note: str               # Maximum pitch (e.g., "C6")
    
    # Histogram parameters
    histogram_bins_high: int     # High-res bins (default 100)
    histogram_bins_low: int      # Low-res bins (default 33)
    smoothing_sigma: float       # Gaussian smoothing width
    
    # Peak detection
    tolerance_cents: float       # ±cents for note mapping
    peak_tolerance_cents: float  # Cross-resolution validation
    
    # Note detection
    note_min_duration: float     # Minimum note duration (s)
    pitch_change_threshold: float # Semitone threshold
    derivative_threshold: float  # For stationary points
    
    # Phrase detection
    phrase_max_gap: float        # Max gap between notes
    phrase_min_length: int       # Min notes per phrase
    
    # ML scoring
    use_ml_model: bool           # Use trained model
    model_path: str              # Path to .pkl model
```

**Properties:**
- `filename` → Base name of audio file
- `stem_dir` → Directory for separated stems
- `vocals_path` → Path to vocals stem
- `accompaniment_path` → Path to accompaniment stem

#### Functions

##### `load_config_from_cli() -> PipelineConfig`
Parse command-line arguments and return configuration.

##### `create_config(audio_path, output_dir, **kwargs) -> PipelineConfig`
Create configuration programmatically.

```python
config = create_config(
    audio_path="song.mp3",
    output_dir="./results",
    vocal_confidence=0.95,
    histogram_bins_high=120,
)
```

---

### audio.py

Audio processing: stem separation and pitch extraction.

#### Classes

##### `PitchData`

Container for pitch extraction results.

```python
@dataclass
class PitchData:
    timestamps: np.ndarray      # Time axis (seconds)
    pitch_hz: np.ndarray        # Detected pitch, 0 for unvoiced
    confidence: np.ndarray      # Detection confidence [0-1]
    voicing: np.ndarray         # Boolean voicing mask
    valid_freqs: np.ndarray     # Voiced frequencies only
    midi_vals: np.ndarray       # Voiced MIDI values
    frame_period: float         # Frame period (default 0.01s)
    audio_path: str             # Source audio path
```

**Properties:**
- `voiced_mask` → Boolean mask for voiced frames
- `voiced_times` → Timestamps for voiced frames
- `cent_vals` → Cent values (0-1200) for voiced frames

**Methods:**
- `apply_confidence_threshold(threshold)` → Return new PitchData with filtered voicing

#### Functions

##### `separate_stems(audio_path, output_dir, engine, model, device) -> Tuple[str, str]`

Separate audio into vocals and accompaniment.

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_path` | str | Input audio file |
| `output_dir` | str | Base output directory |
| `engine` | str | 'demucs' or 'spleeter' |
| `model` | str | Demucs model name |
| `device` | str | 'cuda', 'cpu', or None |

**Returns:** `(vocals_path, accompaniment_path)`

**Caching:** Skips separation if stems already exist.

##### `extract_pitch(audio_path, output_dir, prefix, fmin, fmax, confidence_threshold, force_recompute) -> PitchData`

Extract pitch using SwiftF0.

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_path` | str | Audio file to analyze |
| `output_dir` | str | Directory for cached CSV |
| `prefix` | str | Output file prefix |
| `fmin` | float | Minimum frequency (Hz) |
| `fmax` | float | Maximum frequency (Hz) |
| `confidence_threshold` | float | Minimum confidence |
| `force_recompute` | bool | Ignore cache |

**Returns:** `PitchData` instance

**Caching:** Saves/loads from `{prefix}_pitch_data.csv`

##### `load_pitch_from_csv(csv_path, audio_path) -> PitchData`

Load cached pitch data from CSV file.

---

### analysis.py

Histogram construction, peak detection, and GMM analysis.

#### Classes

##### `HistogramData`

Dual-resolution cent histograms.

```python
@dataclass
class HistogramData:
    high_res: np.ndarray         # 100-bin histogram
    low_res: np.ndarray          # 33-bin histogram
    smoothed_high: np.ndarray    # Gaussian-smoothed high-res
    smoothed_low: np.ndarray     # Gaussian-smoothed low-res
    bin_centers_high: np.ndarray # Cent positions (100 values)
    bin_centers_low: np.ndarray  # Cent positions (33 values)
    high_res_norm: np.ndarray    # Normalized high-res
    smoothed_high_norm: np.ndarray  # Normalized smoothed
```

##### `PeakData`

Peak detection results.

```python
@dataclass
class PeakData:
    high_res_indices: np.ndarray  # Peak indices in high-res
    high_res_cents: np.ndarray    # Peak cent positions
    low_res_indices: np.ndarray   # Peak indices in low-res
    low_res_cents: np.ndarray     # Low-res cent positions
    validated_indices: np.ndarray # Cross-validated peaks
    validated_cents: np.ndarray   # Validated cent positions
    pitch_classes: Set[int]       # Mapped pitch classes (0-11)
    peak_details: List[dict]      # Per-peak metadata
```

##### `GMMResult`

Gaussian Mixture Model results per peak.

```python
@dataclass
class GMMResult:
    peak_idx: int               # Peak index
    peak_cent: float            # Peak position (cents)
    nearest_note: int           # Nearest semitone (0-11)
    means: np.ndarray           # GMM component means
    sigmas: np.ndarray          # GMM component std devs
    weights: np.ndarray         # Component weights
    primary_mean: float         # Dominant component mean
    primary_sigma: float        # Dominant component sigma
    deviation_from_note: float  # Cents from ideal position
```

#### Functions

##### `compute_cent_histograms(pitch_data, bins_high, bins_low, sigma) -> HistogramData`

Build dual-resolution cent histograms with Gaussian smoothing.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pitch_data` | — | PitchData with midi_vals |
| `bins_high` | 100 | High-resolution bin count |
| `bins_low` | 33 | Low-resolution bin count |
| `sigma` | 0.8 | Gaussian smoothing width |

##### `detect_peaks(histogram, tolerance_cents, peak_tolerance_cents, ...) -> PeakData`

Detect and cross-validate peaks across resolutions.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tolerance_cents` | 35.0 | Window for mapping to semitones |
| `peak_tolerance_cents` | 45.0 | Cross-resolution validation |
| `prominence_high_factor` | 0.03 | High-res prominence threshold |
| `prominence_low_factor` | 0.01 | Low-res prominence threshold |

**Algorithm:**
1. Detect peaks in high-res smoothed histogram
2. Detect peaks in low-res smoothed histogram
3. Validate: keep high-res peaks within `peak_tolerance_cents` of any low-res peak
4. Map validated peaks to pitch classes (0-11)

##### `fit_gmm_to_peaks(histogram, peak_indices, window_cents, n_components) -> List[GMMResult]`

Fit Gaussian Mixture Models around each peak for microtonal analysis.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_cents` | 150.0 | Window around each peak |
| `n_components` | 1 | GMM components per peak |

---

### raga.py

Raga database, candidate generation, feature extraction, and scoring.

#### Classes

##### `RagaDatabase`

Load and query raga definitions.

```python
class RagaDatabase:
    def __init__(self, csv_path: str)
    
    def get_raga_mask(self, raga_name: str) -> Tuple[int, ...]
    def get_raga_notes(self, raga_name: str, tonic: int) -> Set[int]
    def fuzzy_match(self, name: str) -> Optional[str]
    def get_interval_lookup(self) -> Dict[Tuple, List[dict]]
```

**Attributes:**
- `all_ragas` → List of all raga entries
- `name_to_mask` → Name → binary mask mapping
- `interval_lookup` → Canonical intervals → [raga entries]

##### `Candidate`

A (raga, tonic) candidate.

```python
@dataclass
class Candidate:
    tonic: int                  # Pitch class (0-11)
    mask: Tuple[int, ...]       # 12-bit binary mask
    raga_names: List[str]       # Matching raga names
    intervals: Tuple[int, ...]  # Interval pattern
```

##### `Features`

Features for scoring a candidate.

```python
@dataclass
class Features:
    # 8 hand-crafted features
    match_mass: float        # Mass on allowed notes
    extra_mass: float        # Mass outside raga notes
    presence: float          # Average note strength
    loglike: float           # Normalized log-likelihood
    complexity: float        # Raga size factor
    size_penalty: float      # Peak count mismatch
    tonic_salience: float    # Accompaniment at tonic
    primary_score: float     # Sa + Pa/Ma strength
    
    # Histogram features
    melody_histogram: np.ndarray   # 12-bin (rotated)
    accomp_histogram: np.ndarray   # 12-bin (rotated)
    
    def to_vector(include_histograms=True) -> np.ndarray
```

##### `RagaScorer`

Score candidates using ML model or manual weights.

```python
class RagaScorer:
    DEFAULT_WEIGHTS = {
        "match_mass": 1.3302,
        "presence": -1.3302,
        "loglike": 1.1739,
        ...
    }
    
    def __init__(self, model_path=None, use_ml=True)
    def score(self, features: Features) -> float
    def rank_candidates(self, candidates) -> pd.DataFrame
```

#### Functions

##### `generate_candidates(pitch_classes, raga_db) -> List[Candidate]`

Generate all (raga, tonic) candidates from detected pitch classes.

**Algorithm:**
1. For each detected pitch class as potential tonic
2. Rotate interval pattern to start from that tonic
3. Look up canonical form in database
4. Return matching raga names

##### `extract_features(candidate, pitch_data_melody, pitch_data_accomp, peak_count) -> Features`

Extract 8 hand-crafted + 24 histogram features.

| Feature | Description |
|---------|-------------|
| `match_mass` | Fraction of melody on raga's allowed notes |
| `extra_mass` | 1 - match_mass |
| `presence` | Mean relative strength of raga notes |
| `loglike` | Normalized log-likelihood of raga notes |
| `complexity` | (raga_size - 5) / 12 |
| `size_penalty` | \|detected_peaks - raga_size\| / 4 |
| `tonic_salience` | Accompaniment strength at candidate tonic |
| `primary_score` | Sa mass + max(Pa, Ma) mass |

---

### sequence.py

Note detection, sargam conversion, phrase detection, and transcription.

#### Constants

```python
OFFSET_TO_SARGAM = {
    0: "Sa", 1: "re", 2: "Re", 3: "ga", 4: "Ga",
    5: "ma", 6: "Ma", 7: "Pa", 8: "dha", 9: "Dha",
    10: "ni", 11: "Ni"
}
```

#### Classes

##### `Note`

A detected musical note.

```python
@dataclass
class Note:
    start: float          # Start time (seconds)
    end: float            # End time (seconds)
    pitch_midi: float     # MIDI note number
    pitch_hz: float       # Frequency (Hz)
    confidence: float     # Detection confidence
    sargam: str           # Sargam label (populated later)
    pitch_class: int      # Pitch class (0-11)
    
    @property
    def duration(self) -> float
    
    def with_sargam(self, tonic) -> Note
```

##### `Phrase`

A group of consecutive notes.

```python
@dataclass
class Phrase:
    notes: List[Note]
    
    @property
    def start(self) -> float
    @property
    def end(self) -> float
    @property
    def duration(self) -> float
    @property
    def sargam_string(self) -> str
    
    def interval_sequence(self) -> List[int]
```

#### Functions

##### `tonic_to_midi_class(tonic) -> int`

Convert tonic (MIDI, note name, or Hz) to pitch class (0-11).

```python
tonic_to_midi_class("C#")   # → 1
tonic_to_midi_class(60)     # → 0 (C)
tonic_to_midi_class(440.0)  # → 9 (A)
```

##### `midi_to_sargam(midi_note, tonic, include_octave=True) -> str`

Convert MIDI note to sargam notation.

```python
midi_to_sargam(60, 60)      # → "Sa"
midi_to_sargam(64, 60)      # → "Ga"
midi_to_sargam(72, 60)      # → "Sa·"  (upper octave)
midi_to_sargam(48, 60)      # → "Sa'"  (lower octave)
```

##### `sargam_to_midi(sargam, tonic, octave=4) -> int`

Convert sargam notation to MIDI note.

##### `detect_notes(pitch_data, config) -> List[Note]`

Main note detection entry point. Currently uses stationary point method.

##### `smooth_pitch_contour(pitch_data, method, sigma, snap_to_semitones) -> PitchData`

Smooth pitch contour using Gaussian or median filtering.

| Parameter | Options | Description |
|-----------|---------|-------------|
| `method` | 'gaussian', 'median' | Smoothing algorithm |
| `sigma` | — | Kernel width (samples) |
| `snap_to_semitones` | bool | Round to nearest semitone |

##### `detect_stationary_points(pitch_data, min_duration, pitch_threshold, derivative_threshold) -> List[Note]`

Detect notes by finding regions where pitch derivative is low.

##### `detect_melodic_peaks(pitch_data, prominence, min_duration) -> List[Note]`

Alternative: detect notes using local maxima/minima.

##### `detect_phrases(notes, max_gap, min_length) -> List[Phrase]`

Group notes into phrases based on temporal gaps.

##### `cluster_phrases(phrases, similarity_threshold) -> Dict[int, List[Phrase]]`

Group similar phrases together (repeated motifs).

##### `compute_transition_matrix(notes, tonic) -> np.ndarray`

Compute 12×12 sargam transition probability matrix.

```python
matrix = compute_transition_matrix(notes, tonic=0)
# matrix[i, j] = P(note j | note i)
```

##### `compute_bigram_counts(notes, tonic) -> Dict[Tuple[str, str], int]`

Count sargam bigram occurrences.

##### `notes_to_intervals(notes) -> List[int]`

Convert note sequence to interval sequence (semitones).

##### `transcribe_notes(notes, tonic) -> str`

Convert notes to space-separated sargam string.

---

### output.py

Visualization and HTML report generation.

#### Classes

##### `AnalysisResults`

Container for all pipeline results.

```python
@dataclass
class AnalysisResults:
    config: PipelineConfig
    pitch_data_vocals: PitchData
    pitch_data_accomp: PitchData
    histogram_vocals: HistogramData
    histogram_accomp: HistogramData
    peaks_vocals: PeakData
    candidates: pd.DataFrame
    detected_tonic: int
    detected_raga: str
    gmm_results: List[GMMResult]
    notes: List[Note]
    phrases: List[Phrase]
    phrase_clusters: Dict[int, List[Phrase]]
    transition_matrix: np.ndarray
    plot_paths: Dict[str, str]
```

#### Functions

##### Static Plots (PNG)

```python
plot_histograms(histogram, peaks, output_path, title) -> str
plot_gmm_overlay(histogram, gmm_results, output_path) -> str
plot_note_segments(pitch_data, notes, output_path, tonic) -> str
```

##### Interactive Visualization

```python
create_pitch_contour_plotly(pitch_data, tonic, raga_notes, audio_duration) -> str
```

Returns JSON string for Plotly figure with:
- Pitch contour line
- Sargam reference lines (color-coded by octave)
- Synchronized cursor line

##### HTML Report

```python
generate_html_report(results: AnalysisResults, output_path: str) -> str
```

Generates comprehensive HTML report with:
1. Metadata section
2. Interactive audio player + pitch contour (vocals)
3. Interactive audio player + pitch contour (accompaniment)
4. Histogram plots
5. Peak detection summary
6. Raga candidate ranking (top 20)
7. Musical transcription (sargam notation)
8. Phrase analysis
9. Transition matrix heatmap
10. GMM microtonal analysis

**Audio Synchronization:**
- Playhead updates cursor position in Plotly chart
- Click on chart seeks audio to that timestamp

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              INPUT                                       │
│                         audio_path (MP3/WAV)                             │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  audio.separate_stems()                                                  │
│  ➜ vocals.mp3, accompaniment.mp3                                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  audio.extract_pitch()                                                   │
│  ➜ PitchData (vocals), PitchData (accompaniment)                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
┌─────────────────────────────┐       ┌─────────────────────────────┐
│   HISTOGRAM PATH            │       │   SEQUENCE PATH              │
└─────────────────────────────┘       └─────────────────────────────┘
            │                                       │
            ▼                                       ▼
┌─────────────────────────────┐       ┌─────────────────────────────┐
│ analysis.compute_histograms │       │ sequence.detect_notes        │
│ ➜ HistogramData             │       │ ➜ List[Note]                 │
└──────────────┬──────────────┘       └──────────────┬──────────────┘
               │                                      │
               ▼                                      ▼
┌─────────────────────────────┐       ┌─────────────────────────────┐
│ analysis.detect_peaks       │       │ sequence.detect_phrases      │
│ ➜ PeakData + pitch_classes  │       │ ➜ List[Phrase]               │
└──────────────┬──────────────┘       └──────────────┬──────────────┘
               │                                      │
               ▼                                      ▼
┌─────────────────────────────┐       ┌─────────────────────────────┐
│ raga.generate_candidates    │       │ sequence.cluster_phrases     │
│ ➜ List[Candidate]           │       │ ➜ Dict[cluster → phrases]    │
└──────────────┬──────────────┘       └──────────────┬──────────────┘
               │                                      │
               ▼                                      ▼
┌─────────────────────────────┐       ┌─────────────────────────────┐
│ raga.extract_features       │       │ sequence.transition_matrix   │
│ ➜ Features per candidate    │       │ ➜ 12×12 probability matrix   │
└──────────────┬──────────────┘       └──────────────┬──────────────┘
               │                                      │
               ▼                                      │
┌─────────────────────────────┐                       │
│ raga.RagaScorer.rank        │                       │
│ ➜ pd.DataFrame (ranked)     │                       │
└──────────────┬──────────────┘                       │
               │                                      │
               └───────────────────┬──────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  output.generate_html_report()                                           │
│  ➜ report.html + all plots                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Output Files

All outputs are saved to `{output_dir}/{demucs_model}/{filename}/`:

| File | Module | Description |
|------|--------|-------------|
| `vocals.mp3` | audio | Separated vocal stem |
| `accompaniment.mp3` | audio | Separated accompaniment |
| `vocals_pitch_data.csv` | audio | Cached pitch extraction |
| `accompaniment_pitch_data.csv` | audio | Cached accompaniment pitch |
| `histogram_vocals.png` | output | Dual-resolution histogram plot |
| `gmm_overlay.png` | output | Histogram with GMM fits |
| `note_segments.png` | output | Detected notes visualization |
| `candidates.csv` | raga | Ranked (raga, tonic) candidates |
| `transcription.txt` | sequence | Sargam notation output |
| **`report.html`** | output | **Complete interactive report** |

---

## Examples

### Programmatic Usage

```python
from raga_pipeline import *

# Full pipeline
config = create_config("my_song.mp3", "./results")

# Step by step
vocals, accomp = separate_stems(config.audio_path, config.output_dir)
pitch_v = extract_pitch(vocals, config.stem_dir, "vocals", 61.7, 1046.5, 0.98)
pitch_a = extract_pitch(accomp, config.stem_dir, "accomp", 61.7, 1046.5, 0.80)

hist = compute_cent_histograms(pitch_v)
peaks = detect_peaks(hist)

raga_db = RagaDatabase("raga_list_final.csv")
candidates = generate_candidates(peaks.pitch_classes, raga_db)

scorer = RagaScorer()
features_list = [(c, extract_features(c, pitch_v, pitch_a, len(peaks.validated_indices))) 
                 for c in candidates]
ranking = scorer.rank_candidates(features_list)

print(f"Top raga: {ranking.iloc[0]['raga']}")
print(f"Tonic: {ranking.iloc[0]['tonic_name']}")
```

### Custom Note Detection

```python
from raga_pipeline.sequence import (
    smooth_pitch_contour,
    detect_stationary_points,
    detect_melodic_peaks,
)

# Smooth with different methods
smoothed_gaussian = smooth_pitch_contour(pitch_data, method='gaussian', sigma=2.0)
smoothed_median = smooth_pitch_contour(pitch_data, method='median', sigma=5)

# Different detection algorithms
notes_stationary = detect_stationary_points(smoothed_gaussian, 
    min_duration=0.15, derivative_threshold=0.1)
notes_peaks = detect_melodic_peaks(smoothed_gaussian, prominence=1.5)

# Compare
print(f"Stationary: {len(notes_stationary)} notes")
print(f"Peak-based: {len(notes_peaks)} notes")
```

---

## Troubleshooting

### "Raga database not found"
Ensure `raga_list_final.csv` is in one of:
- `raga-detection/data/`
- `raga-detection/main notebooks/`
- Or specify with `--raga-db /path/to/file.csv`

### "No voiced frames detected"
- Lower the confidence threshold: `--vocal-confidence 0.90`
- Check if audio has actual melodic content
- Try preprocessing to remove noise

### "CUDA out of memory"
Demucs requires significant GPU memory. Solutions:
- Use CPU: set `DEMUCS_DEVICE='cpu'` in config
- Use smaller model: `--demucs-model mdx`
- Use Spleeter instead: `--separator spleeter`

---

## License

MIT License. See LICENSE file for details.
