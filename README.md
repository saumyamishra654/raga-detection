# Raga Detection Project
---
A comprehensive system for automatic Hindustani raga identification from audio recordings. The project combines signal processing, machine learning, and music theory to identify ragas and their tonics (Sa) from vocals and accompaniment.

**Core Pipeline:** Audio file → stem separation (htdemucs/Spleeter) → framewise pitch extraction (SwiftF0) → pitch class histogram analysis → stable peak detection → raga template matching → joint raga-tonic scoring using machine learning models.

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)  
3. [Requirements](#3-requirements)  
4. [Getting Started](#4-getting-started)  
5. [Core Notebooks](#5-core-notebooks)  
6. [Training Scripts](#6-training-scripts)  
7. [Model Architecture](#7-model-architecture)  
8. [Pipeline Stages](#8-pipeline-stages)  
9. [Data Formats](#9-data-formats)  
10. [Key Parameters](#10-key-parameters)  
11. [Troubleshooting](#11-troubleshooting)
12. [Current Work](#12-current-work)

---

## 1) Project Overview

This project implements an automated raga detection system for Hindustani classical music that solves two interconnected problems:

**Raga Identification:** Matching detected pitch patterns against a database of 79 ragas with 264 name variants.

**Tonic (Sa) Detection:** Identifying the reference pitch (tonic) for the performance, essential for accurate raga identification.

### Key Features
- **Source Separation:** Uses htdemucs/Spleeter to isolate vocals and accompaniment
- **Pitch Extraction:** SwiftF0 for high-precision framewise pitch detection (10ms resolution)
- **Peak Validation:** Cross-resolution histogram analysis with circular peak detection
- **Machine Learning:** Trained models (Logistic Regression + MLP) for optimal feature weighting
- **Ground Truth Dataset:** 1095 annotated songs across 120 unique ragas

### Performance Metrics
- **Logistic Regression** (8 hand-crafted features): Average rank 16.5, Top-1 accuracy 22%
- **MLP** (32 features: 12 melody + 12 accompaniment + 8 handcrafted): Average rank 30.7, Top-1 accuracy 38%
- **Tonic Detection:** Enhanced MLP with 24 features (12 salience + 12 raga-aware)

---

## 2) Repository Structure

```
raga-detection/
├── README.md                         # This file
├── main notebooks/
│   ├── ssje_tweaked_wit_peaks.ipynb # Primary analysis notebook (pitch histograms, peaks, scoring)
│   ├── note_sequence_playground.ipynb # Note sequence analysis experiments (WIP)
│   ├── joint_ranking_real.py        # Logistic Regression training for joint raga-tonic scoring
│   ├── train_raga_mlp.py           # MLP training (32 features: histograms + handcrafted)
│   ├── check_raga_name_mismatches.py # Database consistency checker & fuzzy matcher
│   ├── ground_truth.csv            # Annotated dataset (1095 songs, 120 ragas)
│   ├── raga_list_final.csv         # Raga database (79 ragas, 264 name variants, 12-bit masks)
│   └── raga_hash_table_processed.csv # Preprocessed raga lookup table
├── scripts/
│   ├── train_tonic_mlp_enhanced.py # Enhanced tonic detection (24 features with raga context)
│   ├── train_tonic_mlp_full.py     # Full tonic detection (30 features with statistics)
│   ├── train_raga_weights.py       # Legacy manual weight optimization
│   ├── evaluate_joint_models.py    # Model comparison and benchmarking
│   ├── extract_raga_features.py    # Feature extraction utilities
│   ├── stems.py                    # Stem separation batch processing
│   └── ytdl.py                     # YouTube audio downloader
├── results/                        # Pipeline execution logs
│   ├── median_pitch_extraction_log.txt
│   └── median_pitch_results.csv
├── analysis_results/               # Logistic Regression evaluation outputs
├── analysis_results_mlp/           # MLP evaluation outputs
├── *.pkl files                     # Trained models and scalers
│   ├── joint_score_model.pkl      # Logistic Regression model
│   ├── joint_score_scaler.pkl
│   ├── raga_mlp_model.pkl          # MLP model
│   ├── raga_mlp_scaler.pkl
│   ├── tonic_mlp_enhanced_model.pkl # Enhanced tonic detector
│   └── tonic_mlp_enhanced_scaler.pkl
└── pretrained_models/              # Demucs/Spleeter checkpoints (2stems)
```

---

## 3) Requirements

### Python Environment
**Recommended:** Python 3.10 with conda or venv

### Core Dependencies
```bash
# Scientific computing
pip install numpy pandas scipy scikit-learn

# Audio processing
pip install librosa soundfile

# Visualization
pip install matplotlib seaborn

# Model persistence
pip install joblib

# Stem separation (choose one)
pip install demucs      # Preferred - htdemucs model (better quality)
pip install spleeter    # Alternative - 2stems model

# Pitch detection (custom implementation required)
pip install swift_f0    # SwiftF0 wrapper included in notebooks

# Optional
pip install tensorflow  # Required for Spleeter
pip install yt-dlp     # For YouTube downloads (scripts/ytdl.py)
```

### Installation Notes

**macOS Spleeter Issues:**
- Use Python 3.10 (avoid 3.11+)
- Create fresh venv: `python3.10 -m venv spleeter_env`
- See [GitHub issue #696](https://github.com/deezer/spleeter/issues/696) for dependency fixes
- **Recommended:** Use Demucs instead (`pip install demucs`)

**SwiftF0:**
- Requires local wrapper implementation (included in notebooks)
- Provides 10ms frame resolution with confidence scores
- Outputs CSV with columns: time, pitch_hz, confidence, voicing

---

## 4) Getting Started

### Quick Start - Notebook Analysis
1. Open `main notebooks/ssje_tweaked_wit_peaks.ipynb`
2. Set `audio_path` to your audio file (`.mp3`, `.wav`, etc.)
3. Configure `instrument_mode`:
   - `'sitar'` - restricts tonic search space
   - `'sarod'` - different tonic priors
   - `'autodetect'` - tries all 12 tonics (default)
4. Run cells top-to-bottom
5. Outputs saved in `separated_stems/<filename>/`
6. Force recompute: `analyze_or_load_with_plots(..., force_recompute=True)`

---

## 5) Core Notebooks

### `ssje_tweaked_wit_peaks.ipynb`
**Primary analysis notebook** implementing the complete raga detection pipeline.

**What it does:**
- Stem separation using htdemucs (2-stem) or Spleeter
- Pitch extraction with SwiftF0 (vocals: conf > 0.98, accompaniment: conf > 0.8)
- Dual-resolution cent histograms (100-bin high-res, 33-bin low-res)
- Cross-validated peak detection with circular boundary handling
- Pitch class mapping with ±35¢ tolerance windows
- Raga template matching against database
- Candidate scoring using 8 hand-crafted features
- Accompaniment-based tonic salience
- Final ranking with diagnostic visualizations

**Key Functions:**
- `analyze_or_load_with_plots()`: Pitch extraction with automatic caching
- `compute_pitch_distribution()`: **Unweighted** cent histogram aggregation
- `get_detected_peak_count()`: Dual-resolution peak validation (high-res must align with low-res within 45¢)
- `calculate_features()`: Extracts 8 features (match_mass, presence, loglike, extra_mass, complexity, size_penalty, tonic_salience, primary_score)

**Outputs (in `separated_stems/<filename>/`):**
- `vocals.wav`, `accompaniment.wav` (stems)
- `<prefix>_pitch_data.csv` (framewise pitch)
- `<prefix>_midi.csv`, `<prefix>_midi.mid` (MIDI export)
- Diagnostic plots: pitch contours, note segments, cent histograms, frequency histograms
- `df_final`: Ranked DataFrame of (raga, tonic) candidates

### `note_sequence_playground.ipynb`
**Experimental notebook** for temporal pattern analysis.

**Goals:**
- Note transition probability modeling
- Phrase-level raga identification
- Temporal dynamics beyond static pitch distributions

**Status:** Work in progress - will be updated as concrete results emerge.

---

## 6) Training Scripts

### Raga Detection Models

#### `joint_ranking_real.py` - Logistic Regression
**Joint raga-tonic scoring using 8 hand-crafted features.**

**Features:**
1. `match_mass`: Fraction of melody distribution on raga's allowed notes
2. `presence`: Average relative strength of raga notes vs peak
3. `loglike`: Normalized log-likelihood of note set
4. `extra_mass`: 1 - match_mass (mass outside raga notes)
5. `complexity`: Raga size penalty (normalized to 0-1)
6. `size_penalty`: Mismatch between expected and detected peak count
7. `tonic_salience`: Accompaniment strength at candidate tonic
8. `primary_score`: Sa + max(Pa, Ma, Madhyam) if in raga

**Training Details:**
- **Dataset:** 277 positive samples + subsampled negatives (0.5%)
- **Algorithm:** Logistic Regression with `class_weight='balanced'`
- **Output:** `joint_score_model.pkl`, `joint_score_scaler.pkl`

**Optimized Weights:**
```python
match_mass:     1.3302   # How well melody aligns with raga notes
presence:      -1.3302   # Penalty for weak/missing expected notes
loglike:        1.1739   # Statistical fit quality
tonic_salience: 0.9387   # Accompaniment support for tonic
primary_score:  0.5454   # Tonic (Sa) + fifth/fourth bonus
extra_mass:     0.1752   # Tolerance for notes outside raga
complexity:     0.1409   # Slight preference for richer ragas
size_penalty:  -0.0340   # Minor penalty for peak count mismatch
```

**Performance:**
- Average Rank: **136 → 16.5** (8.2× improvement over manual weights)
- Top-1 Accuracy: **0% → 22%**

#### `train_raga_mlp.py` - Multi-Layer Perceptron
**Deep learning approach using raw histogram features + handcrafted features.**

**Input Features (32 total):**
- 12 melody pitch class bins (rotated by candidate tonic)
- 12 accompaniment pitch class bins (rotated by candidate tonic)
- 8 hand-crafted features (same as Logistic Regression)

**Architecture:**
- Input: 32 features
- Hidden Layer 1: 64 neurons (ReLU)
- Hidden Layer 2: 32 neurons (ReLU)
- Output: 1 neuron (sigmoid for binary classification)

**Training Details:**
- **Dataset:** 286 samples
- **Optimizer:** Adam
- **Regularization:** Early stopping (validation_fraction=0.1)
- **Convergence:** Iteration 25 (loss=0.1580)
- **Output:** `raga_mlp_model.pkl`, `raga_mlp_scaler.pkl`

**Performance:**
- Average Rank: **136 → 30.7** (4.4× improvement)
- Top-1 Accuracy: **0% → 38%** (best performing)

**Key Insight:** Direct access to pitch class distributions allows the MLP to learn nuanced patterns beyond hand-crafted features, achieving 73% better Top-1 accuracy than Logistic Regression.

### Tonic Detection Models

#### `train_tonic_mlp_enhanced.py` - Raga-Aware Tonic Detection
**Enhanced tonic detector incorporating raga matching context.**

**Input Features (24 total):**
- 12 accompaniment salience features (normalized histogram per tonic)
- 6 best-raga features per tonic (match_mass, extra_mass, presence, loglike, complexity, raga_size)
- 6 aggregated raga statistics (mean, std of best-raga features across tonics)

**Architecture:**
- Input: 24 features
- Hidden Layer 1: 64 neurons (ReLU)
- Hidden Layer 2: 32 neurons (ReLU)
- Output: 12 neurons (softmax for tonic classes 0-11)

**Output:** `tonic_mlp_enhanced_model.pkl`, `tonic_mlp_enhanced_scaler.pkl`

#### `train_tonic_mlp_full.py` - Full Feature Tonic Detection
**Comprehensive tonic detection with statistical features.**

**Input Features (30 total):**
- 24 features from enhanced model
- 6 statistical features: pitch_range, voiced_ratio, peak_sharpness, harmonic_clarity, stability, entropy

**Output:** `tonic_mlp_full_model.pkl`, `tonic_mlp_full_scaler.pkl`

### Utility Scripts

#### `check_raga_name_mismatches.py`
**Database consistency checker and fuzzy matcher.**

**Functionality:**
- Identifies exact matches vs fuzzy matches
- Detects spelling variants (Shri/Shree, Shuddh/Shuddha)
- Finds case mismatches
- Lists missing ragas
- Handles smart quotes and CSV parsing issues

**Output:** `raga_rename_suggestions.csv` with (old_name, new_name, count, match_type)

**Usage:**
```bash
python check_raga_name_mismatches.py
# Review raga_rename_suggestions.csv
# Fix ground_truth.csv or add entries to raga_list_final.csv
```

#### `evaluate_joint_models.py`
**Model comparison and benchmarking tool.**

Compares Logistic Regression vs MLP on test set with metrics:
- Average rank
- Top-1/Top-5/Top-10 accuracy
- Per-raga performance breakdown

---

## 7) Model Architecture

### Feature Engineering Philosophy

**Hand-Crafted Features (8):**
Designed based on music theory and domain knowledge.

1. **match_mass**: Fraction of melody on raga's allowed notes (higher = better fit)
2. **extra_mass**: 1 - match_mass (penalty for extraneous notes)
3. **presence**: Average relative strength of raga notes vs peak (uniformity check)
4. **loglike**: Normalized log-likelihood (statistical goodness-of-fit)
5. **complexity**: Raga size factor `(size - 5) / 12` (accounts for scale richness)
6. **size_penalty**: `|detected_peaks - raga_size| / 4` (peak count alignment)
7. **tonic_salience**: Accompaniment strength at candidate Sa (tonic validation)
8. **primary_score**: Sa mass + max(Pa, Ma, Madhyam) if in raga (melodic anchor)

**Raw Histogram Features (24):**
Let the model discover patterns from data.

- 12 melody pitch class bins (rotated by candidate tonic)
- 12 accompaniment pitch class bins (rotated by candidate tonic)

**Why Both?**
- Hand-crafted: Encode expert knowledge, interpretable
- Raw histograms: Capture subtle patterns humans might miss
- Combination: Best of both worlds (38% Top-1 accuracy in MLP)

### Training Strategy

**Positive Samples:**
- Ground truth (raga, tonic) pairs from `ground_truth.csv`
- 50 songs processed → 50 positive samples

**Negative Samples:**
- All other (raga_mask, tonic) combinations per song
- ~79 ragas × 12 tonics = ~948 candidates per song
- Subsampled at 0.5% to balance classes (≈5 negatives per song)

**Class Balancing:**
- Logistic Regression: `class_weight='balanced'`
- MLP: Early stopping prevents overfitting to majority class

**Regularization:**
- MLP: Early stopping (validation_fraction=0.1, tol=0.0001, n_iter_no_change=10)
- StandardScaler: Normalize features before training

---

## 8) Pipeline Stages

### Stage 1: Audio Preprocessing

**1. Stem Separation**
- **Input:** Audio file (MP3, WAV, FLAC, etc.)
- **Tools:** htdemucs (preferred) or Spleeter 2-stem
- **Output:** `vocals.wav`, `accompaniment.wav`
- **Caching:** Stored in `separated_stems/<filename>/` (skips if exists)

**2. Pitch Extraction (SwiftF0)**
- **Input:** Separated stems
- **Processing:** Per-frame pitch detection (10ms hop size)
- **Confidence Thresholds:**
  - Vocals: 0.98 (strict - reduce false positives)
  - Accompaniment: 0.8 (lenient - capture harmonic support)
- **Output:** `<stem>_pitch_data.csv` with columns:
  - `time`: Timestamp (seconds)
  - `pitch_hz`: Detected frequency
  - `confidence`: Model confidence [0-1]
  - `voicing`: Boolean voiced/unvoiced decision

### Stage 2: Pitch Analysis

**3. Histogram Construction**
- **Convert:** Hz → MIDI → cents (mod 1200)
- **Binning:** 
  - High-res: 100 bins across octave (12¢ resolution)
  - Low-res: 33 bins across octave (36¢ resolution)
- **Weighting:** **Unweighted** histograms (matching notebook implementation)
- **Aggregation:** ±35¢ windows around note centers for 12 pitch classes

**4. Peak Detection**
- **Smoothing:** Gaussian filter (σ=0.8) with circular wrap mode
- **High-Res Peaks:** 
  - `scipy.signal.find_peaks` with prominence = max(1.0, 0.03 × hist_max)
  - Distance = 2 bins minimum separation
  - Manual edge checks for circular boundaries (bins 0 and 99)
- **Low-Res Peaks:**
  - Prominence = max(0, 0.01 × hist_max)
  - Distance = 1 bin
  - Edge checks for bins 0 and 32
- **Cross-Validation:** High-res peak must align with low-res peak within 45¢

**5. Pitch Class Mapping**
- **Alignment:** Map validated peaks to nearest semitone centers
- **Tolerance:** ±35¢ window for acceptance
- **Output:** `pc_cand` = set of detected pitch classes (0-11)

### Stage 3: Raga Matching

**6. Template Lookup**
- **Database:** 79 ragas with 12-bit binary masks
- **Matching:** Compare detected pitch classes against raga templates
- **Fuzzy Matching:** Handle spelling variants (optional, for consistency checking)

**7. Feature Extraction**
- **Candidate Generation:** For each (raga, tonic) pair:
  - Rotate melody/accompaniment distributions by tonic
  - Calculate 8 hand-crafted features
  - Extract 24 raw histogram features (for MLP)
- **Filtering:** Optionally filter tonics by accompaniment mean salience

### Stage 4: Scoring & Ranking

**8. Model Prediction**
- **Standardization:** Apply trained scaler (zero mean, unit variance)
- **Logistic Regression:** `decision_function()` for ranking scores
- **MLP:** `predict_proba()[:, 1]` for ranking scores

**9. Final Ranking**
- **Sort:** Candidates by score (descending)
- **Output:** Top-K predictions with:
  - Raga name
  - Tonic (0-11, where 0=C)
  - Confidence score
  - Feature breakdown
- **Visualization:** Histograms, peak annotations, candidate comparison

---

## 9) Data Formats

### Ground Truth CSV (`ground_truth.csv`)
```csv
Filename,Raga,Tonic
song1.mp3,Yaman,C
song2.mp3,Bhairav,D
song3.mp3,Darbari,G
...
```

**Specifications:**
- **1095 entries** spanning **120 unique ragas**
- **Tonic Format:** Musical note names (C, C#, Db, D, ..., B)
- **Mapping:** C=0, C#/Db=1, D=2, ..., B=11
- **Common Issues:** Spelling variants (Shri/Shree), suffix differences (Yaman Kalyan vs Yaman)

### Raga Database (`raga_list_final.csv`)
```csv
0,1,2,3,4,5,6,7,8,9,10,11,names
1,0,1,0,1,1,0,1,0,1,0,1,"[""Yaman""]"
1,1,0,1,0,1,0,1,1,0,1,0,"[""Bhairav""]"
1,0,1,0,1,1,0,1,1,0,1,0,"[""Kafi, Sindhura Kafi""]"
...
```

**Specifications:**
- **79 rows** (unique raga templates)
- **Columns 0-11:** Binary pitch class mask (1=allowed, 0=absent)
  - 0=Sa, 1=Komal Re, 2=Shuddh Re, ..., 11=Komal Ni
- **names column:** JSON-encoded list of raga name variants
  - Supports `["Name1, Name2"]` or `["Name1", "Name2"]` formats
  - **264 total name variants** across all ragas

**Parsing Notes:**
- Handle smart quotes (`"` vs `"` from Excel/Mac)
- Strip nested quotes and brackets
- Case-insensitive matching recommended
- Common variants: Shri/Shree, Shuddh/Shuddha, suffix additions (Kanada, Kalyan, Sarang)

### Pitch Data CSV (SwiftF0 Output)
```csv
time,pitch_hz,confidence,voicing
0.000,220.5,0.99,True
0.010,221.2,0.98,True
0.020,0.0,0.23,False
...
```

**Specifications:**
- **Frame Rate:** 100 Hz (10ms hop size)
- **Columns:**
  - `time`: Timestamp in seconds
  - `pitch_hz`: Detected frequency (0 if unvoiced)
  - `confidence`: Model confidence score [0-1]
  - `voicing`: Boolean (True=voiced, False=unvoiced)

---

## 10) Key Parameters

### Peak Detection Tuning
```python
# Histogram resolution
num_bins_high_res = 100        # Bins across octave (12¢ resolution)
num_bins_low_res = 33          # Coarse validation (36¢ resolution)

# Smoothing
sigma = 0.8                    # Gaussian kernel width (circular wrap)

# Peak thresholds
prom_high = max(1.0, 0.03 * hist_max)  # High-res prominence
prom_low = max(0, 0.01 * hist_max)     # Low-res prominence
dist_high = 2                  # Min separation (high-res bins)
dist_low = 1                   # Min separation (low-res bins)

# Alignment tolerances
tolerance_cents = 35           # Peak-to-semitone window
peak_tolerance_cents = 45      # High-low res validation window
```

**Tuning Guidance:**
- **Increase `sigma`** (e.g., 1.0-1.2): Smoother histograms, fewer spurious peaks, may miss subtle notes
- **Decrease `prom_high`** (e.g., 0.02): Detect weaker peaks, more false positives
- **Increase `tolerance_cents`** (e.g., 40-50): More lenient alignment, may include microtones
- **Decrease `peak_tolerance_cents`** (e.g., 35-40): Stricter cross-validation, fewer peaks validated

### Feature Extraction
```python
WINDOW_CENTS = 35.0            # Note center aggregation window
BIN_SIZE_CENTS = 1.0           # Histogram resolution
EPS = 1e-12                    # Numerical stability epsilon
```

### Training Hyperparameters

**Logistic Regression:**
```python
max_iter = 1000
class_weight = 'balanced'
negative_subsample_rate = 0.005  # 0.5% of negatives
```

**MLP:**
```python
hidden_layer_sizes = (64, 32)
activation = 'relu'
solver = 'adam'
early_stopping = True
validation_fraction = 0.1
n_iter_no_change = 10
tol = 0.0001
max_iter = 500
random_state = 42
```

### Manual Scoring Weights (Legacy)
```python
OLD_WEIGHTS = {
    'match_mass': 0.40,
    'presence': 0.25,
    'loglike': 1.0,
    'extra_mass': -1.10,
    'complexity': -0.4,
    'size_penalty': -0.25,
    'tonic_salience': 0.12,
    'primary_score': 0.0
}
```
These were replaced by optimized weights from Logistic Regression training.

---

## 11) Troubleshooting

### Common Issues

**Problem: No stems created**
- **Check:** `separated_stems/<filename>/` directory exists
- **Verify:** Spleeter/Demucs installed (`pip list | grep demucs`)
- **Ensure:** FFmpeg available (`ffmpeg -version`)
- **Solution:** Reinstall demucs or use absolute paths

**Problem: Empty pitch arrays / No voiced frames**
- **Cause:** Confidence thresholds too strict or poor audio quality
- **Solution:** Lower thresholds:
  - Vocals: 0.95 (from 0.98)
  - Accompaniment: 0.7 (from 0.8)
- **Force recompute:** `analyze_or_load_with_plots(..., force_recompute=True)`
- **Check:** Audio sample rate ≥ 22kHz recommended

**Problem: No peaks validated**
- **Cause:** Histogram too flat or thresholds too strict
- **Diagnosis:** Inspect cent histogram plots
- **Solutions:**
  - Lower `prom_high`: `0.02 * hist_max` (from 0.03)
  - Increase `tolerance_cents`: 40-50 (from 35)
  - Increase `peak_tolerance_cents`: 50 (from 45)
  - Check recording duration (minimum 30s recommended)

**Problem: Training fails with "0 samples"**
- **Cause:** Raga name mismatches between ground truth and database
- **Diagnosis:** Run `check_raga_name_mismatches.py`
- **Solution:** Fix spelling in `ground_truth.csv` or add variants to database
- **Common Fixes:**
  - Shree → Shri
  - Shuddh Sarang → Shuddha Sarang
  - Handle suffix differences (Yaman Kalyan → Yaman)

**Problem: Poor model performance (Avg rank > 50)**
- **Check:** Tonic annotations (common error source)
- **Verify:** Algorithm matches notebook (weighted vs unweighted histograms)
- **Increase:** Training dataset size (current: 50 songs, expand to 200+)
- **Debug:** Run evaluation on individual songs to identify systematic errors

**Problem: Fragile notebook cell ordering**
- **Cause:** Many variables created across cells
- **Solution:** Restart kernel + run all (Kernel → Restart & Run All)
- **Alternative:** Wrap pipeline in functions to avoid global state

### Platform-Specific Issues

**macOS Spleeter Installation:**
```bash
# Create fresh Python 3.10 environment
python3.10 -m venv spleeter_env
source spleeter_env/bin/activate

# Install dependencies with pinned versions
pip install tensorflow==2.5.0
pip install spleeter==2.3.0

# If still failing, use Demucs instead
pip install demucs
```

**Reference:** [Spleeter Issue #696](https://github.com/deezer/spleeter/issues/696)

### Database Format Issues

**Supported Formats:**
1. **Mask Column:** `"1,0,1,0,..."` (12 comma-separated values)
2. **12 Separate Columns:** Named `0` through `11` with 0/1 values

**Common Parsing Failures:**
- Smart quotes (`"` instead of `"`) from Excel/Mac
- Nested quote escaping: `"[""Raga""]"` vs `"["Raga"]"`
- Mixed formats within same file

**Debug Steps:**
```bash
# Check raw CSV format
head -5 "main notebooks/raga_list_final.csv"

# Search for specific raga
grep -i "yaman" "main notebooks/raga_list_final.csv"

# Run consistency checker
python check_raga_name_mismatches.py
```

---

## 12) Current Work & Future Directions

### Active Development

**Note Sequence Analysis (`note_sequence_playground.ipynb`):**
- Temporal pattern recognition beyond static pitch distributions
- Note transition probability modeling
- Phrase-level raga identification
- **Status:** Experimental - concrete results pending

**Database Expansion:**
- Adding more raga variants (target: 100+ ragas)
- Compound raga support (Yaman Kalyan as distinct from Yaman)
- Regional variants (Carnatic ragas, folk scales)

### Known Limitations

1. **Raga Ambiguity:**
   - Some ragas share identical note sets (e.g., Yaman vs Shuddh Kalyan)
   - Differentiation requires temporal/ornamental analysis
   - Current approach: Falls back to tonic salience and statistical features

2. **Tonic Drift:**
   - Long recordings (>10 minutes) may have slight tonic variations
   - Current approach: Uses median pitch, may miss drift
   - Future: Sliding window tonic detection

3. **Training Data:**
   - Limited to 1095 songs from specific artists/styles
   - May not generalize to all performance styles
   - Expansion needed for robustness

4. **Computational Cost:**
   - Stem separation: ~1-2 minutes per 5-minute song
   - Pitch extraction: ~30 seconds per stem
   - Feature extraction: ~5 seconds per song
   - Total: ~3-4 minutes per song

### Potential Improvements

**1. Temporal Features:**
- Note duration distributions
- Inter-onset intervals
- Transition matrices (note → next note probabilities)

**2. Gamaka Patterns:**
- Model ornamentations characteristic to ragas
- Microtonal inflections (meend, kan)
- Vibrato patterns

**3. Multi-Task Learning:**
- Joint training of tonic + raga models
- Shared representations across tasks
- Transfer learning from larger datasets

**4. Attention Mechanisms:**
- Focus on characteristic phrases (pakad)
- Weight important temporal segments
- Ignore alap vs gat differences

**5. Data Augmentation:**
- Pitch shifting (train on multiple tonics)
- Time stretching (tempo invariance)
- Noise injection (robustness)

---

**Last Updated:** December 2025  
**Repository:** raga-detection
