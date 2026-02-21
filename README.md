# Raga Detection System

A robust, pipeline-based system for automatic Hindustani raga identification and detailed melodic analysis. This project evolves from experimental notebooks into a structured, production-ready Python package.

## Key Features

*   **Raga & Tonic Identification**: Automatically detects the raga and tonic (Sa) from audio recordings using template matching and machine learning models.
*   **Flexible Audio Processing**: Handles **Vocal**, **Instrumental**, and **Mixed** recordings with integrated stem separation (using `htdemucs` or `spleeter`).
*   **High-Precision Analysis**:
    *   **Pitch Extraction**: Uses SwiftF0 for accurate framewise pitch tracking (10ms resolution).
    *   **Unified Transcription**: A hybrid engine combining **Stationary Point** detection and **Inflection Point** analysis to capture both stable notes and rapid melodic runs (tans). 
    *   **Energy-Based Gating**: Integrated RMS energy analysis for automatic noise/silence removal and breath detection.
    *   **Microtonal Analysis**: GMM-based analysis of pitch distributions.
*   **Interactive Reporting**: Generates detailed HTML reports with:
    *   **Multi-Track Audio Sync**: Synchronized visualization with **Original**, **Vocals**, and **Accompaniment** stems.
    *   **Bidirectional Navigation**: Click-to-seek functionality (Plot -> Audio) and a real-time cursor that follows the audio playback (Audio -> Plot).
    *   **Energy Analysis**: Visual distribution of note energy to help fine-tune transcription accuracy.
    *   **Dynamic Scaling**: Visualizations automatically adapt to the singer's vocal range.
*   **Docs**: See `raga_pipeline/DOCUMENTATION.md` for the full CLI reference and output layout.

---

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd raga-detection
    ```

2.  **Environment Setup**:
    It is recommended to use Conda.
    ```bash
    conda create -n raga python=3.10
    conda activate raga
    pip install -r requirements.txt
    ```

    *Note: Ensure `ffmpeg` is installed on your system for audio processing.*

---

## Usage

The project is designed to be run via `run_pipeline.sh`, which can optionally activate Conda before launching `driver.py`. The pipeline supports three modes: **Preprocess**, **Detect**, and **Analyze**.

`run_pipeline.sh` environment controls:
- `RAGA_CONDA_SH`: explicit path to `conda.sh` (optional)
- `RAGA_CONDA_ENV`: env name to activate (default: `raga`)
- `RAGA_SKIP_ENV_ACTIVATE=1`: skip Conda activation
- `RAGA_PYTHON_BIN`: Python executable to run (default: `python3`)

### Local Parameter-Tuning App (macOS)

You can launch a local web app that wraps the same pipeline and exposes schema-driven parameter controls for `preprocess`, `detect`, and `analyze`.

```bash
./run_local_app.sh
```

Then open: `http://127.0.0.1:8765/app`

What it provides:
- Dynamic form generation from the live argparse schema (`config.py`) so new CLI flags show up automatically.
- Serial background job execution with logs, progress (`[STEP x/y]` parsing), and cancel support.
- Artifact discovery from existing output folders by selected audio filename stem.
- Top-right quick actions to open detect/analyze reports (when available) in a new tab.
- Rerun support with last-used parameters plus optional raw `extra args` passthrough.
- Auto-prefill of the next mode from printed pipeline suggestions (`preprocess -> detect`, `detect -> analyze`).
- Optional fields support blank input (parser defaults apply), and dependent fields only appear when relevant (e.g., `vocalist_gender` only when `source_type=vocal`).
- Drag-and-drop upload for the `--audio` field (detect/analyze): dropped files are uploaded to local app storage and the resolved path is auto-filled.
- Audio browser for `--audio`: defaults to `../audio_test_files`, lets you pick a custom directory, and shows a song dropdown from that directory.
- Selected audio directory is persisted locally in the browser and reused on next load.
- Batch trigger for the selected audio directory (`raga_pipeline.batch` in `auto` mode from the local app).
- `--raga` field is backed by the raga database CSV list (default or current `--raga-db`) and supports typeahead + fuzzy correction on blur.
- After drag/drop upload, the raw audio filepath input is hidden from the form view.

### Mode 0: Preprocess (`preprocess`)

Downloads audio from YouTube and saves it locally as an MP3 so it can be used by `detect` and `analyze`.

```bash
./run_pipeline.sh preprocess \
    --yt "https://www.youtube.com/watch?v=..." \
    --filename "my_song_name" \
    --start-time "0:30" \
    --end-time "2:00"
```

`--audio-dir` defaults to `../audio_test_files` (you can override it when needed).

`--start-time` and `--end-time` are optional. If omitted, preprocess defaults to full track (`0:00` to track end). Time format supports `SS`, `MM:SS`, or `HH:MM:SS`.

Prerequisites:
- `ffmpeg` and `ffprobe` available in `PATH`
- Python package `yt-dlp` installed (imported as `yt_dlp`)

Validation rules:
- `start-time < end-time`
- both times must be within track duration

This mode saves `../audio_test_files/my_song_name.mp3` by default and prints a copyable `detect` command for the next step.

### Phase 1: Detection (`detect`)

Runs stem separation, pitch extraction, and attempts to identify the Raga and Tonic.

```bash
./run_pipeline.sh detect \
  --audio "path/to/song.mp3" \
  --source-type instrumental

# Constrained detection (limit scoring to specific tonics or ragas)
./run_pipeline.sh detect \
    --audio "path/to/song.mp3" \
    --tonic "C,D#" \
    --raga "Bhairavi"
```

**Common Options:**
*   `--output`: Parent output directory. Defaults to `batch_results`.
    - Outputs are written under `<output>/<demucs-model>/<audio_filename>/...` (or `<output>/spleeter/<audio_filename>/...`).
*   `--source-type`: `mixed` (default), `vocal`, or `instrumental`.
    - If `--vocalist-gender` is provided, `--source-type` is auto-set to `vocal`.
*   `--melody-source`: `separated` (default) or `composite`. Use `composite` when the separated melody stem is unreliable.
*   `--vocalist-gender`: When provided, `--source-type` is auto-set to `vocal`.
*   `--instrument-type`: Instrument type hint for tonic biasing in instrumental mode (default: `autodetect`).
*   `--separator`: `demucs` (default) or `spleeter`.
*   `--demucs-model`: Demucs model name (default: `htdemucs`).
*   `--fmin-note` / `--fmax-note`: Override default pitch extraction range (e.g. `G1` to `C6`).
*   `--vocal-confidence` / `--accomp-confidence`: Confidence thresholds (0-1) for pitch extraction.
*   `--prominence-high` / `--prominence-low`: Fine-tune peak detection sensitivity (factors applied to prominence thresholds).
*   `--bias-rotation`: Disable histogram bias rotation (bias rotation is enabled by default).
*   `--force` / `-f`: Force pitch recomputation (stems are reused if present).
*   `--raga-db`: Override path to the raga database CSV.
*   `--tonic`: Constrain scoring to one or more tonics (comma-separated, e.g. `C,D#`).
*   `--raga`: Constrain scoring to a specific raga name.

**Output:**
*   Written under `<output>/<demucs-model>/<song_name>/` (or `<output>/spleeter/<song_name>/`):
    *   `detection_report.html`: Summary report for Phase 1.
    *   `candidates.csv`: Ranked list of likely ragas/tonics.
    *   `histogram_melody.png`, `histogram_accompaniment.png`: Pitch distribution histograms.
    *   `stationary_note_histogram_duration_weighted.png` and `stationary_note_histogram_duration_weighted.csv`: Duration-weighted stationary-note distribution (12 fixed pitch classes, octave-wrapped).
    *   Cached pitch CSVs (e.g. `composite_pitch_data.csv`, `melody_pitch_data.csv` or `vocals_pitch_data.csv`, `accompaniment_pitch_data.csv`).
    *   Stems (`vocals.mp3`, `accompaniment.mp3`).

### Phase 2: Analysis (`analyze`)

Performs deep sequence analysis, phrasing, and generating the interactive report. **Requires a specific Tonic and Raga** (either from Phase 1 results or manually provided).

```bash
./run_pipeline.sh analyze \
  --audio "path/to/song.mp3" \
  --tonic "C#" \
  --raga "Bhairavi"
```

Important:
- The subcommand is spelled `analyze` (not `analyse`).
- `analyze` expects cached pitch data in the output stem directory. Run `detect` first with the same `--audio`, `--output`, `--separator`, and `--demucs-model`.

**Output:**
*   `analysis_report.html`: The Phase 2 report featuring:
    *   Interactive, scrollable pitch plot with audio sync.
    *   Phrase analysis and common motifs.
    *   Raga correction statistics.
    *   Transition matrices.

**Analyze Options:**
*   `--keep-impure-notes`: Keep notes not in the raga (default behavior is to discard them during correction).
*   `--transcription-smoothing-ms`: Transcription smoothing sigma (ms). Use `0` to disable.
*   `--transcription-min-duration`: Minimum duration (s) for a transcribed note (default: `0.02`).
*   `--transcription-stability-threshold`: Max pitch change (semitones/sec) to be considered stable (default: `4.0`).
*   `--energy-metric`: `rms` (peak-normalized, default) or `log_amp` (dBFS with percentile normalization).
*   `--energy-threshold`: Energy gate (0-1) for removing low-energy notes. Default: 0.0.
*   `--silence-threshold`: Energy threshold (0-1) to split phrases on sustained silence. Default: 0.10 (set `0` to disable).
*   `--silence-min-duration`: Minimum silence duration (seconds) required to split phrases. Default: 0.25.
*   `--phrase-min-duration`: Exclude phrases shorter than this duration (seconds). Default: 1.0.
*   `--phrase-min-notes`: Exclude phrases with fewer notes than this count. Default: 0 (disabled).
*   `--no-rms-overlay`: Disable the energy overlay on pitch plots in the HTML report.
*   `--no-smoothing`: Force transcription smoothing off (equivalent to `--transcription-smoothing-ms 0`).
*   Snapping behavior: Transcription uses chromatic snapping internally; raga correction then removes or keeps non-raga notes depending on `--keep-impure-notes`.

### Batch Processing

Batch mode walks a directory of `.mp3`, `.wav`, `.flac`, or `.m4a` files and launches `driver.py` using the current Python interpreter.

```bash
# 1) Create a blank ground-truth CSV inside the input directory:
python -m raga_pipeline.batch /path/to/audio_dir --init-csv

# This creates: /path/to/audio_dir/<audio_dir_name>_gt.csv
# Fill in at least `raga` and `tonic` for files you want to run in analyze mode.

# 2) Run batch processing (auto mode):
python -m raga_pipeline.batch /path/to/audio_dir

# Force detect-only (ignores ground truth):
python -m raga_pipeline.batch /path/to/audio_dir --mode detect
```

Batch options:
*   `--ground-truth` / `-g`: Ground truth CSV path (default: auto-detected as described above).
*   `--output` / `-o`: Output directory (default: `<repo>/batch_results`).
*   `--max-files`: Process at most N pending files in this run (`0` means all).
*   `--progress-file`: Optional checkpoint JSON path.
*   `--exit-99-on-remaining`: Exit with code `99` if files remain (for PBS resubmission loops).
*   `--silent` / `-s`: Suppress console output; logs are still written.

Outputs:
*   Per-file logs: `<output>/logs/<filename>.log`
*   Per-file pipeline outputs: `<output>/<demucs-model>/<song_name>/...`

---

## Project Structure

*   **`raga_pipeline/`**: Core Python package containing module logic.
    *   `sequence.py`: Note detection, clustering (KMeans/DBSCAN), and pattern analysis.
    *   `output.py`: Visualization and HTML report generation.
    *   `audio.py`, `raga.py`, `analysis.py`: Core processing modules.
*   **`driver.py`**: Main entry point script.
*   **`run_pipeline.sh`**: Helper script to run the driver in the correct environment.
*   **`batch_results/`**: Default directory where pipeline outputs are stored.
*   **`main notebooks/`**: (Legacy) Original research notebooks. Use the pipeline for production tasks.

---

## Troubleshooting

*   **"Raga not found" warning**: Ensure the raga name passed to `--raga` matches one of the standard names in the database (`raga_list_final.csv`).
*   **Sync Calculation Off**: If the interactive plot cursor is drifting, ensure you are using the latest version of `output.py` which accounts for plotting margins.
*   **Low Confidence**: For noisy recordings, try adjusting thresholds in `config.py` or use `--vocal-confidence` CLI args.

---
