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
*   **Technical Primer**: A multi-page [Technical Case Study](primer/index.html) explaining the system's inner workings for developers.

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

The project is designed to be run via the `run_pipeline.sh` script, which handles environment activation and path setup. The pipeline operates in two phases: **Detect** and **Analyze**.

### Phase 1: Detection (`detect`)

Runs stem separation, pitch extraction, and attempts to identify the Raga and Tonic.

```bash
./run_pipeline.sh detect \
  --audio "path/to/song.mp3" \
  --source-type instrumental
```

**Common Options:**
*   `--output`: Defaults to `./results` (optional).
*   `--source-type`: `mixed` (default), `vocal`, or `instrumental`.
*   `--melody-source`: `separated` (default) or `composite`. Use `composite` for instrumental recordings with poor stem separation.
*   `--vocalist-gender`: When provided, `--source-type` is auto-set to `vocal`.
*   `--separator`: `demucs` (default) or `spleeter`.
*   `--fmin-note` / `--fmax-note`: Override default pitch extraction range (e.g. `G1` to `C6`).
*   `--prominence-high` / `--prominence-low`: Fine-tune peak detection sensitivity.
*   `--bias-rotation`: Rotate histograms by median GMM deviation before scoring/plots.

**Output:**
*   `detection_report.html`: Summary of detected candidates.
*   `candidates.csv`: Ranked list of likely ragas.
*   Pitch data and stems saved in `results/htdemucs/<song_name>/` (stems are saved as MP3).

### Phase 2: Analysis (`analyze`)

Performs deep sequence analysis, phrasing, and generating the interactive report. **Requires a specific Tonic and Raga** (either from Phase 1 results or manually provided).

```bash
./run_pipeline.sh analyze \
  --audio "path/to/song.mp3" \
  --tonic "C#" \
  --raga "Bhairavi"
```

**Output:**
*   `analysis_report.html`: The **Final Report** featuring:
    *   Interactive, scrollable pitch plot with audio sync.
    *   Phrase analysis and common motifs.
    *   Raga correction statistics.
    *   Transition matrices.

### Batch Processing

Batch mode walks a directory of `.mp3`, `.wav`, `.flac`, or `.m4a` files and runs `run_pipeline.sh` on each one. The script now defaults to storing the ground-truth CSV alongside the input directory using the directory name (for example, `audio_test_files_gt.csv`). Run `python -m raga_pipeline.batch <audio_dir> --init-csv` from the project root to create that file, then rerun without `--init-csv` to process the directory; the batch tool will also read the same directory-level CSV unless you override `--ground-truth`.

---

## Project Structure

*   **`raga_pipeline/`**: Core Python package containing module logic.
    *   `sequence.py`: Note detection, clustering (KMeans/DBSCAN), and pattern analysis.
    *   `output.py`: Visualization and HTML report generation.
    *   `audio.py`, `raga.py`, `analysis.py`: Core processing modules.
*   **`driver.py`**: Main entry point script.
*   **`run_pipeline.sh`**: Helper script to run the driver in the correct environment.
*   **`results/`**: Directory where all pipeline outputs are stored.
*   **`main notebooks/`**: (Legacy) Original research notebooks. Use the pipeline for production tasks.

---

## Troubleshooting

*   **"Raga not found" warning**: Ensure the raga name passed to `--raga` matches one of the standard names in the database (`raga_list_final.csv`).
*   **Sync Calculation Off**: If the interactive plot cursor is drifting, ensure you are using the latest version of `output.py` which accounts for plotting margins.
*   **Low Confidence**: For noisy recordings, try adjusting thresholds in `config.py` or use `--vocal-confidence` CLI args.

---
