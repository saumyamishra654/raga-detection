# Raga Detection System

A robust, pipeline-based system for automatic Hindustani raga identification and detailed melodic analysis. This project evolves from experimental notebooks into a structured, production-ready Python package.

## Key Features

*   **Raga & Tonic Identification**: Automatically detects the raga and tonic (Sa) from audio recordings using template matching and machine learning models.
*   **Flexible Audio Processing**: Handles **Vocal**, **Instrumental**, and **Mixed** recordings with integrated stem separation (using `htdemucs` or `spleeter`).
*   **High-Precision Analysis**:
    *   **Pitch Extraction**: Uses SwiftF0 for accurate framewise pitch tracking (10ms resolution).
    *   **Advanced Phrasing**: Utilizing **KMeans** and **DBSCAN** clustering to identify musical phrases and motifs.
    *   **Microtonal Analysis**: GMM-based analysis of pitch distributions.
*   **Interactive Reporting**: Generates detailed HTML reports with:
    *   **Scrollable Pitch Contours**: Wide, zoomable visualizations of the singer's pitch curves.
    *   **Audio Sync**: Click-to-seek functionality and a synchronized cursor that follows the audio playback.
    *   **Dynamic Scaling**: Visualizations automatically adapt to the singer's vocal range.

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
  --output "./results" \
  --source-type mixed
```

**Common Options:**
*   `--source-type`: `mixed` (default), `vocal`, or `instrumental`.
*   `--vocalist-gender`: `male` or `female` (helps with tonic detection bias).
*   `--instrument-type`: `sitar`, `sarod`, `bansuri`, etc. (for instrumental mode).

**Output:**
*   `detection_report.html`: Summary of detected candidates.
*   `candidates.csv`: Ranked list of likely ragas.
*   Pitch data and stems saved in `results/htdemucs/<song_name>/`.

### Phase 2: Analysis (`analyze`)

Performs deep sequence analysis, phrasing, and generating the interactive report. **Requires a specific Tonic and Raga** (either from Phase 1 results or manually provided).

```bash
./run_pipeline.sh analyze \
  --audio "path/to/song.mp3" \
  --output "./results" \
  --tonic "C#" \
  --raga "Bhairavi"
```

**Output:**
*   `analysis_report.html`: The **Final Report** featuring:
    *   Interactive, scrollable pitch plot with audio sync.
    *   Phrase analysis and common motifs.
    *   Raga correction statistics.
    *   Transition matrices.

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
