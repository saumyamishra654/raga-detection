# Raga Pipeline Documentation

User-facing documentation for running the raga detection pipeline.

This repository is driven by `driver.py` and a thin wrapper script `run_pipeline.sh`. The CLI is organized around three subcommands:
- `preprocess`: download YouTube audio to an MP3
- `detect`: Phase 1 detection (raga/tonic candidates + summary report)
- `analyze`: Phase 2 analysis (requires tonic+raga; generates the final report)

Note: the subcommand is spelled `analyze` (not `analyse`).

---

## Installation

Python dependencies:
```bash
pip install -r requirements.txt
```

System dependencies:
- `ffmpeg` and `ffprobe` in `PATH` (used for trimming, stem encoding, and duration checks)

Optional dependencies:
- Demucs (for `--separator demucs`, default)
- Spleeter (for `--separator spleeter`)
- `yt-dlp` (Python package; required for `preprocess --yt`)

---

## Running The Pipeline

Recommended (uses the wrapper script):
```bash
./run_pipeline.sh <subcommand> [args...]
```

`run_pipeline.sh` supports configurable activation:
- `RAGA_CONDA_SH`: explicit path to `conda.sh` (optional)
- `RAGA_CONDA_ENV`: env name to activate (default: `raga`)
- `RAGA_SKIP_ENV_ACTIVATE=1`: skip Conda activation
- `RAGA_PYTHON_BIN`: Python executable to run (default: `python3`)

You can always bypass the wrapper and run:
```bash
python driver.py <subcommand> [args...]
```

---

## Local App (Parameter Tuning UI)

The repository also includes a local-only FastAPI app for macOS development:

```bash
./run_local_app.sh
```

Open:
- `http://127.0.0.1:8765/app`

Design notes:
- Uses the same pipeline code path (`parse_config_from_argv(...)` -> `driver.run_pipeline(...)`).
- Uses parser introspection (`build_cli_parser()` + `get_mode_schema(...)`) as the single source of truth, so parameter changes in `config.py` flow into the UI automatically.
- Runs jobs serially in a local background worker and persists status/log snapshots in `.local_app_data/`.
- Serves generated artifacts directly and exposes quick-open detect/analyze report actions.
- Parses printed "next step" commands from logs and auto-loads suggested params into the next mode form (`preprocess -> detect`, `detect -> analyze`).
- Optional fields can be left blank to fall back to parser defaults; dependent fields are conditionally shown based on related selections.
- Detect/analyze `--audio` input supports drag-and-drop uploads; uploaded files are saved in local app storage and the resulting absolute path is used by pipeline runs.
- Detect/analyze `--audio` also includes a directory-driven file picker: default source directory is `../audio_test_files`, you can override it, and files are listed as dropdown options.
- The chosen audio directory is saved in browser local storage for future local app sessions.
- When an audio file is selected, local app auto-discovers matching artifacts by filename stem in output folders and loads them.
- Local app can launch batch processing (`raga_pipeline.batch`) for the selected directory.
- `--raga` input uses values from the raga DB CSV (`names` column fallback heuristics), with browser typeahead and fuzzy matching on blur; options refresh when `--raga-db` changes.
- After successful audio upload, the raw file path textbox is hidden from the visible form while retained as the submitted value.

---

## CLI Reference

### Preprocess (`preprocess`)

Downloads audio from a YouTube URL and saves an MP3 to `--audio-dir`.

```bash
./run_pipeline.sh preprocess \
  --yt "https://www.youtube.com/watch?v=..." \
  --filename "my_song_name" \
  --audio-dir "../audio_test_files" \
  --start-time "0:30" \
  --end-time "2:00"
```

Arguments:
- `--yt` (required): YouTube URL
- `--filename` (required): output base name (saved as `<filename>.mp3`)
- `--audio-dir` (default: `../audio_test_files`): output directory
- `--start-time`, `--end-time` (optional): trim range; supports `SS`, `MM:SS`, `HH:MM:SS`
- `--output` / `-o` (default: `batch_results`): only used as the suggested `--output` in the printed next-step `detect` command

Behavior notes:
- Fails if the target MP3 already exists.
- Validates trim times against the downloaded track duration.
- Prints a copyable `detect` command for the next step.

### Detect (`detect`)

Phase 1: stem separation, pitch extraction, histograms/peaks, and raga candidate ranking.

```bash
./run_pipeline.sh detect \
  --audio "/path/to/song.mp3"
```

Constrain scoring to specific tonics/ragas:
```bash
./run_pipeline.sh detect \
  --audio "/path/to/song.mp3" \
  --tonic "C,D#" \
  --raga "Bhairavi"
```

Common arguments (detect + analyze):
- `--audio` / `-a` (required): input audio file path
- `--output` / `-o` (default: `batch_results`): parent output directory
  - Outputs are written under:
    - Demucs: `<output>/<demucs-model>/<audio_filename>/...`
    - Spleeter: `<output>/spleeter/<audio_filename>/...`
- `--separator` (default: `demucs`): `demucs` or `spleeter`
- `--demucs-model` (default: `htdemucs`): Demucs model name
- `--source-type` (default: `mixed`): `mixed`, `instrumental`, `vocal`
  - If `--vocalist-gender` is provided, `source-type` is auto-set to `vocal`.
- `--melody-source` (default: `separated`): `separated` or `composite`
  - `composite` uses the original mix pitch/energy as the melody track for analysis.
- `--vocalist-gender` (optional): `male` or `female`
- `--instrument-type` (default: `autodetect`): `autodetect`, `sitar`, `sarod`, `bansuri`, `slide_guitar`
- `--fmin-note` (default: `G1`), `--fmax-note` (default: `C6`): pitch extraction range
- `--vocal-confidence` (default: `0.98`), `--accomp-confidence` (default: `0.80`): pitch confidence thresholds
- `--prominence-high` (default: `0.01`), `--prominence-low` (default: `0.03`): peak prominence factors
- `--bias-rotation`: disables histogram bias rotation (bias rotation is enabled by default)
- `--force` / `-f`: force pitch recomputation (stems are reused if present)
- `--raga-db`: override the raga database CSV path
- `--skip-separation`: currently parsed but not used by `driver.py` (stem separation still runs)

Detect-only arguments:
- `--tonic`: force tonic constraint (comma-separated allowed, e.g. `C,D#`)
- `--raga`: force raga constraint

Outputs (written into the stem directory):
- `detection_report.html`: Phase 1 HTML summary
- `candidates.csv`: ranked raga/tonic candidates
- `histogram_melody.png`, `histogram_accompaniment.png`
- `stationary_note_histogram_duration_weighted.png` and `stationary_note_histogram_duration_weighted.csv`
- cached pitch CSVs:
  - `composite_pitch_data.csv` (always computed in detect)
  - `melody_pitch_data.csv` (or `vocals_pitch_data.csv` when `--source-type vocal`)
  - `accompaniment_pitch_data.csv` (when accompaniment stem exists)
- stems:
  - `vocals.mp3`, `accompaniment.mp3`

At the end of `detect`, the pipeline prints a copyable `analyze` command using the top candidate.

### Analyze (`analyze`)

Phase 2: note transcription, phrase segmentation, motif/pattern analysis, and the final interactive report.

```bash
./run_pipeline.sh analyze \
  --audio "/path/to/song.mp3" \
  --tonic "C#" \
  --raga "Bhairavi"
```

Analyze-specific arguments:
- `--tonic` (required): tonic name, e.g. `C`, `D#`
- `--raga` (required): raga name
- `--keep-impure-notes`: keep notes not in the raga (default: discard during correction)
- `--transcription-smoothing-ms` (default: `0.0`): smoothing sigma in ms
- `--transcription-min-duration` (default: `0.02`): minimum stable note duration (seconds)
- `--transcription-stability-threshold` (default: `4.0`): max pitch change (semitones/sec) for stable regions
- `--energy-metric` (default: `rms`): `rms` or `log_amp`
- `--energy-threshold` (default: `0.0`): normalized threshold for filtering notes
- `--silence-threshold` (default: `0.10`): threshold for silence-based phrase splits (set `0` to disable)
- `--silence-min-duration` (default: `0.25`): minimum silence duration (seconds) to split phrases
- `--no-rms-overlay`: disable the energy overlay in report plots
- `--no-smoothing`: forces transcription smoothing off (`--transcription-smoothing-ms 0`)

Prerequisite:
- `analyze` expects cached pitch data in the stem directory. Run `detect` first with matching `--audio`, `--output`, `--separator`, and `--demucs-model`.

Primary output:
- `analysis_report.html` in the stem directory

---

## Batch Processing

Batch mode processes a directory of audio files by invoking `driver.py` with the current Python interpreter per file.

```bash
python -m raga_pipeline.batch /path/to/audio_dir --init-csv
python -m raga_pipeline.batch /path/to/audio_dir
```

Arguments:
- `input_dir` (positional): directory containing audio files
- `--init-csv`: create a blank ground-truth CSV for files in `input_dir`
- `--ground-truth` / `-g`: ground-truth CSV path
  - default: `__AUTO__`, resolved to `<input_dir>/<input_dir_name>_gt.csv`
- `--output` / `-o`: output directory (default: `<repo>/batch_results`)
- `--mode` / `-m`: `auto` (default) or `detect`
  - `auto`: if a ground-truth row has both `raga` and `tonic`, runs `analyze`; otherwise runs `detect`
  - `detect`: always runs `detect` and ignores ground truth for mode selection
- `--max-files`: process at most N pending files in this run (`0` means all pending)
- `--progress-file`: optional checkpoint JSON path
- `--exit-99-on-remaining`: exit `99` when files remain (useful for PBS auto-resubmission wrappers)
- `--silent` / `-s`: suppress console output (logs are still written)

Ground-truth CSV format:
- required header: a filename/name column, plus optional `raga` and `tonic`
- supported optional metadata columns: `instrument_type`, `vocalist_gender`, `source_type`, `melody_source`

Logs:
- written to `<output>/logs/<filename>.log`

---

## Output Layout And Caching

Given:
- `--output batch_results`
- `--separator demucs`
- `--demucs-model htdemucs`
- `--audio /path/to/song.mp3` (filename base `song`)

The stem directory is:
- `batch_results/htdemucs/song/`

Within that folder, the pipeline caches stems and pitch CSVs and writes HTML reports and plots. Re-running with the same options reuses cached stems/pitch unless `--force` is passed.
