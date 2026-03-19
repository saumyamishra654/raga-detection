# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hindustani raga detection pipeline: ingests audio (YouTube/file/microphone), performs stem separation + pitch extraction, identifies ragas via histogram scoring, transcribes notes in sargam notation, and generates interactive HTML reports. Includes a local FastAPI web UI for parameter tuning and a corpus-level motif mining system.

## Conventions

- **NEVER use emojis** in code, frontend, or output.
- **After ANY code change**, update `raga_pipeline/CHANGELOG.md` with a dated entry (format: `## YYYY-MM-DD`).
- **After code changes**, update `raga_pipeline/LLM_REFERENCE.md` to stay in sync with the codebase.
- **Read `raga_pipeline/LLM_REFERENCE.md` first** before reading source code for orientation — it's the authoritative quick reference.
- Prefer `./run_pipeline.sh` for CLI runs (supports env config via `RAGA_CONDA_SH`, `RAGA_CONDA_ENV`, `RAGA_SKIP_ENV_ACTIVATE`, `RAGA_PYTHON_BIN`).

## Commands

### Running the pipeline
```bash
# Via conda wrapper (expects `raga` conda env at /opt/miniconda3)
./run_pipeline.sh preprocess --ingest yt --yt "URL" --filename "name"
./run_pipeline.sh detect --audio path/to/song.mp3
./run_pipeline.sh analyze --audio path/to/song.mp3 --tonic "C#" --raga "Bhairavi"

# Direct (activate your own env first)
python driver.py preprocess|detect|analyze [options]
```

### Running tests
```bash
# All tests
python -m pytest tests/

# Single test file
python -m pytest tests/test_cli_schema_args.py

# Single test method
python -m pytest tests/test_cli_schema_args.py::TestClassName::test_method_name -v
```

pytest is configured with `pythonpath = .` in `pytest.ini`. Tests use `unittest` framework with mocking and `@unittest.skipUnless` for optional dependencies.

### Type checking
```bash
python -m mypy raga_pipeline/ --python-version 3.10
```

### Batch processing
```bash
python -m raga_pipeline.batch /path/to/audio_dir            # auto mode
python -m raga_pipeline.batch /path/to/audio_dir --mode detect
python -m raga_pipeline.batch /path/to/audio_dir --init-csv  # create ground-truth CSV
```

### Motif mining (corpus-level)
```bash
# Mine motifs from a corpus of labeled recordings
python -m raga_pipeline.motifs mine \
    --gt ground_truth.csv --results-dir batch_results/ --output motif_index.json

# Score a new transcription against a mined motif index
python -m raga_pipeline.motifs score \
    --index motif_index.json --transcription transcribed_notes.csv --tonic C#
```

### Local web app
```bash
./run_local_app.sh   # FastAPI on http://127.0.0.1:8765/app
```

## Architecture

### Three-phase pipeline (driver.py)

1. **Preprocess** -- Audio ingestion (YouTube download, file upload, tanpura-assisted recording). Outputs MP3 to `audio_dir`.
2. **Detect** (Phase 1) -- Stem separation (Demucs/Spleeter) -> pitch extraction (SwiftF0) -> dual-resolution histogram -> peak detection -> GMM microtonal analysis -> raga candidate scoring. Outputs `detection_report.html`, `candidates.csv`, cached pitch CSVs.
3. **Analyze** (Phase 2) -- Note transcription (stationary + inflection point hybrid) -> phrase segmentation -> raga correction -> in-recording motif/pattern analysis -> transition matrices. Outputs `analysis_report.html`, interactive pitch plot, `transcribed_notes.csv`. Requires cached pitch data from Detect.

### Core modules (raga_pipeline/)

| Module | Role |
|---|---|
| `config.py` | `PipelineConfig` dataclass (50+ params), CLI parser (`build_cli_parser`), validation |
| `audio.py` | Stem separation, pitch extraction (`PitchData`), YouTube/recording ingest |
| `analysis.py` | Histogram construction (`HistogramData`), peak detection (`PeakData`), GMM fitting |
| `raga.py` | `RagaDatabase`, `RagaScorer` (8-coefficient scoring), candidate generation, aaroh/avroh DB |
| `sequence.py` | `Note`/`Phrase` detection, sargam conversion, phrase segmentation, transition matrices, in-recording motif helpers |
| `transcription.py` | `TranscriptionEvent`, stationary point detection, note snapping |
| `output.py` | HTML report generation (detection + analysis), all visualizations, `AnalysisResults`/`AnalysisStats` |
| `batch.py` | Directory walking, ground-truth CSV handling, job orchestration, HPC chunking |
| `motifs.py` | **Corpus-level motif mining and scoring** (standalone CLI, separate from analyze flow) |
| `runtime_fingerprint.py` | Stage-level code hashing for report reproducibility/freshness tracking |
| `cli_schema.py` | Schema introspection for dynamic UI generation (`get_mode_schema`, `list_modes`) |

### Motif mining system (motifs.py) -- current active work

Separate pipeline from per-recording analyze. Operates offline on a labeled corpus to build a raga-specific motif index, then scores new transcriptions against it.

**Token encoding:** Each note becomes `"{normalized_sargam}:{pitch_class_int}"` (e.g., `sa:0`, `re:2`). Octave-invariant by design -- strips octave markers, lowercases sargam, uses `round(pitch_midi) % 12`.

**Two phases:**

1. **Mine** (`mine_motifs`): Reads ground-truth CSV + `transcribed_notes.csv` (or edited) from prior analyze runs. Per recording: tokenize notes -> extract n-grams (sliding window, default length 3-8) -> accumulate per-raga stats. Candidate discovery via `_discover_candidates` (globs for transcription CSVs, matches to GT rows by stem). Motifs kept only if `recording_support >= min_recording_support` (default 3). Scored by: coverage (support/total recordings), specificity (raga support/global support), entropy (Shannon across ragas), weight (`specificity / (1 + entropy)`). Output: JSON index with per-raga motif lists ranked by `(recording_support DESC, weight DESC, total_occurrences DESC, length DESC)`.

2. **Score** (`score_transcription`): Tokenize new transcription -> extract n-gram positions -> for each raga in index, accumulate `weight * occurrences` for matching motifs. Returns ranked raga list, per-motif breakdown, and `phrase_overlays` showing which time spans had motif hits.

**In-recording motifs** (different from corpus mining): `sequence.py` has `extract_melodic_sequences` + `find_common_patterns` for single-recording pattern analysis during the analyze phase. `cluster_phrases` groups phrases by identical interval sequences.

### Code navigation (where to edit what)

| Intent | Primary file(s) |
|---|---|
| Pipeline step order, cache loading, cross-module plumbing | `driver.py` |
| CLI flags, defaults, validation | `raga_pipeline/config.py` |
| Audio ingest, separation, pitch extraction | `raga_pipeline/audio.py` |
| Histograms, peaks, GMM | `raga_pipeline/analysis.py` |
| Tonic bias, scoring, raga correction, aaroh/avroh DB | `raga_pipeline/raga.py` |
| Transcription (stationary/inflection), energy gating, snapping | `raga_pipeline/transcription.py` |
| Phrases, silence splitting, in-recording motifs, aaroh/avroh conformance | `raga_pipeline/sequence.py` |
| Corpus-level motif mining and scoring | `raga_pipeline/motifs.py` |
| HTML reports, plots, scrollable pitch plot, karaoke | `raga_pipeline/output.py` |
| Local app API endpoints, report serving, transcription editor | `local_app/server.py` |
| Job queue, cancellation, pipeline invocation | `local_app/jobs.py` |
| Parser-driven UI schemas, argv conversion | `raga_pipeline/cli_schema.py`, `raga_pipeline/cli_args.py` |

### Test routing

| Area | Test file(s) |
|---|---|
| CLI/schema contracts | `tests/test_cli_schema_args.py` |
| Driver detect/analyze/preprocess | `tests/test_driver_*.py` |
| Local app APIs/jobs/editor | `tests/test_local_app.py` |
| Report HTML regressions | `tests/test_output_*.py` |
| Transcription correctness | `tests/test_transcription_*.py` |
| Sequence/phrase logic | `tests/test_sequence_*.py` |
| Raga correction | `tests/test_raga_correction_rounding.py` |
| Motif mining CLI | `tests/test_motifs_cli.py` |
| Runtime fingerprinting | `tests/test_runtime_fingerprint_*.py` |

### Local web app (local_app/)

FastAPI server (`server.py`) with single-page frontend. Dynamically generates forms from CLI argparse schema (same parser as CLI). Background job queue with progress tracking. Artifact discovery, transcription editor (versioned save/load/default/regenerate/delete via API), drag-and-drop upload. Analyze workspace embeds report iframe + editor panel side by side.

### Output directory structure
```
<output>/<separator>/<model>/<song_name>/
├── detection_report.html, analysis_report.html
├── analysis_report.meta.json  (transcription_edit_payload for editor)
├── candidates.csv
├── *_pitch_data.csv (cached)
├── vocals.mp3, accompaniment.mp3 (stems)
├── histogram_*.png, transition_matrix.png
├── transcribed_notes.csv
└── transcription_edits/  (versioned edited transcriptions)
```

### Caching
- Stems are cached unless `--force-stems` (also requires `--force`)
- Pitch CSVs are cached unless `--force`
- Analyze expects cached pitch data from a prior Detect run with matching `--separator`/`--demucs-model`

## Key Design Patterns

- **Lazy imports** for heavy optional dependencies (torch, demucs, spleeter) -- check import patterns before adding new module-level imports
- **Dataclasses** for all structured data (`PipelineConfig`, `PitchData`, `HistogramData`, `Note`, `Phrase`, etc.)
- **Sargam notation**: S=Sa, r=komal Re, R=shuddha Re, g=komal Ga, G=shuddha Ga, m=shuddha Ma, M=tivra Ma, P=Pa, d=komal Dha, D=shuddha Dha, n=komal Ni, N=shuddha Ni
- **Raga database** (`raga_list_final.csv`): 110+ ragas with aroha/avroh patterns in sargam notation
- **Safe edit workflow**: update `config.py` first (if CLI-configurable) -> `driver.py` plumbing -> module internals -> local app adapters -> tests -> `LLM_REFERENCE.md` + `CHANGELOG.md`

## Related documentation

| File | Purpose |
|---|---|
| `raga_pipeline/LLM_REFERENCE.md` | Authoritative quick reference -- read first, update with changes |
| `raga_pipeline/CHANGELOG.md` | Daily work log -- update after every change |
| `TODO.md` | Project roadmap and backlog |
| `codebase_study_plan.md` | Deep architectural walk-throughs (reports, transcription, motifs, CLI flows) |

## Active work

- Corpus-level motif mining system (`motifs.py`) -- mine and score phases
- Sequence mining for raga identification (see `sequence-mining-plan.md` in TODO)
- Tuning gating for different instrument types

## Environment

- Python 3.10 (Conda recommended)
- Requires `ffmpeg`/`ffprobe` in PATH
- `swift-f0` installed separately for pitch extraction (env vars: `RAGA_SWIFTF0_PROVIDER`, `RAGA_SWIFTF0_STRICT_PROVIDER`, `RAGA_SWIFTF0_PROVIDER_LOGS`)
- `torch` + `demucs` for stem separation (optional, with CPU/MPS fallback on Apple Silicon)
