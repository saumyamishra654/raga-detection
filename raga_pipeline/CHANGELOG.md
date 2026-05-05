# Raga Pipeline Changelog

> **For LLM Agents:** Update this file with every significant code change. Add entries under the current date. Create a new date section if one doesn't exist for today. Keep entries concise but informative.

---

## 2026-05-05

### Refactor

- **tokenize_notes_for_lm:** Extracted note-to-token logic into `_tokenize_note()` helper. Added optional `phrases: List[Phrase]` parameter so pre-computed phrase boundaries (e.g. from `detect_phrases_by_silence`) flow through to LM tokenization instead of the internal 0.25s gap heuristic. Flat note list path unchanged for backward compat.

### Experiment

- **Exp 30 (RMS phrase boundaries + weight sweep):** RMS energy-based phrase boundaries improve LM-only accuracy +1.3pp (256->260 at GT tonic). With retuned weights (alpha=0.3, beta=0.5), combined top-1 reaches **90.6%** (269/297), up from 88.9% baseline. New best honest result. Results at `results/pipeline_loo_rms_phrases/` and `results/score_weight_sweep/`.

## 2026-05-04

### Added

- **config.py:** Added `phrase_method` field to `PipelineConfig` (default `"rms"`) and corresponding `--phrase-method` CLI argument with choices `rms` / `gap`. Wired through `_config_from_parsed_args`. This controls which phrase detection method the driver dispatches to: `"rms"` for energy-based silence detection (new default) or `"gap"` for legacy inter-note gap clustering.
- **runtime_fingerprint.py:** Added `detect_phrases_by_silence` selector to analyze-phase fingerprint tracking so changes to the new primary phrase detection function are reflected in runtime hashes.

### Documentation

- **CLAUDE.md:** Added "Phrase splitting" subsection documenting the two-stage phrase segmentation pipeline (inter-note gap KMeans method + RMS energy silence splitting), config defaults, and post-processing steps.

### Refactor

- **RMS-primary phrase splitting:** Added `detect_phrases_by_silence()` as the primary phrase detection method (default `--phrase-method rms`). Takes flat note list + energy track, splits at sustained silence regions. Legacy gap-based method (`detect_phrases`) available via `--phrase-method gap`. Replaced two-stage detect+split chain in driver with single dispatch.

## 2026-04-26

### Experiment

- **Exp 22b (n-gram order sweep, honest uncorrected):** Reran `sweep_ngram_order.py` on verified uncorrected stems (`separated_stems_nocorrection/htdemucs`). Original Exp 22 was LEAKED (default `--stems-root` pointed to corrected `separated_stems/htdemucs`). Honest results: order=6 optimal at 87.5% top-1 (was 96.6%), order=1 drops to 68.7% (was 91.2%). Peak shifted from order=7 to order=6; orders 7-8 overfit on noisier uncorrected data. Results at `results/ngram_order_sweep_uncorrected/`.

## 2026-04-25

### Experiment

- **Exp 23b (truly uncorrected baseline):** Re-ran pipeline LOO with verified uncorrected data from `separated_stems_nocorrection/htdemucs/`. Result: **88.9% combined top-1** (264/297), 94.9% top-3, 92.0% given-tonic. Verified: pitch data identical within 0.005 Hz, transcriptions genuinely different for 291/297 recordings. **New honest best result**, beating Exp 26 per-hyp correction (85.5%) by 3.4pp due to distribution matching.
- **Exp 29 (top-3 rerank):** Top-3 histogram candidates at detected/given tonic, per-hyp corrected, LM rerank. Auto: 70.7%, given: 72.7%. Negative result: top-3 recall ceiling only 86.9% (histogram misses GT in 39/297 cases), too aggressive a filter.
- **Exp 23 leakage confirmed:** Original 89.6% "uncorrected baseline" used corrected transcriptions from default stems directory. Verified by comparing 300 recordings: 291 differ, 6 identical (already in-scale), 3 missing. The ~0.7pp difference (89.6% vs 88.9%) reflects the small benefit of pre-corrected transcriptions.
- **Exp 28c (alignment, 30-raga filter):** Full 297-recording LOO with corrected-train alignment LM, top-10 histogram candidates, 30-raga filter. Sub_fraction delta = -0.089 (GT needs fewer substitutions). Best: logistic 66.7%, C2 (LM - 2.0*sub) 63.6%, both beating hist-only 56.6%. GT recall 97.6% (290/297). Still below 88.9% baseline.

### Production Integration

- **Trained production LM model** (order=7, 30 ragas, all 297 recordings) at `raga_pipeline/models/compmusic_ngram_model.json`. Uses uncorrected transcriptions from `separated_stems_nocorrection` to match Exp 23b distribution.
- **LM scoring now enabled by default:** `use_lm_scoring=True`, `lm_skip_correction=True` in `config.py`. Pipeline automatically loads model from `raga_pipeline/models/`.
- **LM winner feeds back into `results.detected_raga`:** After Step 5.5 re-ranking, the combined top-1 raga/tonic replaces the histogram top-1, so `detection_report.meta.json` and the subsequent analyze step use the LM-reranked result.
- **Updated `_find_lm_model_path()`** to search `raga_pipeline/models/compmusic_ngram_model.json` first.
- **Synced pipeline to raga-web-app:** Copied `language_model/` module, trained model, `driver.py`, `config.py`, `sequence.py`. Updated `api/routes/results.py` to prefer `lm_candidates.csv`.
- **Web app: raga dropdown now shows 30 CompMusic ragas** (reads from trained LM model instead of full 113-raga CSV). Falls back to CSV if model not found.
- **Web app: dual candidates display** -- results page shows both "Combined (Histogram + LM)" and "Histogram Only" candidate tables side by side. API returns `histogramCandidates` alongside `candidates`.
- **Fixed CLI defaults:** Changed `--use-lm-scoring` and `--lm-skip-correction` from `store_true` (always-false default) to `BooleanOptionalAction` with `default=True`. Use `--no-use-lm-scoring` to disable.
- **Fixed meta.json hero display:** Web app now reads `selected_raga` (LM winner) over `top_raga` (histogram) in results endpoint and job runner.
- **YouTube start/end time trimming:** Upload page YouTube tab now has optional start/end time inputs (MM:SS format). Backend passes them through to `download_youtube_audio()` which trims via ffmpeg post-download.
- **Browser microphone recording with tanpura:** New "Record" tab in upload page. Uses MediaRecorder API for browser-side recording. Optional tanpura drone playback (12 keys) during recording via HTML5 Audio streaming from `/api/tanpura-audio/{key}`. Tanpura key auto-sets tonic. Recording uploads as file and creates song in library with `source: "recording"`.
- **Tanpura API endpoints:** `/api/tanpura-tracks` lists available keys, `/api/tanpura-audio/{key}` streams tanpura MP3. Tanpura assets copied from pipeline repo.
- **Recording source in library:** Songs from recordings show green "Rec" badge. Appear under "Uploaded" filter alongside file uploads.

### Feature

- New `scripts/sweep_top3_rerank_loo.py`: top-K rerank at detected/given tonic with per-hyp correction and LM reranking. Supports auto (detected tonic) and given (GT tonic) strategies.
- Added `--gated-only`, `--top-k`, and auto 30-raga filter to `sweep_alignment_loo.py`
- Added sub_fraction penalty methods (C2, C3) and hist-only baseline (G) to Phase 2 scoring
- Added sub_fraction to logistic regression features and LM diagnostic printout

### Documentation

- Comprehensive update to `docs/reports/2026-04-18-advanced-scoring-experiments.md`: renamed to Exp 16-29, added Exp 23b and 29, marked Exp 23 as LEAKED, updated cross-experiment summary with corrected rankings, added error decomposition and confusion pair analysis, added same-scale raga group table, added improvement strategies section

## 2026-04-24

### Experiment

- **Exp 28a (alignment diagnostic):** Lambda_match sweep on 15 recordings. Discovered lm_per_token discards substitution cost -- sub_fraction is the discriminative feature (w_sub=1 gives 100% on 15 rec).
- **Exp 28b (alignment, no filter):** 297-recording LOO, top-10, 78-raga histogram. Sub_fraction delta -0.048. Best C2 (LM - 5.0*sub) 69.7%. Hist-only depressed to 56.6% due to LM filtering artifacts (78 DB ragas vs 30 LM ragas).

## 2026-04-23

### Feature

- New `raga_pipeline/language_model/alignment.py`: beam DP alignment scorer for noisy-channel LM scoring. Skip/match/substitute transitions with configurable penalties (lambda_skip, lambda_match, lambda_sub). Phrase-local alignment with beam pruning.
- `scripts/sweep_alignment_loo.py`: LOO evaluation script with two-phase design (feature collection + fold-safe scoring methods). Includes hist-only baseline, z-scored fusion, logistic regression, RRF.
- Added `NgramModel.vocabulary` property and `NgramModel.remove_raga()` method.

## 2026-04-19

### Feature

- New offline sweep script `scripts/sweep_perhyp_correction_loo.py`: per-hypothesis correction LOO experiment. Trains n-gram LM on GT-raga-corrected transcriptions, then at test time corrects the raw uncorrected transcription per each candidate (tonic, raga) pair before scoring against the LM. Phase 1 (calibration) checks LM calibration within same-scale adversarial groups. Phase 2 (full LOO) runs histogram + per-hypothesis-corrected LM combined scoring with checkpoint/resume support. Eliminates train/test distribution mismatch from corrected-train approach.

## 2026-04-18

### Feature

- Added standalone `autoresearch_transcription_repo/` scaffold for autonomous transcription-parameter search (Karpathy-style autoresearch loop). Includes Optuna-based trial runner (`run_study.py`), per-recording evaluator with correction-burden objective (`src/autoresearch_tuner/evaluator.py`), trial logging (`trial_metrics.jsonl`), reproducible defaults (`configs/default_study.json`), and an explicit agent system prompt (`prompts/system_prompt.md`).

- Enhanced `autoresearch_transcription_repo/` objective modes with an F1-style retention/deletion target (`f1_retention_deletion`) and optional F1+token blend (`f1_retention_deletion_plus_token`). The default study config now uses F1 retention/deletion optimization, while preserving the original weighted-sum objective as a selectable mode.

- Finalized a fixed reusable system prompt at `autoresearch_transcription_repo/prompts/system_prompt.md` with a single objective definition (`1 - retention_deletion_f1`), explicit leakage guards, fixed tuning scope, and required reporting outputs for thesis-safe runs.

- Fixed autoresearch loop initialization crash in `autoresearch_transcription_repo/src/autoresearch_tuner/study.py` by using the repository's supported `RagaDatabase` constructor API (with `from_csv` compatibility fallback), resolving `AttributeError: type object 'RagaDatabase' has no attribute 'from_csv'` during `run_study.py` startup.

- Reduced noisy per-recording console output during raga correction by gating `get_raga_notes(...)` match logs behind `RAGA_LOG_RAGA_MATCHES=1` in `raga_pipeline/raga.py` (default now silent). This keeps long tuning loops readable without changing correction behavior.

- Added a true LLM-driven autoresearch path in `autoresearch_transcription_repo/` with `run_llm_autoresearch.py` and `src/autoresearch_tuner/llm_loop.py`. The new loop queries an LLM for the next parameter proposal, evaluates it through the existing pipeline/evaluator, and keeps/discards based on objective improvement while journaling every iteration (`llm_autoresearch_journal.jsonl`) and persisting best state (`llm_autoresearch_state.json`). Added companion config `configs/llm_autoresearch.json` and Karpathy-style instruction file `program.md`.

- New offline sweep script `scripts/sweep_cadence_lm.py` (Experiment 21): trains per-raga n-gram LM on cadential phrases only (last 4 notes before each return to Sa) and evaluates via LOO whether cadence-restricted models add discriminative power. Cadence extraction: phrase split at 0.5s gap, keep Sa-terminated phrases with >= 2 pre-Sa notes, tokenize with `tokenize_notes_for_lm`. Model tiers: trigram (add_k=0.1) for ragas with >= 20 cadence tokens, bigram backoff (add_k=0.5) for 10-20 tokens, skip below 10. Outputs `results/cadence_lm/{progress,summary,cadence_examples}.csv`. Checkpoint/resume support.

- New offline sweep script `scripts/sweep_gmm_fingerprint.py` (Experiment 20): extracts per-swara within-note features (width, shruti offset, skew) from both raw frame-level f0 statistics and existing GMM fits.  Phase C1 builds LOO raga templates and runs Welch t-tests with BH correction on confused pairs; Phase C2 (conditional) performs LOO distance-based classification using weighted squared deviation.  60 features (5 per PC x 12 PCs). Checkpoint/resume support.  Outputs `results/gmm_fingerprint/{fingerprints,confused_pair_analysis,loo_integration,summary}.csv`.

- New offline sweep script `scripts/sweep_positional_pch.py` (Experiment 18): evaluates three positional pitch-class histogram features -- nyas PCH (phrase-ending), phrase-start PCH, and octave-stratified PCH (36-dim) -- as standalone raga discriminators via leave-one-out cosine similarity against per-raga mean templates. Sweeps `phrase_gap_sec` in {0.25, 0.5, 1.0}. Reports accuracy under both GT-tonic and detected-tonic (from saturation calibration) regimes. Checkpoint/resume support. Outputs `results/positional_pch/{progress,summary}.csv`.

- New offline sweep script `scripts/sweep_microtonal_pch.py` (Experiment 19): tests whether 24-TET or 36-TET pitch-class histograms capture microtonal raga differences better than standard 12-TET. Leave-one-out evaluation using GT tonic, cosine similarity against per-raga mean templates. Checkpoint/resume support. Outputs `results/microtonal_pch/{progress,summary}.csv`.

### Fix

- Raise fit_norm clip from [-1.0, 1.0] to [-2.0, 2.0] in `raga.py` (Exp 16 calibration). The band-pass tonic boost was saturating candidates at the clip boundary, creating large tied clusters and suppressing histogram-only raga accuracy from ~74% to ~24%. Combined score was unaffected because LM dominates, but the saturation would interfere with new additive scoring terms.

### Enhancement

- `--lm-model` now auto-discovers `compmusic_ngram_model_uncorrected.json` (preferred) or `compmusic_ngram_model.json` from the project root or `raga_pipeline/models/` when `--use-lm-scoring` is passed without an explicit path. New `PipelineConfig._find_lm_model_path()` mirrors the existing `_find_model_path()` pattern.

## 2026-04-17

### Feature

- New offline analysis script `scripts/sweep_truncation.py`: for each recording in the GT CSV and each time window (first_300/first_600/first_900/full/last_300), slices cached pitch and transcription CSVs, reruns `RagaScorer.score(...)` with pipeline-parity defaults (confidence thresholds 0.95/0.80, `use_confidence_weights=True`, `prominence_high=0.01`, `prominence_low=0.03`, raw `len(peaks.validated_indices)`, gender-filtered tonic candidates, 30-raga filter from GT), and scores the sliced transcription against the pre-trained LM. Writes resume-safe `results/truncation_sweep/{progress,summary}.csv`.

### Results (Experiment 15)

- Truncation sweep, 298 recordings x 5 windows. Tonic top-1: first_300=88.9% -> full=92.3% (-3.4 pp). End-to-end (LM + detected tonic): first_300=88.6% -> full=92.3% (-3.7 pp). Conclusion: **truncation hurts**, full window is best.
- Per-raga: Todi, Alhaiya Bilawal, Kedar, Marwa, Puriya Dhanashree all drop 20+ pp when truncated to first 5 min -- their drut/antara carries raga-defining content the alap alone cannot establish.
- Incidental finding: `fit_score` saturation from the band-pass boost (`ACC_BAND_WEIGHT=0.80`) causes large tied clusters at `fit_score=1000`. Pipeline end-to-end accuracy unaffected (normalization within combined score squashes the ties), but histogram-alone is no longer a clean signal. Flagged for post-thesis cleanup.

### Documentation

- Added Experiment 15 section to `docs/reports/2026-04-01-language-model-experiments.md`; updated Final Summary and Open Questions.

---

## 2026-04-16

### Feature

- Added band-pass accompaniment tonic boost to `RagaScorer` (`raga_pipeline/raga.py`). Builds a second accompaniment pitch-class histogram from voiced frames whose `pitch_hz` falls in `[ACC_BAND_MIN_HZ, ACC_BAND_MAX_HZ)` (defaults 100-300 Hz, the tanpura fundamental range) and adds `ACC_BAND_WEIGHT (0.80) * H_acc_band[tonic] / max(H_acc_band)` to the fit score. Replaces the previous full-range `TONIC_SALIENCE_WEIGHT` term (now unused, retained for backward compat). Raises tonic top-1 detection from 74.2% to 94.3% on the CompMusic 298-recording subset (offline eval with gender range filter).
- Added Sa-Pa accompaniment pair boost (`SAPAPAIR_WEIGHT = 0.20`): rewards candidate tonics where the full-range accompaniment histogram has mass at both `tonic` and `(tonic+7) % 12`, capturing the tanpura's Sa/Pa drone pair (committed separately in `efd132c`).
- New `candidates.csv` diagnostic columns: `tonic_sal_full_norm`, `tonic_sal_band_norm`, `band_frame_count`.
- Added checkpoint/resume support to `run_pipeline_loo.py` and `run_pipeline_loo_known_tonic.py` via `run_pipeline_loo_progress.csv` and `run_pipeline_loo_known_tonic_progress.csv`. Reruns skip already-processed filenames; summary metrics are rebuilt from checkpoint rows. Failures are recorded as `FAILED` rows and excluded from accuracy counts.
- New offline analysis script `scripts/sweep_tonic_bandpass.py`: sweeps frequency bands and weights against existing `nometa/*/candidates.csv` and accompaniment pitch CSVs, writes `results/bandpass_tonic/{band_sweep,weight_sweep}_results.csv`.

### Results

- Pipeline LOO (uncorrected LM, detected tonic) top-1: 84.2% (Exp 13) -> **87.2%** (Exp 14, 260/298).
- Pipeline LOO top-3: 93.0% -> **95.0%** (283/298).
- Closes 56% of the gap to the known-tonic ceiling (89.6% top-1).

### Documentation

- Added Experiment 14 section and updated Final Summary / Open Questions in `docs/reports/2026-04-01-language-model-experiments.md`.

---

## 2026-03-30

- Added n-gram language model for raga detection (`raga_pipeline/language_model/`).
  - Shared tokenizer `tokenize_notes_for_lm` in `sequence.py`: octave-marked sargam tokens with `<BOS>` phrase boundaries.
  - `NgramModel` class: per-raga n-gram counts (order 1..N), add-k smoothing, interpolated scoring, JSON serialization.
  - `train_model`: builds models from ground-truth CSV + transcription CSVs (reuses `motifs.py` candidate discovery).
  - `score_transcription`: classifies recordings with optional segment-level confidence curves.
  - `evaluate_model`: leave-one-out cross-validation with order sweep support.
  - CLI: `python -m raga_pipeline.language_model train|score|evaluate`.
  - 25 tests across 6 test files (`tests/test_lm_*.py`).

---

## 2026-03-26

### Feature

- Added `--compare-extractors` flag for analyze mode: runs both SwiftF0 and pYIN, binary-searches confidence thresholds to produce matching raw note counts, and shows both transcriptions in the HTML report with a toggle (karaoke, transcription table, correction summary, pattern analysis per extractor). - `raga_pipeline/config.py`, `driver.py`, `raga_pipeline/output.py`

### Removal

- Removed CREPE pitch extractor entirely (unresolvable TF 2.11 + numpy 2.x + protobuf conflicts). Cleaned from: `audio.py`, `config.py`, `driver.py`, `cli_schema.py`, `runtime_fingerprint.py`, tests, and docs. `--pitch-extractor` choices are now `{swiftf0, pyin}`.

### Bugfix

- Fixed pYIN producing near-empty histograms by adding extractor-specific confidence threshold defaults. CLI `--vocal-confidence` and `--accomp-confidence` now default to None (auto-resolved per extractor): swiftf0=0.95/0.80, pyin=0.15/0.05. Explicit user overrides still take precedence. - `raga_pipeline/config.py`
- Fixed `voiced_times` / `midi_vals` dimension mismatch that crashed analyze-mode plotting with pYIN. `extract_pitch()` now bakes the confidence threshold into the `voicing` array stored in PitchData (raw voicing is still saved to CSV for cache). - `raga_pipeline/audio.py`
- Fixed motif token assertions to match case-preserving sargam tokens (e.g. `Sa:0`, `Re:2` not `sa:0`, `re:2`). Sargam casing encodes komal/shuddha/tivra distinctions. - `tests/test_motifs_cli.py`

### Testing

- Added tests for extractor-specific confidence resolution, explicit override precedence. Removed CREPE tests. - `tests/test_pitch_extractors.py`, `tests/test_cli_schema_args.py`

---

## 2026-03-22

### Feature

- Added multi-backend pitch extraction support: pYIN (via librosa, configurable hop) and CREPE (deep learning, configurable hop, requires TensorFlow) alongside existing SwiftF0. New CLI flags: `--pitch-extractor {swiftf0,pyin,crepe}`, `--pitch-hop-ms` (frame hop in ms for pyin/crepe), `--crepe-model {tiny,small,medium,large,full}`. Cache CSVs are extractor-namespaced to prevent collisions. Unified CSV export replaces SwiftF0-specific export_to_csv path. - `raga_pipeline/audio.py`, `raga_pipeline/config.py`, `driver.py`, `raga_pipeline/cli_schema.py`, `raga_pipeline/runtime_fingerprint.py`

### Testing

- Added unit tests for pitch extractor dispatch, pYIN/CREPE backend output, cache filename generation, and CLI argument parsing. - `tests/test_pitch_extractors.py`

### Documentation

- Updated LLM reference with pitch extractor selection docs, new config params, and extractor-specific CSV naming. - `raga_pipeline/LLM_REFERENCE.md`

## 2026-03-01

### Bugfix

- Fixed transcription-editor phrase merge ordering in both local app and report-embedded templates by removing placeholder `start/end=0` merge rows, recomputing merged phrase bounds from note timings, and reinserting merged phrases at the earliest selected timeline index. - `local_app/static/transcription_editor.js`, `raga_pipeline/output.py`
- Hardened local-report asset fallback rewriting to only use basename fallback when exactly one unique local candidate exists, preventing ambiguous wrong-audio rewrites. - `local_app/server.py`
- Restored the in-panel job status progress bar and hardened the header progress widget with JS inline fallback styling so it no longer degrades to plain top-left `0%` text when stale CSS is served. - `local_app/templates/index.html`, `local_app/static/app.js`

### Feature

- Made non-required local-app parameters collapsible per schema section via `Optional parameters (N)` details blocks while keeping required fields always visible. - `local_app/static/app.js`, `local_app/static/style.css`
- Tuned pitch defaults for finer taan/small-movement capture (`--fmax-note` default `D6`, `--vocal-confidence` default `0.95`) and expanded CLI guidance text for taan-sensitive tuning across pitch/transcription controls. - `raga_pipeline/config.py`
- Added explicit Demucs device diagnostics (requested vs selected backend, CUDA/MPS availability, MPS fallback env visibility), plus guarded MPS runtime retry on CPU with clear effective-device logging. - `raga_pipeline/audio.py`
- Updated phrase-karaoke base note styling to blue while preserving sung/current highlight emphasis. - `raga_pipeline/output.py`
- Refactored preprocess ingest semantics to canonical required values (`yt`, `recording`, `tanpura_recording`), removed public `--record-mode`, added legacy alias normalization, and enforced conditional requirements for `--yt`/`--tanpura-key`. - `raga_pipeline/config.py`, `driver.py`
- Added detect-only `--force-stems` (requires `--force`) and wired it through local UI dependency logic and runtime stem recomputation bypass. - `raga_pipeline/config.py`, `raga_pipeline/audio.py`, `driver.py`, `local_app/static/app.js`, `raga_pipeline/cli_schema.py`
- Updated local preprocess recording API and UI to key off canonical ingest modes (with `record_mode` compatibility alias), including tanpura-required validation in tanpura recording mode. - `local_app/server.py`, `local_app/static/app.js`
- Moved visual job progress from the large in-panel progress bar to a compact top-left circular header icon with state-aware styling (`idle/running/completed/failed/cancelled`) and live percentage updates. - `local_app/templates/index.html`, `local_app/static/style.css`, `local_app/static/app.js`

### Documentation

- Updated CLI option docs with taan-focused tuning guidance and revised defaults for pitch extraction/transcription controls. - `README.md`

## 2026-02-26

### Documentation

- Added a comprehensive "Code Navigation: Where To Edit What" section to LLM reference, mapping common change intents to the exact files/symbols to modify across pipeline, reports, and local app layers. - `raga_pipeline/LLM_REFERENCE.md`
- Aligned HPC usage docs and PBS wrapper with legacy cluster-script conventions (env activation, optional ffmpeg PATH, exit-99 resubmission loop) while keeping current `raga_pipeline.batch` flags/behavior (`--max-files`, `--progress-file`, `--exit-99-on-remaining`). - `hpc/README.md`, `hpc/pipeline_batch.pbs`
- Updated HPC wrapper/docs to match Ashoka PBS GPU-node usage: explicit `ngpus` request in resource line, project-code submission variant (`qsub -P`), and queue monitoring commands (`qstat`, `qstat -n`, `qstat -f`, `qdel`, `qstat -Q`). - `hpc/pipeline_batch.pbs`, `hpc/README.md`
- Finalized HPC wrapper defaults for your cluster path layout and reliability: set fresh default output dir (`batch_results_v2`), switched resubmission script path to absolute, and enforced repo-root execution for module-based batch runs. - `hpc/pipeline_batch.pbs`, `hpc/README.md`

## 2026-02-25

### Feature

- Propagated GMM bias rotation into analyze-phase transcription snapping so stationary and inflection note pitches are bias-corrected before chromatic/raga snapping. - `raga_pipeline/transcription.py`, `driver.py`
- Applied GMM bias rotation to analyze pitch-contour rendering (static `pitch_sargam.png` and scrollable plot/hover MIDI), keeping contour and transcription aligned with rotated histogram reference. - `raga_pipeline/output.py`, `driver.py`
- Added a backlog task to align pitch-class guide lines/snapping with detected histogram peak centers rather than fixed semitone bins. - `TODO.md`

## 2026-02-24

### Feature

- Enhanced scroll-plot hover inspector to show the pitch-track value at the hovered timestamp (`Pitch @ t`, including sargam/western note + MIDI) above nearest transcription-note details. - `raga_pipeline/output.py`
- Added dotted hover cross-guides in the scroll plot: vertical guide from x-axis up to the hovered pitch point and horizontal guide from that point to the y-axis. - `raga_pipeline/output.py`
- Darkened dotted hover cross-guide styling to improve contrast over the scroll-plot background. - `raga_pipeline/output.py`
- Moved transcription editing UI out of `analysis_report.html` and into local app Analyze workspace (embedded report iframe + in-app editor mount), while keeping versioned save/default/regenerate/delete APIs and edited-report generation behavior. - `local_app/templates/index.html`, `local_app/static/app.js`, `local_app/static/transcription_editor.js`, `local_app/server.py`
- Added report-metadata seed payload (`transcription_edit_payload`) and new local-app base payload endpoint (`GET /api/transcription-edits/{dir_token}/{report_name}/base`) so the app editor can initialize without parsing report HTML. - `raga_pipeline/output.py`, `local_app/server.py`, `local_app/schemas.py`
- Extended audio-artifact discovery response with `analyze_report_context` (`url`, `dir_token`, `report_name`) so frontend can bind editor APIs without URL parsing heuristics. - `local_app/server.py`

### Testing

- Updated scroll-inspector HTML coverage to assert hover pitch payload/hooks and rendering helpers for `Pitch @ t` tooltip output. - `tests/test_output_scroll_inspector.py`
- Extended scroll-inspector HTML checks for hover guide DOM elements and guide-mapping helpers (`midiToY`, `showHoverGuides`, `hideHoverGuides`). - `tests/test_output_scroll_inspector.py`
- Replaced in-report editor HTML assertions with metadata-seed and report-read-only coverage, and added local app tests for analyze workspace context payload, `/base` ready/legacy flows, and app template inclusion of the editor workspace/script. - `tests/test_output_transcription_editor.py`, `tests/test_local_app.py`

### Documentation

- Updated LLM reference to document the new local-app Analyze transcription editor flow, read-only analyze reports, and `analysis_report.meta.json` payload seeding contract. - `raga_pipeline/LLM_REFERENCE.md`

## 2026-02-23

### Bugfix

- Fixed raga correction so accepted notes always retain the snapped/rounded pitch (`corrected_midi`) instead of leaking raw microtonal values into final transcription, and refreshed derived note metadata (`pitch_hz`, `pitch_class`, `sargam`) after correction. - `raga_pipeline/raga.py`
- Unified transcription energy gating so inflection notes now sample nearest-frame energy, respect `--energy-threshold` inside transcription, and no longer disappear due to downstream zero-energy filtering. - `raga_pipeline/transcription.py`, `driver.py`
- Relaxed analyze phrase defaults so short audible fragments are retained by default (`phrase_min_duration=0.2`, `phrase_min_notes=1`). - `raga_pipeline/config.py`
- Prevented native drag-and-drop of the scrollable pitch image during range selection, and normalized inspector note labels to strip octave markers so notes render as base sargam symbols only. - `raga_pipeline/output.py`
- Improved note-fragment consolidation with short-gap dropout healing in `merge_consecutive_notes`, and added phrase-level collapsing so consecutive same-note segments inside each phrase are emitted as a single note in final transcription outputs. - `raga_pipeline/sequence.py`, `driver.py`
- Updated inspector note breakdown to a scrollable table capped at 5 visible rows (Note/Duration/MIDI columns) for easier reading of dense selections. - `raga_pipeline/output.py`
- Fixed regenerated analyze reports that could lose the original-audio link when metadata stored repo-relative audio paths by hardening media-path resolution and normalizing report metadata audio paths to absolute values. - `local_app/server.py`, `raga_pipeline/output.py`
- Fixed regenerate fallback context when report metadata is missing/incomplete by recovering raga/tonic and audio source paths directly from existing report HTML (subtitle + audio source tags), preventing `Unknown (Tonic: Unknown)` regressions and broken original-audio links. - `local_app/server.py`

### Feature

- Added an interactive inspector to analyze scroll plots with click point-query (seek + sampled overlay energy + active/nearest transcription note) and drag-range query (time span, overlay energy min/mean/max, overlapping transcription rows + summary). Selection state now resets when switching track/overlay panels. - `raga_pipeline/output.py`
- Added an in-report transcription editor with undo/redo, point/range-aware note selection, add-note from drag range (sargam + octave -> MIDI preview), note delete/range delete, phrase merge/split, snapped phrase-bound edits, note resizing, manual versioned saves, and auto-load of latest saved edit state. - `raga_pipeline/output.py`
- Added local app transcription-edit APIs with version history, latest/version fetch, and save-time generation of versioned edited reports plus JSON/CSV artifacts. - `local_app/server.py`, `local_app/schemas.py`
- Added analysis-report metadata sidecar generation and report regeneration hooks so edited transcription versions can be re-rendered with updated phrase/note overlays and persisted as separate HTML versions. - `raga_pipeline/output.py`, `local_app/server.py`
- Removed the transcription-editor phrase-summary preview strip (`P1: ... | P2: ...`) from the report UI to keep only the phrase-note table view. - `raga_pipeline/output.py`
- Updated transcription-editor version workflow so `Save` now updates the selected/current version in-place (accumulating edits), added explicit `Save as New Version`, added selector option `Create new version...`, and added `Delete Version` with backend file cleanup. - `raga_pipeline/output.py`, `local_app/server.py`, `local_app/schemas.py`
- Added default transcription selection (`original` or saved version), wired `/local-report/.../analysis_report.html` to serve the selected default report variant, and exposed default selection metadata in transcription-edit API responses. - `local_app/server.py`, `local_app/schemas.py`
- Added explicit regenerate endpoint for saved edit versions and editor toolbar actions to regenerate selected version reports, set selected/original as default, and auto-load the configured default transcription in the editor. - `local_app/server.py`, `raga_pipeline/output.py`

### Testing

- Added regression coverage for raga-correction rounding behavior to ensure microtonal accepted notes are emitted on snapped pitch and corrected notes refresh stale sargam labels. - `tests/test_raga_correction_rounding.py`
- Added transcription-energy regression tests for inflection energy sampling and threshold behavior. - `tests/test_transcription_energy_gating.py`
- Added schema/config default coverage for updated analyze phrase thresholds. - `tests/test_cli_schema_args.py`
- Added HTML-generation tests for scroll-plot inspector elements, note payload serialization, and pointer-selection event hooks. - `tests/test_output_scroll_inspector.py`
- Added regression tests for consecutive-note merge behavior (dropout-healing and phrase-level collapse behavior). - `tests/test_sequence_merge_consecutive.py`
- Extended inspector HTML test coverage for the new scrollable note table structure. - `tests/test_output_scroll_inspector.py`
- Added local app API tests for transcription edit save/list/latest/version flows, including versioned report artifact generation checks. - `tests/test_local_app.py`
- Added HTML-generation coverage for the transcription editor controls, selection hooks, and save/load JS wiring. - `tests/test_output_transcription_editor.py`
- Extended transcription-edit tests to cover in-place version updates (`target_version_id`), explicit new-version creation (`create_new_version`), and version deletion endpoint behavior. - `tests/test_local_app.py`, `tests/test_output_transcription_editor.py`
- Extended transcription-edit test coverage for default-selection API behavior, source-report default routing, and per-version regenerate endpoint behavior, plus editor HTML checks for new default/regenerate controls. - `tests/test_local_app.py`, `tests/test_output_transcription_editor.py`

### Documentation

- Updated analyze option docs and LLM reference for per-track energy-threshold semantics, single-pass transcription energy gating, and new phrase defaults. - `README.md`, `raga_pipeline/LLM_REFERENCE.md`

## 2026-02-21

### Refactor

- Removed redundant CLI arg helpers and unused imports in schema/analysis utilities to simplify argv generation and peak detection internals. - `raga_pipeline/cli_args.py`, `raga_pipeline/analysis.py`
- Consolidated repeated array-alignment and local-import patterns in audio processing helpers. - `raga_pipeline/audio.py`
- Streamlined advanced phrase clustering internals by removing unused label/index extraction paths and dead temporary variables. - `raga_pipeline/sequence.py`
- Refactored batch command construction with shared metadata-arg handling and deterministic task ordering. - `raga_pipeline/batch.py`
- Refactored `run_pipeline.sh` into an environment-configurable wrapper suitable for HPC and non-Conda launches. - `run_pipeline.sh`
- Refactored batch runner to invoke `driver.py` via current Python interpreter instead of shelling out to `run_pipeline.sh`. - `raga_pipeline/batch.py`

### Bugfix

- Fixed config fallback defaults so `bias_rotation` stays enabled unless explicitly disabled and invalid modes fail early. - `raga_pipeline/config.py`
- Preserved `PitchData.energy` through `smooth_pitch_contour` output to avoid dropping note-energy context in legacy note-detection paths. - `raga_pipeline/sequence.py`
- Preserved note metadata (`energy`, `sargam`) during raga-note correction rewrites. - `raga_pipeline/raga.py`
- Honored `transcription_min_duration` alias when provided and cleaned tonic-resolution logic for mixed tonic input formats. - `raga_pipeline/transcription.py`
- Added pipeline preflight dependency checks for HPC runs (ffmpeg/ffprobe/SwiftF0/separator modules/raga DB path) with fail-fast diagnostics. - `driver.py`
- Accepted browser recorder `.webm/.weba` uploads in local app audio upload endpoint to prevent preprocess recording ingest failures. - `local_app/server.py`, `tests/test_local_app.py`
- Added dedicated in-app tanpura dropdown with preview play/stop controls for preprocess recording, synchronized with `--tanpura-key`. - `local_app/static/app.js`, `local_app/static/style.css`
- Prevented broken detection-report stem players when detect is run without separation by only rendering existing audio tracks. - `raga_pipeline/output.py`
- Added analyze-mode cache fallback to `composite_pitch_data.csv` when stem pitch cache is missing (skip-separation artifact compatibility). - `driver.py`
- Fixed transcription/export mismatch where stable stationary events could be dropped downstream by using snapped pitch as canonical note pitch in `transcribe_to_notes`, and aligned scroll-plot stationary overlays with transcription energy-threshold filtering. - `raga_pipeline/transcription.py`, `raga_pipeline/output.py`, `tests/test_transcription_stationary.py`

### Feature

- Added resumable batch checkpointing (`--progress-file`), chunked processing (`--max-files`), and scheduler-loop signaling via `--exit-99-on-remaining`. - `raga_pipeline/batch.py`
- Added PBS-ready HPC wrapper template and usage notes for cluster execution. - `hpc/pipeline_batch.pbs`, `hpc/README.md`
- Added preprocess recording ingest (`--ingest record`) with song/tanpura-vocal modes, tanpura tonic pass-through into suggested detect commands, and macOS-first interactive CLI mic recording with optional ffplay tanpura loop. - `raga_pipeline/config.py`, `raga_pipeline/audio.py`, `driver.py`
- Added local app tanpura track API and preprocess browser recording controls (MediaRecorder + tanpura playback) that upload into `--recorded-audio`. - `local_app/server.py`, `local_app/static/app.js`, `local_app/static/style.css`
- Added tanpura registry/path helpers and canonical tonic mapping utilities for preprocess workflows and local app tanpura catalog responses. - `raga_pipeline/audio.py`, `local_app/server.py`
- Implemented detect-wide skip-separation mode: `--skip-separation` now skips stem separation, requires `--tonic`, and auto-forces `melody_source=composite`. - `raga_pipeline/config.py`, `driver.py`, `local_app/static/app.js`
- Tanpura-vocal preprocess now auto-suggests detect with `--skip-separation` alongside tonic in both CLI and local app Next flow. - `driver.py`, `local_app/static/app.js`, `tests/test_driver_preprocess.py`
- Demucs auto-device selection now prefers Apple Metal (`mps`) when CUDA is unavailable, with MPS CPU fallback enabled by default. - `raga_pipeline/audio.py`
- Added SwiftF0 provider runtime controls wired via env vars (`RAGA_SWIFTF0_PROVIDER`, `RAGA_SWIFTF0_STRICT_PROVIDER`, `RAGA_SWIFTF0_PROVIDER_LOGS`) with fork-style constructor kwargs and legacy-constructor fallback compatibility. - `raga_pipeline/audio.py`
- Added SwiftF0 validation helper scripts for provider speed benchmarking and cross-provider parity checks. - `tools/swiftf0_provider_benchmark.py`, `tools/swiftf0_parity_check.py`
- Added upstream contribution tracking checklist and submission outline for SwiftF0 CoreML/provider support PR workflow. - `SWIFTF0_UPSTREAM_PR_TRACK.md`

### Documentation

- Updated CLI docs for HPC-friendly wrapper env vars and new batch flags. - `README.md`, `raga_pipeline/DOCUMENTATION.md`, `raga_pipeline/LLM_REFERENCE.md`
- Documented new preprocess ingest/record/tanpura workflow and CLI validation behavior. - `README.md`, `raga_pipeline/DOCUMENTATION.md`, `raga_pipeline/LLM_REFERENCE.md`
- Documented skip-separation detect semantics, tonic requirement, composite fallback behavior, and updated local-app recording path notes. - `README.md`, `raga_pipeline/DOCUMENTATION.md`, `raga_pipeline/LLM_REFERENCE.md`
- Clarified skip-separation help text to indicate denoising requires leaving skip-separation unchecked in UI. - `raga_pipeline/config.py`
- Documented SwiftF0 provider env controls, fork-pin install workflow, and benchmark/parity command examples. - `README.md`, `raga_pipeline/DOCUMENTATION.md`, `raga_pipeline/LLM_REFERENCE.md`, `requirements.txt`, `requirements-swiftf0-fork.txt`
- Wired `requirements-swiftf0-fork.txt` to the user fork URL (`saumyamishra654/swift-f0`) for direct git-based SwiftF0 installs. - `requirements-swiftf0-fork.txt`

### Testing

- Added preprocess schema/config validation coverage, preprocess command-generation tests, and local app tanpura endpoint + preprocess-record payload tests. - `tests/test_cli_schema_args.py`, `tests/test_driver_preprocess.py`, `tests/test_local_app.py`
- Added detect skip-separation config validation tests, detect skip-path driver tests, and local app schema coverage for `skip_separation`. - `tests/test_cli_schema_args.py`, `tests/test_driver_detect_skip_separation.py`, `tests/test_local_app.py`
- Added unit tests for SwiftF0 provider env parsing, fork-style constructor kwargs, and legacy constructor fallback behavior. - `tests/test_swiftf0_provider_config.py`

## 2026-02-16

### Feature

- Analyze report sargam display now uses a recording-relative median Sa anchor and suppresses octave suffixes by default; octave markers appear only for notes that fall 3+ octaves below the recording anchor. Applied to phrase transcription and scroll-plot legend labels. - `raga_pipeline/output.py`
- Reintroduced the top Phrase Karaoke section in analyze reports with cumulative note highlighting in the phrase rows (sung notes stay lit), and removed the horizontal ticker strip to keep the UI focused on phrase-wise transcription. - `raga_pipeline/output.py`
- Added analyze-mode note-duration histogram generation (`note_duration_histogram.png`) from merged transcribed notes and surfaced it in the analysis report visualizations section. - `driver.py`, `raga_pipeline/output.py`

### Bugfix

- Reduced click-seek lag in scrollable pitch timeline by using explicit plotted x-axis bounds for time mapping, clamped margin-aware x<->time conversion, and immediate cursor/scroll realignment on click/seek events. - `raga_pipeline/output.py`
- Fixed non-moving static pitch-analysis playhead by switching to robust active-audio tracking plus event-driven cursor updates (`play`, `timeupdate`, `seeked`, `loadedmetadata`) with rAF fallback. - `raga_pipeline/output.py`
- Fixed post-seek playhead freeze by prioritizing the currently playing audio element after seek (instead of latching onto paused tracks that also emit `seeked`) and adding safer active-audio reassignment on pause/end. - `raga_pipeline/output.py`
- Reduced seek-time lag regressions from karaoke cumulative highlighting by chunking sung-note class updates over animation frames and using instant phrase-list scroll alignment on `seeked` events. - `raga_pipeline/output.py`
- Further reduced accumulated seek latency by coalescing karaoke sync into a single `requestAnimationFrame` scheduler (`latest update wins`) and lowering per-frame cumulative highlight work size. - `raga_pipeline/output.py`
- Optimized local report asset URL rewriting to short-circuit large embedded `data:` URIs before URL parsing, preventing analyze-report load instability/perf spikes in local app. - `local_app/server.py`

## 2026-02-15

### Feature

- Added analyze-report playback speed controls (`1x`, `0.5x`, `0.25x`) that apply to Original, Vocals, and Accompaniment players for easier transcription verification. - `raga_pipeline/output.py`
- Added configurable phrase exclusion thresholds in analyze mode: `--phrase-min-duration` (default `1.0s`) and `--phrase-min-notes` (default `0`). - `raga_pipeline/config.py`, `driver.py`
- Enforced a final phrase-filter pass after optional silence-based splitting so short/sparse phrases are excluded from downstream clustering/report sections. - `driver.py`
- Enforced mutually-exclusive audio playback in analyze reports so starting one track pauses the others. - `raga_pipeline/output.py`
- Enforced mutually-exclusive audio playback in detection reports so starting one track pauses the others. - `raga_pipeline/output.py`
- Added parser refactor and new reusable config APIs (`build_cli_parser`, `parse_config_from_argv`) so CLI and local UI share one argument source of truth. - `raga_pipeline/config.py`, `raga_pipeline/__init__.py`
- Added argparse-driven schema and argv utilities for app integration (`get_mode_schema`, `params_to_argv`). - `raga_pipeline/cli_schema.py`, `raga_pipeline/cli_args.py`
- Added local FastAPI app scaffolding with serial job queue, cancellation handling, logs/progress parsing, artifact discovery, and report embedding support. - `local_app/server.py`, `local_app/jobs.py`, `local_app/schemas.py`, `local_app/templates/index.html`, `local_app/static/app.js`, `local_app/static/style.css`
- Added local app launcher script (`run_local_app.sh`) and runtime dependencies (`fastapi`, `uvicorn`, `jinja2`). - `run_local_app.sh`, `requirements.txt`
- Local app now parses the printed next-step commands from pipeline logs and auto-loads/focuses the next mode form (`preprocess -> detect`, `detect -> analyze`). - `local_app/static/app.js`
- Local app optional fields now support explicit blank state (use parser defaults), with conditional visibility for dependent fields (`vocalist_gender`/`instrument_type` by `source_type`). - `local_app/static/app.js`
- Added drag-and-drop audio upload support in local app detect/analyze forms, backed by a new upload endpoint that stores files in local app data and auto-fills `--audio`. - `local_app/server.py`, `local_app/static/app.js`, `local_app/static/style.css`, `tests/test_local_app.py`, `requirements.txt`

### Documentation

- Documented new analyze options for phrase filtering in CLI docs. - `README.md`
- Updated LLM reference defaults and report capabilities (phrase filtering defaults and playback-speed controls). - `raga_pipeline/LLM_REFERENCE.md`
- Documented local app usage and parser-sync architecture. - `README.md`, `raga_pipeline/DOCUMENTATION.md`, `raga_pipeline/LLM_REFERENCE.md`

### Testing

- Added unit/integration-style tests for parser/schema/argv conversion and local app job/API behavior. - `tests/test_cli_schema_args.py`, `tests/test_local_app.py`

## 2026-02-14

### Documentation

- Updated CLI docs to match current argparse surface area (subcommands, flags, and defaults like `--output batch_results`). - `README.md`, `raga_pipeline/DOCUMENTATION.md`
- Clarified detect/analyze output directory layout and report filenames (`detection_report.html`, `analysis_report.html`) and batch runner behavior/logging. - `README.md`, `raga_pipeline/DOCUMENTATION.md`
- Fixed `LLM_REFERENCE.md` defaults that drifted from the code (`bias_rotation`, pitch range notes) and bumped the last-updated date. - `raga_pipeline/LLM_REFERENCE.md`

### Bugfix

- URL-encoded HTML audio source paths so filenames/paths with spaces load correctly in report audio players (Original/Vocals/Accompaniment sources). - `raga_pipeline/output.py`
- Added explicit HTML MIME mapping for `.m4a`/`.mp4`/`.aac` audio sources to improve browser loading in generated reports. - `raga_pipeline/output.py`
- Trimmed leading/trailing whitespace from CLI path-like inputs (`--audio`, `--audio-dir`, `--filename`) before path validation to prevent false `FileNotFoundError` when accidental spaces are passed in shell arguments. - `raga_pipeline/config.py`

## 2026-02-13

### Feature

- Updated detect-mode stationary-note histogram output to octave-wrapped western pitch classes (12 fixed bins: C to B), and CSV export now writes 12 pitch-class columns. - `driver.py`, `output.py`
- Added a second detect-mode stationary-note histogram weighted by note duration, with 12-column octave-wrapped CSV export of per-pitch-class total seconds. - `driver.py`, `output.py`
- Removed detect-mode unweighted stationary-note histogram from generation/report; duration-weighted histogram is now the only stationary-note graph shown. - `driver.py`, `output.py`
- Added standalone `preprocess` mode for YouTube ingest (`--yt --audio-dir --filename`) with MP3 download, fail-if-exists behavior, and copyable next-step `detect` command output. - `config.py`, `audio.py`, `driver.py`, `requirements.txt`
- Set `preprocess` default `--audio-dir` to `../audio_test_files` so only `--yt` and `--filename` are required for common usage. - `config.py`
- Hardened YouTube preprocess downloads with multi-strategy yt-dlp fallbacks (client/protocol variations) and cleaner user-facing failure handling without traceback spam. - `audio.py`, `driver.py`
- Added preprocess trim flags `--start-time` and `--end-time` with validation (`start < end`, both <= track duration) and full-track defaults when omitted. - `config.py`, `audio.py`, `driver.py`
- Fixed preprocess trim flow to ensure post-download trimming always executes (removed early return before trim validation). - `audio.py`

### Bugfix

- Fixed Optional audio-path typing issues in report generation (`output.py`) by explicitly requiring concrete audio paths before `relpath`/`basename` calls. - `output.py`

### Documentation

- Updated docs/tracker wording for octave-wrapped stationary-note histogram behavior. - `README.md`, `LLM_REFERENCE.md`, `TODO.md`
- Documented preprocess workflow and three-mode pipeline behavior. - `README.md`, `LLM_REFERENCE.md`, `TODO.md`
- Updated preprocess usage docs with the new default audio directory. - `README.md`, `TODO.md`
- Documented preprocess trimming flags and validation behavior. - `README.md`, `LLM_REFERENCE.md`, `TODO.md`

## 2026-02-12

### Feature

- Added portable audio source fallback in generated HTML reports so audio players try the local folder copy first and then the existing relative path fallback; this makes zipped report bundles open more reliably on other machines. - `output.py`

### Documentation

- Updated project task tracker with completed portable report-audio fallback item. - `TODO.md`

## 2026-02-11

### Planning

- Created comprehensive web application implementation plan covering architecture, tech stack (FastAPI, React, React Native/Expo, Supabase, Cloudflare R2), infrastructure (RunPod GPU workers), API design, frontend/mobile development, security, cost estimates (~$330/month MVP), and 3-4 month timeline. - `WEB_APP_IMPLEMENTATION_PLAN.md`, `TODO.md`
- Created focused thesis demo implementation plan (FastAPI + React) with 8-week roadmap, Railway + Vercel deployment, sample library strategy, and $10-15/month cost estimate for undergraduate class demonstration. - `THESIS_DEMO_PLAN.md`, `TODO.md`
- Created comprehensive pitch matching game implementation plan for future project: interactive music learning tool with real-time pitch tracking (<50ms latency), DTW-based multi-dimensional scoring (pitch, timing, gamakas, duration, contour), Guitar Hero-style scrolling notation, post-analysis feedback system. Leverages existing pipeline components (Demucs, SwiftF0) with Web Audio API + React + PixiJS frontend. 12-17 week timeline, $10-20/month hosting. Universal application for any musical tradition. - `PITCH_MATCHING_GAME_PLAN.md`, `TODO.md`

### Feature

- Added aaroh/avroh directional pattern database utilities: subset builder aligned to `raga_list_final.csv`, pattern loader, and raga name resolver. - `raga.py`
- Added aaroh/avroh conformance checker based on directional incoming-edge evidence from detected phrases. - `sequence.py`
- Integrated checker into analyze-mode pattern analysis and console summary output (no optional flag; runs when pattern DB entry exists). - `driver.py`, `sequence.py`
- Added analysis report section for aaroh/avroh checker (reference pattern, score, mismatches, per-note directional evidence). - `output.py`
- Created aligned subset file for aaroh/avroh patterns used by the checker. - `raga_pipeline/data/aarohavroha_subset.csv`
- Added detect-mode tonic/raga constraints with comma-separated tonic support to limit candidate scoring. - `config.py`, `raga.py`, `driver.py`
- Added `--bias-rotation` flag that shifts all histogram peaks by the median deviation from ideal semitone positions, compensating for non-standard tuning systems and recording pitch offsets. - `config.py`, `raga.py`, `driver.py`

### Bugfix

- Fixed mypy type errors across pipeline modules (`driver.py`, `batch.py`, `sequence.py`, `analysis.py`, `transcription.py`, `raga.py`, `output.py`) including optional-path narrowing, list/ndarray inference, union narrowing for track specs, and note reconstruction typing. - `driver.py`, `raga_pipeline/*.py`
- Fixed analyze-mode tonic parsing by removing a shadowing import that caused `UnboundLocalError`. - `driver.py`
- Skipped hidden/AppleDouble files (e.g. `._*.mp3`) in batch processing to avoid ffprobe failures. - `raga_pipeline/batch.py`

### Tooling

- Added `mypy.ini` to standardize local type checking and ignore missing third-party stubs while enforcing internal module typing. - `mypy.ini`

### Bugfix

- Updated dataclass `field(default_factory=...)` usage to explicit typed helper callables to satisfy stricter Pyright/Pylance checks in `analysis.py` and `output.py`. - `analysis.py`, `output.py`

## 2026-02-09

- Added energy-over-time plot beneath static pitch analysis in analyze reports. - `output.py`
- Batch ground-truth CSV now defaults to `<input_dir>_gt.csv` stored alongside the input directory to match batch invocation targets. - `raga_pipeline/batch.py`
- Save Demucs/Spleeter stems as MP3 (with legacy WAV fallback and conversion). - `audio.py`, `config.py`

### Bugfix

- Limit bias rotation to melody scoring/plots and leave accompaniment histograms unrotated. - `raga.py`, `driver.py`
- Auto-select `source_type=vocal` when `--vocalist-gender` is provided. - `config.py`

## 2026-02-10

### Feature

- Added optional RMS energy overlay on pitch plots (Plotly and scrollable analysis plot). - `output.py`, `config.py`
- Added silence-based phrase splitting using RMS thresholds and minimum silence duration (configurable via CLI). - `sequence.py`, `driver.py`, `config.py`
- Added karaoke-style scrolling transcription section to analysis reports. - `output.py`
- Added `--energy-metric rms|log_amp` CLI option. `log_amp` computes 20*log10(RMS) dBFS with percentile normalisation (p5-p95 mapped to 0-1), giving better dynamic range representation for quiet passages. - `audio.py`, `config.py`, `driver.py`
- Defaulted analyze-mode silence threshold to 0.10 to match typical phrase-splitting usage. - `config.py`

### Documentation

- Updated README flags for energy overlay, silence splitting, and energy metric options. - `README.md`
- Updated LLM_REFERENCE.md: corrected date to 2026-02-10, added energy_metric parameter to extract_pitch signature, fixed output filenames (detection_report.html), corrected note_min_duration default (0.1), updated batch usage command to python -m syntax, and clarified cache file names. - `LLM_REFERENCE.md`

### Bugfix

- Improved scrollable plot sync reliability by binding to audio timeupdate events. - `output.py`
- Filtered karaoke transcription to exclude unknown/non-standard sargam labels. - `output.py`
- Default silence-based phrase splitting to transcription energy threshold when unset. - `driver.py`
- Linked analysis report audio tracks with scroll plot and karaoke using a shared sync group and smooth cursor updates. - `output.py`
- Added analyze-mode cache fallback to accept legacy `vocals_pitch_data.csv` when `melody_pitch_data.csv` is absent. - `driver.py`
- Fixed analyze-mode cache validation indentation to prevent `IndentationError`. - `driver.py`

### Feature

- Updated transcription snapping to always snap to the nearest chromatic target; in raga mode, fall back to the second-closest in-raga target or drop the note. - `transcription.py`

### Planning

- Created comprehensive planning document for aaroh/avroh reconstruction algorithm that analyzes directional note context to infer ascending/descending scale structures. - `aaroh-avroh-reconstruction.md`, `TODO.md`
- Created comprehensive planning document for sequence mining system to discover characteristic melodic patterns (pakads, motifs) across recordings, with special focus on discriminating ragas with identical aaroh/avroh patterns. Uses overcomplete motif dictionary with entropy-based weighting for educational value and robustness. Includes multi-resolution pattern representation, two-stage matching (n-gram filtering + DTW refinement), three aggregation methods (weighted voting, TF-IDF, probabilistic), and dissimilarity constraints for confused raga clusters. - `sequence-mining-plan.md`, `TODO.md`

## 2026-01-28

### Feature: Advanced Instrumental Mode
- **Always-on Composite Pitch**: Extracting pitch from original mix now happens by default (`composite_pitch_data.csv`). - `driver.py`
- **Decoupled Melody Source**: Added `--melody-source` flag for switching between separated stems and original mix. - `config.py`, `driver.py`
- **Forced Separation for Accompaniment**: Instrumental mode now always runs separation to facilitate high-quality tonic detection from the drone. - `driver.py`
- **Exposed Core Parameters**: Exposed `--fmin-note`, `--fmax-note`, `--prominence-high`, and `--prominence-low` to the CLI. - `config.py`
- **Updated Transcription Defaults**: Set new defaults optimized for raw analysis: `energy_threshold=0.0`, `min_duration=0.02s`, `stability_threshold=4.0`, and `no-smoothing` (smoothing=0ms) by default. - `config.py`

### Bugfix
- **Peak detection discrepancy**: Fixed major discrepancy in `solost.mp3` by aligning `histogram_bins_low` (33), extending `fmax`, and tuning prominence. - `config.py`, `analysis.py`
- **Raga Database Cleanup**: Stripped smart quotes and brackets from `raga_list_final.csv` to fix messy report names. - `main notebooks/raga_list_final.csv`

### Feature: Enhanced CLI Usability
- **CLI Defaults**: Defaulted `--output` to `results` and made it optional. - `config.py`
- **Dynamic Cache Pathing**: `analyze` mode now correctly resolves prefixed pitch files (e.g. `melody_` vs `vocals_`). - `driver.py`
- **Peak Merging**: Added logic to `detect_peaks` to merge multiple peaks mapping to the same pitch class, retaining only the best candidate. - `analysis.py`

### Feature: Batch Processing
- **Batch Script**: Added `raga_pipeline/batch.py` to process directory of audio files. Supports automatic `analyze`/`detect` mode switching based on ground truth CSV. - `raga_pipeline/batch.py`

### Bugfix
- **Spleeter Path Bug**: Fixed incorrect stem directory lookup for Spleeter engine. - `audio.py`
- **Pandas FutureWarnings**: Switched to `.iloc` for Series positional indexing. - `raga.py`
- **Docstring Maintenance**: Fixed misleading comments in `driver.py`.

---

## 2026-01-25

### Feature: Unified Transcription & RMS Energy
- **Unified Transcription Method**: Implemented a hybrid transcription approach combining **Stationary Points** (for stable notes) and **Inflection Points** (for fast runs/transients). This significantly improves detection of ornaments (murki/tan). - `transcription.py`
- **RMS Energy Integration**: Added per-event RMS energy calculation using `librosa.feature.rms`.
- **Note Gating/Filtering**: Integrated energy-based filtering in `driver.py` (controlled via `--energy-threshold`) to automatically remove silent segments, breaths, and background noise from the transcription.
- **CSV Data Extension**: Updated `transcribed_notes.csv` to include an `energy` column for downstream analysis. - `output.py`

### Feature: Technical Primer & Case Study
- **Technical Primer (Case Study)**: Created a multi-page interactive technical primer in the `primer/` directory. It explains the system architecture, audio/pitch extraction, analysis logic, scoring engine, and transcription methods for Python developers.
- **Backend Migration Plan**: Formulated a detailed technical strategy for transitioning the pipeline into a FastAPI/Celery-based production backend.

### Feature: Advanced Visualization & Multi-Track Sync
- **Consolidated Multi-Track Sync**: Both the interactive **Plotly Graph** and the **Scrollable Pitch Plot** now synchronize with all three audio tracks: Original, Vocals, and Accompaniment. - `output.py`
- **Bidirectional Navigation**: 
    - **Audio -> Plot**: Playhead movement in any player updates both cursors.
    - **Plot -> Audio**: Clicking on either plot seeks all three audio tracks simultaneously to stay in sync.
- **Energy Distribution Histogram**: Added a dedicated visualization for note energy distribution in the report, helping users calibrate the detection threshold. - `output.py`

### Bugfix
- **AttributeError**: Resolved crash in report generation due to missing `duration` property in `PitchData` - `audio.py`
- **Plot Injection**: Fixed issue where the interactive Plotly player would occasionally fail to render due to missing container wrappers.
- **Param Conflict**: Fixed duplicate `derivative_threshold` argument bug in `driver.py`.

---

## 2026-01-24

### Feature: Advanced Phrase Clustering
- Integrated `cluster_notes_into_phrases` incorporating KMeans and DBSCAN clustering methods - `sequence.py`
- Refactored `detect_phrases` to utilize the new clustering logic - `sequence.py`
- Added robust thresholding logic (IQR-based) for phrase break identification - `sequence.py`

### Feature: Interactive Visualization
- Implemented synchronized, scrollable pitch contour visualization in HTML reports - `output.py`
- Added dynamic Y-axis scaling to prevent vocal range clipping - `output.py`
- Implemented linear mapping with margin correction for precise audio-visual synchronization - `output.py`
- Added configurable X-axis time markers (set to 2-second intervals) - `output.py`
- Integrated interactive "click-to-seek" functionality in the pitch plot - `output.py`

### Bugfix: Sync Offset
- Resolved ~2 second constant synchronization lag by accounting for Matplotlib figure margins in the JS tracking engine - `output.py`

---

## 2026-01-17

### Documentation
- Created `LLM_REFERENCE.md` - comprehensive quick reference for LLM agents with complete system architecture, module breakdowns, data structures, and troubleshooting guide
- Created `NOTEBOOK_VS_PIPELINE_COMPARISON.md` - detailed audit comparing `ssje_tweaked_wit_peaks.ipynb` and `note_sequence_playground.ipynb` with Python pipeline implementation
- Created `.gitignore` - excludes Python cache, output directories, intermediate files (stems, pitch CSVs), generated plots and reports
- Created `CHANGELOG.md` (this file) - daily work record for project history
- Updated `.gitignore` to include documentation files (`LLM_REFERENCE.md`, `CHANGELOG.md`, `NOTEBOOK_VS_PIPELINE_COMPARISON.md`), ignore `.pkl` model files, and refine CSV ignore rules to protect repository data.

### Identified Gaps (from comparison audit)
- Pipeline is 95% complete for histogram-based raga detection
- Pipeline is ~60% complete for sequence analysis
- Missing features identified:
  - Aaroh/Avroh pattern extraction
  - Melodic sequence n-gram analysis
  - Common pattern frequency analysis
  - Octave range filtering
  - Instrument mode tonic filtering

### Feature (Phase 1): Configuration Extension
- Added `source_type` field: "mixed", "instrumental", "vocal" - `config.py`
- Added `vocalist_gender` field: "male", "female" for tonic bias - `config.py`
- Added `instrument_type` field: "sitar", "sarod", "bansuri", "slide_guitar" - `config.py`
- Removed `full` mode, only `detect` and `analyze` supported - `config.py`
- Disabled ML model (`use_ml_model=False`) - `config.py`
- Updated CLI arguments with new source type options - `config.py`

### Feature (Phase 2): Tonic Bias Constants
- Added `TONIC_BIAS` dictionary with instrument/gender-specific tonic ranges - `raga.py`
- Added `get_tonic_candidates()` function for filtering candidates - `raga.py`
- Female vocalists: bias toward A-A# (pitch class 7-0)
- Male vocalists: bias toward D-D# (pitch class 11-5)
- Sitar: C#-D#, Sarod: A#-D, Bansuri: D-F, Slide guitar: C#-D#

### Feature (Phase 3): Stem Separation Skip
- Added `load_audio_direct()` function for instrumental/vocal sources - `audio.py`
- Updated `driver.py` to skip stem separation based on source_type
- Conditional accompaniment pitch extraction and histogram plotting
- Uses local `stem_dir` variable for correct output paths

### Feature (Phase 4): Scoring Without Accompaniment
- Updated `score_candidates_full()` to accept Optional accompaniment - `raga.py`
- Added `tonic_candidates` parameter for explicit tonic bias
- Tonic salience filtering only applies when accompaniment is available
- Falls back to all tonics or explicit tonic candidates when no accompaniment

### Feature (Phases 5-8): Pattern Analysis
- Aaroh/avroh extraction already exists in `build_transition_matrix_corrected()` - `sequence.py`
- `extract_melodic_sequences()` and `find_common_patterns()` already implemented - `sequence.py`
- Added `filter_notes_by_octave_range()` for pitch detection error filtering - `sequence.py`
- Pattern analysis module not needed (functionality already in sequence.py and driver.py)

### Feature (Phase 9): Driver Integration
- Updated `driver.py` to use `source_type` instead of `instrument_mode`
- Full pipeline works with detect/analyze modes (full mode removed)
- Pattern analysis used in analyze mode via existing functions

### Feature (Phase 10): Documentation
- Updated `LLM_REFERENCE.md` with new config fields, TONIC_BIAS, load_audio_direct, run_pipeline.sh
- Added directive to always use `./run_pipeline.sh` for correct conda environment

- Added directive to always use `./run_pipeline.sh` for correct conda environment

### Feature (Update): Mixed Mode Refinement
- Added `--skip-separation` flag for instrumental mode
- Fixed: Vocal mode now runs stem separation + gender bias (does not skip separation)
- Fixed: Tonic candidates bias was calculated but not passed to scorer - fixed in `driver.py` and `raga.py`
- Updated driver to suggest next commands using preserved input paths

### Verification
- Tested sitar instrumental: tonic candidates [1,2,3] -> detected Devshri/G#
- Tested female vocal: tonic candidates [7,8,9,10,11,0] -> detected Kalavati/D
- Tested male vocal: tonic candidates [11,0,1,2,3,4,5] -> detected Bhairavi/C#
- Verified optional skip separation behavior for instrumental
- Verified mixed (default) mode still works correctly with accompaniment

### Feature (Phase 11): Advanced Pattern Analysis
- Added `extract_aaroh_avroh_runs()`: Identifies long ascending/descending runs (>3 notes) - `sequence.py`
- Added `analyze_raga_patterns()`: Aggregates motifs and run stats - `sequence.py`
- Integrated pattern analysis into `driver.py` analyze mode and `output.py` HTML report.
- Verified on test files: successfully extracts Aaroh/Avroh runs and common motifs.

### Bugfix: Spleeter Output Paths
- Fixed `stem_dir` in `config.py` to use `spleeter/` subdirectory when using Spleeter.
- Fixed `_separate_spleeter` in `audio.py` to correctly locate Spleeter's output files.
- Spleeter and Demucs results now go to separate directories for comparison.

---

## Format Guidelines

Each entry should follow this format:

```markdown
## YYYY-MM-DD

### Category (e.g., Feature, Bugfix, Documentation, Refactor)
- Brief description of change
- File(s) affected: `filename.py`
- Related issue or reason for change

### Another Category
- Another change
```

### Categories to Use
- **Feature** - New functionality
- **Bugfix** - Bug fixes
- **Documentation** - Doc updates
- **Refactor** - Code restructuring without behavior change
- **Performance** - Optimizations
- **Testing** - Test additions/changes
- **Config** - Configuration changes
- **Breaking** - Breaking changes (highlight these prominently)

---
