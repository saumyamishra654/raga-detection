# LM-Enhanced Detect Mode -- Design Spec

## Overview

Add an optional n-gram language model re-ranking stage to the existing detect pipeline. When `--use-lm-scoring` is enabled, detect runs a single chromatic transcription after histogram scoring, then re-tokenizes under each surviving tonic hypothesis and scores all (tonic, raga) candidates against the trained LM. The LM score replaces the histogram score as the primary ranking.

This removes the circularity in the current evaluation setup: transcription is chromatic (no raga-aware correction), and the LM selects the raga rather than being told the raga in advance.

## Goals

- Integrate LM scoring as an optional second stage of detect (not a separate pipeline)
- Zero raga bias in transcription: chromatic snapping only, no raga correction
- Exhaustive scoring across all raga models for each tonic candidate from detect
- LM score becomes the primary ranking; histogram score preserved as metadata
- Minimal overhead: the expensive work (separation, pitch extraction) is already done by detect

## Non-goals

- Replacing the histogram scorer (it remains the primary scorer when LM is not enabled)
- Training the LM (handled by existing `python -m raga_pipeline.language_model train`)
- Multi-hypothesis transcription (not needed -- chromatic transcription is tonic-independent; only tokenization changes per tonic)

---

## Architecture

### Where it fits in the detect pipeline

```
[Steps 1-5: existing detect -- unchanged]
    Stem separation -> pitch extraction -> histograms -> peaks -> candidate scoring
    -> candidates.csv with histogram-ranked (tonic, raga) pairs
    -> surviving tonic set (typically 4-6 tonics)

[Step 5.5: LM re-ranking -- NEW, only when --use-lm-scoring]
    -> Load NgramModel from --lm-model JSON
    -> Run chromatic transcription on melody pitch data (already in memory)
       using transcribe_to_notes with snap_mode='chromatic', no raga correction
    -> Extract unique tonics from candidate list
    -> For each tonic: tokenize transcription via tokenize_notes_for_lm(notes, tonic_midi)
    -> For each (tonic, raga) pair in candidates:
         score = model.score_sequence(raga, tokens_for_tonic)
    -> Re-rank candidates by LM score descending
    -> Update candidates DataFrame with lm_score and lm_rank columns

[Steps 6-7: existing report generation -- uses re-ranked candidates]
```

### Key insight: one transcription, many scorings

The chromatic transcription produces Note objects with fixed MIDI pitches. The tonic only affects tokenization (which MIDI pitch maps to which sargam token and octave marker). So:

1. **One transcription** (~20-30 seconds) -- the only significant new compute
2. **One tokenization per unique tonic** (~milliseconds each, typically 4-6 tonics)
3. **One LM scoring per (tonic, raga) pair** (~milliseconds each, typically 200-500 pairs)

Total LM overhead: ~20-30 seconds for transcription + negligible time for scoring.

---

## CLI Interface

```bash
python driver.py detect \
    --audio song.mp3 \
    --use-lm-scoring \
    --lm-model compmusic_ngram_model.json
```

New flags (detect mode only):

| Flag | Default | Description |
|------|---------|-------------|
| `--use-lm-scoring` | `False` | Enable LM re-ranking after histogram scoring |
| `--lm-model` | None | Path to trained n-gram model JSON (required when `--use-lm-scoring` is set) |

Validation: if `--use-lm-scoring` is set, `--lm-model` must be provided and the file must exist.

---

## Output Changes

### candidates.csv

When LM scoring is active, add columns:

| Column | Description |
|--------|-------------|
| `lm_score` | Average log-likelihood per token under the raga's LM |
| `lm_rank` | Rank by LM score (1 = best) |

Rows are sorted by `lm_score` descending (LM ranking is primary). The existing `score` column (histogram score) is preserved.

When LM scoring is not active, these columns are absent (backwards compatible).

### detection_report.html

When LM scoring is active, the top raga display uses the LM ranking. The report should indicate that LM scoring was used. Histogram scores remain visible in the detailed candidate table.

### Printed summary

```
  Top Tonic: C#  (LM-scored)
  Top Raga: Puriya Dhanashree (LM), Basant (histogram)
```

---

## Transcription Details

The LM re-ranking step needs to run transcription internally, using the same code path as analyze but without raga correction. Specifically:

- Calls `transcription.transcribe_to_notes()` with `snap_mode='chromatic'`
- Uses melody pitch data already loaded in detect (from `PitchData` for the melody track)
- Uses detect's tonic for transcription parameters (derivative threshold, min duration, etc.) from the existing `PipelineConfig`
- Does NOT call `apply_raga_correction_to_notes`
- Does NOT generate phrases, patterns, reports, or any analyze-phase output
- The notes are used solely for tokenization and LM scoring

This is essentially `--transcription-only` + `--skip-raga-correction` behavior, invoked programmatically within the detect flow.

---

## Implementation Scope

### Files to modify

| File | Change |
|------|--------|
| `raga_pipeline/config.py` | Add `use_lm_scoring: bool`, `lm_model_path: Optional[str]` to `PipelineConfig`; add CLI args to detect parser; add validation |
| `driver.py` | Add step 5.5 between candidate scoring and report generation in detect flow |

### Files to use (no changes needed)

| File | What we import |
|------|---------------|
| `raga_pipeline/transcription.py` | `transcribe_to_notes()` |
| `raga_pipeline/sequence.py` | `tokenize_notes_for_lm()` |
| `raga_pipeline/language_model/__init__.py` | `NgramModel.from_dict()`, `score_sequence()` |

### No new files needed

All the building blocks exist. This is a wiring task inside `driver.py`.

---

## Dependencies

- Trained n-gram model JSON (from `python -m raga_pipeline.language_model train`)
- No new Python dependencies

---

## Future Extensions (not in scope)

- Combined histogram + LM scoring (weighted blend)
- Segment-level LM confidence in the detect report
- Automatic model path discovery (search standard locations)
- LM scoring in the local web app
