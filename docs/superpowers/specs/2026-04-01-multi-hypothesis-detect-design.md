# Multi-Hypothesis LM-Enhanced Detect -- Design Spec

## Overview

Extend the detect pipeline with an optional n-gram language model re-ranking stage. For each (tonic, raga) candidate from the histogram scorer, apply that raga's correction to a chromatic transcription, score the corrected transcription against the LM, and record the correction's deletion rate as an independent signal. The final ranking uses both the LM score and the deletion rate.

This removes the circularity problem: every hypothesis gets its own raga correction applied symmetrically, and the LM (trained on corrected transcriptions) scores each hypothesis on its own terms. The deletion rate penalizes hypotheses that require heavy surgery to fit.

## Flow

```
[Steps 1-5: existing detect -- unchanged]
    Stem separation -> pitch extraction -> histograms -> peaks -> candidate scoring
    -> candidates with surviving tonic set (typically 4-6 tonics)

[Step 5.5: LM re-ranking -- NEW, only when --use-lm-scoring]

    5.5a) Load NgramModel from --lm-model JSON

    5.5b) Run ONE chromatic transcription on melody pitch data
          (snap_mode='chromatic', no raga correction)
          -> raw_notes (fixed MIDI pitches)

    5.5c) For each unique tonic in candidates:
          -> (nothing to transcribe -- raw_notes are tonic-independent)

    5.5d) For each (tonic, raga) candidate:
          -> Apply raga correction for this specific raga to raw_notes
             -> corrected_notes, correction_stats (including deletion count)
          -> Compute deletion_rate = correction_stats['discarded'] / correction_stats['total']
          -> Tokenize corrected_notes with this tonic via tokenize_notes_for_lm
          -> lm_score = model.score_sequence(raga, tokens)
          -> Store: lm_score, deletion_rate per candidate

    5.5e) Write lm_candidates.csv with lm_score, deletion_rate, lm_rank
          Sorted by lm_score descending
          Original candidates.csv is NOT modified

[Steps 6-7: existing report generation -- unchanged, uses original candidates]
```

## Key Design Decisions

### Train on corrected, score on corrected

The LM is trained on raga-corrected transcriptions (the standard analyze output). At inference, each hypothesis gets its own raga correction applied before scoring. This makes training and scoring symmetric -- the model sees the same kind of data at both stages.

### Deletion rate as independent signal

The correction's deletion rate (fraction of chromatic notes discarded to fit the raga) is stored alongside the LM score but NOT combined into it. The two signals are kept independent so that:
- Their relative importance can be tuned empirically later
- The deletion rate can be adjusted for raga scale size (pentatonic ragas naturally delete more)
- A combined score can be computed downstream: e.g., `combined = lm_score + alpha * (1 - deletion_rate)`

### One transcription, many corrections

The chromatic transcription produces fixed MIDI pitches. Each (tonic, raga) hypothesis applies its own raga correction to those same raw notes. The correction is cheap (~milliseconds per hypothesis). Tokenization is also per-hypothesis since both the tonic (for sargam mapping) and the raga (for which notes survive correction) differ.

### Phrase-aware n-grams (prerequisite)

N-grams must NOT cross phrase boundaries. The tokenizer should produce per-phrase token sequences, and n-gram counting/scoring should respect phrase boundaries. This is a separate fix to the existing LM infrastructure that should be done before this spec is implemented.

## CLI Interface

```bash
python driver.py detect \
    --audio song.mp3 \
    --use-lm-scoring \
    --lm-model compmusic_ngram_model.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--use-lm-scoring` | `False` | Enable LM re-ranking after histogram scoring |
| `--lm-model` | None | Path to trained n-gram model JSON (required when `--use-lm-scoring`) |

## Output

### candidates.csv (when LM scoring active)

Existing columns preserved. New columns added:

| Column | Description |
|--------|-------------|
| `lm_score` | Average log-likelihood per token under the raga's LM |
| `lm_rank` | Rank by LM score (1 = best) |
| `deletion_rate` | Fraction of chromatic notes discarded by raga correction (0.0 = perfect fit, 1.0 = all notes removed) |
| `notes_before_correction` | Total chromatic notes (same for all candidates with same tonic) |
| `notes_after_correction` | Notes remaining after raga correction |

Rows sorted by `lm_score` descending. Written to `lm_candidates.csv` (separate file, does not modify `candidates.csv`).

### detection_report.html

Unchanged. The existing report continues to show histogram rankings only. LM results are in `lm_candidates.csv` only (no report changes in this spec).

## Files to Modify

| File | Change |
|------|--------|
| `raga_pipeline/config.py` | Add `use_lm_scoring`, `lm_model_path` to PipelineConfig + CLI args |
| `driver.py` | Add step 5.5 in detect flow: chromatic transcription, per-hypothesis correction + tokenization + LM scoring |

## Files Used (no changes)

| File | What we import |
|------|---------------|
| `raga_pipeline/transcription.py` | `transcribe_to_notes()` |
| `raga_pipeline/raga.py` | `apply_raga_correction_to_notes()`, `get_raga_notes()` |
| `raga_pipeline/sequence.py` | `tokenize_notes_for_lm()` |
| `raga_pipeline/language_model/__init__.py` | `NgramModel` |

## Prerequisites

- Phrase-boundary-aware n-gram counting (fix the 49% cross-boundary n-gram issue)
- Retrained model on phrase-aware corrected transcriptions

## Future Extensions

- Empirically determine optimal deletion_rate coefficient (alpha) via grid search on LOO CV
- Adjust deletion_rate normalization for raga scale size (pentatonic vs heptatonic)
- Combined scoring formula: `combined = lm_score + alpha * log(1 - deletion_rate)`
- Segment-level LM confidence in detect report
