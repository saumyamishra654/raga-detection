# N-gram Language Model for Raga Detection -- Design Spec

## Overview

Build per-raga character n-gram language models from transcribed sargam sequences, then classify new recordings by which raga model assigns the highest probability to the transcription. This is the direct analogue of **language identification from romanized text** -- ragas are "languages", sargam tokens are "characters", and we learn each raga's sequential grammar.

This is **Approach 1** from the broader language model investigation (see `docs/language-model-raga-detection-approaches.md`). Approaches 2 (PPM) and 3 (discriminative reweighting) build on this foundation.

## Goals

- Thesis experiment: compare n-gram language model accuracy against existing histogram scorer and motif mining on the 300-song/30-raga CompuMusic subset
- Whole-recording classification (top-1, top-3, MRR) as the headline metric
- Segment-level confidence curves showing how raga identity emerges over time
- Full musicological interpretability: inspectable per-raga trigram tables

## Non-goals

- Pipeline integration (future work if results are promising)
- Replacing the histogram scorer or motif mining
- Neural/deep learning approaches (reserved for Approach 2/3 or later)

---

## Token Representation

### Alphabet

The 12 sargam tokens, tonic-relative: `Sa, re, Re, ga, Ga, ma, Ma, Pa, dha, Dha, ni, Ni`

Case encodes quality: lowercase = komal, uppercase = shuddha/tivra (except `Sa` and `Pa` which are always uppercase).

### Octave marking

- **Middle octave** (containing Sa, determined by median pitch of the recording): bare tokens -- `Sa`, `Re`, `Ga`
- **One octave below**: apostrophe suffix -- `Sa'`, `Ni'`, `Dha'`
- **One octave above**: double apostrophe -- `Sa''`, `Re''`, `Ga''`
- **Beyond**: clip to these three octaves

The reference octave is the octave of the median MIDI pitch of voiced frames in the transcription (consistent with the normalization already used in `output.py` for report display).

This yields a practical alphabet of ~36 tokens (12 x 3 octaves), though most ragas concentrate on ~15-20.

### Special tokens

- `<BOS>`: inserted at phrase boundaries (beginning of each phrase). Lets the model learn phrase-initial patterns (e.g., "Yaman phrases typically open with `Ni' Re Ga`").
- No `<EOS>` needed -- phrases just end.

### Tokenization source

Input is `transcribed_notes.csv` (or the latest edited version from `transcription_edits/` if present). Each row has `start_time, end_time, midi_pitch, sargam, confidence, ...`. The tokenizer:

1. Reads the note list
2. Computes the reference octave from median pitch
3. Converts each note to `sargam + octave_marker`
4. Inserts `<BOS>` at phrase boundaries (detected from silence gaps in the note timing, using the same phrase boundary logic as `sequence.py`)

---

## Shared Tokenizer

Lives in `raga_pipeline/` so it can be reused by motif mining and future approaches.

### Location

Extend or create a tokenization utility accessible from both `motifs.py` and the new language model code. Two options:

- **Add to `sequence.py`**: functions `tokenize_notes_for_lm(notes, tonic_midi, phrase_gap_sec=0.25) -> List[str]` and `compute_reference_octave(notes) -> int`
- **New file `raga_pipeline/tokenizer.py`**: if the scope grows beyond a couple of functions

Start with adding to `sequence.py` since it already owns note/phrase data structures. Extract to a separate file if it gets large.

### Interface

```python
def tokenize_notes_for_lm(
    notes: List[Note],
    tonic_midi: float,
    phrase_gap_sec: float = 0.25,
) -> List[str]:
    """Convert note list to LM tokens with octave markers and phrase boundaries.

    Returns list of tokens like ['<BOS>', 'Sa', 'Re', 'Ga', 'ma', 'Pa', '<BOS>', 'Ni\\'', 'Re', ...]
    """
```

---

## N-gram Model

### Training

For each raga r with recordings R_1..R_k:

1. Tokenize each recording's transcription
2. Pool all tokens for raga r
3. Count n-grams for orders 1 through `--order` (default 5)
4. Apply smoothing
5. Compute interpolation weights (lambdas)

### Smoothing

Two options, selectable via `--smoothing`:

- **`add-k`** (default k=0.01): simple, fast, good baseline. `P(w|ctx) = (count(ctx,w) + k) / (count(ctx) + k*V)` where V is vocabulary size.
- **`kneser-ney`**: better theoretical properties. Uses continuation counts for lower-order distributions. More code but well-documented algorithm.

Start with `add-k`. Switch to Kneser-Kney if evaluation shows it matters.

### Interpolation

Final probability for a token given context (example for order 5):

```text
P(w | c4..c1) = L5*P_5gram + L4*P_4gram + L3*P_3gram + L2*P_bigram + L1*P_unigram
```

Default: equal-weight interpolation across all orders. Optionally tune via held-out likelihood on training data (grid search or EM).

### Model serialization

JSON file with this structure:

```json
{
  "metadata": {
    "order": 5,
    "smoothing": "add-k",
    "smoothing_params": {"k": 0.01},
    "lambdas": [0.2, 0.2, 0.2, 0.2, 0.2],
    "vocabulary": ["Sa", "re", "Re", "ga", "Ga", "ma", "Ma", "Pa", "dha", "Dha", "ni", "Ni", "Sa'", "..."],
    "training_recordings": 270,
    "training_ragas": 30,
    "timestamp": "2026-03-30T..."
  },
  "ragas": {
    "Yaman": {
      "recording_count": 10,
      "total_tokens": 45000,
      "unigrams": {"Sa": 5200, "Re": 4100, "Ga": 6300, "...": "..."},
      "bigrams": {"Sa|Re": 2100, "Re|Ga": 3400, "...": "..."},
      "trigrams": {"Sa|Re|Ga": 1800, "Ni'|Re|Ga": 950, "...": "..."},
      "4grams": {"Sa|Re|Ga|ma": 900, "...": "..."},
      "5grams": {"Sa|Re|Ga|ma|Pa": 400, "...": "..."}
    },
    "Bhairav": { "..." : "..." }
  }
}
```

Counts stored rather than probabilities -- probabilities are computed at score time so smoothing parameters can be adjusted without retraining. Human-readable: you can open the file and inspect Yaman's top trigrams directly.

---

## Scoring

### Whole-recording score

Given a tokenized transcription T = [t_1, ..., t_N] and a raga model r:

```
score(r, T) = (1/N) * sum_{i=1}^{N} log P_r(t_i | context_i)
```

Where `context_i` is the preceding `order-1` tokens (with `<BOS>` padding at phrase starts). This is the average log-likelihood per token (negative perplexity in log space).

Predicted raga = `argmax_r score(r, T)`.

### Segment-level scoring

Slide a window of `--segment-window` tokens (default 100) across the transcription with 50% overlap. Compute `score(r, segment)` for each raga at each position. Output:

- Per-segment top-3 ragas with scores
- Time-mapped back to the recording via note timestamps
- Enables "confidence over time" visualization

### Score output

```json
{
  "transcription": "path/to/transcribed_notes.csv",
  "tonic": "C#",
  "total_tokens": 12000,
  "rankings": [
    {"raga": "Yaman", "score": -2.31, "rank": 1},
    {"raga": "Kalyan", "score": -2.58, "rank": 2},
    "..."
  ],
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 45.2,
      "token_range": [0, 100],
      "top_ragas": [
        {"raga": "Yaman", "score": -2.45},
        {"raga": "Kalyan", "score": -2.51}
      ]
    },
    "..."
  ]
}
```

---

## Evaluation

### Method

Leave-one-out cross-validation per raga: for each recording, train on all other recordings of that raga (and all recordings of other ragas), score the held-out recording, record the predicted raga and rank of true raga.

### Metrics

- **Top-1 accuracy**: fraction of recordings where correct raga ranks first
- **Top-3 accuracy**: fraction where correct raga is in top 3
- **Mean reciprocal rank (MRR)**: average of `1/rank_of_correct_raga`
- **Per-raga accuracy**: breakdown by raga (identifies which ragas are hard)
- **Confusion pairs**: which raga pairs are most often confused (e.g., Yaman/Kalyan)

### Order sweep

The `--sweep-orders` flag in `evaluate` runs the full leave-one-out evaluation at each specified order and produces a summary table: accuracy vs. n-gram order. This shows where returns diminish and how much sequential context ragas need for discrimination. Literature suggests orders 4-5 should outperform trigrams for music; the sweep confirms this empirically on our corpus.

### Baselines for comparison

1. **Histogram scorer** (existing 8-coefficient `RagaScorer`)
2. **Motif mining scorer** (existing `motifs.py`)
3. **N-gram language model** (this work)
4. **Unigram-only ablation**: same tokenization but only unigram frequencies (no sequence information). Isolates whether improvement comes from sequential modelling or from better input representation.

### Evaluation output

CSV with columns: `filename, true_raga, predicted_raga, correct, true_raga_rank, score_top1, score_top2, score_top3, raga_top1, raga_top2, raga_top3`

Plus summary statistics printed to stdout.

---

## CLI Interface

Entry point: `python -m raga_pipeline.language_model <subcommand>`

### `train`

```bash
python -m raga_pipeline.language_model train \
    --gt ground_truth.csv \
    --results-dir batch_results/ \
    --output raga_ngram_model.json \
    --order 3 \
    --smoothing add-k \
    --smoothing-k 0.01 \
    --min-recordings 3 \
    --lambdas 0.6,0.3,0.1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--gt` | required | Ground-truth CSV (filename, raga, tonic) |
| `--results-dir` | required | Batch results directory with transcription CSVs |
| `--output` | `raga_ngram_model.json` | Output model path |
| `--order` | `5` | Maximum n-gram order |
| `--smoothing` | `add-k` | Smoothing method (`add-k` or `kneser-ney`) |
| `--smoothing-k` | `0.01` | k parameter for add-k smoothing |
| `--min-recordings` | `3` | Skip ragas with fewer recordings |
| `--lambdas` | equal weight | Interpolation weights (highest order first, comma-separated) |

### `score`

```bash
python -m raga_pipeline.language_model score \
    --model raga_ngram_model.json \
    --transcription transcribed_notes.csv \
    --tonic C# \
    --segments \
    --segment-window 100 \
    --output score_result.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Trained model JSON |
| `--transcription` | required | Transcribed notes CSV |
| `--tonic` | required | Tonic note name |
| `--segments` | `False` | Emit segment-level scores |
| `--segment-window` | `100` | Tokens per segment |
| `--output` | stdout | Output path (JSON) |

### `evaluate`

```bash
python -m raga_pipeline.language_model evaluate \
    --gt ground_truth.csv \
    --results-dir batch_results/ \
    --order 5 \
    --smoothing add-k \
    --output eval_results.csv \
    --sweep-orders 2,3,4,5,6
```

| Flag | Default | Description |
|------|---------|-------------|
| `--gt` | required | Ground-truth CSV |
| `--results-dir` | required | Batch results directory |
| `--order` | `5` | Maximum n-gram order |
| `--smoothing` | `add-k` | Smoothing method |
| `--smoothing-k` | `0.01` | k parameter |
| `--min-recordings` | `3` | Skip ragas with fewer recordings |
| `--lambdas` | equal weight | Interpolation weights |
| `--sweep-orders` | None | Comma-separated orders to sweep (e.g., `2,3,4,5,6`); runs full eval for each |
| `--output` | `eval_results.csv` | Output CSV path |

---

## File Layout

```
raga_pipeline/
    sequence.py              # extended: tokenize_notes_for_lm(), compute_reference_octave()
    language_model/
        __init__.py           # NgramModel class, train/score/evaluate logic
        __main__.py           # CLI entry point (python -m raga_pipeline.language_model)

experiments/
    language_model/
        README.md             # experiment notes, how to reproduce
        eval_results/         # output CSVs, plots
        notebooks/            # analysis notebooks (confusion matrices, confidence curves)
```

The `raga_pipeline/language_model.py` module contains the model logic (NgramModel class, training, scoring, serialization). The CLI entry point uses the same `argparse` pattern as `motifs.py`. The `experiments/` directory holds evaluation outputs and analysis -- not checked into the pipeline's test/validation loop.

---

## Candidate Discovery

Reuses `motifs.py:_discover_candidates` logic: glob for `transcribed_notes.csv` (and edited versions) under `--results-dir`, match to ground-truth rows by filename stem. If this function isn't already factored out for reuse, extract it as a shared utility during implementation.

---

## Dependencies

No new dependencies. Uses only:
- `json` (model serialization)
- `csv` (transcription/GT reading)
- `math` (log probabilities)
- `collections` (Counter, defaultdict)
- `argparse` (CLI)
- Existing `raga_pipeline` types (`Note`, `Phrase`)

---

## Future Extensions (not in scope for this spec)

- Approach 2 (PPM/variable-length Markov): swap NgramModel internals, same CLI
- Approach 3 (discriminative reweighting): post-processing pass on trained model
- Pipeline integration: wire `score` into `driver.py` analyze phase
- Duration-weighted tokens: weight n-gram counts by note duration (longer notes contribute more)
- Interval-augmented tokens: `Sa+2>Re` encoding pitch interval alongside sargam (for Approach 2/3)
