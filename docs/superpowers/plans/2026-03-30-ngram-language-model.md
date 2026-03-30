# N-gram Language Model for Raga Detection -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build per-raga n-gram language models from sargam transcriptions to classify recordings by raga, with evaluation against existing histogram/motif approaches.

**Architecture:** Shared tokenizer in `raga_pipeline/sequence.py` converts Note lists to octave-marked sargam tokens. New `raga_pipeline/language_model/` package contains the NgramModel class (train/score/serialize) and CLI entry point. Evaluation scripts live alongside the model code. Reuses `motifs.py` candidate discovery and ground-truth parsing.

**Tech Stack:** Python 3.10, stdlib only (json, csv, math, collections, argparse). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-30-ngram-language-model-raga-detection.md`

---

### Task 1: Shared Tokenizer -- `tokenize_notes_for_lm`

**Files:**
- Modify: `raga_pipeline/sequence.py` (add function at end of file)
- Create: `tests/test_lm_tokenizer.py`

This tokenizer converts a list of Note objects into octave-marked sargam tokens with `<BOS>` phrase boundary markers. It is the shared foundation for both the language model and future sequence-based approaches.

**Token format:** `Sa`, `Re`, `ga`, `Ga`, `ma`, `Ma`, `Pa`, `dha`, `Dha`, `ni`, `Ni` with octave markers: bare = middle octave, `'` suffix = one below, `''` = two below, `''` = one above (using double single-quote to distinguish from lower). We will use the convention: lower octave = `'`, upper octave = `''`, to keep tokens ASCII-clean and avoid the `·` middle-dot used by `midi_to_sargam`.

Updated convention to avoid ambiguity: lower = `_lo` suffix, upper = `_hi` suffix? No -- let's stay close to the spec. Use: bare = middle, single `'` = lower, double `'` = upper. The tokenizer always produces these. Context length makes collisions impossible (a token is always a single unit).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lm_tokenizer.py
import unittest
from raga_pipeline.sequence import Note, tokenize_notes_for_lm


class TestTokenizeNotesForLM(unittest.TestCase):
    """Tests for the shared LM tokenizer."""

    def _make_note(self, start, end, midi, sargam="", pitch_class=-1, confidence=0.9):
        return Note(
            start=start, end=end, pitch_midi=midi,
            pitch_hz=440.0, confidence=confidence,
            sargam=sargam, pitch_class=pitch_class,
        )

    def test_basic_middle_octave(self):
        """Notes in the middle octave get bare sargam tokens."""
        # Tonic = C4 = MIDI 60, so Sa=60, Re=62, Ga=64
        notes = [
            self._make_note(0.0, 0.5, 60),   # Sa
            self._make_note(0.5, 1.0, 62),   # Re
            self._make_note(1.0, 1.5, 64),   # Ga
        ]
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(tokens, ["<BOS>", "Sa", "Re", "Ga"])

    def test_lower_octave_marking(self):
        """Notes below the reference octave get ' suffix."""
        # Tonic = C4 = 60. Ni below = MIDI 59 -> Ni, one octave below tonic octave
        # Actually MIDI 59 is B3, which is Ni in the octave below
        notes = [
            self._make_note(0.0, 0.5, 60),   # Sa (middle)
            self._make_note(0.5, 1.0, 59),   # Ni (lower)
            self._make_note(1.0, 1.5, 57),   # Dha (lower)
        ]
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(tokens, ["<BOS>", "Sa", "Ni'", "Dha'"])

    def test_upper_octave_marking(self):
        """Notes above the reference octave get '' suffix."""
        # Sa one octave above = MIDI 72
        notes = [
            self._make_note(0.0, 0.5, 72),   # Sa (upper)
            self._make_note(0.5, 1.0, 74),   # Re (upper)
        ]
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(tokens, ["<BOS>", "Sa''", "Re''"])

    def test_phrase_boundaries(self):
        """Silence gaps > phrase_gap_sec insert <BOS> tokens."""
        notes = [
            self._make_note(0.0, 0.5, 60),   # Sa
            self._make_note(0.5, 1.0, 62),   # Re
            # 1.0s gap (> 0.25 default)
            self._make_note(2.0, 2.5, 64),   # Ga
            self._make_note(2.5, 3.0, 65),   # ma
        ]
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60, phrase_gap_sec=0.25)
        self.assertEqual(tokens, ["<BOS>", "Sa", "Re", "<BOS>", "Ga", "ma"])

    def test_empty_notes(self):
        """Empty note list returns empty token list."""
        tokens = tokenize_notes_for_lm([], tonic_midi=60)
        self.assertEqual(tokens, [])

    def test_komal_shuddha_encoding(self):
        """Komal/shuddha distinction preserved via sargam case."""
        # Tonic C4=60: komal Re = MIDI 61, shuddha Re = MIDI 62
        notes = [
            self._make_note(0.0, 0.5, 61),   # re (komal)
            self._make_note(0.5, 1.0, 62),   # Re (shuddha)
        ]
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(tokens, ["<BOS>", "re", "Re"])

    def test_single_note(self):
        """Single note produces BOS + one token."""
        notes = [self._make_note(0.0, 0.5, 67)]  # Pa
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(tokens, ["<BOS>", "Pa"])

    def test_clipping_extreme_octaves(self):
        """Notes more than 1 octave away are clipped to the boundary octave."""
        # Two octaves below = MIDI 36 (C2 if tonic=C4)
        notes = [self._make_note(0.0, 0.5, 36)]  # Sa, 2 octaves below -> clip to Sa'
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(tokens, ["<BOS>", "Sa'"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lm_tokenizer.py -v`
Expected: `ImportError: cannot import name 'tokenize_notes_for_lm' from 'raga_pipeline.sequence'`

- [ ] **Step 3: Implement `tokenize_notes_for_lm`**

Add to `raga_pipeline/sequence.py` (at the end, before any `if __name__` block):

```python
def tokenize_notes_for_lm(
    notes: List[Note],
    tonic_midi: float,
    phrase_gap_sec: float = 0.25,
) -> List[str]:
    """Convert note list to LM tokens with octave markers and phrase boundaries.

    Token format:
        - Middle octave (containing tonic): bare sargam, e.g. ``Sa``, ``Re``
        - One octave below: single apostrophe suffix, e.g. ``Ni'``, ``Dha'``
        - One octave above: double apostrophe suffix, e.g. ``Sa''``
        - Beyond: clipped to nearest boundary octave

    Phrase boundaries (gaps > *phrase_gap_sec* between consecutive notes)
    insert a ``<BOS>`` token.

    Returns:
        List of string tokens, e.g.
        ``['<BOS>', 'Sa', 'Re', 'Ga', '<BOS>', 'Ni\\'', 'Re', ...]``
    """
    if not notes:
        return []

    tonic_pc = int(round(tonic_midi)) % 12
    # Reference octave: the octave of the tonic MIDI value
    tonic_octave = int(round(tonic_midi)) // 12

    tokens: List[str] = []
    prev_end: Optional[float] = None

    for note in notes:
        # Insert phrase boundary on gap
        if prev_end is None or (note.start - prev_end) > phrase_gap_sec:
            tokens.append("<BOS>")

        midi_rounded = int(round(note.pitch_midi))
        offset = (midi_rounded - int(round(tonic_midi))) % 12
        sargam = OFFSET_TO_SARGAM.get(offset, f"?{offset}")

        note_octave = midi_rounded // 12
        octave_diff = note_octave - tonic_octave

        # Clip to [-1, +1] range
        if octave_diff <= -1:
            sargam += "'"
        elif octave_diff >= 1:
            sargam += "''"
        # else: middle octave, bare sargam

        tokens.append(sargam)
        prev_end = note.end

    return tokens
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lm_tokenizer.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add raga_pipeline/sequence.py tests/test_lm_tokenizer.py
git commit -m "feat: add tokenize_notes_for_lm shared tokenizer for LM raga detection"
```

---

### Task 2: NgramModel class -- counting and probability

**Files:**
- Create: `raga_pipeline/language_model/__init__.py`
- Create: `tests/test_ngram_model.py`

The core model class that stores n-gram counts per raga, computes smoothed interpolated probabilities, and serializes to/from JSON.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ngram_model.py
import unittest
import math
from raga_pipeline.language_model import NgramModel


class TestNgramModel(unittest.TestCase):
    """Tests for NgramModel counting and probability computation."""

    def test_add_sequence_counts_ngrams(self):
        """Adding a token sequence populates unigram through n-gram counts."""
        model = NgramModel(order=3)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga", "ma", "Pa"])
        counts = model.get_counts("Yaman")

        # Unigrams: each token counted once (including <BOS>)
        self.assertEqual(counts[1][("Sa",)], 1)
        self.assertEqual(counts[1][("Re",)], 1)
        self.assertEqual(counts[1][("<BOS>",)], 1)

        # Bigrams
        self.assertEqual(counts[2][("<BOS>", "Sa")], 1)
        self.assertEqual(counts[2][("Sa", "Re")], 1)
        self.assertEqual(counts[2][("Re", "Ga")], 1)

        # Trigrams
        self.assertEqual(counts[3][("<BOS>", "Sa", "Re")], 1)
        self.assertEqual(counts[3][("Sa", "Re", "Ga")], 1)

    def test_add_multiple_sequences_accumulates(self):
        """Multiple sequences for the same raga accumulate counts."""
        model = NgramModel(order=2)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga"])
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "ma"])
        counts = model.get_counts("Yaman")

        self.assertEqual(counts[2][("Sa", "Re")], 2)
        self.assertEqual(counts[2][("Re", "Ga")], 1)
        self.assertEqual(counts[2][("Re", "ma")], 1)

    def test_log_prob_smoothed_nonzero(self):
        """Smoothed probability is nonzero even for unseen n-grams."""
        model = NgramModel(order=2, smoothing="add-k", smoothing_k=0.01)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga"])
        model.finalize()

        # Seen bigram
        lp_seen = model.log_prob("Yaman", "Re", ("Sa",))
        # Unseen bigram
        lp_unseen = model.log_prob("Yaman", "Pa", ("Sa",))

        self.assertGreater(lp_seen, lp_unseen)
        self.assertTrue(math.isfinite(lp_unseen))
        self.assertLess(lp_unseen, 0)  # log prob is negative

    def test_score_sequence(self):
        """score_sequence returns average log-likelihood per token."""
        model = NgramModel(order=2, smoothing="add-k", smoothing_k=0.01)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga", "ma", "Pa"] * 20)
        model.add_sequence("Bhairav", ["<BOS>", "Sa", "re", "Ga", "ma", "Pa"] * 20)
        model.finalize()

        yaman_seq = ["<BOS>", "Sa", "Re", "Ga", "ma", "Pa"]
        bhairav_seq = ["<BOS>", "Sa", "re", "Ga", "ma", "Pa"]

        # Yaman sequence should score higher under Yaman model
        score_yaman_yaman = model.score_sequence("Yaman", yaman_seq)
        score_bhairav_yaman = model.score_sequence("Bhairav", yaman_seq)
        self.assertGreater(score_yaman_yaman, score_bhairav_yaman)

        # Bhairav sequence should score higher under Bhairav model
        score_bhairav_bhairav = model.score_sequence("Bhairav", bhairav_seq)
        score_yaman_bhairav = model.score_sequence("Yaman", bhairav_seq)
        self.assertGreater(score_bhairav_bhairav, score_yaman_bhairav)

    def test_ragas_list(self):
        """Model tracks which ragas have been added."""
        model = NgramModel(order=2)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re"])
        model.add_sequence("Bhairav", ["<BOS>", "Sa", "re"])
        self.assertEqual(sorted(model.ragas()), ["Bhairav", "Yaman"])

    def test_empty_sequence_ignored(self):
        """Adding an empty sequence does not crash or create a raga entry."""
        model = NgramModel(order=2)
        model.add_sequence("Yaman", [])
        self.assertEqual(model.ragas(), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ngram_model.py -v`
Expected: `ModuleNotFoundError: No module named 'raga_pipeline.language_model'`

- [ ] **Step 3: Implement NgramModel**

```python
# raga_pipeline/language_model/__init__.py
"""
Per-raga n-gram language models for raga detection.

CLI entry point:
    python -m raga_pipeline.language_model train|score|evaluate ...
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


class NgramModel:
    """N-gram language model with per-raga count tables and smoothed scoring."""

    def __init__(
        self,
        order: int = 5,
        smoothing: str = "add-k",
        smoothing_k: float = 0.01,
        lambdas: Optional[List[float]] = None,
    ):
        self.order = order
        self.smoothing = smoothing
        self.smoothing_k = smoothing_k
        # Interpolation weights: one per order (index 0 = unigram, ..., index order-1 = highest)
        if lambdas is not None:
            self.lambdas = list(lambdas)
        else:
            # Equal weight across all orders
            self.lambdas = [1.0 / order] * order
        # Per-raga n-gram counts: raga -> order -> ngram_tuple -> count
        self._counts: Dict[str, Dict[int, Counter[Tuple[str, ...]]]] = {}
        # Per-raga context counts (denominator): raga -> order -> context_tuple -> count
        self._context_counts: Dict[str, Dict[int, Counter[Tuple[str, ...]]]] = {}
        self._vocabulary: set[str] = set()
        self._recording_counts: Dict[str, int] = defaultdict(int)
        self._token_counts: Dict[str, int] = defaultdict(int)
        self._finalized = False

    def add_sequence(self, raga: str, tokens: List[str]) -> None:
        """Add a tokenized recording to the raga's model."""
        if not tokens:
            return
        self._finalized = False
        if raga not in self._counts:
            self._counts[raga] = {n: Counter() for n in range(1, self.order + 1)}
            self._context_counts[raga] = {n: Counter() for n in range(1, self.order + 1)}

        self._recording_counts[raga] += 1
        self._token_counts[raga] += len(tokens)

        for token in tokens:
            self._vocabulary.add(token)

        for n in range(1, self.order + 1):
            for i in range(n - 1, len(tokens)):
                ngram = tuple(tokens[i - n + 1 : i + 1])
                self._counts[raga][n][ngram] += 1
                if n > 1:
                    context = ngram[:-1]
                    self._context_counts[raga][n][context] += 1

    def finalize(self) -> None:
        """Pre-compute any derived structures after all sequences added."""
        self._finalized = True

    def ragas(self) -> List[str]:
        """Return list of ragas in the model."""
        return sorted(self._counts.keys())

    def get_counts(self, raga: str) -> Dict[int, Counter[Tuple[str, ...]]]:
        """Return raw n-gram counts for a raga (for testing/inspection)."""
        return self._counts.get(raga, {})

    def vocabulary_size(self) -> int:
        """Return size of observed vocabulary."""
        return len(self._vocabulary)

    def log_prob(self, raga: str, token: str, context: Tuple[str, ...] = ()) -> float:
        """Compute interpolated smoothed log probability of token given context.

        Uses the interpolation weights (lambdas) to blend orders.
        Context should be up to (order-1) preceding tokens.
        """
        V = max(self.vocabulary_size(), 1)
        raga_counts = self._counts.get(raga)
        raga_ctx_counts = self._context_counts.get(raga)
        if raga_counts is None or raga_ctx_counts is None:
            return math.log(1.0 / V)

        prob = 0.0
        for n in range(1, self.order + 1):
            lam = self.lambdas[n - 1]
            if lam == 0:
                continue

            if n == 1:
                # Unigram
                count = raga_counts[1].get((token,), 0)
                total = sum(raga_counts[1].values())
                if self.smoothing == "add-k":
                    p = (count + self.smoothing_k) / (total + self.smoothing_k * V)
                else:
                    p = (count + self.smoothing_k) / (total + self.smoothing_k * V)
            else:
                # N-gram with context
                ctx_len = n - 1
                if len(context) >= ctx_len:
                    ctx = context[-ctx_len:]
                else:
                    ctx = context
                ngram = ctx + (token,)
                if len(ngram) == n:
                    count = raga_counts[n].get(ngram, 0)
                    ctx_total = raga_ctx_counts[n].get(ctx, 0)
                    if self.smoothing == "add-k":
                        p = (count + self.smoothing_k) / (ctx_total + self.smoothing_k * V)
                    else:
                        p = (count + self.smoothing_k) / (ctx_total + self.smoothing_k * V)
                else:
                    # Not enough context for this order, skip
                    continue

            prob += lam * p

        if prob <= 0:
            return math.log(1e-20)
        return math.log(prob)

    def score_sequence(self, raga: str, tokens: List[str]) -> float:
        """Average log-likelihood per token of a sequence under a raga model.

        Higher is better (less surprising).
        """
        if not tokens:
            return float("-inf")

        total_ll = 0.0
        count = 0
        for i, token in enumerate(tokens):
            context = tuple(tokens[max(0, i - self.order + 1) : i])
            total_ll += self.log_prob(raga, token, context)
            count += 1

        return total_ll / count if count > 0 else float("-inf")

    def rank_ragas(self, tokens: List[str]) -> List[Tuple[str, float]]:
        """Score tokens against all ragas, return sorted (raga, score) pairs."""
        scores = []
        for raga in self.ragas():
            score = self.score_sequence(raga, tokens)
            scores.append((raga, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to a JSON-compatible dictionary."""
        ragas_data = {}
        for raga in self.ragas():
            raga_entry: Dict[str, Any] = {
                "recording_count": self._recording_counts.get(raga, 0),
                "total_tokens": self._token_counts.get(raga, 0),
            }
            for n in range(1, self.order + 1):
                key = f"{n}grams"
                raga_entry[key] = {
                    "|".join(ngram): count
                    for ngram, count in self._counts[raga][n].most_common()
                }
            ragas_data[raga] = raga_entry

        return {
            "metadata": {
                "order": self.order,
                "smoothing": self.smoothing,
                "smoothing_params": {"k": self.smoothing_k},
                "lambdas": self.lambdas,
                "vocabulary": sorted(self._vocabulary),
                "training_recordings": sum(self._recording_counts.values()),
                "training_ragas": len(self._counts),
            },
            "ragas": ragas_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NgramModel":
        """Deserialize model from a dictionary (as loaded from JSON)."""
        meta = data["metadata"]
        model = cls(
            order=meta["order"],
            smoothing=meta["smoothing"],
            smoothing_k=meta["smoothing_params"]["k"],
            lambdas=meta["lambdas"],
        )
        model._vocabulary = set(meta.get("vocabulary", []))

        for raga, raga_data in data["ragas"].items():
            model._counts[raga] = {n: Counter() for n in range(1, model.order + 1)}
            model._context_counts[raga] = {n: Counter() for n in range(1, model.order + 1)}
            model._recording_counts[raga] = raga_data.get("recording_count", 0)
            model._token_counts[raga] = raga_data.get("total_tokens", 0)

            for n in range(1, model.order + 1):
                key = f"{n}grams"
                if key in raga_data:
                    for ngram_str, count in raga_data[key].items():
                        ngram = tuple(ngram_str.split("|"))
                        model._counts[raga][n][ngram] = count
                        if n > 1:
                            context = ngram[:-1]
                            model._context_counts[raga][n][context] += count

        model._finalized = True
        return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ngram_model.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add raga_pipeline/language_model/__init__.py tests/test_ngram_model.py
git commit -m "feat: add NgramModel class with counting, smoothing, and scoring"
```

---

### Task 3: Model serialization round-trip

**Files:**
- Modify: `tests/test_ngram_model.py` (add serialization tests)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ngram_model.py`:

```python
import json
import tempfile
from pathlib import Path


class TestNgramModelSerialization(unittest.TestCase):
    """Tests for model save/load round-trip."""

    def test_to_dict_and_back(self):
        """Model survives a to_dict -> from_dict round-trip."""
        model = NgramModel(order=3, smoothing="add-k", smoothing_k=0.05)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga", "ma", "Pa"])
        model.add_sequence("Bhairav", ["<BOS>", "Sa", "re", "Ga", "ma", "dha"])
        model.finalize()

        data = model.to_dict()
        restored = NgramModel.from_dict(data)

        # Same ragas
        self.assertEqual(sorted(restored.ragas()), sorted(model.ragas()))

        # Same scores
        seq = ["<BOS>", "Sa", "Re", "Ga"]
        for raga in model.ragas():
            self.assertAlmostEqual(
                restored.score_sequence(raga, seq),
                model.score_sequence(raga, seq),
                places=10,
            )

    def test_json_round_trip(self):
        """Model survives JSON serialization to disk and back."""
        model = NgramModel(order=2)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga"] * 10)
        model.finalize()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model.to_dict(), f)
            path = Path(f.name)

        try:
            with path.open() as f:
                loaded = NgramModel.from_dict(json.load(f))
            self.assertEqual(loaded.ragas(), model.ragas())
            seq = ["<BOS>", "Sa", "Re"]
            self.assertAlmostEqual(
                loaded.score_sequence("Yaman", seq),
                model.score_sequence("Yaman", seq),
                places=10,
            )
        finally:
            path.unlink()

    def test_metadata_fields(self):
        """Serialized dict contains expected metadata."""
        model = NgramModel(order=5, smoothing="add-k", smoothing_k=0.02)
        model.add_sequence("Yaman", ["<BOS>", "Sa", "Re"])
        data = model.to_dict()

        self.assertEqual(data["metadata"]["order"], 5)
        self.assertEqual(data["metadata"]["smoothing"], "add-k")
        self.assertEqual(data["metadata"]["smoothing_params"]["k"], 0.02)
        self.assertEqual(data["metadata"]["training_ragas"], 1)
        self.assertIn("Sa", data["metadata"]["vocabulary"])
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_ngram_model.py -v`
Expected: All 9 tests PASS (these should pass immediately with the Task 2 implementation)

- [ ] **Step 3: Commit**

```bash
git add tests/test_ngram_model.py
git commit -m "test: add serialization round-trip tests for NgramModel"
```

---

### Task 4: Training pipeline -- `train_model` function

**Files:**
- Modify: `raga_pipeline/language_model/__init__.py` (add `train_model` function)
- Create: `tests/test_lm_train.py`

This function reads ground-truth CSV + transcription CSVs from a results directory (reusing `motifs.py` candidate discovery), tokenizes each recording, and builds the NgramModel.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lm_train.py
import csv
import json
import tempfile
import unittest
from pathlib import Path

from raga_pipeline.language_model import train_model


def _write_gt_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write a ground-truth CSV with (filename, raga, tonic) rows."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Filename", "Raga", "Tonic"])
        writer.writeheader()
        for filename, raga, tonic in rows:
            writer.writerow({"Filename": filename, "Raga": raga, "Tonic": tonic})


def _write_transcription_csv(path: Path, rows: list[tuple[float, float, float, str]]) -> None:
    """Write a transcription CSV with (start, end, pitch_midi, sargam) rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "end", "pitch_midi", "sargam"])
        writer.writeheader()
        for start, end, midi, sargam in rows:
            writer.writerow({
                "start": f"{start:.3f}",
                "end": f"{end:.3f}",
                "pitch_midi": f"{midi:.1f}",
                "sargam": sargam,
            })


class TestTrainModel(unittest.TestCase):
    """Tests for the train_model function."""

    def test_basic_training(self):
        """train_model builds a model from GT CSV and transcription CSVs."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Write ground truth
            gt_path = tmp_path / "gt.csv"
            _write_gt_csv(gt_path, [
                ("song_a.mp3", "Yaman", "C"),
                ("song_b.mp3", "Yaman", "C"),
                ("song_c.mp3", "Yaman", "C"),
                ("song_d.mp3", "Bhairav", "C"),
                ("song_e.mp3", "Bhairav", "C"),
                ("song_f.mp3", "Bhairav", "C"),
            ])

            # Create results directory structure matching motifs.py conventions
            results = tmp_path / "results"
            # Yaman songs: Sa Re Ga ma Pa (C=60: 60,62,64,65,67)
            yaman_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                           enumerate([(60, "Sa"), (62, "Re"), (64, "Ga"), (65, "ma"), (67, "Pa")] * 20)]
            for name in ["song_a", "song_b", "song_c"]:
                _write_transcription_csv(
                    results / "demucs" / "htdemucs" / name / "transcribed_notes.csv",
                    yaman_notes,
                )

            # Bhairav songs: Sa re Ga ma Pa (C=60: 60,61,64,65,67)
            bhairav_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                            enumerate([(60, "Sa"), (61, "re"), (64, "Ga"), (65, "ma"), (67, "Pa")] * 20)]
            for name in ["song_d", "song_e", "song_f"]:
                _write_transcription_csv(
                    results / "demucs" / "htdemucs" / name / "transcribed_notes.csv",
                    bhairav_notes,
                )

            # Train
            output_path = tmp_path / "model.json"
            result = train_model(
                ground_truth=str(gt_path),
                results_dir=str(results),
                output=str(output_path),
                order=3,
                min_recordings=1,
            )

            # Verify model was saved
            self.assertTrue(output_path.exists())

            # Verify it can be loaded and has both ragas
            with output_path.open() as f:
                data = json.load(f)
            self.assertIn("Yaman", data["ragas"])
            self.assertIn("Bhairav", data["ragas"])
            self.assertEqual(data["metadata"]["order"], 3)

    def test_min_recordings_filter(self):
        """Ragas with fewer recordings than min_recordings are excluded."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_path = tmp_path / "gt.csv"
            _write_gt_csv(gt_path, [
                ("song_a.mp3", "Yaman", "C"),
                ("song_b.mp3", "Rare", "C"),
            ])

            results = tmp_path / "results"
            notes = [(i * 0.3, (i + 1) * 0.3, 60, "Sa") for i in range(10)]
            for name in ["song_a", "song_b"]:
                _write_transcription_csv(
                    results / "demucs" / "htdemucs" / name / "transcribed_notes.csv",
                    notes,
                )

            output_path = tmp_path / "model.json"
            train_model(
                ground_truth=str(gt_path),
                results_dir=str(results),
                output=str(output_path),
                order=2,
                min_recordings=2,
            )

            with output_path.open() as f:
                data = json.load(f)
            # Both have only 1 recording, so both excluded with min_recordings=2
            self.assertEqual(len(data["ragas"]), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lm_train.py -v`
Expected: `ImportError: cannot import name 'train_model' from 'raga_pipeline.language_model'`

- [ ] **Step 3: Implement `train_model`**

Add to `raga_pipeline/language_model/__init__.py`:

```python
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

# Add these imports at the top of the file alongside existing ones
# (they may already be imported, just ensure they're present)


def _load_notes_from_csv(csv_path: Path, tonic_midi: float) -> List[str]:
    """Read a transcription CSV and return LM tokens.

    Reads pitch_midi and sargam columns, converts to Note objects,
    then delegates to tokenize_notes_for_lm.
    """
    from raga_pipeline.sequence import Note, tokenize_notes_for_lm

    notes: List[Note] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [h.strip().lower() for h in (reader.fieldnames or [])]
        if not fieldnames:
            return []

        # Flexible column lookup
        def _find(candidates: List[str]) -> Optional[str]:
            for orig in (reader.fieldnames or []):
                if orig.strip().lower() in candidates:
                    return orig
            return None

        start_col = _find(["start", "start_time", "starttime"])
        end_col = _find(["end", "end_time", "endtime"])
        midi_col = _find(["pitch_midi", "pitchmidi", "midi"])
        sargam_col = _find(["sargam"])

        if not midi_col and not sargam_col:
            return []

        for row in reader:
            try:
                start = float(row[start_col]) if start_col else 0.0
                end = float(row[end_col]) if end_col else start + 0.1
                midi = float(row[midi_col]) if midi_col else 0.0
            except (ValueError, TypeError, KeyError):
                continue

            sargam = row.get(sargam_col, "") if sargam_col else ""
            notes.append(Note(
                start=start, end=end, pitch_midi=midi,
                pitch_hz=0.0, confidence=1.0, sargam=sargam,
            ))

    if not notes:
        return []

    return tokenize_notes_for_lm(notes, tonic_midi=tonic_midi)


# Tonic name -> MIDI pitch class mapping (reuse from motifs.py pattern)
_TONIC_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}


def _tonic_name_to_midi(tonic: str) -> float:
    """Convert tonic name like 'C#' to a MIDI note in octave 4."""
    pc = _TONIC_MAP.get(tonic.strip(), None)
    if pc is None:
        raise ValueError(f"Unknown tonic: {tonic!r}")
    return 60.0 + pc  # Octave 4


def train_model(
    ground_truth: str,
    results_dir: str,
    output: str,
    order: int = 5,
    smoothing: str = "add-k",
    smoothing_k: float = 0.01,
    lambdas: Optional[List[float]] = None,
    min_recordings: int = 3,
    transcription_source: str = "auto",
    quiet: bool = False,
) -> Dict[str, Any]:
    """Train n-gram language models from a corpus of transcriptions.

    Reads ground-truth CSV and discovers transcription CSVs under results_dir
    (same candidate discovery as motifs.py). Builds one NgramModel with
    per-raga n-gram tables.

    Returns the serialized model dict (also written to output path).
    """
    # Import candidate discovery from motifs
    from raga_pipeline.motifs import _discover_candidates, _read_ground_truth_rows

    gt_path = Path(ground_truth)
    results_path = Path(results_dir)
    output_path = Path(output)

    gt_rows = _read_ground_truth_rows(gt_path)
    stem_map, basename_map = _discover_candidates(results_path)

    model = NgramModel(
        order=order,
        smoothing=smoothing,
        smoothing_k=smoothing_k,
        lambdas=lambdas,
    )

    raga_recording_counts: Dict[str, int] = defaultdict(int)
    missing: List[str] = []

    for row in gt_rows:
        # Find transcription CSV for this recording
        stem_key = row.filename.lower()
        # Strip extension for stem matching
        for ext in (".mp3", ".wav", ".m4a", ".flac"):
            if stem_key.endswith(ext):
                stem_key = stem_key[: -len(ext)]
                break

        candidates = stem_map.get(stem_key) or basename_map.get(stem_key)
        if not candidates:
            # Try with original filename too
            candidates = basename_map.get(row.filename.lower())
        if not candidates:
            missing.append(row.filename)
            continue

        candidate = candidates[0]
        csv_path, _ = candidate.resolve(transcription_source)
        if csv_path is None or not csv_path.exists():
            missing.append(row.filename)
            continue

        tonic_midi = _tonic_name_to_midi(row.tonic)
        tokens = _load_notes_from_csv(csv_path, tonic_midi)
        if not tokens:
            continue

        raga_recording_counts[row.raga] += 1
        model.add_sequence(row.raga, tokens)

    # Remove ragas below min_recordings threshold
    ragas_to_remove = [
        raga for raga, count in raga_recording_counts.items()
        if count < min_recordings
    ]
    for raga in ragas_to_remove:
        if raga in model._counts:
            del model._counts[raga]
            del model._context_counts[raga]
        if raga in model._recording_counts:
            del model._recording_counts[raga]
        if raga in model._token_counts:
            del model._token_counts[raga]

    model.finalize()

    result = model.to_dict()
    result["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["metadata"]["ground_truth"] = str(gt_path)
    result["metadata"]["results_dir"] = str(results_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if not quiet:
        print(f"Trained {len(model.ragas())} ragas, order={order}")
        if missing:
            print(f"  {len(missing)} recordings not found")

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lm_train.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add raga_pipeline/language_model/__init__.py tests/test_lm_train.py
git commit -m "feat: add train_model function for corpus-level LM training"
```

---

### Task 5: Scoring function -- `score_transcription`

**Files:**
- Modify: `raga_pipeline/language_model/__init__.py` (add `score_transcription` function)
- Create: `tests/test_lm_score.py`

Loads a trained model, tokenizes a new transcription, and returns ranked ragas with optional segment-level scores.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lm_score.py
import csv
import json
import tempfile
import unittest
from pathlib import Path

from raga_pipeline.language_model import NgramModel, score_transcription


def _make_model_file(tmp_path: Path) -> Path:
    """Create a trained model JSON for testing."""
    model = NgramModel(order=3, smoothing="add-k", smoothing_k=0.01)
    # Yaman: Sa Re Ga ma Pa
    model.add_sequence("Yaman", ["<BOS>", "Sa", "Re", "Ga", "ma", "Pa"] * 50)
    # Bhairav: Sa re Ga ma dha
    model.add_sequence("Bhairav", ["<BOS>", "Sa", "re", "Ga", "ma", "dha"] * 50)
    model.finalize()

    model_path = tmp_path / "model.json"
    with model_path.open("w") as f:
        json.dump(model.to_dict(), f)
    return model_path


def _write_transcription(path: Path, notes: list[tuple[float, float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "end", "pitch_midi", "sargam"])
        writer.writeheader()
        for start, end, midi, sargam in notes:
            writer.writerow({
                "start": f"{start:.3f}", "end": f"{end:.3f}",
                "pitch_midi": f"{midi:.1f}", "sargam": sargam,
            })


class TestScoreTranscription(unittest.TestCase):
    """Tests for score_transcription function."""

    def test_correct_raga_ranks_first(self):
        """A Yaman-like transcription should rank Yaman first."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = _make_model_file(tmp_path)

            # Yaman-like transcription: Sa Re Ga ma Pa
            yaman_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                           enumerate([(60, "Sa"), (62, "Re"), (64, "Ga"), (65, "ma"), (67, "Pa")] * 10)]
            trans_path = tmp_path / "transcription.csv"
            _write_transcription(trans_path, yaman_notes)

            result = score_transcription(
                model_path=str(model_path),
                transcription_path=str(trans_path),
                tonic="C",
            )

            self.assertEqual(result["rankings"][0]["raga"], "Yaman")
            self.assertEqual(result["rankings"][0]["rank"], 1)
            self.assertGreater(result["rankings"][0]["score"], result["rankings"][1]["score"])

    def test_segment_level_scores(self):
        """With segments=True, result includes segment-level breakdown."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = _make_model_file(tmp_path)

            notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                     enumerate([(60, "Sa"), (62, "Re"), (64, "Ga")] * 100)]
            trans_path = tmp_path / "transcription.csv"
            _write_transcription(trans_path, notes)

            result = score_transcription(
                model_path=str(model_path),
                transcription_path=str(trans_path),
                tonic="C",
                segments=True,
                segment_window=50,
            )

            self.assertIn("segments", result)
            self.assertGreater(len(result["segments"]), 0)
            seg = result["segments"][0]
            self.assertIn("start_time", seg)
            self.assertIn("end_time", seg)
            self.assertIn("top_ragas", seg)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lm_score.py -v`
Expected: `ImportError: cannot import name 'score_transcription' from 'raga_pipeline.language_model'`

- [ ] **Step 3: Implement `score_transcription`**

Add to `raga_pipeline/language_model/__init__.py`:

```python
def score_transcription(
    model_path: str,
    transcription_path: str,
    tonic: str,
    segments: bool = False,
    segment_window: int = 100,
    top_k: int = 5,
    output: Optional[str] = None,
) -> Dict[str, Any]:
    """Score a transcription against a trained n-gram model.

    Returns ranked ragas with scores, and optionally segment-level breakdown.
    """
    model_file = Path(model_path)
    with model_file.open() as f:
        model = NgramModel.from_dict(json.load(f))

    tonic_midi = _tonic_name_to_midi(tonic)
    tokens = _load_notes_from_csv(Path(transcription_path), tonic_midi)

    if not tokens:
        return {"error": "no tokens extracted from transcription"}

    # Read note timestamps for segment time-mapping
    note_times: List[Tuple[float, float]] = []
    with Path(transcription_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fnames = [h.strip().lower() for h in (reader.fieldnames or [])]
        start_col = next((h for h in (reader.fieldnames or []) if h.strip().lower() in ("start", "start_time")), None)
        end_col = next((h for h in (reader.fieldnames or []) if h.strip().lower() in ("end", "end_time")), None)
        for row in reader:
            try:
                s = float(row[start_col]) if start_col else 0.0
                e = float(row[end_col]) if end_col else s + 0.1
                note_times.append((s, e))
            except (ValueError, TypeError, KeyError):
                note_times.append((0.0, 0.0))

    # Whole-recording ranking
    rankings_raw = model.rank_ragas(tokens)
    rankings = [
        {"raga": raga, "score": round(score, 4), "rank": i + 1}
        for i, (raga, score) in enumerate(rankings_raw[:top_k])
    ]

    result: Dict[str, Any] = {
        "transcription": str(transcription_path),
        "tonic": tonic,
        "total_tokens": len(tokens),
        "rankings": rankings,
    }

    # Segment-level scoring
    if segments and len(tokens) > 0:
        step = max(1, segment_window // 2)
        segs: List[Dict[str, Any]] = []

        for seg_start in range(0, len(tokens), step):
            seg_end = min(seg_start + segment_window, len(tokens))
            seg_tokens = tokens[seg_start:seg_end]
            if not seg_tokens:
                break

            seg_rankings = model.rank_ragas(seg_tokens)

            # Map token indices back to note timestamps
            # Tokens include <BOS> markers which don't map to notes,
            # so we track non-BOS token positions
            non_bos_start = sum(1 for t in tokens[:seg_start] if t != "<BOS>")
            non_bos_end = sum(1 for t in tokens[:seg_end] if t != "<BOS>")

            start_time = note_times[min(non_bos_start, len(note_times) - 1)][0] if note_times else 0.0
            end_time = note_times[min(non_bos_end - 1, len(note_times) - 1)][1] if note_times else 0.0

            segs.append({
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "token_range": [seg_start, seg_end],
                "top_ragas": [
                    {"raga": r, "score": round(s, 4)}
                    for r, s in seg_rankings[:3]
                ],
            })

        result["segments"] = segs

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lm_score.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add raga_pipeline/language_model/__init__.py tests/test_lm_score.py
git commit -m "feat: add score_transcription function with segment-level scoring"
```

---

### Task 6: Evaluation function -- `evaluate_model`

**Files:**
- Modify: `raga_pipeline/language_model/__init__.py` (add `evaluate_model` function)
- Create: `tests/test_lm_evaluate.py`

Leave-one-out cross-validation with order sweep support.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lm_evaluate.py
import csv
import tempfile
import unittest
from pathlib import Path

from raga_pipeline.language_model import evaluate_model


def _write_gt_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Filename", "Raga", "Tonic"])
        writer.writeheader()
        for filename, raga, tonic in rows:
            writer.writerow({"Filename": filename, "Raga": raga, "Tonic": tonic})


def _write_transcription_csv(path: Path, rows: list[tuple[float, float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "end", "pitch_midi", "sargam"])
        writer.writeheader()
        for start, end, midi, sargam in rows:
            writer.writerow({
                "start": f"{start:.3f}", "end": f"{end:.3f}",
                "pitch_midi": f"{midi:.1f}", "sargam": sargam,
            })


class TestEvaluateModel(unittest.TestCase):
    """Tests for leave-one-out evaluation."""

    def _build_corpus(self, tmp_path: Path):
        """Build a small 2-raga corpus with 3 recordings each."""
        gt_path = tmp_path / "gt.csv"
        results = tmp_path / "results"

        gt_rows = []
        # Yaman (Sa Re Ga ma Pa): distinct from Bhairav (Sa re Ga ma dha)
        yaman_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                       enumerate([(60, "Sa"), (62, "Re"), (64, "Ga"), (65, "ma"), (67, "Pa")] * 30)]
        bhairav_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                         enumerate([(60, "Sa"), (61, "re"), (64, "Ga"), (65, "ma"), (68, "dha")] * 30)]

        for i in range(3):
            name = f"yaman_{i}.mp3"
            gt_rows.append((name, "Yaman", "C"))
            _write_transcription_csv(
                results / "demucs" / "htdemucs" / f"yaman_{i}" / "transcribed_notes.csv",
                yaman_notes,
            )

        for i in range(3):
            name = f"bhairav_{i}.mp3"
            gt_rows.append((name, "Bhairav", "C"))
            _write_transcription_csv(
                results / "demucs" / "htdemucs" / f"bhairav_{i}" / "transcribed_notes.csv",
                bhairav_notes,
            )

        _write_gt_csv(gt_path, gt_rows)
        return gt_path, results

    def test_basic_evaluation(self):
        """Evaluation produces results CSV with expected columns."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_path, results = self._build_corpus(tmp_path)
            output_path = tmp_path / "eval.csv"

            summary = evaluate_model(
                ground_truth=str(gt_path),
                results_dir=str(results),
                output=str(output_path),
                order=2,
                min_recordings=1,
            )

            self.assertTrue(output_path.exists())
            self.assertIn("top1_accuracy", summary)
            self.assertIn("top3_accuracy", summary)
            self.assertIn("mrr", summary)

            # With very distinct ragas and enough data, expect decent accuracy
            self.assertGreater(summary["top1_accuracy"], 0.5)

    def test_sweep_orders(self):
        """Order sweep produces per-order results."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_path, results = self._build_corpus(tmp_path)
            output_path = tmp_path / "eval.csv"

            summary = evaluate_model(
                ground_truth=str(gt_path),
                results_dir=str(results),
                output=str(output_path),
                order=3,
                min_recordings=1,
                sweep_orders=[2, 3],
            )

            self.assertIn("sweep_results", summary)
            self.assertEqual(len(summary["sweep_results"]), 2)
            self.assertIn(2, [s["order"] for s in summary["sweep_results"]])
            self.assertIn(3, [s["order"] for s in summary["sweep_results"]])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lm_evaluate.py -v`
Expected: `ImportError: cannot import name 'evaluate_model' from 'raga_pipeline.language_model'`

- [ ] **Step 3: Implement `evaluate_model`**

Add to `raga_pipeline/language_model/__init__.py`:

```python
def _run_leave_one_out(
    gt_rows: list,
    recordings: Dict[str, Tuple[str, str, List[str]]],
    order: int,
    smoothing: str,
    smoothing_k: float,
    lambdas: Optional[List[float]],
    min_recordings: int,
) -> List[Dict[str, Any]]:
    """Run leave-one-out CV. Returns per-recording result dicts.

    recordings: mapping of filename -> (raga, tonic, tokens)
    """
    # Group by raga
    raga_to_filenames: Dict[str, List[str]] = defaultdict(list)
    for fname, (raga, tonic, tokens) in recordings.items():
        raga_to_filenames[raga].append(fname)

    results = []
    filenames = list(recordings.keys())

    for held_out in filenames:
        held_raga, held_tonic, held_tokens = recordings[held_out]
        if not held_tokens:
            continue

        # Build model excluding held-out recording
        model = NgramModel(
            order=order, smoothing=smoothing,
            smoothing_k=smoothing_k, lambdas=lambdas,
        )
        raga_counts: Dict[str, int] = defaultdict(int)

        for fname, (raga, tonic, tokens) in recordings.items():
            if fname == held_out:
                continue
            if tokens:
                model.add_sequence(raga, tokens)
                raga_counts[raga] += 1

        # Remove ragas below threshold
        for raga in list(model._counts.keys()):
            if raga_counts.get(raga, 0) < min_recordings:
                del model._counts[raga]
                del model._context_counts[raga]

        model.finalize()

        if not model.ragas():
            continue

        ranked = model.rank_ragas(held_tokens)
        raga_names = [r for r, _ in ranked]
        true_rank = (raga_names.index(held_raga) + 1) if held_raga in raga_names else len(ranked) + 1

        results.append({
            "filename": held_out,
            "true_raga": held_raga,
            "predicted_raga": ranked[0][0] if ranked else "",
            "correct": ranked[0][0] == held_raga if ranked else False,
            "true_raga_rank": true_rank,
            "score_top1": round(ranked[0][1], 4) if ranked else 0.0,
            "raga_top1": ranked[0][0] if ranked else "",
            "raga_top2": ranked[1][0] if len(ranked) > 1 else "",
            "raga_top3": ranked[2][0] if len(ranked) > 2 else "",
            "score_top2": round(ranked[1][1], 4) if len(ranked) > 1 else 0.0,
            "score_top3": round(ranked[2][1], 4) if len(ranked) > 2 else 0.0,
        })

    return results


def _compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy/MRR from per-recording results."""
    if not results:
        return {"top1_accuracy": 0.0, "top3_accuracy": 0.0, "mrr": 0.0, "total": 0}

    n = len(results)
    top1 = sum(1 for r in results if r["correct"]) / n
    top3 = sum(1 for r in results if r["true_raga_rank"] <= 3) / n
    mrr = sum(1.0 / r["true_raga_rank"] for r in results) / n

    return {
        "top1_accuracy": round(top1, 4),
        "top3_accuracy": round(top3, 4),
        "mrr": round(mrr, 4),
        "total": n,
    }


def evaluate_model(
    ground_truth: str,
    results_dir: str,
    output: str,
    order: int = 5,
    smoothing: str = "add-k",
    smoothing_k: float = 0.01,
    lambdas: Optional[List[float]] = None,
    min_recordings: int = 3,
    transcription_source: str = "auto",
    sweep_orders: Optional[List[int]] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Leave-one-out cross-validation evaluation.

    Returns summary dict with accuracy metrics. Writes per-recording CSV.
    """
    from raga_pipeline.motifs import _discover_candidates, _read_ground_truth_rows

    gt_path = Path(ground_truth)
    results_path = Path(results_dir)
    output_path = Path(output)

    gt_rows = _read_ground_truth_rows(gt_path)
    stem_map, basename_map = _discover_candidates(results_path)

    # Load all recordings once
    recordings: Dict[str, Tuple[str, str, List[str]]] = {}
    for row in gt_rows:
        stem_key = row.filename.lower()
        for ext in (".mp3", ".wav", ".m4a", ".flac"):
            if stem_key.endswith(ext):
                stem_key = stem_key[: -len(ext)]
                break

        candidates = stem_map.get(stem_key) or basename_map.get(stem_key)
        if not candidates:
            candidates = basename_map.get(row.filename.lower())
        if not candidates:
            continue

        csv_path, _ = candidates[0].resolve(transcription_source)
        if csv_path is None or not csv_path.exists():
            continue

        tonic_midi = _tonic_name_to_midi(row.tonic)
        tokens = _load_notes_from_csv(csv_path, tonic_midi)
        if tokens:
            recordings[row.filename] = (row.raga, row.tonic, tokens)

    # Run evaluation (with optional order sweep)
    if sweep_orders:
        sweep_results = []
        all_results = []
        for o in sweep_orders:
            lam = [1.0 / o] * o if lambdas is None else lambdas
            results = _run_leave_one_out(
                gt_rows, recordings, order=o, smoothing=smoothing,
                smoothing_k=smoothing_k, lambdas=lam, min_recordings=min_recordings,
            )
            summary = _compute_summary(results)
            summary["order"] = o
            sweep_results.append(summary)
            all_results.extend(results)

            if not quiet:
                print(f"  Order {o}: top1={summary['top1_accuracy']:.1%}, "
                      f"top3={summary['top3_accuracy']:.1%}, MRR={summary['mrr']:.3f}")

        # Write CSV from the last order's results (or all)
        _write_eval_csv(output_path, all_results)
        final_summary = _compute_summary(all_results)
        final_summary["sweep_results"] = sweep_results
        return final_summary

    else:
        lam = lambdas if lambdas else [1.0 / order] * order
        results = _run_leave_one_out(
            gt_rows, recordings, order=order, smoothing=smoothing,
            smoothing_k=smoothing_k, lambdas=lam, min_recordings=min_recordings,
        )
        _write_eval_csv(output_path, results)
        summary = _compute_summary(results)

        if not quiet:
            print(f"Top-1: {summary['top1_accuracy']:.1%}, "
                  f"Top-3: {summary['top3_accuracy']:.1%}, "
                  f"MRR: {summary['mrr']:.3f} ({summary['total']} recordings)")

        return summary


def _write_eval_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    """Write per-recording evaluation results to CSV."""
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename", "true_raga", "predicted_raga", "correct", "true_raga_rank",
        "score_top1", "score_top2", "score_top3", "raga_top1", "raga_top2", "raga_top3",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lm_evaluate.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add raga_pipeline/language_model/__init__.py tests/test_lm_evaluate.py
git commit -m "feat: add evaluate_model with leave-one-out CV and order sweep"
```

---

### Task 7: CLI entry point

**Files:**
- Create: `raga_pipeline/language_model/__main__.py`
- Create: `tests/test_lm_cli.py`

The `python -m raga_pipeline.language_model` CLI with train/score/evaluate subcommands, mirroring `motifs.py` patterns.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lm_cli.py
import csv
import json
import tempfile
import unittest
from pathlib import Path

from raga_pipeline.language_model.__main__ import main


def _write_gt_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Filename", "Raga", "Tonic"])
        writer.writeheader()
        for filename, raga, tonic in rows:
            writer.writerow({"Filename": filename, "Raga": raga, "Tonic": tonic})


def _write_transcription_csv(path: Path, rows: list[tuple[float, float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "end", "pitch_midi", "sargam"])
        writer.writeheader()
        for start, end, midi, sargam in rows:
            writer.writerow({
                "start": f"{start:.3f}", "end": f"{end:.3f}",
                "pitch_midi": f"{midi:.1f}", "sargam": sargam,
            })


class TestLMCli(unittest.TestCase):
    """Smoke tests for the CLI entry point."""

    def test_train_and_score_smoke(self):
        """CLI train then score runs without error."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            gt_path = tmp_path / "gt.csv"
            _write_gt_csv(gt_path, [
                ("song_a.mp3", "Yaman", "C"),
                ("song_b.mp3", "Yaman", "C"),
                ("song_c.mp3", "Yaman", "C"),
            ])

            results = tmp_path / "results"
            notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                     enumerate([(60, "Sa"), (62, "Re"), (64, "Ga")] * 30)]
            for name in ["song_a", "song_b", "song_c"]:
                _write_transcription_csv(
                    results / "demucs" / "htdemucs" / name / "transcribed_notes.csv",
                    notes,
                )

            model_path = tmp_path / "model.json"

            # Train
            rc = main([
                "train",
                "--gt", str(gt_path),
                "--results-dir", str(results),
                "--output", str(model_path),
                "--order", "2",
                "--min-recordings", "1",
                "--quiet",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(model_path.exists())

            # Score
            trans_path = results / "demucs" / "htdemucs" / "song_a" / "transcribed_notes.csv"
            rc = main([
                "score",
                "--model", str(model_path),
                "--transcription", str(trans_path),
                "--tonic", "C",
            ])
            self.assertEqual(rc, 0)

    def test_evaluate_smoke(self):
        """CLI evaluate runs without error."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            gt_path = tmp_path / "gt.csv"
            _write_gt_csv(gt_path, [
                ("s1.mp3", "Yaman", "C"),
                ("s2.mp3", "Yaman", "C"),
                ("s3.mp3", "Bhairav", "C"),
                ("s4.mp3", "Bhairav", "C"),
            ])

            results = tmp_path / "results"
            y_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                       enumerate([(60, "Sa"), (62, "Re"), (64, "Ga")] * 30)]
            b_notes = [(i * 0.3, (i + 1) * 0.3, m, s) for i, (m, s) in
                       enumerate([(60, "Sa"), (61, "re"), (64, "Ga")] * 30)]

            for name in ["s1", "s2"]:
                _write_transcription_csv(
                    results / "demucs" / "htdemucs" / name / "transcribed_notes.csv", y_notes)
            for name in ["s3", "s4"]:
                _write_transcription_csv(
                    results / "demucs" / "htdemucs" / name / "transcribed_notes.csv", b_notes)

            eval_path = tmp_path / "eval.csv"
            rc = main([
                "evaluate",
                "--gt", str(gt_path),
                "--results-dir", str(results),
                "--output", str(eval_path),
                "--order", "2",
                "--min-recordings", "1",
                "--quiet",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(eval_path.exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lm_cli.py -v`
Expected: `ModuleNotFoundError: No module named 'raga_pipeline.language_model.__main__'`

- [ ] **Step 3: Implement `__main__.py`**

```python
# raga_pipeline/language_model/__main__.py
"""CLI entry point: python -m raga_pipeline.language_model train|score|evaluate"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from raga_pipeline.language_model import (
    evaluate_model,
    score_transcription,
    train_model,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-raga n-gram language models for raga detection.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train = subparsers.add_parser("train", help="Train n-gram models from a labeled corpus.")
    train.add_argument("--gt", required=True, help="Path to ground-truth CSV.")
    train.add_argument("--results-dir", required=True, help="Root directory with transcription CSVs.")
    train.add_argument("--output", default="raga_ngram_model.json", help="Output model JSON path.")
    train.add_argument("--order", type=int, default=5, help="Maximum n-gram order (default: 5).")
    train.add_argument("--smoothing", choices=["add-k", "kneser-ney"], default="add-k", help="Smoothing method.")
    train.add_argument("--smoothing-k", type=float, default=0.01, help="k for add-k smoothing (default: 0.01).")
    train.add_argument("--min-recordings", type=int, default=3, help="Skip ragas with fewer recordings.")
    train.add_argument(
        "--lambdas", default=None,
        help="Interpolation weights, comma-separated highest-order-first (default: equal).",
    )
    train.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    # --- score ---
    score = subparsers.add_parser("score", help="Score a transcription against a trained model.")
    score.add_argument("--model", required=True, help="Path to trained model JSON.")
    score.add_argument("--transcription", required=True, help="Path to transcription CSV.")
    score.add_argument("--tonic", required=True, help="Tonic note name (e.g., C#, D).")
    score.add_argument("--segments", action="store_true", help="Include segment-level scores.")
    score.add_argument("--segment-window", type=int, default=100, help="Tokens per segment (default: 100).")
    score.add_argument("--top-k", type=int, default=5, help="Number of top ragas (default: 5).")
    score.add_argument("--output", default=None, help="Output JSON path (default: stdout).")

    # --- evaluate ---
    evaluate = subparsers.add_parser("evaluate", help="Leave-one-out cross-validation.")
    evaluate.add_argument("--gt", required=True, help="Path to ground-truth CSV.")
    evaluate.add_argument("--results-dir", required=True, help="Root directory with transcription CSVs.")
    evaluate.add_argument("--output", default="eval_results.csv", help="Output CSV path.")
    evaluate.add_argument("--order", type=int, default=5, help="Maximum n-gram order (default: 5).")
    evaluate.add_argument("--smoothing", choices=["add-k", "kneser-ney"], default="add-k", help="Smoothing method.")
    evaluate.add_argument("--smoothing-k", type=float, default=0.01, help="k for add-k smoothing.")
    evaluate.add_argument("--min-recordings", type=int, default=3, help="Skip ragas with fewer recordings.")
    evaluate.add_argument(
        "--lambdas", default=None,
        help="Interpolation weights, comma-separated (default: equal).",
    )
    evaluate.add_argument(
        "--sweep-orders", default=None,
        help="Comma-separated orders to sweep, e.g. '2,3,4,5,6'.",
    )
    evaluate.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    return parser


def _parse_lambdas(raw: Optional[str], order: int) -> Optional[list[float]]:
    if raw is None:
        return None
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != order:
        raise ValueError(f"--lambdas must have {order} values (one per order), got {len(parts)}")
    return parts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        lambdas = _parse_lambdas(args.lambdas, args.order)
        train_model(
            ground_truth=args.gt,
            results_dir=args.results_dir,
            output=args.output,
            order=args.order,
            smoothing=args.smoothing,
            smoothing_k=args.smoothing_k,
            lambdas=lambdas,
            min_recordings=args.min_recordings,
            quiet=args.quiet,
        )
        return 0

    if args.command == "score":
        result = score_transcription(
            model_path=args.model,
            transcription_path=args.transcription,
            tonic=args.tonic,
            segments=args.segments,
            segment_window=args.segment_window,
            top_k=args.top_k,
            output=args.output,
        )
        if not args.output:
            print(json.dumps(result, indent=2))
        return 0

    if args.command == "evaluate":
        lambdas = _parse_lambdas(args.lambdas, args.order) if args.lambdas else None
        sweep = [int(x) for x in args.sweep_orders.split(",")] if args.sweep_orders else None
        evaluate_model(
            ground_truth=args.gt,
            results_dir=args.results_dir,
            output=args.output,
            order=args.order,
            smoothing=args.smoothing,
            smoothing_k=args.smoothing_k,
            lambdas=lambdas,
            min_recordings=args.min_recordings,
            sweep_orders=sweep,
            quiet=args.quiet,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lm_cli.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Run all LM tests together**

Run: `python -m pytest tests/test_lm_tokenizer.py tests/test_ngram_model.py tests/test_lm_train.py tests/test_lm_score.py tests/test_lm_evaluate.py tests/test_lm_cli.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add raga_pipeline/language_model/__main__.py tests/test_lm_cli.py
git commit -m "feat: add CLI entry point for language model (train/score/evaluate)"
```

---

### Task 8: Update documentation

**Files:**
- Modify: `raga_pipeline/CHANGELOG.md`
- Modify: `raga_pipeline/LLM_REFERENCE.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CHANGELOG.md**

Add a new dated entry at the top of `raga_pipeline/CHANGELOG.md`:

```markdown
## 2026-03-30

- Added n-gram language model for raga detection (`raga_pipeline/language_model/`).
  - Shared tokenizer `tokenize_notes_for_lm` in `sequence.py` with octave-marked sargam tokens.
  - `NgramModel` class with add-k smoothing, interpolated scoring, JSON serialization.
  - `train_model`: builds per-raga models from ground-truth CSV + transcription CSVs.
  - `score_transcription`: classifies recordings with optional segment-level confidence.
  - `evaluate_model`: leave-one-out cross-validation with order sweep support.
  - CLI: `python -m raga_pipeline.language_model train|score|evaluate`.
```

- [ ] **Step 2: Update LLM_REFERENCE.md**

Add a new section after the batch processing section:

```markdown
### 9. Language Model for Raga Detection (New)

N-gram language model approach to raga classification. Builds per-raga probability distributions over sargam token sequences, classifies by perplexity.

**Token format:** Octave-marked sargam: `Sa`, `Re'` (lower), `Ga''` (upper), with `<BOS>` at phrase boundaries.

**CLI:**
```bash
python -m raga_pipeline.language_model train --gt gt.csv --results-dir results/ --output model.json
python -m raga_pipeline.language_model score --model model.json --transcription notes.csv --tonic C#
python -m raga_pipeline.language_model evaluate --gt gt.csv --results-dir results/ --sweep-orders 2,3,4,5,6
```

**Shared tokenizer:** `sequence.py:tokenize_notes_for_lm(notes, tonic_midi, phrase_gap_sec)` -- reusable across motif mining and language model.
```

- [ ] **Step 3: Update CLAUDE.md**

Add to the core modules table:

```markdown
| `language_model/` | Per-raga n-gram LM training, scoring, evaluation, CLI |
```

Add to the code navigation table:

```markdown
| N-gram language model training, scoring, evaluation | `raga_pipeline/language_model/` |
```

Add to the test routing table:

```markdown
| Language model tokenizer/model/CLI | `tests/test_lm_*.py` |
```

- [ ] **Step 4: Commit**

```bash
git add raga_pipeline/CHANGELOG.md raga_pipeline/LLM_REFERENCE.md CLAUDE.md
git commit -m "docs: add language model to CHANGELOG, LLM_REFERENCE, and CLAUDE.md"
```

---

### Task 9: Run full test suite

**Files:** None (verification only)

- [ ] **Step 1: Run all project tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS (both new LM tests and existing tests)

- [ ] **Step 2: Run type checking**

Run: `python -m mypy raga_pipeline/language_model/`
Expected: No errors (or only pre-existing warnings from `ignore_missing_imports`)

- [ ] **Step 3: Final commit if any fixes needed**

If any tests or type errors were found and fixed, commit the fixes.
