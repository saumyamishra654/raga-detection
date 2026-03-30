"""Per-raga n-gram language models for raga detection. CLI entry point: python -m raga_pipeline.language_model train|score|evaluate ..."""

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class NgramModel:
    """Per-raga n-gram language model with add-k smoothing and linear interpolation.

    Stores raw n-gram counts per raga (not probabilities) so that the model
    can be serialized and later updated if needed.  Probabilities are computed
    on-the-fly from counts at scoring time.

    Attributes:
        order: Maximum n-gram order (1 = unigram only, 2 = bigram, ...).
        smoothing: Smoothing scheme; currently only "add-k" is supported.
        smoothing_k: The k parameter for add-k smoothing.
        lambdas: Interpolation weights (length == order); index 0 = unigram.
    """

    # Separator used when serialising n-gram tuples to JSON keys.
    _KEY_SEP = "|"

    def __init__(
        self,
        order: int = 5,
        smoothing: str = "add-k",
        smoothing_k: float = 0.01,
        lambdas: Optional[List[float]] = None,
    ) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.smoothing = smoothing
        self.smoothing_k = smoothing_k

        if lambdas is not None:
            if len(lambdas) != order:
                raise ValueError("lambdas must have length == order")
            total = sum(lambdas)
            self.lambdas = [w / total for w in lambdas]
        else:
            self.lambdas = [1.0 / order] * order

        # _counts[raga][n][ngram_tuple] -> int  (n in 1..order)
        self._counts: Dict[str, Dict[int, Dict[tuple, int]]] = {}
        # _context_counts[raga][n][context_tuple] -> int  (n in 2..order)
        self._context_counts: Dict[str, Dict[int, Dict[tuple, int]]] = {}
        # per-raga global token counts (for unigram denominator)
        self._token_counts: Dict[str, int] = {}
        # per-raga recording counts
        self._recording_counts: Dict[str, int] = {}
        # global vocabulary (union across all ragas)
        self._vocabulary: set = set()
        self._finalized: bool = False

    # ------------------------------------------------------------------
    # Building the model
    # ------------------------------------------------------------------

    def add_sequence(self, raga: str, tokens: List[str]) -> None:
        """Add a tokenised recording to the model for *raga*.

        Empty sequences are silently ignored (no raga entry is created).
        """
        if not tokens:
            return

        # Initialise per-raga structures on first access.
        if raga not in self._counts:
            self._counts[raga] = {n: defaultdict(int) for n in range(1, self.order + 1)}
            self._context_counts[raga] = {n: defaultdict(int) for n in range(2, self.order + 1)}
            self._token_counts[raga] = 0
            self._recording_counts[raga] = 0

        self._recording_counts[raga] += 1
        self._finalized = False

        counts = self._counts[raga]
        ctx_counts = self._context_counts[raga]

        for i, token in enumerate(tokens):
            self._vocabulary.add(token)
            self._token_counts[raga] += 1

            # Count n-grams of every order that end at position i.
            for n in range(1, self.order + 1):
                if i - n + 1 < 0:
                    break
                ngram = tuple(tokens[i - n + 1 : i + 1])
                counts[n][ngram] += 1

                # Context count: the (n-1)-gram that precedes the last token.
                if n >= 2:
                    context = ngram[:-1]
                    ctx_counts[n][context] += 1

    def finalize(self) -> None:
        """Mark the model as ready for scoring.  No-op if already finalised."""
        self._finalized = True

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def ragas(self) -> List[str]:
        """Return sorted list of raga names that have training data."""
        return sorted(self._counts.keys())

    def get_counts(self, raga: str) -> Dict[int, Dict[tuple, int]]:
        """Return the raw counts dict for *raga* (used in tests).

        Keys are n-gram orders (1..order); values are dicts mapping
        n-gram tuples to integer counts.

        For order-2+ counts the dict is keyed by full n-gram tuple;
        for order-1 counts it is keyed by 1-tuple e.g. ``("Sa",)``.
        """
        if raga not in self._counts:
            return {}
        # Return plain dicts (not defaultdicts) for clean test assertions.
        return {n: dict(d) for n, d in self._counts[raga].items()}

    def vocabulary_size(self) -> int:
        """Return the number of distinct tokens seen across all ragas."""
        return len(self._vocabulary)

    # ------------------------------------------------------------------
    # Probability computation
    # ------------------------------------------------------------------

    def _smoothed_prob_n(self, raga: str, n: int, token: str, context: tuple) -> float:
        """Add-k smoothed probability P_n(token | context) for a specific order.

        For the unigram case (n=1) the context is ignored.
        """
        k = self.smoothing_k
        V = max(self.vocabulary_size(), 1)
        raga_counts = self._counts.get(raga, {})

        if n == 1:
            # Unigram: count(token) / total_tokens, with add-k smoothing.
            total = self._token_counts.get(raga, 0)
            count = raga_counts.get(1, {}).get((token,), 0)
            return (count + k) / (total + k * V)
        else:
            # Higher order: count(context + token) / count(context).
            ctx_count = self._context_counts.get(raga, {}).get(n, {}).get(context, 0)
            ngram = context + (token,)
            ng_count = raga_counts.get(n, {}).get(ngram, 0)
            return (ng_count + k) / (ctx_count + k * V)

    def log_prob(self, raga: str, token: str, context: tuple) -> float:
        """Compute interpolated smoothed log P(token | context) for *raga*.

        *context* should be the (n-1)-gram immediately preceding *token*
        (for the highest order used).  Shorter contexts are derived from it
        by taking the appropriate suffix.

        Returns log probability (natural log, always negative).
        """
        interpolated = 0.0
        for n in range(1, self.order + 1):
            lam = self.lambdas[n - 1]
            if n == 1:
                ctx = ()
            else:
                # Use the last (n-1) tokens of the provided context.
                ctx = context[-(n - 1):] if len(context) >= n - 1 else context
            p_n = self._smoothed_prob_n(raga, n, token, ctx)
            interpolated += lam * p_n

        # Guard against numerical underflow / zero probability.
        if interpolated <= 0.0:
            return -1e30
        return math.log(interpolated)

    def score_sequence(self, raga: str, tokens: List[str]) -> float:
        """Return the average log-likelihood per token for *tokens* under *raga*.

        Tokens at position i are scored using the preceding (order-1) tokens
        as context.  The sequence must have at least 2 tokens to be meaningful.

        Returns 0.0 for sequences shorter than 2 tokens.
        """
        if len(tokens) < 2:
            return 0.0

        total_ll = 0.0
        scored = 0
        for i in range(1, len(tokens)):
            token = tokens[i]
            # Context: up to (order-1) preceding tokens.
            ctx_start = max(0, i - (self.order - 1))
            context = tuple(tokens[ctx_start:i])
            total_ll += self.log_prob(raga, token, context)
            scored += 1

        return total_ll / scored if scored > 0 else 0.0

    def rank_ragas(self, tokens: List[str]) -> List[Tuple[str, float]]:
        """Score *tokens* against every raga; return sorted (raga, score) pairs.

        Highest score first.
        """
        scores = [(raga, self.score_sequence(raga, tokens)) for raga in self.ragas()]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the model to a JSON-compatible dict.

        N-gram tuple keys are joined with ``|`` (e.g. ``"Sa|Re|Ga"``).
        Counts (not probabilities) are stored so that the model can be
        fully reconstructed.
        """
        ragas_data: dict = {}
        for raga in self._counts:
            raga_entry: dict = {
                "counts": {},
                "context_counts": {},
                "token_count": self._token_counts.get(raga, 0),
                "recording_count": self._recording_counts.get(raga, 0),
            }
            for n, ngram_dict in self._counts[raga].items():
                raga_entry["counts"][str(n)] = {
                    self._KEY_SEP.join(k): v for k, v in ngram_dict.items()
                }
            for n, ctx_dict in self._context_counts[raga].items():
                raga_entry["context_counts"][str(n)] = {
                    self._KEY_SEP.join(k): v for k, v in ctx_dict.items()
                }
            ragas_data[raga] = raga_entry

        return {
            "metadata": {
                "order": self.order,
                "smoothing": self.smoothing,
                "smoothing_params": {"k": self.smoothing_k},
                "lambdas": self.lambdas,
                "training_ragas": len(self._counts),
                "vocabulary": sorted(self._vocabulary),
            },
            "ragas": ragas_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NgramModel":
        """Deserialise a model previously produced by :meth:`to_dict`."""
        meta = data["metadata"]
        model = cls(
            order=meta["order"],
            smoothing=meta["smoothing"],
            smoothing_k=meta["smoothing_params"]["k"],
            lambdas=meta.get("lambdas"),
        )
        model._vocabulary = set(meta.get("vocabulary", []))

        for raga, raga_entry in data.get("ragas", {}).items():
            model._counts[raga] = {}
            model._context_counts[raga] = {}
            model._token_counts[raga] = raga_entry.get("token_count", 0)
            model._recording_counts[raga] = raga_entry.get("recording_count", 0)

            for n_str, ngram_dict in raga_entry.get("counts", {}).items():
                n = int(n_str)
                model._counts[raga][n] = defaultdict(int)
                for key, count in ngram_dict.items():
                    ngram_tuple = tuple(key.split(cls._KEY_SEP))
                    model._counts[raga][n][ngram_tuple] = count

            for n_str, ctx_dict in raga_entry.get("context_counts", {}).items():
                n = int(n_str)
                model._context_counts[raga][n] = defaultdict(int)
                for key, count in ctx_dict.items():
                    ctx_tuple = tuple(key.split(cls._KEY_SEP))
                    model._context_counts[raga][n][ctx_tuple] = count

        model._finalized = True
        return model


# =============================================================================
# Training helpers
# =============================================================================

_TONIC_MAP: Dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "F": 5, "E#": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}


def _tonic_name_to_midi(tonic: str) -> float:
    """Convert a tonic name to a MIDI note number in octave 4.

    E.g. "C" -> 60.0, "C#" -> 61.0, "G" -> 67.0.
    Returns 60.0 (middle C) as a fallback for unknown tonic names.
    """
    pc = _TONIC_MAP.get(tonic.strip())
    if pc is None:
        return 60.0
    # Octave 4 in MIDI convention: C4 = 60.
    return float(60 + pc)


def _normalize_col_header(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _find_col(fieldnames: List[str], aliases: List[str]) -> Optional[str]:
    norm: Dict[str, str] = {_normalize_col_header(n): n for n in fieldnames if n}
    for alias in aliases:
        resolved = norm.get(_normalize_col_header(alias))
        if resolved is not None:
            return resolved
    return None


def _load_notes_from_csv(csv_path: Path, tonic_midi: float) -> List[str]:
    """Read a transcription CSV, build Note objects, and return LM tokens.

    Supports flexible column names:
        - start time: ``start``, ``start_time``
        - end time: ``end``, ``end_time``
        - pitch: ``pitch_midi``, ``midi``
        - sargam: ``sargam``

    Delegates to :func:`raga_pipeline.sequence.tokenize_notes_for_lm`.
    """
    from raga_pipeline.sequence import Note, tokenize_notes_for_lm  # lazy import

    notes: List[Note] = []
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            start_col = _find_col(fieldnames, ["start", "start_time"])
            end_col = _find_col(fieldnames, ["end", "end_time"])
            midi_col = _find_col(fieldnames, ["pitch_midi", "midi"])
            sargam_col = _find_col(fieldnames, ["sargam"])

            for row in reader:
                try:
                    start = float(row[start_col]) if start_col else 0.0
                    end = float(row[end_col]) if end_col else 0.0
                    pitch_midi = float(row[midi_col]) if midi_col else 60.0
                    sargam = str(row[sargam_col]).strip() if sargam_col else ""
                    # pitch_hz is not used by the tokenizer but required by the dataclass.
                    pitch_hz = 440.0 * (2.0 ** ((pitch_midi - 69.0) / 12.0))
                    notes.append(
                        Note(
                            start=start,
                            end=end,
                            pitch_midi=pitch_midi,
                            pitch_hz=pitch_hz,
                            confidence=1.0,
                            sargam=sargam,
                        )
                    )
                except (ValueError, KeyError, TypeError):
                    continue
    except OSError:
        return []

    return tokenize_notes_for_lm(notes, tonic_midi)


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
    """Train a per-raga n-gram language model from a labeled corpus.

    Reads a ground-truth CSV (columns: Filename, Raga, Tonic) and discovers
    transcription CSVs under *results_dir* using the same candidate-discovery
    logic as :mod:`raga_pipeline.motifs`.  Each recording is tokenized via
    :func:`_load_notes_from_csv` and added to the :class:`NgramModel`.

    Ragas whose recording count falls below *min_recordings* are removed from
    the model before serialization.

    Args:
        ground_truth: Path to the ground-truth CSV file.
        results_dir: Root directory that contains per-recording sub-directories
            with ``transcribed_notes.csv`` files.
        output: Path where the serialized JSON model will be written.
        order: Maximum n-gram order (default 5).
        smoothing: Smoothing scheme passed to :class:`NgramModel` (default "add-k").
        smoothing_k: Add-k smoothing parameter (default 0.01).
        lambdas: Optional interpolation weights; length must equal *order*.
        min_recordings: Minimum number of recordings required to keep a raga
            in the model (default 3).
        transcription_source: ``"auto"``, ``"edited"``, or ``"original"``
            (default ``"auto"``).
        quiet: Suppress progress messages when True.

    Returns:
        The serialized model dict (same as what is written to *output*).
    """
    from raga_pipeline.motifs import _discover_candidates, _read_ground_truth_rows  # lazy import

    gt_path = Path(ground_truth)
    rd_path = Path(results_dir)
    out_path = Path(output)

    gt_rows = _read_ground_truth_rows(gt_path)
    stem_map, basename_map = _discover_candidates(rd_path)

    model = NgramModel(
        order=order,
        smoothing=smoothing,
        smoothing_k=smoothing_k,
        lambdas=lambdas,
    )

    skipped = 0
    processed = 0

    for row in gt_rows:
        filename = row.filename
        raga = row.raga
        tonic = row.tonic

        if not filename or not raga:
            skipped += 1
            continue

        # Match filename to a transcription candidate using stem lookup.
        base = os.path.basename(filename)
        stem = Path(base).stem.lower()

        # Try stem_map first, then basename_map.
        candidates = stem_map.get(stem) or basename_map.get(base.lower())
        if not candidates:
            if not quiet:
                print(f"[train_model] no candidate for '{filename}', skipping")
            skipped += 1
            continue
        if len(candidates) > 1:
            if not quiet:
                print(f"[train_model] ambiguous match for '{filename}' ({len(candidates)} candidates), skipping")
            skipped += 1
            continue

        candidate = candidates[0]
        csv_path, _source = candidate.resolve(transcription_source)
        if csv_path is None or not csv_path.exists():
            if not quiet:
                print(f"[train_model] transcription CSV missing for '{filename}', skipping")
            skipped += 1
            continue

        tonic_midi = _tonic_name_to_midi(tonic) if tonic else 60.0
        tokens = _load_notes_from_csv(csv_path, tonic_midi)
        if not tokens:
            if not quiet:
                print(f"[train_model] empty token sequence for '{filename}', skipping")
            skipped += 1
            continue

        model.add_sequence(raga, tokens)
        processed += 1

    # Remove ragas below the minimum recording threshold.
    ragas_to_remove = [
        r for r in list(model._recording_counts)
        if model._recording_counts[r] < min_recordings
    ]
    for raga in ragas_to_remove:
        model._counts.pop(raga, None)
        model._context_counts.pop(raga, None)
        model._token_counts.pop(raga, None)
        model._recording_counts.pop(raga, None)

    model.finalize()

    serialized = model.to_dict()
    # Augment metadata with provenance information.
    serialized["metadata"]["ground_truth"] = str(gt_path.resolve())
    serialized["metadata"]["results_dir"] = str(rd_path.resolve())
    serialized["metadata"]["trained_at"] = datetime.now(tz=timezone.utc).isoformat()
    serialized["metadata"]["training_ragas"] = len(model._counts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(serialized, fh, indent=2)

    if not quiet:
        print(
            f"[train_model] processed={processed} skipped={skipped} "
            f"ragas={len(model._counts)} -> {out_path}"
        )

    return serialized
