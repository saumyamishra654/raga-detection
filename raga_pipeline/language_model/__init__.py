"""Per-raga n-gram language models for raga detection. CLI entry point: python -m raga_pipeline.language_model train|score|evaluate ..."""

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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

        # Lambda convention: index 0 = unigram weight, index 1 = bigram, ...,
        # index (order-1) = highest-order weight.  The CLI presents these in
        # reversed order (highest-order-first) and __main__._parse_lambdas
        # reverses them before passing here.
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

    def add_sequence(self, raga: str, phrases: List[List[str]]) -> None:
        """Add a tokenised recording (as phrase-separated sequences) to the model.

        *phrases* is a list of phrase token lists, each starting with ``<BOS>``.
        N-grams are counted within each phrase only -- never crossing phrase
        boundaries.

        Also accepts a flat ``List[str]`` for backwards compatibility (treated
        as a single phrase).

        Empty input is silently ignored (no raga entry is created).
        """
        # Backwards compat: flat list of strings -> single phrase
        if phrases and isinstance(phrases[0], str):
            phrases = [phrases]  # type: ignore[list-item]

        if not phrases or all(len(p) == 0 for p in phrases):
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

        for phrase in phrases:
            for i, token in enumerate(phrase):
                self._vocabulary.add(token)
                self._token_counts[raga] += 1

                # Count n-grams of every order that end at position i.
                for n in range(1, self.order + 1):
                    if i - n + 1 < 0:
                        break
                    ngram = tuple(phrase[i - n + 1 : i + 1])
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

    def score_sequence(self, raga: str, phrases: List[List[str]]) -> float:
        """Return the average log-likelihood per token under *raga*.

        *phrases* is a list of phrase token lists (each starting with
        ``<BOS>``).  Scoring is done within each phrase -- context never
        crosses phrase boundaries.

        Also accepts a flat ``List[str]`` for backwards compatibility
        (treated as a single phrase).

        Returns 0.0 for empty or trivially short input.
        """
        # Backwards compat: flat list -> single phrase
        if phrases and isinstance(phrases[0], str):
            phrases = [phrases]  # type: ignore[list-item]

        total_ll = 0.0
        scored = 0
        for phrase in phrases:
            if len(phrase) < 2:
                continue
            for i in range(1, len(phrase)):
                token = phrase[i]
                ctx_start = max(0, i - (self.order - 1))
                context = tuple(phrase[ctx_start:i])
                total_ll += self.log_prob(raga, token, context)
                scored += 1

        return total_ll / scored if scored > 0 else 0.0

    def rank_ragas(self, phrases: List[List[str]]) -> List[Tuple[str, float]]:
        """Score *tokens* against every raga; return sorted (raga, score) pairs.

        Highest score first.
        """
        scores = [(raga, self.score_sequence(raga, phrases)) for raga in self.ragas()]
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


def _find_col(fieldnames: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    norm: Dict[str, str] = {_normalize_col_header(n): n for n in fieldnames if n}
    for alias in aliases:
        resolved = norm.get(_normalize_col_header(alias))
        if resolved is not None:
            return resolved
    return None


def _load_raw_notes_from_csv(csv_path: Path) -> List:
    """Read a transcription CSV and return raw Note objects (no tokenization).

    Returns an empty list on any read/parse error.
    """
    from raga_pipeline.sequence import Note  # lazy import

    notes: List = []
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

    return notes


def _load_notes_from_csv(csv_path: Path, tonic_midi: float) -> List[List[str]]:
    """Read a transcription CSV and return phrase-separated LM tokens.

    Delegates to :func:`_load_raw_notes_from_csv` then
    :func:`raga_pipeline.sequence.tokenize_notes_for_lm`.
    """
    from raga_pipeline.sequence import tokenize_notes_for_lm  # lazy import

    notes = _load_raw_notes_from_csv(csv_path)
    if not notes:
        return []
    return tokenize_notes_for_lm(notes, tonic_midi)


def _load_note_timestamps_from_csv(csv_path: Path) -> List[Tuple[float, float]]:
    """Read (start, end) timestamp pairs from a transcription CSV.

    Returns a list parallel to the notes in the file (one entry per valid row).
    Rows that cannot be parsed are skipped, matching the skip logic in
    :func:`_load_notes_from_csv`.
    """
    timestamps: List[Tuple[float, float]] = []
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
                    # Validate that pitch and sargam columns parse too so we
                    # stay in sync with the note-loading logic above.
                    if midi_col:
                        float(row[midi_col])
                    if sargam_col:
                        str(row[sargam_col]).strip()
                    timestamps.append((start, end))
                except (ValueError, KeyError, TypeError):
                    continue
    except OSError:
        pass
    return timestamps


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

    Loads the model from *model_path*, tokenizes the transcription CSV at
    *transcription_path* using *tonic* as the reference pitch, and ranks all
    ragas in the model by average log-likelihood.

    Args:
        model_path: Path to a JSON model file produced by :func:`train_model`.
        transcription_path: Path to a ``transcribed_notes.csv`` file.
        tonic: Tonic note name (e.g. ``"C"``, ``"C#"``).
        segments: When True, also return segment-level rankings using a
            sliding window of *segment_window* tokens with 50 % overlap.
        segment_window: Number of tokens per segment window (default 100).
        top_k: Maximum number of ragas to include in the overall ranking
            (default 5).  Segment-level results always include 3 ragas.
        output: Optional path to write the result dict as JSON.

    Returns:
        A dict with the following keys:

        - ``"rankings"``: list of ``{"raga": str, "score": float, "rank": int}``
          sorted highest-score-first, limited to *top_k* entries.
        - ``"segments"`` (only when *segments* is True): list of segment dicts,
          each with ``"start_time"``, ``"end_time"``,
          ``"token_range": [start_idx, end_idx]``, and
          ``"top_ragas": [{"raga": str, "score": float}, ...]`` (top 3).
        - ``"error"`` (only on failure): human-readable error string.
    """
    # Load model.
    try:
        with open(model_path, "r", encoding="utf-8") as fh:
            model = NgramModel.from_dict(json.load(fh))
    except (OSError, KeyError, ValueError) as exc:
        return {"error": f"failed to load model: {exc}"}

    # Tokenize transcription (phrase-separated).
    tonic_midi = _tonic_name_to_midi(tonic)
    trans_path = Path(transcription_path)
    phrases = _load_notes_from_csv(trans_path, tonic_midi)

    if not phrases:
        return {"error": "no tokens extracted from transcription"}

    total_tokens = sum(len(p) for p in phrases)

    # Whole-recording ranking.
    ranked = model.rank_ragas(phrases)
    rankings = [
        {"raga": raga, "score": round(score, 4), "rank": i + 1}
        for i, (raga, score) in enumerate(ranked[:top_k])
    ]

    result: Dict[str, Any] = {"rankings": rankings}

    # Segment-level scoring: group consecutive phrases into windows of
    # ~segment_window tokens, score each window as a unit.
    if segments:
        timestamps = _load_note_timestamps_from_csv(trans_path)

        # Build per-phrase time spans from note timestamps.
        note_idx = 0
        phrase_times: List[Tuple[float, float]] = []
        for phrase in phrases:
            n_notes = len([t for t in phrase if t != "<BOS>"])
            if n_notes > 0 and note_idx < len(timestamps):
                p_start = timestamps[note_idx][0]
                p_end = timestamps[min(note_idx + n_notes - 1, len(timestamps) - 1)][1]
                phrase_times.append((p_start, p_end))
            else:
                phrase_times.append((0.0, 0.0))
            note_idx += n_notes

        seg_results: List[Dict[str, Any]] = []
        i = 0
        while i < len(phrases):
            # Accumulate phrases until we reach ~segment_window tokens.
            window_phrases: List[List[str]] = []
            window_tokens = 0
            j = i
            while j < len(phrases) and window_tokens < segment_window:
                window_phrases.append(phrases[j])
                window_tokens += len(phrases[j])
                j += 1

            if not window_phrases:
                break

            seg_start_time = phrase_times[i][0]
            seg_end_time = phrase_times[j - 1][1]

            seg_ranked = model.rank_ragas(window_phrases)
            top_ragas = [
                {"raga": r, "score": round(s, 4)}
                for r, s in seg_ranked[:3]
            ]

            seg_results.append({
                "start_time": seg_start_time,
                "end_time": seg_end_time,
                "phrase_range": [i, j],
                "token_count": window_tokens,
                "top_ragas": top_ragas,
            })

            # Advance by half the phrases in the window (50% overlap).
            step = max(1, (j - i) // 2)
            i += step

        result["segments"] = seg_results

    # Optional file output.
    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    return result


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


# =============================================================================
# Evaluation helpers
# =============================================================================


def _run_leave_one_out(
    gt_rows: List[Any],
    recordings: Dict[str, Tuple[str, str, List[str]]],
    order: int,
    smoothing: str,
    smoothing_k: float,
    lambdas: Optional[List[float]],
    min_recordings: int,
) -> List[Dict[str, Any]]:
    """Run leave-one-out cross-validation over *recordings*.

    *recordings* maps filename -> (raga, tonic, tokens).

    For each held-out recording a fresh :class:`NgramModel` is trained on all
    other recordings, ragas below *min_recordings* (after excluding the
    held-out recording) are removed, and the held-out tokens are scored.

    Returns a list of per-recording result dicts.
    """
    results: List[Dict[str, Any]] = []
    filenames = list(recordings.keys())

    for held_out_filename in filenames:
        true_raga, _tonic, held_out_tokens = recordings[held_out_filename]

        # Build model from all recordings except the held-out one.
        loo_lambdas = [1.0 / order] * order if lambdas is None else lambdas
        model = NgramModel(
            order=order,
            smoothing=smoothing,
            smoothing_k=smoothing_k,
            lambdas=loo_lambdas,
        )

        # Count recordings per raga excluding held-out (for threshold check).
        raga_rec_counts: Dict[str, int] = defaultdict(int)
        for fname, (raga, _tonic2, tokens) in recordings.items():
            if fname == held_out_filename:
                continue
            model.add_sequence(raga, tokens)
            raga_rec_counts[raga] += 1

        # Remove ragas below the minimum recording threshold.
        ragas_to_remove = [r for r, cnt in raga_rec_counts.items() if cnt < min_recordings]
        for raga in ragas_to_remove:
            model._counts.pop(raga, None)
            model._context_counts.pop(raga, None)
            model._token_counts.pop(raga, None)
            model._recording_counts.pop(raga, None)

        model.finalize()

        ranked = model.rank_ragas(held_out_tokens)

        # Determine rank of true raga.
        true_raga_rank = None
        for rank_idx, (raga, _score) in enumerate(ranked, start=1):
            if raga == true_raga:
                true_raga_rank = rank_idx
                break
        # If true raga was removed (below min_recordings), rank is worst + 1.
        if true_raga_rank is None:
            true_raga_rank = len(ranked) + 1

        predicted_raga = ranked[0][0] if ranked else ""
        correct = predicted_raga == true_raga

        row: Dict[str, Any] = {
            "filename": held_out_filename,
            "true_raga": true_raga,
            "predicted_raga": predicted_raga,
            "correct": correct,
            "true_raga_rank": true_raga_rank,
            "score_top1": round(ranked[0][1], 4) if len(ranked) > 0 else 0.0,
            "score_top2": round(ranked[1][1], 4) if len(ranked) > 1 else 0.0,
            "score_top3": round(ranked[2][1], 4) if len(ranked) > 2 else 0.0,
            "raga_top1": ranked[0][0] if len(ranked) > 0 else "",
            "raga_top2": ranked[1][0] if len(ranked) > 1 else "",
            "raga_top3": ranked[2][0] if len(ranked) > 2 else "",
        }
        results.append(row)

    return results


def _run_leave_one_out_lm_deletion(
    gt_rows: list,
    recordings: Dict[str, Tuple[str, str, List[List[str]]]],
    raw_notes_map: Dict[str, list],
    order: int,
    smoothing: str,
    smoothing_k: float,
    lambdas: Optional[List[float]],
    min_recordings: int,
    raga_db: Any,
    lm_deletion_lambda: float = 2.0,
    lm_deletion_slope: float = -0.0684,
    lm_deletion_intercept: float = 0.6640,
) -> List[Dict[str, Any]]:
    """LOO CV with LM + deletion-residual scoring (no histogram).

    For each held-out recording, trains an NgramModel on the rest, then
    for each candidate raga: applies raga correction, computes deletion
    residual, tokenizes, scores with LM, and combines the signals.
    """
    from raga_pipeline.raga import apply_raga_correction_to_notes, get_raga_notes
    from raga_pipeline.sequence import tokenize_notes_for_lm

    raga_to_filenames: Dict[str, List[str]] = defaultdict(list)
    for fname, (raga, tonic, tokens) in recordings.items():
        raga_to_filenames[raga].append(fname)

    results: List[Dict[str, Any]] = []
    filenames = list(recordings.keys())

    for held_out in filenames:
        held_raga, held_tonic, held_tokens = recordings[held_out]
        held_raw_notes = raw_notes_map.get(held_out, [])
        if not held_tokens or not held_raw_notes:
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

        # Score each candidate raga with LM + deletion-residual scoring
        tonic_pc = _TONIC_MAP.get(held_tonic.strip(), 0) if isinstance(held_tonic, str) else int(held_tonic)
        tonic_midi = 60.0 + tonic_pc
        candidate_scores: List[Tuple[str, float]] = []

        for cand_raga in model.ragas():
            try:
                corrected_notes, corr_stats, _ = apply_raga_correction_to_notes(
                    held_raw_notes, raga_db, cand_raga, tonic_pc,
                    max_distance=1.0, keep_impure=False,
                )
            except Exception:
                continue

            total = corr_stats.get("total", len(held_raw_notes))
            discarded = corr_stats.get("discarded", 0)
            deletion_rate = discarded / total if total > 0 else 1.0

            scale_size = len(get_raga_notes(raga_db, cand_raga, tonic_pc))
            expected_del = lm_deletion_slope * scale_size + lm_deletion_intercept
            del_residual = deletion_rate - expected_del

            phrases = tokenize_notes_for_lm(corrected_notes, tonic_midi)
            lm_score = model.score_sequence(cand_raga, phrases) if phrases else -999.0

            combined = lm_score - lm_deletion_lambda * del_residual
            candidate_scores.append((cand_raga, combined))

        if not candidate_scores:
            continue

        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        raga_names = [r for r, _ in candidate_scores]
        true_rank = (raga_names.index(held_raga) + 1) if held_raga in raga_names else len(raga_names) + 1

        results.append({
            "filename": held_out,
            "true_raga": held_raga,
            "predicted_raga": candidate_scores[0][0] if candidate_scores else "",
            "correct": candidate_scores[0][0] == held_raga if candidate_scores else False,
            "true_raga_rank": true_rank,
            "score_top1": round(candidate_scores[0][1], 4) if candidate_scores else 0.0,
            "raga_top1": candidate_scores[0][0] if candidate_scores else "",
            "raga_top2": candidate_scores[1][0] if len(candidate_scores) > 1 else "",
            "raga_top3": candidate_scores[2][0] if len(candidate_scores) > 2 else "",
            "score_top2": round(candidate_scores[1][1], 4) if len(candidate_scores) > 1 else 0.0,
            "score_top3": round(candidate_scores[2][1], 4) if len(candidate_scores) > 2 else 0.0,
        })

    return results


def _compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute top-1 accuracy, top-3 accuracy, and MRR from per-recording results."""
    total = len(results)
    if total == 0:
        return {"top1_accuracy": 0.0, "top3_accuracy": 0.0, "mrr": 0.0, "total": 0}

    top1_correct = sum(1 for r in results if r["correct"])
    top3_correct = sum(1 for r in results if r["true_raga_rank"] <= 3)
    mrr = sum(1.0 / r["true_raga_rank"] for r in results) / total

    return {
        "top1_accuracy": top1_correct / total,
        "top3_accuracy": top3_correct / total,
        "mrr": round(mrr, 4),
        "total": total,
    }


def _write_eval_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    """Write per-recording evaluation results to a CSV file."""
    fieldnames = [
        "filename", "true_raga", "predicted_raga", "correct",
        "true_raga_rank", "score_top1", "score_top2", "score_top3",
        "raga_top1", "raga_top2", "raga_top3",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


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
    scoring_mode: str = "lm",
    raga_db_path: Optional[str] = None,
    lm_deletion_lambda: float = 2.0,
    lm_deletion_slope: float = -0.0684,
    lm_deletion_intercept: float = 0.6640,
) -> Dict[str, Any]:
    """Evaluate a per-raga n-gram model using leave-one-out cross-validation.

    Loads the ground-truth CSV and discovers transcription CSVs under
    *results_dir* (same logic as :func:`train_model`).  All recordings are
    tokenized once, then leave-one-out CV is run at *order* (or at each order
    in *sweep_orders* if provided).

    Args:
        ground_truth: Path to the ground-truth CSV file (Filename, Raga, Tonic).
        results_dir: Root directory with per-recording sub-directories.
        output: Path where the per-recording result CSV will be written.
        order: N-gram order to use when *sweep_orders* is None (default 5).
        smoothing: Smoothing scheme for :class:`NgramModel` (default "add-k").
        smoothing_k: Add-k smoothing parameter (default 0.01).
        lambdas: Optional interpolation weights (length must equal *order*).
        min_recordings: Minimum recordings per raga to include in each LOO fold
            (default 3).
        transcription_source: "auto", "edited", or "original" (default "auto").
        sweep_orders: When provided, run LOO CV at each listed order and return
            per-order results under ``"sweep_results"`` in the summary.
        quiet: Suppress progress messages when True.

    Returns:
        Summary dict with ``"top1_accuracy"``, ``"top3_accuracy"``, ``"mrr"``,
        ``"total"`` (for the primary/last order).  When *sweep_orders* is
        provided, also includes ``"sweep_results"``: a list of per-order dicts
        each containing the above keys plus ``"order"``.
    """
    from raga_pipeline.motifs import _discover_candidates, _read_ground_truth_rows  # lazy import

    gt_path = Path(ground_truth)
    rd_path = Path(results_dir)
    out_path = Path(output)

    gt_rows = _read_ground_truth_rows(gt_path)
    stem_map, basename_map = _discover_candidates(rd_path)

    # Tokenize all recordings once.
    recordings: Dict[str, Tuple[str, str, List[str]]] = {}
    skipped = 0

    for row in gt_rows:
        filename = row.filename
        raga = row.raga
        tonic = row.tonic

        if not filename or not raga:
            skipped += 1
            continue

        base = os.path.basename(filename)
        stem = Path(base).stem.lower()

        candidates = stem_map.get(stem) or basename_map.get(base.lower())
        if not candidates:
            if not quiet:
                print(f"[evaluate_model] no candidate for '{filename}', skipping")
            skipped += 1
            continue
        if len(candidates) > 1:
            if not quiet:
                print(f"[evaluate_model] ambiguous match for '{filename}' ({len(candidates)} candidates), skipping")
            skipped += 1
            continue

        candidate = candidates[0]
        csv_path, _source = candidate.resolve(transcription_source)
        if csv_path is None or not csv_path.exists():
            if not quiet:
                print(f"[evaluate_model] transcription CSV missing for '{filename}', skipping")
            skipped += 1
            continue

        tonic_midi = _tonic_name_to_midi(tonic) if tonic else 60.0
        tokens = _load_notes_from_csv(csv_path, tonic_midi)
        if not tokens:
            if not quiet:
                print(f"[evaluate_model] empty token sequence for '{filename}', skipping")
            skipped += 1
            continue

        recordings[filename] = (raga, tonic, tokens)

    # For lm-deletion scoring: also load raw notes and raga DB
    raw_notes_map: Dict[str, list] = {}
    raga_db = None
    if scoring_mode == "lm-deletion":
        from raga_pipeline.raga import RagaDatabase
        if raga_db_path is None:
            # Auto-discover raga DB
            for candidate_path in [
                Path("main notebooks/raga_list_final.csv"),
                Path(__file__).parent.parent / "raga_list_final.csv",
                Path(__file__).parent.parent.parent / "main notebooks" / "raga_list_final.csv",
            ]:
                if candidate_path.exists():
                    raga_db_path = str(candidate_path)
                    break
        if raga_db_path:
            raga_db = RagaDatabase(raga_db_path)
        else:
            raise ValueError("--raga-db required for scoring-mode=lm-deletion (auto-discovery failed)")

        # Load raw notes for each recording
        for row in gt_rows:
            filename = row.filename
            if filename not in recordings:
                continue
            base = os.path.basename(filename)
            stem = Path(base).stem.lower()
            candidates_list = stem_map.get(stem) or basename_map.get(base.lower())
            if not candidates_list or len(candidates_list) != 1:
                continue
            csv_path, _ = candidates_list[0].resolve(transcription_source)
            if csv_path and csv_path.exists():
                raw_notes_map[filename] = _load_raw_notes_from_csv(csv_path)

    if not quiet:
        print(f"[evaluate_model] loaded={len(recordings)} skipped={skipped}")

    # Choose LOO function based on scoring mode
    def _run_loo(o: int, lam: Optional[List[float]]) -> List[Dict[str, Any]]:
        if scoring_mode == "lm-deletion" and raga_db is not None:
            return _run_leave_one_out_lm_deletion(
                gt_rows=gt_rows, recordings=recordings,
                raw_notes_map=raw_notes_map, order=o,
                smoothing=smoothing, smoothing_k=smoothing_k,
                lambdas=lam, min_recordings=min_recordings,
                raga_db=raga_db,
                lm_deletion_lambda=lm_deletion_lambda,
                lm_deletion_slope=lm_deletion_slope,
                lm_deletion_intercept=lm_deletion_intercept,
            )
        return _run_leave_one_out(
            gt_rows=gt_rows, recordings=recordings, order=o,
            smoothing=smoothing, smoothing_k=smoothing_k,
            lambdas=lam, min_recordings=min_recordings,
        )

    if sweep_orders is not None:
        sweep_results: List[Dict[str, Any]] = []
        last_loo_results: List[Dict[str, Any]] = []
        for sweep_order in sweep_orders:
            loo_results = _run_loo(sweep_order, None)
            order_summary = _compute_summary(loo_results)
            order_summary["order"] = sweep_order
            sweep_results.append(order_summary)
            last_loo_results = loo_results
            if not quiet:
                print(
                    f"[evaluate_model] order={sweep_order} "
                    f"top1={order_summary['top1_accuracy']:.3f} "
                    f"top3={order_summary['top3_accuracy']:.3f} "
                    f"mrr={order_summary['mrr']:.3f}"
                )

        _write_eval_csv(out_path, last_loo_results)
        summary = _compute_summary(last_loo_results)
        summary["sweep_results"] = sweep_results
        return summary

    else:
        loo_results = _run_loo(order, lambdas)
        _write_eval_csv(out_path, loo_results)
        summary = _compute_summary(loo_results)

        if not quiet:
            print(
                f"[evaluate_model] top1={summary['top1_accuracy']:.3f} "
                f"top3={summary['top3_accuracy']:.3f} "
                f"mrr={summary['mrr']:.3f} "
                f"total={summary['total']}"
            )

        return summary
