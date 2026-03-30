"""Per-raga n-gram language models for raga detection. CLI entry point: python -m raga_pipeline.language_model train|score|evaluate ..."""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


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
