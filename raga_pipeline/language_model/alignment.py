"""Noisy-channel alignment scorer for n-gram language models.

Scores uncorrected (noisy) token sequences against corrected-trained LMs
using phrase-local beam DP with skip and substitution costs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from raga_pipeline.language_model import NgramModel

# Sargam token -> pitch class (0-11), matching sequence.OFFSET_TO_SARGAM
_SARGAM_TO_PC: Dict[str, int] = {
    "Sa": 0, "re": 1, "Re": 2, "ga": 3, "Ga": 4,
    "ma": 5, "Ma": 6, "Pa": 7, "dha": 8, "Dha": 9,
    "ni": 10, "Ni": 11,
}


def token_pitch_info(token: str) -> Optional[Tuple[int, int]]:
    """Extract (pitch_class, octave) from an LM sargam token.

    Octave convention: 0 = middle (bare), -1 = lower ('), +1 = upper ('').
    Handles direction suffixes (/U, /D, /=) by stripping them first.
    Returns None for <BOS>, unknown, or empty tokens.
    """
    if not token or token == "<BOS>":
        return None

    # Strip direction suffix if present (e.g. "Re/U" -> "Re")
    base = token
    for suffix in ("/U", "/D", "/="):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    # Strip octave suffixes (order matters: check '' before ')
    octave = 0
    if base.endswith("''"):
        octave = 1
        base = base[:-2]
    elif base.endswith("'"):
        octave = -1
        base = base[:-1]

    pc = _SARGAM_TO_PC.get(base)
    if pc is None:
        return None
    return (pc, octave)


def pitch_distance(pc_a: int, oct_a: int, pc_b: int, oct_b: int) -> int:
    """Circular pitch-class distance plus octave penalty.

    Returns min semitone distance on the pitch-class circle (0-6)
    plus abs(octave difference).
    """
    raw = abs(pc_a - pc_b)
    pc_dist = min(raw, 12 - raw)
    return pc_dist + abs(oct_a - oct_b)


def build_substitution_map(
    vocabulary: Set[str],
    max_distance: int = 2,
) -> Dict[str, List[Tuple[str, int]]]:
    """Build token -> [(target_token, distance)] for all pairs within max_distance.

    Excludes <BOS> and tokens with unknown pitch class.  Self-substitutions
    (distance 0) are excluded.
    """
    # Pre-compute pitch info for all valid tokens
    token_info: Dict[str, Tuple[int, int]] = {}
    for tok in vocabulary:
        info = token_pitch_info(tok)
        if info is not None:
            token_info[tok] = info

    result: Dict[str, List[Tuple[str, int]]] = {}
    tokens = list(token_info.keys())

    for tok_a in tokens:
        pc_a, oct_a = token_info[tok_a]
        subs: List[Tuple[str, int]] = []
        for tok_b in tokens:
            if tok_b == tok_a:
                continue
            pc_b, oct_b = token_info[tok_b]
            dist = pitch_distance(pc_a, oct_a, pc_b, oct_b)
            if 0 < dist <= max_distance:
                subs.append((tok_b, dist))
        if subs:
            result[tok_a] = subs

    return result


@dataclass
class AlignmentConfig:
    """Hyperparameters for the alignment scorer."""
    lambda_skip: float = 0.5       # penalty per skipped token
    lambda_match: float = 2.0      # reward per matched token (compensates negative log-probs)
    lambda_sub: float = 0.3        # penalty per semitone of substitution distance
    beam_width: int = 200          # max DP states per position
    max_sub_distance: int = 2      # max pitch distance for substitution (0 = disabled)


@dataclass
class AlignmentResult:
    """Result of aligning an observed sequence against a raga LM."""
    lm_per_token: float        # average log-prob per matched token
    n_matched: int             # tokens scored by LM (match + substituted)
    n_skipped: int             # tokens treated as noise
    n_substituted: int         # tokens that were substituted
    skip_fraction: float       # n_skipped / (n_matched + n_skipped)
    total_sub_distance: float  # sum of substitution distances
    raw_lm_sum: float = 0.0   # sum of log-probs (before averaging)


@dataclass
class _BeamEntry:
    """Internal state for one beam hypothesis."""
    score: float        # DP objective value (log-prob sum - penalties)
    lm_sum: float       # sum of log P(z_t | ctx) only (no penalties)
    n_matched: int
    n_skipped: int
    n_subst: int
    sub_dist: float     # total substitution distance


def score_phrase_aligned(
    model: NgramModel,
    raga: str,
    phrase: List[str],
    config: AlignmentConfig,
    sub_map: Optional[Dict[str, List[Tuple[str, int]]]] = None,
) -> AlignmentResult:
    """Score a single phrase using beam-search DP alignment.

    The DP finds the partition of observed tokens into matched, skipped,
    and substituted that maximizes:
        SUM (log P_r(z_t | ctx_t) + lambda_match) - lambda_skip * N_skip - lambda_sub * SUM dist_t

    Context (ctx) is built from matched/substituted tokens only and never
    crosses phrase boundaries.  <BOS> always anchors the context start.

    Parameters
    ----------
    model : NgramModel
        A trained (and finalized) n-gram model.
    raga : str
        Raga to score against.
    phrase : list of str
        Token list starting with ``<BOS>``.
    config : AlignmentConfig
        Hyperparameters.
    sub_map : dict, optional
        Pre-built substitution map (from ``build_substitution_map``).
        If None, substitutions are disabled regardless of max_sub_distance.
    """
    _EMPTY = AlignmentResult(
        lm_per_token=0.0, n_matched=0, n_skipped=0,
        n_substituted=0, skip_fraction=0.0, total_sub_distance=0.0,
    )

    if len(phrase) < 2:
        return _EMPTY

    ctx_len = model.order - 1  # how many prior tokens form context

    # Initial beam: context = (<BOS>,), ready to process token at index 1
    init_ctx = ("<BOS>",)
    beam: Dict[tuple, _BeamEntry] = {
        init_ctx: _BeamEntry(score=0.0, lm_sum=0.0, n_matched=0,
                             n_skipped=0, n_subst=0, sub_dist=0.0),
    }

    def _update(target: Dict[tuple, _BeamEntry], key: tuple, entry: _BeamEntry) -> None:
        existing = target.get(key)
        if existing is None or entry.score > existing.score:
            target[key] = entry

    # Process each observed token after <BOS>
    for i in range(1, len(phrase)):
        token = phrase[i]
        new_beam: Dict[tuple, _BeamEntry] = {}

        for ctx, entry in beam.items():
            # --- Option 1: Skip (noise) ---
            _update(new_beam, ctx, _BeamEntry(
                score=entry.score - config.lambda_skip,
                lm_sum=entry.lm_sum,
                n_matched=entry.n_matched,
                n_skipped=entry.n_skipped + 1,
                n_subst=entry.n_subst,
                sub_dist=entry.sub_dist,
            ))

            # --- Option 2: Match (accept token as-is) ---
            # lambda_match compensates for negative log-probs so that
            # in-scale tokens contribute positively to the DP objective.
            lp = model.log_prob(raga, token, ctx)
            new_ctx = (ctx + (token,))[-ctx_len:] if ctx_len > 0 else ()
            _update(new_beam, new_ctx, _BeamEntry(
                score=entry.score + lp + config.lambda_match,
                lm_sum=entry.lm_sum + lp,
                n_matched=entry.n_matched + 1,
                n_skipped=entry.n_skipped,
                n_subst=entry.n_subst,
                sub_dist=entry.sub_dist,
            ))

            # --- Option 3: Substitute to nearby token ---
            if sub_map and config.max_sub_distance > 0:
                for sub_token, dist in (sub_map.get(token) or []):
                    lp_sub = model.log_prob(raga, sub_token, ctx)
                    sub_ctx = (ctx + (sub_token,))[-ctx_len:] if ctx_len > 0 else ()
                    _update(new_beam, sub_ctx, _BeamEntry(
                        score=entry.score + lp_sub + config.lambda_match - config.lambda_sub * dist,
                        lm_sum=entry.lm_sum + lp_sub,
                        n_matched=entry.n_matched + 1,
                        n_skipped=entry.n_skipped,
                        n_subst=entry.n_subst + 1,
                        sub_dist=entry.sub_dist + dist,
                    ))

        # Beam pruning
        if len(new_beam) > config.beam_width:
            sorted_items = sorted(new_beam.items(), key=lambda x: x[1].score, reverse=True)
            new_beam = dict(sorted_items[:config.beam_width])

        beam = new_beam

    if not beam:
        return _EMPTY

    # Best final state
    best = max(beam.values(), key=lambda e: e.score)

    if best.n_matched == 0:
        # All tokens were skipped -- use a large negative sentinel so that
        # "skip everything" ranks below any hypothesis that actually matched
        # tokens, making lm_per_token safe for ranking comparisons.
        return AlignmentResult(
            lm_per_token=-1e6, n_matched=0, n_skipped=best.n_skipped,
            n_substituted=0, skip_fraction=1.0, total_sub_distance=0.0,
        )

    total = best.n_matched + best.n_skipped
    return AlignmentResult(
        lm_per_token=best.lm_sum / best.n_matched,
        n_matched=best.n_matched,
        n_skipped=best.n_skipped,
        n_substituted=best.n_subst,
        skip_fraction=best.n_skipped / total if total > 0 else 0.0,
        total_sub_distance=best.sub_dist,
        raw_lm_sum=best.lm_sum,
    )


def score_sequence_aligned(
    model: NgramModel,
    raga: str,
    phrases: List[List[str]],
    config: AlignmentConfig,
    sub_map: Optional[Dict[str, List[Tuple[str, int]]]] = None,
) -> AlignmentResult:
    """Score a multi-phrase sequence by aligning each phrase independently.

    Returns aggregated AlignmentResult across all phrases.  If *sub_map* is
    None and config.max_sub_distance > 0, builds the map from the model's
    vocabulary automatically.
    """
    # Backwards compat: flat list -> single phrase
    if phrases and isinstance(phrases[0], str):
        phrases = [phrases]  # type: ignore[list-item]

    # Auto-build substitution map if needed
    if sub_map is None and config.max_sub_distance > 0:
        sub_map = build_substitution_map(model.vocabulary, config.max_sub_distance)

    total_lm = 0.0
    total_matched = 0
    total_skipped = 0
    total_subst = 0
    total_sub_dist = 0.0

    for phrase in phrases:
        r = score_phrase_aligned(model, raga, phrase, config, sub_map)
        total_lm += r.raw_lm_sum
        total_matched += r.n_matched
        total_skipped += r.n_skipped
        total_subst += r.n_substituted
        total_sub_dist += r.total_sub_distance

    total = total_matched + total_skipped
    return AlignmentResult(
        lm_per_token=total_lm / total_matched if total_matched > 0 else -1e6,
        n_matched=total_matched,
        n_skipped=total_skipped,
        n_substituted=total_subst,
        skip_fraction=total_skipped / total if total > 0 else 0.0,
        total_sub_distance=total_sub_dist,
        raw_lm_sum=total_lm,
    )
