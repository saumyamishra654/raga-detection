"""Noisy-channel alignment scorer for n-gram language models.

Scores uncorrected (noisy) token sequences against corrected-trained LMs
using phrase-local beam DP with skip and substitution costs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

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
