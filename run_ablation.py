"""Ablation study: entropy weighting x direction tokens.

Runs 4 LOO evaluations to isolate the contribution of each feature.
Usage: /opt/miniconda3/envs/raga/bin/python run_ablation.py
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Ensure we can import raga_pipeline
sys.path.insert(0, os.path.dirname(__file__))

from raga_pipeline.language_model import (
    NgramModel,
    _load_raw_notes_from_csv,
    _tonic_name_to_midi,
)
from raga_pipeline.motifs import _discover_candidates, _read_ground_truth_rows
from raga_pipeline.sequence import tokenize_notes_for_lm

GT_CSV = "compmusic_gt.csv"
RESULTS_DIR = "/Volumes/Extreme SSD/stems/separated_stems_nocorrection/"
ORDER = 5
MIN_RECORDINGS = 3


def load_recordings(include_direction: bool):
    """Load all recordings as (raga, tonic, phrases) with configurable direction."""
    gt_rows = _read_ground_truth_rows(Path(GT_CSV))
    stem_map, basename_map = _discover_candidates(Path(RESULTS_DIR))

    recordings = {}
    for row in gt_rows:
        filename = row.filename
        raga = row.raga
        tonic = row.tonic
        if not filename or not raga:
            continue

        base = os.path.basename(filename)
        stem = Path(base).stem.lower()
        candidates = stem_map.get(stem) or basename_map.get(base.lower())
        if not candidates or len(candidates) != 1:
            continue

        csv_path, _ = candidates[0].resolve("auto")
        if csv_path is None or not csv_path.exists():
            continue

        tonic_midi = _tonic_name_to_midi(tonic) if tonic else 60.0
        notes = _load_raw_notes_from_csv(csv_path)
        if not notes:
            continue

        phrases = tokenize_notes_for_lm(notes, tonic_midi, include_direction=include_direction)
        if phrases:
            recordings[filename] = (raga, tonic, phrases)

    return recordings


def run_loo(recordings, use_entropy: bool):
    """Run LOO evaluation with optional entropy weighting."""
    filenames = list(recordings.keys())
    raga_to_files = defaultdict(list)
    for fn, (raga, tonic, phrases) in recordings.items():
        raga_to_files[raga].append(fn)

    top1 = top3 = total = 0
    for held_out in filenames:
        held_raga, held_tonic, held_phrases = recordings[held_out]

        model = NgramModel(order=ORDER, smoothing="add-k", smoothing_k=0.01)
        model.use_entropy_weights = use_entropy
        raga_counts = defaultdict(int)

        for fn, (raga, tonic, phrases) in recordings.items():
            if fn == held_out:
                continue
            model.add_sequence(raga, phrases)
            raga_counts[raga] += 1

        for raga in list(model._counts.keys()):
            if raga_counts.get(raga, 0) < MIN_RECORDINGS:
                del model._counts[raga]
                del model._context_counts[raga]

        model.finalize()
        if not model.ragas():
            continue

        total += 1
        ranked = model.rank_ragas(held_phrases)
        raga_names = [r for r, _ in ranked]

        true_rank = None
        for i, (r, s) in enumerate(ranked):
            if held_raga.lower() in r.lower() or r.lower() in held_raga.lower():
                true_rank = i + 1
                break

        if true_rank == 1:
            top1 += 1
        if true_rank and true_rank <= 3:
            top3 += 1

    return top1, top3, total


def main():
    configs = [
        ("baseline (no entropy, no direction)", False, False),
        ("entropy only", True, False),
        ("direction only", False, True),
        ("entropy + direction", True, True),
    ]

    print(f"{'Config':<40} {'Top-1':>8} {'Top-3':>8}")
    print("-" * 58)

    for name, use_entropy, use_direction in configs:
        print(f"Loading recordings (direction={use_direction})...", flush=True)
        recordings = load_recordings(include_direction=use_direction)
        print(f"  {len(recordings)} recordings loaded. Running LOO...", flush=True)
        top1, top3, total = run_loo(recordings, use_entropy=use_entropy)
        print(f"{name:<40} {top1}/{total} ({top1/total:.1%}) {top3}/{total} ({top3/total:.1%})")
        sys.stdout.flush()

    print("\nDone.")


if __name__ == "__main__":
    main()
