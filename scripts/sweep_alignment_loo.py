#!/usr/bin/env python3
"""Experiment 28: Noisy-channel alignment LM scoring.

Phase 1: LOO feature collection. For each recording, for each histogram
candidate, tokenize uncorrected test notes and score against the
corrected-trained LM using alignment.

Phase 2: Score with multiple calibration methods on collected features.

Train side: correct raw notes with GT raga/tonic, tokenize, train LM.
Test side: tokenize raw notes WITHOUT correction, score with alignment.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from raga_pipeline.analysis import compute_cent_histograms, detect_peaks
from raga_pipeline.audio import PitchData, load_pitch_from_csv
from raga_pipeline.config import find_default_raga_db_path
from raga_pipeline.language_model import NgramModel, _load_raw_notes_from_csv, _tonic_name_to_midi
from raga_pipeline.language_model.alignment import (
    AlignmentConfig,
    build_substitution_map,
    score_sequence_aligned,
)
from raga_pipeline.raga import RagaDatabase, RagaScorer, apply_raga_correction_to_notes, get_raga_notes
from raga_pipeline.sequence import tokenize_notes_for_lm

# ---------------------------------------------------------------------------
# Constants (matching v3 pipeline parity)
# ---------------------------------------------------------------------------

FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
TONIC_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}
MALE_TONIC_RANGE = [0, 1, 2, 3, 4, 5, 6]
FEMALE_TONIC_RANGE = [7, 8, 9, 10, 11, 0]

VOCAL_CONFIDENCE = 0.95
ACCOMP_CONFIDENCE = 0.80
USE_CONFIDENCE_WEIGHTS = True
PROMINENCE_HIGH_FACTOR = 0.01
PROMINENCE_LOW_FACTOR = 0.03

SMOOTHING = "add-k"
SMOOTHING_K = 0.01
MIN_RECORDINGS = 3

FEATURE_FIELDS = [
    "rec_idx", "filename", "gt_raga", "gt_tonic",
    "cand_tonic", "cand_raga", "is_gt",
    "hist_score", "hist_gated",
    "lm_per_token", "skip_fraction", "sub_fraction", "scale_size",
    "n_matched", "n_skipped",
]


# ---------------------------------------------------------------------------
# Helpers (from v3)
# ---------------------------------------------------------------------------

def normalize_tonic(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return ""
    s = s[0].upper() + s[1:]
    return FLAT_TO_SHARP.get(s, s)


def gender_to_tonic_range(gender: str) -> List[int]:
    g = (gender or "").strip().upper()
    return FEMALE_TONIC_RANGE if g.startswith("F") else MALE_TONIC_RANGE


def raga_names_match(a: str, b: str) -> bool:
    bn = b.strip().lower().replace(" ", "")
    for alias in str(a).split(","):
        if alias.strip().lower().replace(" ", "") == bn:
            return True
    return False


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# Data loading (train on corrected, test on raw)
# ---------------------------------------------------------------------------

def load_all_data(gt_rows, stems_root, transcription_root, raga_db):
    """Load pitch data, raw notes, and corrected training phrases.

    For each recording:
    - raw_notes: uncorrected Note objects from transcribed_notes.csv
    - train_phrases: GT-raga-corrected and tokenized (for LM training)
    """
    data: Dict[str, dict] = {}
    filenames: List[str] = []
    skipped = 0

    for i, gt in enumerate(gt_rows):
        fname = gt.get("Filename", "").strip()
        raga = gt.get("Raga", "").strip()
        tonic = normalize_tonic(gt.get("Tonic", ""))
        gender = gt.get("Gender", "").strip()
        if not fname or not raga or not tonic:
            skipped += 1
            continue

        rec_dir = stems_root / fname
        vocal_csv = rec_dir / "vocals_pitch_data.csv"
        if not vocal_csv.exists():
            vocal_csv = rec_dir / "melody_pitch_data.csv"
        accomp_csv = rec_dir / "accompaniment_pitch_data.csv"
        trans_csv = transcription_root / fname / "transcribed_notes.csv"

        if not vocal_csv.exists() or not trans_csv.exists():
            skipped += 1
            continue

        try:
            pd_vocal = load_pitch_from_csv(str(vocal_csv)).apply_confidence_threshold(VOCAL_CONFIDENCE)
            pd_accomp = (
                load_pitch_from_csv(str(accomp_csv)).apply_confidence_threshold(ACCOMP_CONFIDENCE)
                if accomp_csv.exists() else
                PitchData(timestamps=np.array([]), pitch_hz=np.array([]),
                          voiced_mask=np.array([], dtype=bool), confidence=np.array([]))
            )

            if len(pd_vocal.midi_vals) > 0:
                histograms = compute_cent_histograms(pd_vocal, use_confidence_weights=USE_CONFIDENCE_WEIGHTS)
                peaks = detect_peaks(histograms, prominence_high_factor=PROMINENCE_HIGH_FACTOR,
                                     prominence_low_factor=PROMINENCE_LOW_FACTOR)
                n_peaks = len(peaks.validated_indices)
            else:
                n_peaks = 0

            raw_notes = _load_raw_notes_from_csv(trans_csv)
            if not raw_notes:
                skipped += 1
                continue

            tonic_midi = _tonic_name_to_midi(tonic)
            tonic_pc = TONIC_MAP.get(tonic, 0)

            # Training: correct with GT raga/tonic, then tokenize
            corrected_notes, gt_stats, _ = apply_raga_correction_to_notes(
                raw_notes, raga_db, raga, tonic_pc, max_distance=1.0, keep_impure=False,
            )
            train_phrases = tokenize_notes_for_lm(corrected_notes, tonic_midi) if corrected_notes else []

            gt_scale_size = len(gt_stats.get("valid_pcs", []))

            if not train_phrases:
                skipped += 1
                continue

            data[fname] = {
                "raga": raga, "tonic": tonic, "tonic_pc": tonic_pc, "gender": gender,
                "pd_vocal": pd_vocal, "pd_accomp": pd_accomp, "n_peaks": n_peaks,
                "raw_notes": raw_notes, "train_phrases": train_phrases,
                "gt_scale_size": gt_scale_size,
            }
            filenames.append(fname)

        except Exception as exc:
            print(f"  [{i+1}] {fname}: load error ({exc})")
            skipped += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(gt_rows)}] loaded {fname}")

    return data, filenames, skipped


# ---------------------------------------------------------------------------
# Phase 1: LOO alignment feature collection
# ---------------------------------------------------------------------------

def collect_features(
    filenames: List[str],
    data: Dict[str, dict],
    raga_db: RagaDatabase,
    raga_filter_str: str,
    lm_order: int,
    align_config: AlignmentConfig,
) -> List[dict]:
    """Run LOO: train corrected LM, score uncorrected test with alignment."""
    all_features: List[dict] = []
    t0 = time.time()

    for rec_idx, held_out in enumerate(filenames):
        rec = data[held_out]
        gt_raga, gt_tonic, gender = rec["raga"], rec["tonic"], rec["gender"]
        raw_notes = rec["raw_notes"]

        # --- Histogram scoring ---
        pd_vocal, pd_accomp, n_peaks = rec["pd_vocal"], rec["pd_accomp"], rec["n_peaks"]
        tonic_candidates = gender_to_tonic_range(gender)
        scorer = RagaScorer(raga_db=raga_db)
        try:
            hist_df = scorer.score(
                pitch_data_vocals=pd_vocal,
                pitch_data_accomp=pd_accomp if len(pd_accomp.timestamps) > 0 else None,
                detected_peak_count=n_peaks, instrument_mode="vocal",
                tonic_candidates=tonic_candidates, bias_cents=None,
                raga_filter=raga_filter_str,
            )
        except Exception:
            continue

        if len(hist_df) == 0:
            continue

        # --- Train LOO LM (corrected) ---
        lambdas = [1.0 / lm_order] * lm_order
        model = NgramModel(order=lm_order, smoothing=SMOOTHING, smoothing_k=SMOOTHING_K, lambdas=lambdas)
        raga_rec_counts: Dict[str, int] = defaultdict(int)
        for fname in filenames:
            if fname == held_out:
                continue
            other = data[fname]
            if other["train_phrases"]:
                model.add_sequence(other["raga"], other["train_phrases"])
                raga_rec_counts[other["raga"]] += 1

        for raga in list(model.ragas()):
            if raga_rec_counts.get(raga, 0) < MIN_RECORDINGS:
                model.remove_raga(raga)
        model.finalize()
        lm_raga_set = set(model.ragas())

        # Pre-build substitution map for this fold's vocabulary
        sub_map = build_substitution_map(model.vocabulary, align_config.max_sub_distance)

        # --- Histogram gate ---
        gated_positive = [(int(r["tonic"]), str(r["raga"]))
                          for _, r in hist_df.iterrows()
                          if float(r.get("fit_score", r.get("score", 0))) > 0]
        if not gated_positive:
            gated_positive = [(int(r["tonic"]), str(r["raga"]))
                              for _, r in hist_df.head(20).iterrows()]
        gated_keys = set()
        for tpc, raga_group in gated_positive:
            for r in raga_group.split(","):
                r = r.strip()
                if r and r in lm_raga_set:
                    gated_keys.add((tpc, r))

        # --- Tokenize test ONCE per candidate tonic (uncorrected!) ---
        unique_tonics = set(int(r["tonic"]) for _, r in hist_df.iterrows())
        test_phrases_cache: Dict[int, List[List[str]]] = {}
        for tpc in unique_tonics:
            tonic_midi = 60.0 + tpc
            test_phrases_cache[tpc] = tokenize_notes_for_lm(raw_notes, tonic_midi)

        # --- Score all candidates with alignment ---
        for _, cand_row in hist_df.iterrows():
            cand_tonic = int(cand_row["tonic"])
            raga_group = str(cand_row["raga"])
            hist_score = float(cand_row.get("fit_score", cand_row.get("score", 0.0)))
            individual_ragas = [r.strip() for r in raga_group.split(",") if r.strip()]
            individual_ragas = [r for r in individual_ragas if r in lm_raga_set]

            test_phrases = test_phrases_cache.get(cand_tonic, [])

            for cand_raga in individual_ragas:
                is_gated = (cand_tonic, cand_raga) in gated_keys

                if test_phrases:
                    result = score_sequence_aligned(
                        model, cand_raga, test_phrases, align_config, sub_map,
                    )
                    lm_per_token = result.lm_per_token
                    skip_frac = result.skip_fraction
                    sub_frac = result.n_substituted / result.n_matched if result.n_matched > 0 else 0.0
                    n_matched = result.n_matched
                    n_skipped = result.n_skipped
                else:
                    lm_per_token = -999.0
                    skip_frac = 1.0
                    sub_frac = 0.0
                    n_matched = 0
                    n_skipped = 0

                scale_size = len(get_raga_notes(raga_db, cand_raga, cand_tonic))

                all_features.append({
                    "rec_idx": rec_idx,
                    "filename": held_out,
                    "gt_raga": gt_raga,
                    "gt_tonic": gt_tonic,
                    "cand_tonic": cand_tonic,
                    "cand_raga": cand_raga,
                    "is_gt": int(raga_names_match(cand_raga, gt_raga)),
                    "hist_score": round(hist_score, 4),
                    "hist_gated": int(is_gated),
                    "lm_per_token": round(lm_per_token, 6),
                    "skip_fraction": round(skip_frac, 4),
                    "sub_fraction": round(sub_frac, 4),
                    "scale_size": scale_size,
                    "n_matched": n_matched,
                    "n_skipped": n_skipped,
                })

        elapsed = time.time() - t0
        eta = elapsed / (rec_idx + 1) * (len(filenames) - rec_idx - 1)
        if (rec_idx + 1) % 10 == 0 or rec_idx == 0:
            print(f"  [{rec_idx+1}/{len(filenames)}] {held_out} "
                  f"({len(all_features)} features, {elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    return all_features


# ---------------------------------------------------------------------------
# Phase 2: Scoring methods (all fold-safe)
# ---------------------------------------------------------------------------

def _fold_safe_stats(df, held_out_idx):
    """Compute per-scale-size mu/sigma from train fold only (excl held-out)."""
    train = df[df["rec_idx"] != held_out_idx]
    stats: Dict[int, Tuple[float, float]] = {}
    for k, grp in train.groupby("scale_size"):
        vals = grp["lm_per_token"].values
        stats[int(k)] = (float(np.mean(vals)), max(float(np.std(vals)), 1e-6))
    return stats


def _eval_method(df, n_recs, score_fn, use_gating):
    """Generic evaluator: for each recording, pick the best candidate by score_fn.

    score_fn(row, stats_by_k) -> float.
    Runs both gated and ungated if use_gating == "both".
    """
    results = {}
    for gating_label, do_gate in ([("gated", True), ("ungated", False)]
                                   if use_gating == "both"
                                   else [(use_gating, use_gating == "gated")]):
        correct = 0
        rec_indices = sorted(df["rec_idx"].unique())

        for held_out_idx in rec_indices:
            stats_by_k = _fold_safe_stats(df, held_out_idx)
            grp = df[df["rec_idx"] == held_out_idx]

            if do_gate:
                candidates = grp[grp["hist_gated"] == 1]
                if len(candidates) == 0:
                    candidates = grp.nlargest(20, "hist_score")
            else:
                candidates = grp

            best_score = -1e30
            best_is_gt = False
            for _, row in candidates.iterrows():
                s = score_fn(row, stats_by_k)
                if s > best_score:
                    best_score = s
                    best_is_gt = row["is_gt"] == 1
            if best_is_gt:
                correct += 1

        results[gating_label] = correct / n_recs
    return results


def phase2_score(features_csv: Path, output_dir: Path) -> None:
    """Score collected features with multiple calibration methods.

    All methods that use scale-size z-scoring compute mu/sigma from the
    train fold only (excluding the held-out recording) to prevent leakage.
    All methods are evaluated both gated and ungated for comparability.
    """
    import pandas as pd

    df = pd.read_csv(features_csv)
    n_recs = df["rec_idx"].nunique()
    print(f"\nPhase 2: {len(df)} features across {n_recs} recordings")

    if n_recs == 0:
        print("  No features to score. Need more recordings (MIN_RECORDINGS=3 per raga for LOO).")
        return

    all_results = {}

    # --- Method A: Raw alignment LM (no calibration) ---
    def score_a(row, stats_by_k):
        return row["lm_per_token"]

    all_results["A: Raw alignment LM"] = _eval_method(df, n_recs, score_a, "both")

    # --- Method B: Z-scored LM by scale size (fold-safe) ---
    def score_b(row, stats_by_k):
        k = int(row["scale_size"])
        mu, sigma = stats_by_k.get(k, (0.0, 1.0))
        return (row["lm_per_token"] - mu) / sigma

    all_results["B: Z-scored LM"] = _eval_method(df, n_recs, score_b, "both")

    # --- Method C: Z-scored LM - gamma * skip_fraction (fold-safe) ---
    for gamma in [0.5, 1.0, 2.0, 5.0]:
        def score_c(row, stats_by_k, _g=gamma):
            k = int(row["scale_size"])
            mu, sigma = stats_by_k.get(k, (0.0, 1.0))
            z_lm = (row["lm_per_token"] - mu) / sigma
            return z_lm - _g * row["skip_fraction"]

        all_results[f"C: Z-LM - {gamma}*skip"] = _eval_method(df, n_recs, score_c, "both")

    # --- Method D: RRF (hist rank + alignment LM rank) ---
    for gating_label, do_gate in [("gated", True), ("ungated", False)]:
        correct = 0
        rec_indices = sorted(df["rec_idx"].unique())
        for held_out_idx in rec_indices:
            grp = df[df["rec_idx"] == held_out_idx].copy()
            if do_gate:
                candidates = grp[grp["hist_gated"] == 1]
                if len(candidates) == 0:
                    candidates = grp.nlargest(20, "hist_score")
                candidates = candidates.copy()
            else:
                candidates = grp

            candidates["hist_rank"] = candidates["hist_score"].rank(ascending=False, method="min")
            candidates["lm_rank"] = candidates["lm_per_token"].rank(ascending=False, method="min")
            candidates["rrf"] = 1.0 / (60 + candidates["hist_rank"]) + 1.0 / (60 + candidates["lm_rank"])
            best = candidates.loc[candidates["rrf"].idxmax()]
            if best["is_gt"] == 1:
                correct += 1
        all_results.setdefault("D: RRF (hist+LM)", {})[gating_label] = correct / n_recs

    # --- Method E: Logistic regression (grouped LOO, fold-safe by construction) ---
    for gating_label, do_gate in [("gated", True), ("ungated", False)]:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            all_results.setdefault("E: Logistic", {})[gating_label] = -1.0
            continue

        correct = 0
        rec_indices = sorted(df["rec_idx"].unique())

        for held_out_idx in rec_indices:
            train_df = df[df["rec_idx"] != held_out_idx]
            test_df = df[df["rec_idx"] == held_out_idx]

            if do_gate:
                test_df = test_df[test_df["hist_gated"] == 1]
                if len(test_df) == 0:
                    test_df = df[df["rec_idx"] == held_out_idx].nlargest(20, "hist_score")

            if len(test_df) == 0:
                continue

            feature_cols = ["lm_per_token", "skip_fraction", "scale_size", "hist_score"]
            X_train = train_df[feature_cols].values
            y_train = train_df["is_gt"].values

            if len(np.unique(y_train)) < 2:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)

            lr = LogisticRegression(max_iter=1000, class_weight="balanced")
            lr.fit(X_train_s, y_train)

            X_test = test_df[feature_cols].values
            X_test_s = scaler.transform(X_test)
            probs = lr.predict_proba(X_test_s)[:, 1]

            best_idx = np.argmax(probs)
            if test_df.iloc[best_idx]["is_gt"] == 1:
                correct += 1

        all_results.setdefault("E: Logistic", {})[gating_label] = correct / n_recs

    # --- Method F: Combined z_lm + hist (fold-safe) ---
    def score_f(row, stats_by_k):
        k = int(row["scale_size"])
        mu, sigma = stats_by_k.get(k, (0.0, 1.0))
        z_lm = (row["lm_per_token"] - mu) / sigma
        return 0.5 * row["hist_score"] + 2.0 * z_lm

    all_results["F: 0.5*hist + 2.0*z_lm"] = _eval_method(df, n_recs, score_f, "both")

    # Print results
    print("\n=== Scoring Comparison (gated | ungated) ===")
    comparison_rows = []
    for method, gating_results in all_results.items():
        g = gating_results.get("gated", -1)
        u = gating_results.get("ungated", -1)
        print(f"  {method}: {g*100:.1f}% | {u*100:.1f}%")
        comparison_rows.append({
            "method": method,
            "top1_gated": round(g * 100, 2),
            "top1_ungated": round(u * 100, 2),
        })

    # Save
    comp_path = output_dir / "scoring_comparison.csv"
    write_csv(comp_path, ["method", "top1_gated", "top1_ungated"], comparison_rows)
    print(f"\nSaved: {comp_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-csv", default=str(REPO_ROOT / "compmusic_gt.csv"))
    parser.add_argument("--stems-root",
                        default="/Volumes/Extreme SSD/stems/separated_stems_nocorrection/htdemucs")
    parser.add_argument("--transcription-root",
                        default="/Volumes/Extreme SSD/stems/separated_stems_nocorrection/htdemucs",
                        help="Root for transcribed_notes.csv lookup")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "alignment_loo"))
    parser.add_argument("--lm-order", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--features-csv", type=str, default="",
                        help="Skip Phase 1, load features from this CSV for Phase 2 only")

    # Alignment config
    parser.add_argument("--lambda-skip", type=float, default=0.5)
    parser.add_argument("--lambda-match", type=float, default=2.0)
    parser.add_argument("--lambda-sub", type=float, default=0.3)
    parser.add_argument("--beam-width", type=int, default=200)
    parser.add_argument("--max-sub-distance", type=int, default=2)

    parser.add_argument("--raga-filter", default="")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_csv = Path(args.features_csv) if args.features_csv else output_dir / "features.csv"

    if args.features_csv:
        if not features_csv.exists():
            print(f"[ERROR] --features-csv provided but file not found: {features_csv}")
            sys.exit(1)
        print(f"Skipping Phase 1, loading features from {features_csv}")
        phase2_score(features_csv, output_dir)
        return

    # --- Phase 1 ---
    print("=== Phase 1: LOO Alignment Feature Collection ===")

    raga_db_path = find_default_raga_db_path()
    if not raga_db_path:
        print("[ERROR] No raga database found")
        sys.exit(1)
    raga_db = RagaDatabase(raga_db_path)
    print(f"Raga DB: {raga_db_path} ({len(raga_db.all_ragas)} ragas)")

    with open(args.gt_csv, "r", encoding="utf-8") as fh:
        gt_rows = list(csv.DictReader(fh))
    if args.limit > 0:
        gt_rows = gt_rows[:args.limit]

    stems_root = Path(args.stems_root)
    trans_root = Path(args.transcription_root)
    data, filenames, skipped = load_all_data(gt_rows, stems_root, trans_root, raga_db)
    print(f"Loaded: {len(filenames)} recordings, skipped: {skipped}")

    # Persist evaluated universe for baseline comparability
    eval_list_path = output_dir / "evaluated_filenames.txt"
    eval_list_path.write_text("\n".join(filenames) + "\n", encoding="utf-8")
    print(f"Evaluated universe: {len(filenames)} recordings -> {eval_list_path}")

    align_config = AlignmentConfig(
        lambda_skip=args.lambda_skip,
        lambda_match=args.lambda_match,
        lambda_sub=args.lambda_sub,
        beam_width=args.beam_width,
        max_sub_distance=args.max_sub_distance,
    )
    print(f"Alignment config: {align_config}")

    features = collect_features(
        filenames, data, raga_db, args.raga_filter, args.lm_order, align_config,
    )
    write_csv(features_csv, FEATURE_FIELDS, features)
    print(f"Phase 1 complete: {len(features)} features saved to {features_csv}")

    # --- Phase 2 ---
    phase2_score(features_csv, output_dir)


if __name__ == "__main__":
    main()
