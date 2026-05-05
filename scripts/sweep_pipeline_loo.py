#!/usr/bin/env python3
"""Offline pipeline LOO: histogram + LM combined scoring on cached data.

For each recording: trains LOO LM on 296 others, runs histogram scorer
on cached pitch CSVs, combines via 0.5*norm_hist + 2.0*norm_lm, reports
top-1/top-3 accuracy. Uses the 30-raga GT filter and clip_2.0 fix.

Much faster than run_pipeline_loo.py (no subprocess, no stem separation).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from raga_pipeline.analysis import compute_cent_histograms, detect_peaks  # noqa: E402
from raga_pipeline.audio import PitchData, load_pitch_from_csv  # noqa: E402
from raga_pipeline.config import find_default_raga_db_path  # noqa: E402
from raga_pipeline.language_model import NgramModel, _load_raw_notes_from_csv, _tonic_name_to_midi  # noqa: E402
from raga_pipeline.raga import RagaDatabase, RagaScorer, _normalize_raga_name  # noqa: E402
from raga_pipeline.sequence import tokenize_notes_for_lm, detect_phrases_by_silence  # noqa: E402

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
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

ALPHA = 0.5   # histogram weight
BETA = 2.0    # LM weight

PROGRESS_FIELDS = [
    "filename", "gt_raga", "gt_tonic", "detected_tonic", "tonic_match",
    "hist_top1_raga", "hist_raga_match",
    "lm_top1_raga", "lm_raga_match",
    "combined_top1_raga", "combined_raga_match", "combined_rank",
    "n_candidates", "status",
]

SUMMARY_FIELDS = [
    "metric", "value",
]


def normalize_tonic(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return ""
    s = s[0].upper() + s[1:]
    return FLAT_TO_SHARP.get(s, s)


def gender_to_tonic_range(gender: str) -> List[int]:
    g = (gender or "").strip().upper()
    return FEMALE_TONIC_RANGE if g.startswith("F") else MALE_TONIC_RANGE


def raga_names_match(candidate_cell: str, gt_name: str) -> bool:
    gt_norm = gt_name.strip().lower().replace(" ", "")
    for alias in str(candidate_cell).split(","):
        if alias.strip().lower().replace(" ", "") == gt_norm:
            return True
    return False


def load_checkpoint(path: Path) -> set:
    done: set = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            done.add(row.get("filename", ""))
    return done


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PROGRESS_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PROGRESS_FIELDS})


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def eval_one(
    held_out: str,
    filenames: List[str],
    data: Dict[str, dict],
    raga_db: RagaDatabase,
    raga_filter_str: str,
    lm_order: int,
    force_gt_tonic: bool = False,
) -> dict:
    """Evaluate one held-out recording: histogram + LOO LM + combined."""
    rec = data[held_out]
    gt_raga = rec["raga"]
    gt_tonic = rec["tonic"]
    gender = rec["gender"]

    result = {
        "filename": held_out,
        "gt_raga": gt_raga,
        "gt_tonic": gt_tonic,
        "status": "ok",
    }

    try:
        # --- Histogram scoring ---
        pd_vocal = rec["pd_vocal"]
        pd_accomp = rec["pd_accomp"]
        n_peaks = rec["n_peaks"]
        if force_gt_tonic:
            gt_tonic_pc = TONIC_MAP.get(gt_tonic, 0)
            tonic_candidates = [gt_tonic_pc]
        else:
            tonic_candidates = gender_to_tonic_range(gender)

        scorer = RagaScorer(raga_db=raga_db)
        hist_df = scorer.score(
            pitch_data_vocals=pd_vocal,
            pitch_data_accomp=pd_accomp if len(pd_accomp.timestamps) > 0 else None,
            detected_peak_count=n_peaks,
            instrument_mode="vocal",
            tonic_candidates=tonic_candidates,
            bias_cents=None,
            raga_filter=raga_filter_str,
        )

        if len(hist_df) == 0:
            result["status"] = "no_candidates"
            return result

        hist_top = hist_df.iloc[0]
        detected_tonic = str(hist_top["tonic_name"])
        hist_top1_raga = str(hist_top["raga"])

        result["detected_tonic"] = detected_tonic
        result["tonic_match"] = str(detected_tonic == gt_tonic).lower()
        result["hist_top1_raga"] = hist_top1_raga
        result["hist_raga_match"] = str(raga_names_match(hist_top1_raga, gt_raga)).lower()

        # --- Train LOO LM ---
        lambdas = [1.0 / lm_order] * lm_order
        model = NgramModel(
            order=lm_order,
            smoothing=SMOOTHING,
            smoothing_k=SMOOTHING_K,
            lambdas=lambdas,
        )

        raga_rec_counts: Dict[str, int] = defaultdict(int)
        for fname in filenames:
            if fname == held_out:
                continue
            other = data[fname]
            train_phrases = other.get("train_phrases", other["phrases"])
            if train_phrases:
                model.add_sequence(other["raga"], train_phrases)
                raga_rec_counts[other["raga"]] += 1

        for raga in list(model._counts.keys()):
            if raga_rec_counts.get(raga, 0) < MIN_RECORDINGS:
                model._counts.pop(raga, None)
                model._context_counts.pop(raga, None)
                model._token_counts.pop(raga, None)
                model._recording_counts.pop(raga, None)

        model.finalize()
        lm_raga_set = set(model.ragas())

        # --- LM-only top-1 (using GT tonic) ---
        phrases_gt = rec["phrases"]
        if phrases_gt:
            lm_ranked = model.rank_ragas(phrases_gt)
            lm_top1 = lm_ranked[0][0] if lm_ranked else ""
        else:
            lm_top1 = ""

        result["lm_top1_raga"] = lm_top1
        result["lm_raga_match"] = str(lm_top1 == gt_raga).lower()

        # --- Combined scoring (replicates driver.py logic) ---
        unique_tonics = set(int(r["tonic"]) for _, r in hist_df.iterrows())
        tonic_phrases_cache: Dict[int, list] = {}
        rms_phr = rec.get("rms_phrases", [])
        notes = rec["raw_notes"]
        for tpc in unique_tonics:
            tonic_midi = 60.0 + tpc
            tonic_phrases_cache[tpc] = tokenize_notes_for_lm(
                notes, tonic_midi, phrases=rms_phr,
            ) if notes else []

        lm_rows: List[dict] = []
        for _, cand_row in hist_df.iterrows():
            cand_tonic = int(cand_row["tonic"])
            raga_group = str(cand_row["raga"])
            hist_score = float(cand_row.get("fit_score", cand_row.get("score", 0.0)))
            individual_ragas = [r.strip() for r in raga_group.split(",") if r.strip()]
            individual_ragas = [r for r in individual_ragas if r in lm_raga_set]
            phrases = tonic_phrases_cache.get(cand_tonic, [])

            for cand_raga in individual_ragas:
                lm_score = model.score_sequence(cand_raga, phrases) if phrases else -999.0
                lm_rows.append({
                    "tonic": cand_tonic,
                    "raga": cand_raga,
                    "histogram_score": hist_score,
                    "lm_score": lm_score,
                })

        if not lm_rows:
            result["combined_top1_raga"] = hist_top1_raga
            result["combined_raga_match"] = result["hist_raga_match"]
            result["combined_rank"] = ""
            result["n_candidates"] = 0
            return result

        # Histogram gate
        gated = [r for r in lm_rows if r["histogram_score"] > 0]
        if not gated:
            gated = sorted(lm_rows, key=lambda r: r["histogram_score"], reverse=True)[:20]
        gated_set = {(r["tonic"], r["raga"]) for r in gated}

        # Normalize within gated
        gated_hist = [r["histogram_score"] for r in gated]
        hist_min, hist_max = min(gated_hist), max(gated_hist)
        hist_range = hist_max - hist_min if hist_max > hist_min else 1.0

        gated_lm = [r["lm_score"] for r in lm_rows
                     if (r["tonic"], r["raga"]) in gated_set and r["lm_score"] > -900]
        lm_min = min(gated_lm) if gated_lm else 0.0
        lm_max = max(gated_lm) if gated_lm else 1.0
        lm_range = lm_max - lm_min if lm_max > lm_min else 1.0

        for row in lm_rows:
            is_gated = (row["tonic"], row["raga"]) in gated_set
            if is_gated:
                norm_hist = (row["histogram_score"] - hist_min) / hist_range
                norm_lm = (row["lm_score"] - lm_min) / lm_range
                row["combined_score"] = ALPHA * norm_hist + BETA * norm_lm
            else:
                row["combined_score"] = -999.0

        lm_rows.sort(key=lambda r: r["combined_score"], reverse=True)

        combined_top1 = lm_rows[0]["raga"]
        combined_rank = None
        for i, r in enumerate(lm_rows):
            if raga_names_match(r["raga"], gt_raga):
                combined_rank = i + 1
                break

        result["combined_top1_raga"] = combined_top1
        result["combined_raga_match"] = str(raga_names_match(combined_top1, gt_raga)).lower()
        result["combined_rank"] = combined_rank if combined_rank else len(lm_rows) + 1
        result["n_candidates"] = len(lm_rows)

    except Exception as exc:
        result["status"] = f"error: {exc}"

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-csv", default=str(REPO_ROOT / "compmusic_gt.csv"))
    parser.add_argument("--stems-root",
                        default="/Volumes/Extreme SSD/stems/separated_stems/htdemucs")
    parser.add_argument("--transcription-root", default=None,
                        help="Separate root for transcribed_notes.csv (default: same as --stems-root)")
    parser.add_argument("--raga-db", default=None)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "pipeline_loo"))
    parser.add_argument("--lm-order", type=int, default=7)
    parser.add_argument("--train-corrected", action="store_true",
                        help="Apply GT raga correction to training recordings before LM training. "
                             "Test recordings always use uncorrected transcriptions.")
    parser.add_argument("--force-gt-tonic", action="store_true",
                        help="Force ground-truth tonic for every recording (measures raga-only accuracy).")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    progress_path = output_dir / "progress.csv"
    summary_path = output_dir / "summary.csv"
    stems_root = Path(args.stems_root)
    transcription_root = Path(args.transcription_root) if args.transcription_root else stems_root
    lm_order = args.lm_order

    raga_db_path = args.raga_db or find_default_raga_db_path()
    if not raga_db_path:
        print("ERROR: raga DB not found", file=sys.stderr)
        return 2
    raga_db = RagaDatabase(raga_db_path)
    print(f"Raga DB: {raga_db_path}")

    with open(args.gt_csv, "r", encoding="utf-8") as fh:
        gt_rows = list(csv.DictReader(fh))
    if args.limit > 0:
        gt_rows = gt_rows[:args.limit]

    gt_ragas = sorted({r.get("Raga", "").strip() for r in gt_rows if r.get("Raga")})
    raga_filter_str = ",".join(gt_ragas)
    print(f"GT rows: {len(gt_rows)}, ragas: {len(gt_ragas)}, raga filter: {len(gt_ragas)} ragas")
    train_corrected = args.train_corrected
    force_gt_tonic = args.force_gt_tonic
    print(f"LM order: {lm_order}")
    print(f"Train corrected: {train_corrected}")
    print(f"Force GT tonic: {force_gt_tonic}")
    print(f"Pitch root: {stems_root}")
    print(f"Transcription root: {transcription_root}")

    # Pre-load all data
    data: Dict[str, dict] = {}
    filenames: List[str] = []
    skipped = 0
    t0 = time.time()

    for i, gt in enumerate(gt_rows):
        fname = gt.get("Filename", "").strip()
        raga = gt.get("Raga", "").strip()
        tonic = normalize_tonic(gt.get("Tonic", ""))
        gender = gt.get("Gender", "").strip()
        if not fname or not raga or not tonic:
            skipped += 1
            continue

        rec_dir = stems_root / fname
        trans_dir = transcription_root / fname
        vocal_csv = rec_dir / "vocals_pitch_data.csv"
        if not vocal_csv.exists():
            vocal_csv = rec_dir / "melody_pitch_data.csv"
        accomp_csv = rec_dir / "accompaniment_pitch_data.csv"
        trans_csv = trans_dir / "transcribed_notes.csv"

        if not vocal_csv.exists() or not trans_csv.exists():
            skipped += 1
            continue

        try:
            pd_vocal = load_pitch_from_csv(str(vocal_csv)).apply_confidence_threshold(VOCAL_CONFIDENCE)
            pd_accomp = load_pitch_from_csv(str(accomp_csv)).apply_confidence_threshold(ACCOMP_CONFIDENCE) if accomp_csv.exists() else PitchData(
                timestamps=np.array([]), pitch_hz=np.array([]),
                voiced_mask=np.array([], dtype=bool), confidence=np.array([])
            )

            if len(pd_vocal.midi_vals) > 0:
                histograms = compute_cent_histograms(pd_vocal, use_confidence_weights=USE_CONFIDENCE_WEIGHTS)
                peaks = detect_peaks(histograms, prominence_high_factor=PROMINENCE_HIGH_FACTOR,
                                     prominence_low_factor=PROMINENCE_LOW_FACTOR)
                n_peaks = len(peaks.validated_indices)
            else:
                n_peaks = 0

            raw_notes = _load_raw_notes_from_csv(trans_csv)
            tonic_midi = _tonic_name_to_midi(tonic)

            # RMS phrase boundaries from vocal energy (tonic-independent)
            rms_phrases = detect_phrases_by_silence(
                raw_notes, pd_vocal.energy, pd_vocal.timestamps,
            ) if raw_notes else []
            phrases = tokenize_notes_for_lm(
                raw_notes, tonic_midi, phrases=rms_phrases,
            ) if raw_notes else []

            # Corrected phrases for training (apply GT raga correction then tokenize)
            corrected_phrases = phrases  # default: same as uncorrected
            if train_corrected and raw_notes and raga_db:
                from raga_pipeline.raga import apply_raga_correction_to_notes
                tonic_pc = TONIC_MAP.get(tonic, 0)
                try:
                    corrected_notes, _stats, _ = apply_raga_correction_to_notes(
                        raw_notes, raga_db, raga, tonic_pc,
                        max_distance=1.0, keep_impure=False,
                    )
                    corrected_rms = detect_phrases_by_silence(
                        corrected_notes, pd_vocal.energy, pd_vocal.timestamps,
                    ) if corrected_notes else []
                    corrected_phrases = tokenize_notes_for_lm(
                        corrected_notes, tonic_midi, phrases=corrected_rms,
                    ) if corrected_notes else phrases
                except Exception:
                    corrected_phrases = phrases  # fallback to uncorrected

            data[fname] = {
                "raga": raga, "tonic": tonic, "gender": gender,
                "pd_vocal": pd_vocal, "pd_accomp": pd_accomp,
                "n_peaks": n_peaks, "raw_notes": raw_notes,
                "rms_phrases": rms_phrases,
                "phrases": phrases, "train_phrases": corrected_phrases,
            }
            filenames.append(fname)

        except Exception as exc:
            print(f"  [{i+1}] {fname}: load error ({exc})")
            skipped += 1
            continue

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(gt_rows)}] loaded {fname}")

    load_time = time.time() - t0
    print(f"Loaded {len(filenames)} recordings, skipped {skipped} ({load_time:.1f}s)")

    # Check resume
    done = load_checkpoint(progress_path)
    todo = [f for f in filenames if f not in done]
    if done:
        print(f"Resuming: {len(done)} done, {len(todo)} remaining")

    if not todo:
        print("All recordings already evaluated")
    else:
        print(f"Evaluating {len(todo)} recordings...")
        t_start = time.time()
        for idx, fname in enumerate(todo):
            result = eval_one(fname, filenames, data, raga_db, raga_filter_str, lm_order, force_gt_tonic)
            append_row(progress_path, result)

            if (idx + 1) % 10 == 0 or result.get("combined_raga_match") != "true":
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(todo) - idx - 1) / rate / 60 if rate > 0 else 0
                match = result.get("combined_raga_match", "")
                print(f"  [{idx+1}/{len(todo)}] {fname}: combined={match} ETA {eta:.0f}m")

    # Load all progress for summary
    all_rows: List[dict] = []
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as fh:
            all_rows = list(csv.DictReader(fh))

    ok_rows = [r for r in all_rows if r.get("status") == "ok"]
    n = len(ok_rows)
    if n == 0:
        print("No results to summarize")
        return 0

    tonic_correct = sum(1 for r in ok_rows if r.get("tonic_match") == "true")
    hist_correct = sum(1 for r in ok_rows if r.get("hist_raga_match") == "true")
    lm_correct = sum(1 for r in ok_rows if r.get("lm_raga_match") == "true")
    combined_correct = sum(1 for r in ok_rows if r.get("combined_raga_match") == "true")
    combined_top3 = sum(1 for r in ok_rows
                        if r.get("combined_rank") and int(r.get("combined_rank", 999)) <= 3)

    summary = [
        {"metric": "n_recordings", "value": n},
        {"metric": "lm_order", "value": lm_order},
        {"metric": "tonic_top1", "value": round(tonic_correct / n, 4)},
        {"metric": "hist_raga_top1", "value": round(hist_correct / n, 4)},
        {"metric": "lm_raga_top1_gt_tonic", "value": round(lm_correct / n, 4)},
        {"metric": "combined_top1", "value": round(combined_correct / n, 4)},
        {"metric": "combined_top3", "value": round(combined_top3 / n, 4)},
    ]
    write_csv(summary_path, SUMMARY_FIELDS, summary)

    print(f"\n=== Pipeline LOO Results (order={lm_order}) ===")
    print(f"  Recordings:        {n}")
    print(f"  Tonic top-1:       {tonic_correct}/{n} ({tonic_correct/n*100:.1f}%)")
    print(f"  Hist raga top-1:   {hist_correct}/{n} ({hist_correct/n*100:.1f}%)")
    print(f"  LM raga top-1:     {lm_correct}/{n} ({lm_correct/n*100:.1f}%) [GT tonic]")
    print(f"  Combined top-1:    {combined_correct}/{n} ({combined_correct/n*100:.1f}%)")
    print(f"  Combined top-3:    {combined_top3}/{n} ({combined_top3/n*100:.1f}%)")

    errors = [r for r in ok_rows if r.get("combined_raga_match") != "true"]
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for r in sorted(errors, key=lambda x: x["filename"]):
            print(f"    {r['filename']:<30s} true={r['gt_raga']:<20s} "
                  f"pred={r.get('combined_top1_raga',''):<20s} rank={r.get('combined_rank','')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
