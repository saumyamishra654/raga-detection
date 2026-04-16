#!/usr/bin/env python3
"""Offline truncation sweep for raga detection.

For each recording in the GT CSV and each time window W (in seconds), slice
the cached pitch CSVs and transcribed_notes.csv to the first W seconds
(or full / last N seconds), then rerun histogram-based tonic scoring and
n-gram LM raga scoring. Record per-recording outcomes and per-window
aggregates.

No stem separation or pitch extraction is performed -- this operates purely
on cached CSVs already on the Extreme SSD. Meant to answer:
  "Does restricting analysis to the first N minutes (skipping drut sections)
   change tonic-detection and/or LM raga-detection accuracy?"

Caveat: the LM model was trained on the same 297 recordings we evaluate
against, so absolute accuracy numbers are inflated by training-data
leakage. The DELTA across windows is what we care about.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np

# Make the project importable when run as a bare script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from raga_pipeline.analysis import compute_cent_histograms, detect_peaks  # noqa: E402
from raga_pipeline.audio import PitchData, load_pitch_from_csv  # noqa: E402
from raga_pipeline.config import find_default_raga_db_path  # noqa: E402
from raga_pipeline.language_model import (  # noqa: E402
    NgramModel,
    _load_raw_notes_from_csv,
    _tonic_name_to_midi,
)
from raga_pipeline.raga import RagaDatabase, RagaScorer  # noqa: E402
from raga_pipeline.sequence import tokenize_notes_for_lm  # noqa: E402


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
MALE_TONIC_RANGE = [0, 1, 2, 3, 4, 5, 6]
FEMALE_TONIC_RANGE = [7, 8, 9, 10, 11, 0]

# --- Pipeline parity defaults (mirror raga_pipeline/config.py) ---
# Confidence thresholds applied after load_pitch_from_csv (audio.py:1403-1405).
VOCAL_CONFIDENCE = 0.95
ACCOMP_CONFIDENCE = 0.80
# Histogram / peak-detection defaults (config.py:98, 103-104).
USE_CONFIDENCE_WEIGHTS = True
PROMINENCE_HIGH_FACTOR = 0.01
PROMINENCE_LOW_FACTOR = 0.03

CHECKPOINT_FIELDS = [
    "filename",
    "window_label",
    "window_sec",
    "gt_tonic",
    "gt_raga",
    "detected_tonic",
    "tonic_match",
    "hist_top1_raga",
    "hist_raga_match",
    "lm_top1_raga_gt_tonic",
    "lm_match_gt_tonic",
    "lm_top1_raga_det_tonic",
    "lm_match_det_tonic",
    "n_voiced_frames",
    "n_peaks",
    "n_notes",
    "status",
]


# ---------------------------------------------------------------------------
# Name / tonic helpers
# ---------------------------------------------------------------------------

def normalize_tonic_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return ""
    s = s[0].upper() + s[1:]
    return FLAT_TO_SHARP.get(s, s)


def tonic_name_to_pitch_class(name: str) -> int:
    s = normalize_tonic_name(name)
    if s not in NOTE_NAMES:
        raise ValueError(f"Unknown tonic name: {name!r}")
    return NOTE_NAMES.index(s)


def gender_to_tonic_range(gender: str) -> List[int]:
    g = (gender or "").strip().upper()
    return FEMALE_TONIC_RANGE if g.startswith("F") else MALE_TONIC_RANGE


def raga_names_match(candidate_cell: str, gt_name: str) -> bool:
    """The scorer may emit multiple comma-separated aliases in one row."""
    gt_norm = gt_name.strip().lower().replace(" ", "")
    for alias in str(candidate_cell).split(","):
        if alias.strip().lower().replace(" ", "") == gt_norm:
            return True
    return False


# ---------------------------------------------------------------------------
# Pitch slicing
# ---------------------------------------------------------------------------

def slice_pitch_data(pd_: PitchData, start_sec: float, end_sec: Optional[float]) -> PitchData:
    """Return a new PitchData keeping timestamps in [start_sec, end_sec).

    ``end_sec=None`` means no upper bound. Derived arrays are rebuilt from
    the surviving voiced frames.
    """
    ts = pd_.timestamps
    if len(ts) == 0:
        return pd_
    mask = ts >= start_sec
    if end_sec is not None:
        mask &= ts < end_sec
    new_ts = ts[mask]
    new_hz = pd_.pitch_hz[mask]
    new_conf = pd_.confidence[mask]
    new_voicing = pd_.voicing[mask]
    if len(pd_.energy) == len(ts):
        new_energy = pd_.energy[mask]
    else:
        new_energy = pd_.energy

    voiced = (new_hz > 0) & new_voicing
    valid_freqs = new_hz[voiced]
    midi_vals = librosa.hz_to_midi(valid_freqs) if len(valid_freqs) > 0 else np.array([])

    return PitchData(
        timestamps=new_ts,
        pitch_hz=new_hz,
        confidence=new_conf,
        voicing=new_voicing,
        valid_freqs=valid_freqs,
        midi_vals=midi_vals,
        energy=new_energy,
        frame_period=pd_.frame_period,
        audio_path=pd_.audio_path,
    )


def window_spec(label: str, duration_sec: float, window_sec: int) -> tuple:
    """Return (start_sec, end_sec or None) for a window label applied to a
    recording of length duration_sec.

    Labels: "full" -> full recording. "first_<N>" -> [0, N). "last_<N>" ->
    [duration-N, duration].
    """
    if label == "full":
        return (0.0, None)
    if label.startswith("first_"):
        return (0.0, float(window_sec))
    if label.startswith("last_"):
        start = max(0.0, duration_sec - float(window_sec))
        return (start, None)
    raise ValueError(f"Unknown window label {label!r}")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> Dict[tuple, dict]:
    if not path.exists():
        return {}
    rows: Dict[tuple, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row.get("filename", ""), row.get("window_label", ""))
            rows[key] = row
    return rows


def append_checkpoint_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CHECKPOINT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CHECKPOINT_FIELDS})


# ---------------------------------------------------------------------------
# Scoring wrappers
# ---------------------------------------------------------------------------

def histogram_top1(
    scorer: RagaScorer,
    pd_vocal: PitchData,
    pd_accomp: PitchData,
    tonic_candidates: List[int],
    raga_filter: str,
) -> tuple:
    """Return (top_tonic_name, top_raga_cell, n_validated_peaks) from
    histogram scoring. Mirrors driver.py STEP 3 defaults:
    use_confidence_weights=True, prominence_high=0.01, prominence_low=0.03,
    detected_peak_count = raw len(peaks.validated_indices) (no floor).
    """
    if len(pd_vocal.midi_vals) == 0:
        return None, None, 0

    histograms = compute_cent_histograms(
        pd_vocal, use_confidence_weights=USE_CONFIDENCE_WEIGHTS,
    )
    peaks = detect_peaks(
        histograms,
        prominence_high_factor=PROMINENCE_HIGH_FACTOR,
        prominence_low_factor=PROMINENCE_LOW_FACTOR,
    )
    n_peaks = int(len(peaks.validated_indices))

    df = scorer.score(
        pitch_data_vocals=pd_vocal,
        pitch_data_accomp=pd_accomp if len(pd_accomp.timestamps) > 0 else None,
        detected_peak_count=n_peaks,
        instrument_mode="vocal",
        tonic_candidates=tonic_candidates,
        bias_cents=None,
        raga_filter=raga_filter,
    )
    if len(df) == 0:
        return None, None, n_peaks
    top = df.iloc[0]
    return str(top["tonic_name"]), str(top["raga"]), n_peaks


def lm_top1(
    model: NgramModel,
    notes: list,
    tonic_midi: float,
) -> Optional[str]:
    if not notes:
        return None
    phrases = tokenize_notes_for_lm(notes, tonic_midi, include_direction=False)
    if not phrases:
        return None
    ranked = model.rank_ragas(phrases)
    if not ranked:
        return None
    return ranked[0][0]


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def parse_windows(spec: str) -> List[tuple]:
    """Parse a CLI window spec like 'first_300,first_600,full,last_300' into
    a list of (label, sec) tuples.
    """
    out: List[tuple] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "full" or item == "0":
            out.append(("full", 0))
            continue
        if item.startswith("first_"):
            out.append((item, int(item.split("_", 1)[1])))
            continue
        if item.startswith("last_"):
            out.append((item, int(item.split("_", 1)[1])))
            continue
        # Plain integer -> first_<n>
        out.append((f"first_{int(item)}", int(item)))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-csv", default=str(REPO_ROOT / "compmusic_gt.csv"))
    p.add_argument(
        "--stems-root",
        default="/Volumes/Extreme SSD/stems/separated_stems/htdemucs",
    )
    p.add_argument(
        "--lm-model",
        default=str(REPO_ROOT / "compmusic_ngram_model_uncorrected.json"),
    )
    p.add_argument(
        "--raga-db",
        default=None,
        help="Path to raga_list_final.csv. Auto-resolved via find_default_raga_db_path() if omitted.",
    )
    p.add_argument(
        "--windows",
        default="first_300,first_600,first_900,full,last_300",
        help="Comma-separated window labels: first_<sec>, last_<sec>, full.",
    )
    p.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "truncation_sweep"))
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, process only the first N recordings from the GT CSV.",
    )
    p.add_argument(
        "--progress-name",
        default="progress.csv",
        help="Filename inside --output-dir for the resume-safe progress log.",
    )
    p.add_argument(
        "--summary-name",
        default="summary.csv",
        help="Filename inside --output-dir for the per-window aggregate.",
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    progress_path = output_dir / args.progress_name
    summary_path = output_dir / args.summary_name

    windows = parse_windows(args.windows)
    if not windows:
        print("No windows parsed; aborting.", file=sys.stderr)
        return 2

    # --- Load GT ---
    gt_rows: List[dict] = []
    with open(args.gt_csv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gt_rows.append(row)
    if args.limit > 0:
        gt_rows = gt_rows[: args.limit]
    print(f"GT rows: {len(gt_rows)}")

    # --- Raga filter restricted to the unique GT ragas + their common aliases ---
    gt_ragas = sorted({r.get("Raga", "").strip() for r in gt_rows if r.get("Raga")})
    raga_filter_str = ",".join(gt_ragas)
    print(f"Distinct GT ragas: {len(gt_ragas)}")

    # --- Resources loaded once ---
    raga_db_path = args.raga_db or find_default_raga_db_path()
    if not raga_db_path:
        print("Could not locate raga_list_final.csv", file=sys.stderr)
        return 2
    print(f"Loading raga DB from {raga_db_path} ...")
    raga_db = RagaDatabase(raga_db_path)
    scorer = RagaScorer(raga_db=raga_db)

    print(f"Loading LM from {args.lm_model} ...")
    with open(args.lm_model, "r", encoding="utf-8") as fh:
        model = NgramModel.from_dict(json.load(fh))
    print(f"LM ragas: {len(model.ragas())}")

    # --- Resume ---
    processed = load_checkpoint(progress_path)
    if processed:
        print(f"Resuming: {len(processed)} checkpointed (filename,window) rows.")

    # --- Main loop ---
    stems_root = Path(args.stems_root)
    total_pairs = len(gt_rows) * len(windows)
    done_count = 0
    t0 = time.time()

    for row_idx, row in enumerate(gt_rows):
        fname = row.get("Filename", "").strip()
        gt_raga = row.get("Raga", "").strip()
        gt_tonic = normalize_tonic_name(row.get("Tonic", ""))
        gender = row.get("Gender", "").strip()
        if not fname or not gt_raga or not gt_tonic:
            print(f"[{row_idx}] SKIP: missing GT fields in row {row}")
            continue

        rec_dir = stems_root / fname
        vocal_csv = rec_dir / "vocals_pitch_data.csv"
        accomp_csv = rec_dir / "accompaniment_pitch_data.csv"
        notes_csv = rec_dir / "transcribed_notes.csv"

        # Decide which windows remain for this recording
        remaining = [w for w in windows if (fname, w[0]) not in processed]
        if not remaining:
            done_count += len(windows)
            continue

        # If any of the required files are missing, record a "missing" row per
        # remaining window and move on.
        missing = [p for p in (vocal_csv, accomp_csv, notes_csv) if not p.exists()]
        if missing:
            for label, sec in remaining:
                append_checkpoint_row(progress_path, {
                    "filename": fname,
                    "window_label": label,
                    "window_sec": sec,
                    "gt_tonic": gt_tonic,
                    "gt_raga": gt_raga,
                    "status": "missing",
                })
                done_count += 1
            continue

        # --- Load full pitch / notes once per recording ---
        # Pipeline parity: reapply confidence thresholds after loading cache
        # (raga_pipeline/audio.py:1403-1405).
        try:
            pd_vocal_full = load_pitch_from_csv(str(vocal_csv)).apply_confidence_threshold(VOCAL_CONFIDENCE)
            pd_accomp_full = load_pitch_from_csv(str(accomp_csv)).apply_confidence_threshold(ACCOMP_CONFIDENCE)
            notes_full = _load_raw_notes_from_csv(notes_csv)
        except Exception as exc:
            for label, sec in remaining:
                append_checkpoint_row(progress_path, {
                    "filename": fname,
                    "window_label": label,
                    "window_sec": sec,
                    "gt_tonic": gt_tonic,
                    "gt_raga": gt_raga,
                    "status": f"load_error: {exc}",
                })
                done_count += 1
            continue

        duration = float(pd_vocal_full.timestamps[-1]) if len(pd_vocal_full.timestamps) else 0.0
        tonic_candidates = gender_to_tonic_range(gender)
        gt_tonic_midi = _tonic_name_to_midi(gt_tonic)

        for label, sec in remaining:
            done_count += 1
            start_s, end_s = window_spec(label, duration, sec)
            pd_vocal_w = slice_pitch_data(pd_vocal_full, start_s, end_s)
            pd_accomp_w = slice_pitch_data(pd_accomp_full, start_s, end_s)

            # Notes: overlap the window. A note counts if any portion of it
            # is inside [start_s, end_s). Avoids dropping notes that straddle
            # the boundary (auditor callout).
            if end_s is None:
                notes_w = [n for n in notes_full if n.end > start_s]
            else:
                notes_w = [n for n in notes_full if n.end > start_s and n.start < end_s]

            row_out = {
                "filename": fname,
                "window_label": label,
                "window_sec": sec,
                "gt_tonic": gt_tonic,
                "gt_raga": gt_raga,
                "n_voiced_frames": int(len(pd_vocal_w.midi_vals)),
                "n_notes": int(len(notes_w)),
                "status": "ok",
            }

            # --- Histogram / tonic ---
            try:
                det_tonic, hist_raga_cell, n_peaks = histogram_top1(
                    scorer, pd_vocal_w, pd_accomp_w, tonic_candidates, raga_filter_str,
                )
                row_out["n_peaks"] = n_peaks
            except Exception as exc:
                det_tonic, hist_raga_cell = None, None
                row_out["status"] = f"hist_error: {exc}"
                row_out["n_peaks"] = 0

            row_out["detected_tonic"] = det_tonic or ""
            row_out["tonic_match"] = str(det_tonic == gt_tonic).lower() if det_tonic else ""
            row_out["hist_top1_raga"] = hist_raga_cell or ""
            row_out["hist_raga_match"] = (
                str(raga_names_match(hist_raga_cell, gt_raga)).lower()
                if hist_raga_cell else ""
            )

            # --- LM with GT tonic ---
            try:
                lm_gt_raga = lm_top1(model, notes_w, gt_tonic_midi)
            except Exception as exc:
                lm_gt_raga = None
                row_out["status"] = f"lm_gt_error: {exc}"
            row_out["lm_top1_raga_gt_tonic"] = lm_gt_raga or ""
            row_out["lm_match_gt_tonic"] = (
                str(raga_names_match(lm_gt_raga, gt_raga)).lower() if lm_gt_raga else ""
            )

            # --- LM with histogram-detected tonic ---
            if det_tonic:
                try:
                    det_tonic_midi = _tonic_name_to_midi(det_tonic)
                    lm_det_raga = lm_top1(model, notes_w, det_tonic_midi)
                except Exception as exc:
                    lm_det_raga = None
                    row_out["status"] = f"lm_det_error: {exc}"
            else:
                lm_det_raga = None

            row_out["lm_top1_raga_det_tonic"] = lm_det_raga or ""
            row_out["lm_match_det_tonic"] = (
                str(raga_names_match(lm_det_raga, gt_raga)).lower() if lm_det_raga else ""
            )

            append_checkpoint_row(progress_path, row_out)

            if done_count % 25 == 0 or done_count == total_pairs:
                elapsed = time.time() - t0
                rate = done_count / max(elapsed, 1e-6)
                eta = (total_pairs - done_count) / max(rate, 1e-6) / 60.0
                print(
                    f"[{done_count}/{total_pairs}] {fname} {label}: "
                    f"tonic {det_tonic}->{gt_tonic} ({'OK' if det_tonic == gt_tonic else 'NO'}) "
                    f"| LM_gt={lm_gt_raga} | LM_det={lm_det_raga} "
                    f"| ETA {eta:.1f}m"
                )

    # --- Aggregate summary ---
    rows_all = load_checkpoint(progress_path)
    print(f"Writing summary over {len(rows_all)} checkpointed rows...")

    from collections import defaultdict
    agg: Dict[str, dict] = defaultdict(lambda: {
        "n": 0, "tonic_ok": 0, "hist_ok": 0, "lm_gt_ok": 0, "lm_det_ok": 0,
    })
    for row in rows_all.values():
        if row.get("status") != "ok":
            continue
        label = row["window_label"]
        d = agg[label]
        d["n"] += 1
        if row.get("tonic_match") == "true":
            d["tonic_ok"] += 1
        if row.get("hist_raga_match") == "true":
            d["hist_ok"] += 1
        if row.get("lm_match_gt_tonic") == "true":
            d["lm_gt_ok"] += 1
        if row.get("lm_match_det_tonic") == "true":
            d["lm_det_ok"] += 1

    summary_fields = [
        "window_label",
        "n",
        "tonic_top1_acc",
        "hist_raga_top1_acc",
        "lm_raga_top1_acc_gt_tonic",
        "lm_raga_top1_acc_det_tonic",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        for label, d in agg.items():
            n = max(d["n"], 1)
            writer.writerow({
                "window_label": label,
                "n": d["n"],
                "tonic_top1_acc": round(d["tonic_ok"] / n, 4),
                "hist_raga_top1_acc": round(d["hist_ok"] / n, 4),
                "lm_raga_top1_acc_gt_tonic": round(d["lm_gt_ok"] / n, 4),
                "lm_raga_top1_acc_det_tonic": round(d["lm_det_ok"] / n, 4),
            })
    print(f"Wrote {summary_path}")

    # Console print the summary for convenience.
    print("\n=== Truncation sweep summary ===")
    print(f"{'window':<12} {'n':>4} {'tonic':>8} {'hist':>8} {'lm_gt':>8} {'lm_det':>8}")
    for label, d in agg.items():
        n = max(d["n"], 1)
        print(
            f"{label:<12} {d['n']:>4} "
            f"{d['tonic_ok']/n*100:>7.1f}% "
            f"{d['hist_ok']/n*100:>7.1f}% "
            f"{d['lm_gt_ok']/n*100:>7.1f}% "
            f"{d['lm_det_ok']/n*100:>7.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
