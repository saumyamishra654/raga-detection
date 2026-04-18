#!/usr/bin/env python3
"""Offline fit-score saturation calibration sweep.

This is Experiment 16 from the advanced scoring plan. It replays the
histogram scorer on cached pitch CSVs, exposes the pre-clip fit_norm, and
compares a small set of calibration variants without changing raga.py.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import librosa
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from raga_pipeline.analysis import compute_cent_histograms, detect_peaks  # noqa: E402
from raga_pipeline.audio import PitchData, load_pitch_from_csv  # noqa: E402
from raga_pipeline.config import find_default_raga_db_path  # noqa: E402
from raga_pipeline.raga import RagaDatabase, ScoringParams, _normalize_raga_name  # noqa: E402


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
MALE_TONIC_RANGE = [0, 1, 2, 3, 4, 5, 6]
FEMALE_TONIC_RANGE = [7, 8, 9, 10, 11, 0]

VOCAL_CONFIDENCE = 0.95
ACCOMP_CONFIDENCE = 0.80
USE_CONFIDENCE_WEIGHTS = True
PROMINENCE_HIGH_FACTOR = 0.01
PROMINENCE_LOW_FACTOR = 0.03


@dataclass(frozen=True)
class Variant:
    name: str
    acc_band_weight: float
    sapapair_weight: float = 0.20
    clip_norm: Optional[float] = 1.0
    mode: str = "additive"  # additive or multiplicative


DEFAULT_VARIANTS = [
    Variant("baseline", 0.80),
    Variant("acc_0_40", 0.40),
    Variant("acc_0_20", 0.20),
    Variant("no_band", 0.00),
    Variant("clip_2_0", 0.80, clip_norm=2.0),
    Variant("no_clip", 0.80, clip_norm=None),
    Variant("mult_0_80", 0.80, mode="multiplicative"),
]


PROGRESS_FIELDS = [
    "filename",
    "variant",
    "gt_tonic",
    "gt_raga",
    "detected_tonic",
    "tonic_match",
    "hist_top1_raga",
    "hist_raga_match",
    "n_peaks",
    "n_candidates",
    "n_saturated_candidates",
    "saturated_candidate_frac",
    "top_tie_size",
    "top_fit_score",
    "status",
]


SUMMARY_FIELDS = [
    "variant",
    "n",
    "tonic_top1_acc",
    "hist_raga_top1_acc",
    "candidate_count",
    "saturated_candidate_frac",
    "mean_top_tie_size",
    "p95_top_tie_size",
    "max_top_tie_size",
    "records_with_top_tie_gt1",
]


def normalize_tonic_name(name: str) -> str:
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


def _tonic_to_name(pc: int) -> str:
    return NOTE_NAMES[int(pc) % 12]


def load_checkpoint(path: Path) -> Dict[tuple, dict]:
    if not path.exists():
        return {}
    rows: Dict[tuple, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows[(row.get("filename", ""), row.get("variant", ""))] = row
    return rows


def append_checkpoint_row(path: Path, row: dict) -> None:
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


def _raga_filter_keys(raga_filter: str) -> List[str]:
    return [_normalize_raga_name(part) for part in str(raga_filter).split(",") if part.strip()]


def score_candidates_diagnostic(
    pitch_data_vocals: PitchData,
    pitch_data_accomp: Optional[PitchData],
    raga_db: RagaDatabase,
    detected_peak_count: int,
    tonic_candidates: List[int],
    raga_filter: str,
    variant: Variant,
    params: ScoringParams,
) -> pd.DataFrame:
    """Mirror raga.score_candidates_full with exposed pre-clip fit_norm."""
    eps = params.EPS

    midi_vals_mel = pitch_data_vocals.midi_vals
    cent_vals_mel = (midi_vals_mel % 12) * 100.0

    num_bins = 1200
    bin_edges = np.linspace(0.0, 1200.0, num_bins + 1)
    cent_hist, _ = np.histogram(cent_vals_mel, bins=bin_edges, range=(0.0, 1200.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    def mass_within_window_for_note(note_idx: int) -> float:
        center = (note_idx * 100.0) % 1200.0
        diff = np.abs(bin_centers - center)
        diff = np.minimum(diff, 1200.0 - diff)
        return float(np.sum(cent_hist[diff <= params.WINDOW_CENTS]))

    h_pc_arr = np.array([mass_within_window_for_note(i) for i in range(12)], dtype=float)
    cent_hist_smooth = gaussian_filter1d(cent_hist.astype(float), sigma=5.0, mode="wrap")
    p_pc = (h_pc_arr + eps) / (np.sum(h_pc_arr) + 12 * eps)

    has_accompaniment = pitch_data_accomp is not None and len(pitch_data_accomp.midi_vals) > 0
    h_acc = np.zeros(12, dtype=float)
    h_acc_band = np.zeros(12, dtype=float)
    max_acc = 1.0
    max_acc_band = 1.0
    band_frame_count = 0
    salience_all_tonics = {t: 0 for t in range(12)}
    if has_accompaniment:
        assert pitch_data_accomp is not None
        midi_vals_acc = pitch_data_accomp.midi_vals
        pitch_classes_acc = np.mod(np.round(midi_vals_acc), 12).astype(int)
        h_acc, _ = np.histogram(pitch_classes_acc, bins=12, range=(0, 12))

        band_mask = (
            pitch_data_accomp.voiced_mask
            & np.isfinite(pitch_data_accomp.pitch_hz)
            & (pitch_data_accomp.pitch_hz >= params.ACC_BAND_MIN_HZ)
            & (pitch_data_accomp.pitch_hz < params.ACC_BAND_MAX_HZ)
        )
        band_frame_count = int(np.sum(band_mask))
        if band_frame_count > 0:
            band_midi_vals = librosa.hz_to_midi(pitch_data_accomp.pitch_hz[band_mask])
            band_pitch_classes = np.mod(np.round(band_midi_vals), 12).astype(int)
            h_acc_band, _ = np.histogram(band_pitch_classes, bins=12, range=(0, 12))
        salience_all_tonics = {t: int(h_acc[t]) for t in range(12)}
        max_acc = float(h_acc.max()) if h_acc.size and h_acc.max() > 0 else 1.0
        max_acc_band = float(h_acc_band.max()) if h_acc_band.size and h_acc_band.max() > 0 else 1.0

    filter_keys = _raga_filter_keys(raga_filter)
    rows: List[dict] = []
    for raga_entry in raga_db.all_ragas:
        if filter_keys:
            if not any(_normalize_raga_name(name) in filter_keys for name in raga_entry.get("names", [])):
                continue

        mask_abs = np.array(raga_entry["mask"], dtype=int)
        names = raga_entry["names"]
        if mask_abs.sum() < 2:
            continue

        for tonic in tonic_candidates:
            tonic_sal = float(salience_all_tonics.get(tonic, 0))
            p_rot = np.roll(p_pc, -tonic)
            raga_note_indices = np.where(mask_abs == 1)[0].tolist()
            raga_size = len(raga_note_indices)
            if raga_size < 2:
                continue

            size_diff = abs(raga_size - detected_peak_count)
            size_penalty = params.MATCH_SIZE_GAMMA * (size_diff / 4.0)
            match_mass = float(np.sum(p_rot[raga_note_indices]))
            extra_mass = float(1.0 - match_mass)

            peak = float(np.max(p_rot) + eps)
            pres = p_rot[raga_note_indices] / peak
            observed_note_score = float(np.mean(pres)) if params.USE_PRESENCE_MEAN else float(np.sum(pres) / (np.sqrt(raga_size) + eps))

            sum_logp = float(np.sum(np.log(p_rot[raga_note_indices] + eps)))
            baseline = -np.log(12.0)
            avg_logp = sum_logp / (raga_size + eps)
            loglike_norm = 1.0 + (avg_logp / (-baseline + eps))
            loglike_norm = max(0.0, min(1.0, loglike_norm))
            complexity_pen = max(0.0, (raga_size - 5) / 12.0)

            prim = float(p_rot[0])
            bonus_options = [0.0]
            for idx in (5, 6, 7):
                if mask_abs[idx] == 1:
                    bonus_options.append(float(p_rot[idx]))
            primary_score = prim + max(bonus_options)

            base_fit_norm = (
                params.ALPHA_MATCH * match_mass
                + params.BETA_PRESENCE * observed_note_score
                + params.GAMMA_LOGLIKE * loglike_norm
            ) - (
                params.DELTA_EXTRA * extra_mass
                + params.COMPLEX_PENALTY * complexity_pen
                + size_penalty
            )

            tonic_sal_band = float(h_acc_band[tonic])
            tonic_sal_band_norm = tonic_sal_band / (max_acc_band + eps)
            pa_sal = float(h_acc[(tonic + 7) % 12]) if has_accompaniment else 0.0
            sapapair_norm = (tonic_sal + pa_sal) / (max_acc + eps) if has_accompaniment else 0.0

            if variant.mode == "multiplicative":
                fit_norm_pre_clip = base_fit_norm * (1.0 + variant.acc_band_weight * tonic_sal_band_norm)
                fit_norm_pre_clip += variant.sapapair_weight * sapapair_norm
            else:
                fit_norm_pre_clip = base_fit_norm
                fit_norm_pre_clip += variant.acc_band_weight * tonic_sal_band_norm
                fit_norm_pre_clip += variant.sapapair_weight * sapapair_norm

            if variant.clip_norm is None:
                fit_norm_clipped = fit_norm_pre_clip
                saturated = fit_norm_pre_clip > 1.0
            else:
                fit_norm_clipped = max(-variant.clip_norm, min(variant.clip_norm, fit_norm_pre_clip))
                saturated = fit_norm_pre_clip > variant.clip_norm
            fit_score = float(fit_norm_clipped * params.SCALE)

            detected_peak_pcs = set()
            sorted_indices = np.argsort(h_pc_arr)[::-1]
            for idx in sorted_indices[:detected_peak_count]:
                detected_peak_pcs.add(int(idx))

            valley_count = 0
            for note_idx in raga_note_indices:
                abs_pc = (note_idx + tonic) % 12
                center_bin = int(abs_pc * 100) % 1200
                center_val = cent_hist_smooth[center_bin]
                left_val = cent_hist_smooth[(center_bin - 15) % 1200]
                right_val = cent_hist_smooth[(center_bin + 15) % 1200]
                if center_val < left_val and center_val < right_val and abs_pc not in detected_peak_pcs:
                    valley_count += 1

            rows.append(
                {
                    "raga": ", ".join(names),
                    "tonic": int(tonic),
                    "tonic_name": _tonic_to_name(tonic),
                    "fit_score": fit_score,
                    "fit_norm_pre_clip": float(fit_norm_pre_clip),
                    "fit_norm_clipped": float(fit_norm_clipped),
                    "saturated": bool(saturated),
                    "base_fit_norm": float(base_fit_norm),
                    "primary_score": float(primary_score),
                    "salience": int(tonic_sal),
                    "tonic_sal_band_norm": float(tonic_sal_band_norm),
                    "sapapair_norm": float(sapapair_norm),
                    "band_frame_count": int(band_frame_count),
                    "match_mass": match_mass,
                    "extra_mass": extra_mass,
                    "observed_note_score": observed_note_score,
                    "loglike_norm": loglike_norm,
                    "raga_size": raga_size,
                    "match_diff": size_diff,
                    "valley_penalty": valley_count,
                }
            )

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(
            by=["fit_score", "primary_score", "salience"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        df["rank"] = df.index + 1
    return df


def compute_peaks(pd_vocal: PitchData) -> int:
    histograms = compute_cent_histograms(pd_vocal, use_confidence_weights=USE_CONFIDENCE_WEIGHTS)
    peaks = detect_peaks(
        histograms,
        prominence_high_factor=PROMINENCE_HIGH_FACTOR,
        prominence_low_factor=PROMINENCE_LOW_FACTOR,
    )
    return int(len(peaks.validated_indices))


def summarize_variant(rows: List[dict], candidate_rows: List[dict], variant: str) -> dict:
    ok_rows = [r for r in rows if r.get("variant") == variant and r.get("status") == "ok"]
    cand = [r for r in candidate_rows if r.get("variant") == variant]
    n = len(ok_rows)
    denom = max(n, 1)
    tie_sizes = [int(float(r.get("top_tie_size", 0) or 0)) for r in ok_rows]
    candidate_count = len(cand)
    saturated = sum(1 for r in cand if str(r.get("saturated", "")).lower() == "true")
    return {
        "variant": variant,
        "n": n,
        "tonic_top1_acc": round(sum(r.get("tonic_match") == "true" for r in ok_rows) / denom, 4),
        "hist_raga_top1_acc": round(sum(r.get("hist_raga_match") == "true" for r in ok_rows) / denom, 4),
        "candidate_count": candidate_count,
        "saturated_candidate_frac": round(saturated / max(candidate_count, 1), 4),
        "mean_top_tie_size": round(float(np.mean(tie_sizes)) if tie_sizes else 0.0, 3),
        "p95_top_tie_size": round(float(np.percentile(tie_sizes, 95)) if tie_sizes else 0.0, 3),
        "max_top_tie_size": max(tie_sizes) if tie_sizes else 0,
        "records_with_top_tie_gt1": sum(t > 1 for t in tie_sizes),
    }


def parse_variants(raw: str) -> List[Variant]:
    if raw.strip() == "default":
        return DEFAULT_VARIANTS
    lookup = {v.name: v for v in DEFAULT_VARIANTS}
    out = []
    for name in raw.split(","):
        key = name.strip()
        if not key:
            continue
        if key not in lookup:
            raise ValueError(f"Unknown variant {key!r}. Known: {', '.join(sorted(lookup))}")
        out.append(lookup[key])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-csv", default=str(REPO_ROOT / "compmusic_gt.csv"))
    parser.add_argument("--stems-root", default="/Volumes/Extreme SSD/stems/separated_stems/htdemucs")
    parser.add_argument("--raga-db", default=None)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "saturation_calibration"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--variants", default="default")
    parser.add_argument("--progress-name", default="progress.csv")
    parser.add_argument("--candidate-name", default="fit_score_distribution.csv")
    parser.add_argument("--summary-name", default="summary.csv")
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    output_dir = Path(args.output_dir)
    progress_path = output_dir / args.progress_name
    candidate_path = output_dir / args.candidate_name
    summary_path = output_dir / args.summary_name

    with open(args.gt_csv, "r", encoding="utf-8", newline="") as fh:
        gt_rows = list(csv.DictReader(fh))
    if args.limit > 0:
        gt_rows = gt_rows[: args.limit]

    gt_ragas = sorted({r.get("Raga", "").strip() for r in gt_rows if r.get("Raga")})
    raga_filter_str = ",".join(gt_ragas)
    raga_db_path = args.raga_db or find_default_raga_db_path()
    if not raga_db_path:
        print("Could not locate raga DB", file=sys.stderr)
        return 2

    print(f"GT rows: {len(gt_rows)}")
    print(f"Variants: {', '.join(v.name for v in variants)}")
    print(f"Loading raga DB from {raga_db_path}")
    raga_db = RagaDatabase(raga_db_path)
    params = ScoringParams()
    stems_root = Path(args.stems_root)
    processed = load_checkpoint(progress_path)
    if processed:
        print(f"Resuming: {len(processed)} completed recording/variant rows")

    progress_rows: List[dict] = list(processed.values())
    candidate_rows: List[dict] = []
    if candidate_path.exists() and candidate_path.stat().st_size > 0:
        with candidate_path.open("r", encoding="utf-8", newline="") as fh:
            candidate_rows = list(csv.DictReader(fh))

    t0 = time.time()
    total = len(gt_rows) * len(variants)
    done = len(processed)

    for row_idx, gt in enumerate(gt_rows):
        fname = gt.get("Filename", "").strip()
        gt_raga = gt.get("Raga", "").strip()
        gt_tonic = normalize_tonic_name(gt.get("Tonic", ""))
        gender = gt.get("Gender", "").strip()
        if not fname or not gt_raga or not gt_tonic:
            continue

        rec_dir = stems_root / fname
        vocal_csv = rec_dir / "vocals_pitch_data.csv"
        accomp_csv = rec_dir / "accompaniment_pitch_data.csv"
        missing = [p for p in (vocal_csv, accomp_csv) if not p.exists()]
        if missing:
            for variant in variants:
                key = (fname, variant.name)
                if key in processed:
                    continue
                out = {
                    "filename": fname,
                    "variant": variant.name,
                    "gt_tonic": gt_tonic,
                    "gt_raga": gt_raga,
                    "status": "missing",
                }
                append_checkpoint_row(progress_path, out)
                progress_rows.append(out)
                done += 1
            continue

        try:
            pd_vocal = load_pitch_from_csv(str(vocal_csv)).apply_confidence_threshold(VOCAL_CONFIDENCE)
            pd_accomp = load_pitch_from_csv(str(accomp_csv)).apply_confidence_threshold(ACCOMP_CONFIDENCE)
            n_peaks = compute_peaks(pd_vocal) if len(pd_vocal.midi_vals) > 0 else 0
        except Exception as exc:
            for variant in variants:
                key = (fname, variant.name)
                if key in processed:
                    continue
                out = {
                    "filename": fname,
                    "variant": variant.name,
                    "gt_tonic": gt_tonic,
                    "gt_raga": gt_raga,
                    "status": f"load_error: {exc}",
                }
                append_checkpoint_row(progress_path, out)
                progress_rows.append(out)
                done += 1
            continue

        tonic_candidates = gender_to_tonic_range(gender)
        for variant in variants:
            key = (fname, variant.name)
            if key in processed:
                continue
            done += 1
            try:
                df = score_candidates_diagnostic(
                    pitch_data_vocals=pd_vocal,
                    pitch_data_accomp=pd_accomp,
                    raga_db=raga_db,
                    detected_peak_count=n_peaks,
                    tonic_candidates=tonic_candidates,
                    raga_filter=raga_filter_str,
                    variant=variant,
                    params=params,
                )
                if len(df) == 0:
                    raise RuntimeError("no candidates")

                top = df.iloc[0]
                top_fit = float(top["fit_score"])
                top_tie = int(np.sum(np.isclose(df["fit_score"].astype(float), top_fit, atol=1e-9)))
                n_saturated = int(df["saturated"].sum())
                out = {
                    "filename": fname,
                    "variant": variant.name,
                    "gt_tonic": gt_tonic,
                    "gt_raga": gt_raga,
                    "detected_tonic": str(top["tonic_name"]),
                    "tonic_match": str(str(top["tonic_name"]) == gt_tonic).lower(),
                    "hist_top1_raga": str(top["raga"]),
                    "hist_raga_match": str(raga_names_match(str(top["raga"]), gt_raga)).lower(),
                    "n_peaks": n_peaks,
                    "n_candidates": len(df),
                    "n_saturated_candidates": n_saturated,
                    "saturated_candidate_frac": round(n_saturated / max(len(df), 1), 4),
                    "top_tie_size": top_tie,
                    "top_fit_score": round(top_fit, 6),
                    "status": "ok",
                }
                for cand in df.to_dict("records"):
                    candidate_rows.append(
                        {
                            "filename": fname,
                            "variant": variant.name,
                            "rank": int(cand["rank"]),
                            "tonic": int(cand["tonic"]),
                            "tonic_name": cand["tonic_name"],
                            "raga": cand["raga"],
                            "fit_score": round(float(cand["fit_score"]), 8),
                            "fit_norm_pre_clip": round(float(cand["fit_norm_pre_clip"]), 8),
                            "fit_norm_clipped": round(float(cand["fit_norm_clipped"]), 8),
                            "saturated": bool(cand["saturated"]),
                            "base_fit_norm": round(float(cand["base_fit_norm"]), 8),
                            "primary_score": round(float(cand["primary_score"]), 8),
                            "salience": int(cand["salience"]),
                            "tonic_sal_band_norm": round(float(cand["tonic_sal_band_norm"]), 8),
                            "sapapair_norm": round(float(cand["sapapair_norm"]), 8),
                        }
                    )
            except Exception as exc:
                out = {
                    "filename": fname,
                    "variant": variant.name,
                    "gt_tonic": gt_tonic,
                    "gt_raga": gt_raga,
                    "n_peaks": n_peaks,
                    "status": f"score_error: {exc}",
                }

            append_checkpoint_row(progress_path, out)
            progress_rows.append(out)
            if done % 25 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (total - done) / max(rate, 1e-6) / 60.0
                print(f"[{done}/{total}] {fname} {variant.name} ETA {eta:.1f}m")

    candidate_fields = [
        "filename",
        "variant",
        "rank",
        "tonic",
        "tonic_name",
        "raga",
        "fit_score",
        "fit_norm_pre_clip",
        "fit_norm_clipped",
        "saturated",
        "base_fit_norm",
        "primary_score",
        "salience",
        "tonic_sal_band_norm",
        "sapapair_norm",
    ]
    write_csv(candidate_path, candidate_fields, candidate_rows)

    summary_rows = [summarize_variant(progress_rows, candidate_rows, v.name) for v in variants]
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    print(f"Wrote {summary_path}")
    print(f"Wrote {candidate_path}")
    print("\n=== Saturation calibration summary ===")
    print(f"{'variant':<12} {'n':>4} {'tonic':>8} {'hist':>8} {'sat%':>8} {'tie':>8} {'p95tie':>8}")
    for row in summary_rows:
        print(
            f"{row['variant']:<12} {row['n']:>4} "
            f"{row['tonic_top1_acc']*100:>7.1f}% "
            f"{row['hist_raga_top1_acc']*100:>7.1f}% "
            f"{row['saturated_candidate_frac']*100:>7.1f}% "
            f"{row['mean_top_tie_size']:>8.2f} "
            f"{row['p95_top_tie_size']:>8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
