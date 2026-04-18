#!/usr/bin/env python3
"""Sweep accompaniment band-pass ranges and weights for tonic detection."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


TONIC_TO_PC = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}
ALL_TONICS: Tuple[int, ...] = tuple(range(12))
MALE_TONICS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
FEMALE_TONICS: Tuple[int, ...] = (7, 8, 9, 10, 11, 0)


@dataclass
class Recording:
    filename: str
    true_tonic: int
    allowed_tonics: Tuple[int, ...]
    pitch_hz: np.ndarray
    voiced_runtime: np.ndarray
    voiced_conf05: np.ndarray
    tonic_fits: Dict[int, float]


def _parse_tonic(value: str) -> int:
    key = value.strip().upper()
    return TONIC_TO_PC.get(key, -1)


def _parse_bool_cell(raw: str, fallback: bool) -> bool:
    txt = str(raw).strip().lower()
    if not txt:
        return fallback
    return txt in {"1", "1.0", "true", "t", "yes", "y"}


def _parse_float_cell(raw: str, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _allowed_tonics_for_gender(raw_gender: str) -> Tuple[int, ...]:
    g = str(raw_gender).strip().upper()
    if g.startswith("M"):
        return MALE_TONICS
    if g.startswith("F"):
        return FEMALE_TONICS
    return ALL_TONICS


def _load_recordings(
    gt_csv: Path,
    accomp_root: Path,
    candidates_root: Path,
    accomp_confidence: float,
) -> List[Recording]:
    rows: List[Recording] = []
    with gt_csv.open(newline="", encoding="utf-8") as fh:
        for gt in csv.DictReader(fh):
            filename = str(gt.get("Filename", "")).strip()
            if not filename:
                continue
            true_tonic = _parse_tonic(str(gt.get("Tonic", "")))
            if true_tonic < 0:
                continue

            accomp_csv = accomp_root / filename / "accompaniment_pitch_data.csv"
            cand_csv = candidates_root / filename / "candidates.csv"
            if not accomp_csv.exists() or not cand_csv.exists():
                continue

            hz_vals: List[float] = []
            conf_vals: List[float] = []
            voiced_vals: List[bool] = []
            with accomp_csv.open(newline="", encoding="utf-8") as af:
                for row in csv.DictReader(af):
                    hz = _parse_float_cell(row.get("pitch_hz", row.get("f0", 0.0)))
                    conf = _parse_float_cell(row.get("confidence", row.get("conf", 0.0)))
                    voiced_raw = row.get("voicing", row.get("voiced", ""))
                    voiced = _parse_bool_cell(voiced_raw, fallback=hz > 0.0)
                    hz_vals.append(hz)
                    conf_vals.append(conf)
                    voiced_vals.append(voiced)
            if not hz_vals:
                continue

            tonic_fits: Dict[int, float] = {}
            with cand_csv.open(newline="", encoding="utf-8") as cf:
                for row in csv.DictReader(cf):
                    try:
                        tonic = int(row["tonic"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    score = _parse_float_cell(row.get("fit_score", row.get("score", -1000.0)), default=-1000.0)
                    prev = tonic_fits.get(tonic)
                    if prev is None or score > prev:
                        tonic_fits[tonic] = score
            if not tonic_fits:
                continue

            pitch_hz = np.asarray(hz_vals, dtype=float)
            confidence = np.asarray(conf_vals, dtype=float)
            voicing = np.asarray(voiced_vals, dtype=bool)

            voiced_runtime = (pitch_hz > 0.0) & voicing & (confidence >= accomp_confidence)
            voiced_conf05 = (pitch_hz > 0.0) & voicing & (confidence > 0.5)

            rows.append(
                Recording(
                    filename=filename,
                    true_tonic=true_tonic,
                    allowed_tonics=_allowed_tonics_for_gender(gt.get("Gender", "")),
                    pitch_hz=pitch_hz,
                    voiced_runtime=voiced_runtime,
                    voiced_conf05=voiced_conf05,
                    tonic_fits=tonic_fits,
                )
            )
    return rows


def _build_band_hist(recording: Recording, low_hz: float, high_hz: float, gate_mode: str) -> Tuple[np.ndarray, int]:
    mask_base = recording.voiced_runtime if gate_mode == "runtime" else recording.voiced_conf05
    band_mask = (
        mask_base
        & np.isfinite(recording.pitch_hz)
        & (recording.pitch_hz >= low_hz)
        & (recording.pitch_hz < high_hz)
    )
    frame_count = int(np.sum(band_mask))
    hist = np.zeros(12, dtype=float)
    if frame_count == 0:
        return hist, frame_count

    midi = 69.0 + 12.0 * np.log2(recording.pitch_hz[band_mask] / 440.0)
    pcs = np.mod(np.round(midi).astype(int), 12)
    np.add.at(hist, pcs, 1.0)
    return hist, frame_count


def _pick_from_hist(hist: np.ndarray, allowed_tonics: Sequence[int]) -> int:
    best_tonic = -1
    best_mass = -1.0
    for tonic in allowed_tonics:
        mass = float(hist[tonic])
        if mass > best_mass:
            best_mass = mass
            best_tonic = int(tonic)
    if best_mass <= 0.0:
        return -1
    return best_tonic


def _pick_from_fit(tonic_fits: Dict[int, float], allowed_tonics: Sequence[int]) -> int:
    best_tonic = -1
    best_score = -1e30
    for tonic in allowed_tonics:
        if tonic not in tonic_fits:
            continue
        score = float(tonic_fits[tonic])
        if score > best_score:
            best_score = score
            best_tonic = int(tonic)
    return best_tonic


def _pick_combined(
    tonic_fits: Dict[int, float],
    band_hist: np.ndarray,
    weight: float,
    allowed_tonics: Sequence[int],
) -> int:
    best_tonic = -1
    best_score = -1e30
    band_max = float(np.max(band_hist)) if float(np.max(band_hist)) > 0.0 else 1.0
    for tonic in allowed_tonics:
        fit_score = float(tonic_fits.get(int(tonic), -1000.0))
        band_boost = float(band_hist[int(tonic)]) / band_max
        combined = fit_score + weight * band_boost
        if combined > best_score:
            best_score = combined
            best_tonic = int(tonic)
    return best_tonic


def _empty_metrics() -> Dict[str, int]:
    return {
        "band_only_correct": 0,
        "band_gender_correct": 0,
        "hist_only_correct": 0,
        "hist_gender_correct": 0,
        "combined_only_correct": 0,
        "combined_gender_correct": 0,
        "tracks_with_band_frames": 0,
    }


def _evaluate(
    recordings: Sequence[Recording],
    low_hz: float,
    high_hz: float,
    weight: float,
    gate_mode: str,
) -> Dict[str, float]:
    metrics = _empty_metrics()
    total = len(recordings)
    for rec in recordings:
        band_hist, frame_count = _build_band_hist(rec, low_hz, high_hz, gate_mode)
        if frame_count > 0:
            metrics["tracks_with_band_frames"] += 1

        pred_band_all = _pick_from_hist(band_hist, ALL_TONICS)
        pred_band_gender = _pick_from_hist(band_hist, rec.allowed_tonics)
        pred_hist_all = _pick_from_fit(rec.tonic_fits, ALL_TONICS)
        pred_hist_gender = _pick_from_fit(rec.tonic_fits, rec.allowed_tonics)
        pred_comb_all = _pick_combined(rec.tonic_fits, band_hist, weight, ALL_TONICS)
        pred_comb_gender = _pick_combined(rec.tonic_fits, band_hist, weight, rec.allowed_tonics)

        if pred_band_all == rec.true_tonic:
            metrics["band_only_correct"] += 1
        if pred_band_gender == rec.true_tonic:
            metrics["band_gender_correct"] += 1
        if pred_hist_all == rec.true_tonic:
            metrics["hist_only_correct"] += 1
        if pred_hist_gender == rec.true_tonic:
            metrics["hist_gender_correct"] += 1
        if pred_comb_all == rec.true_tonic:
            metrics["combined_only_correct"] += 1
        if pred_comb_gender == rec.true_tonic:
            metrics["combined_gender_correct"] += 1

    out: Dict[str, float] = {"total": float(total)}
    out.update({k: float(v) for k, v in metrics.items()})
    denom = float(total) if total > 0 else 1.0
    out["band_only_acc"] = metrics["band_only_correct"] / denom
    out["band_gender_acc"] = metrics["band_gender_correct"] / denom
    out["hist_only_acc"] = metrics["hist_only_correct"] / denom
    out["hist_gender_acc"] = metrics["hist_gender_correct"] / denom
    out["combined_only_acc"] = metrics["combined_only_correct"] / denom
    out["combined_gender_acc"] = metrics["combined_gender_correct"] / denom
    out["band_frame_coverage"] = metrics["tracks_with_band_frames"] / denom
    return out


def _generate_bands(hz_min: int, hz_max: int, hz_step: int, min_width: int) -> List[Tuple[int, int]]:
    bands: List[Tuple[int, int]] = []
    for lo in range(hz_min, hz_max, hz_step):
        for hi in range(lo + min_width, hz_max + 1, hz_step):
            if hi <= lo:
                continue
            bands.append((lo, hi))
    return bands


def _parse_weights(raw: str) -> List[float]:
    vals: List[float] = []
    for part in raw.split(","):
        txt = part.strip()
        if not txt:
            continue
        vals.append(float(txt))
    if not vals:
        raise ValueError("Weight list is empty.")
    return vals


def _write_csv(path: Path, rows: Iterable[Dict[str, float]]) -> int:
    row_list = list(rows)
    if not row_list:
        path.write_text("", encoding="utf-8")
        return 0
    fieldnames = list(row_list[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)
    return len(row_list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-csv", type=Path, default=Path("compmusic_gt.csv"))
    parser.add_argument(
        "--accomp-root",
        type=Path,
        default=Path("/Volumes/Extreme SSD/stems/separated_stems/htdemucs"),
        help="Root directory containing <filename>/accompaniment_pitch_data.csv",
    )
    parser.add_argument(
        "--candidates-root",
        type=Path,
        default=Path("/Volumes/Extreme SSD/stems/separated_stems_nometa/htdemucs"),
        help="Root directory containing <filename>/candidates.csv",
    )
    parser.add_argument("--accomp-confidence", type=float, default=0.80)
    parser.add_argument("--hz-min", type=int, default=50)
    parser.add_argument("--hz-max", type=int, default=700)
    parser.add_argument("--hz-step", type=int, default=50)
    parser.add_argument("--min-width", type=int, default=50)
    parser.add_argument("--band-sweep-weight", type=float, default=800.0)
    parser.add_argument("--weight-band-min", type=float, default=100.0)
    parser.add_argument("--weight-band-max", type=float, default=300.0)
    parser.add_argument(
        "--weights",
        type=str,
        default="0,100,200,300,400,500,800,1000,1500,2000",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/bandpass_tonic"))
    args = parser.parse_args()

    weights = _parse_weights(args.weights)
    recordings = _load_recordings(
        gt_csv=args.gt_csv,
        accomp_root=args.accomp_root,
        candidates_root=args.candidates_root,
        accomp_confidence=args.accomp_confidence,
    )
    total = len(recordings)
    print(f"[INFO] Loaded {total} recordings")
    if total == 0:
        raise RuntimeError("No recordings were loaded; check paths and CSV availability.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bands = _generate_bands(args.hz_min, args.hz_max, args.hz_step, args.min_width)
    print(f"[INFO] Evaluating {len(bands)} bands x 2 gate modes")

    band_rows: List[Dict[str, float]] = []
    for gate_mode in ("runtime", "conf05"):
        for low_hz, high_hz in bands:
            metrics = _evaluate(
                recordings=recordings,
                low_hz=float(low_hz),
                high_hz=float(high_hz),
                weight=args.band_sweep_weight,
                gate_mode=gate_mode,
            )
            row: Dict[str, float] = {
                "gate_mode": gate_mode,  # type: ignore[assignment]
                "low_hz": float(low_hz),
                "high_hz": float(high_hz),
                "width_hz": float(high_hz - low_hz),
                "weight": float(args.band_sweep_weight),
            }
            row.update(metrics)
            band_rows.append(row)

    weight_rows: List[Dict[str, float]] = []
    for gate_mode in ("runtime", "conf05"):
        for weight in weights:
            metrics = _evaluate(
                recordings=recordings,
                low_hz=float(args.weight_band_min),
                high_hz=float(args.weight_band_max),
                weight=float(weight),
                gate_mode=gate_mode,
            )
            row = {
                "gate_mode": gate_mode,  # type: ignore[assignment]
                "low_hz": float(args.weight_band_min),
                "high_hz": float(args.weight_band_max),
                "width_hz": float(args.weight_band_max - args.weight_band_min),
                "weight": float(weight),
            }
            row.update(metrics)
            weight_rows.append(row)

    band_csv = args.output_dir / "band_sweep_results.csv"
    weight_csv = args.output_dir / "weight_sweep_results.csv"
    _write_csv(band_csv, band_rows)
    _write_csv(weight_csv, weight_rows)

    print(f"[WRITE] {band_csv}")
    print(f"[WRITE] {weight_csv}")
    for gate_mode in ("runtime", "conf05"):
        gate_band_rows = [r for r in band_rows if r["gate_mode"] == gate_mode]
        gate_band_rows.sort(
            key=lambda r: (r["combined_gender_acc"], r["band_gender_acc"], -r["width_hz"]),
            reverse=True,
        )
        top = gate_band_rows[:5]
        print(f"\nTop bands ({gate_mode}, weight={args.band_sweep_weight:.0f}):")
        for row in top:
            print(
                f"  [{int(row['low_hz'])},{int(row['high_hz'])}) "
                f"combined_gender={row['combined_gender_correct']:.0f}/{int(row['total'])} "
                f"({row['combined_gender_acc']:.1%}) "
                f"band_gender={row['band_gender_correct']:.0f}/{int(row['total'])} "
                f"({row['band_gender_acc']:.1%})"
            )

    for gate_mode in ("runtime", "conf05"):
        gate_weight_rows = [r for r in weight_rows if r["gate_mode"] == gate_mode]
        gate_weight_rows.sort(key=lambda r: (r["combined_gender_acc"], -r["weight"]), reverse=True)
        top = gate_weight_rows[0]
        print(
            f"\nBest weight ({gate_mode}) for [{args.weight_band_min:.0f},{args.weight_band_max:.0f}): "
            f"{top['weight']:.0f} -> "
            f"{top['combined_gender_correct']:.0f}/{int(top['total'])} ({top['combined_gender_acc']:.1%})"
        )


if __name__ == "__main__":
    main()
