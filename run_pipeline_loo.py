"""Honest pipeline LOO evaluation.

For each recording: trains LM on the other 297, runs full detect pipeline
with --use-lm-scoring, checks if correct raga ranks first.

Usage: /opt/miniconda3/envs/raga/bin/python run_pipeline_loo.py
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

GT_CSV = "compmusic_gt.csv"
OUTPUT = "/Volumes/Extreme SSD/stems/separated_stems"
AUDIO_BASE = "/Volumes/Extreme SSD/RagaDataset/compmusic"
UNCORR_DIR = "/Volumes/Extreme SSD/stems/separated_stems_nocorrection"
PYTHON = sys.executable  # should be raga conda env
CHECKPOINT_CSV = "run_pipeline_loo_progress.csv"
CHECKPOINT_FIELDS = [
    "filename",
    "true_raga",
    "pred_raga",
    "rank",
    "evaluated",
    "top1",
    "top3",
    "status",
]

RAGA_DIR_ALIASES = {
    "Alhaiya Bilawal": "Alhaiya Bilaval",
    "Marwa": "Marva",
    "Puriya Dhanashree": "Puriya Dhanashri",
}


def raga_match(true_r, cand_r):
    t = true_r.lower().strip()
    c = cand_r.lower().strip()
    return t == c or t in c or c in t


def train_loo_model(gt_rows, held_out_filename, uncorr_dir, output_path):
    """Train model on all recordings except held_out_filename."""
    from raga_pipeline.language_model import NgramModel, _load_notes_from_csv, _tonic_name_to_midi
    from raga_pipeline.motifs import _discover_candidates

    stem_map, basename_map = _discover_candidates(Path(uncorr_dir))

    model = NgramModel(order=5, smoothing="add-k", smoothing_k=0.01)
    raga_counts = defaultdict(int)

    for row in gt_rows:
        if row["Filename"].strip() == held_out_filename:
            continue

        filename = row["Filename"].strip()
        raga = row["Raga"].strip()
        tonic = row["Tonic"].strip()

        stem = Path(filename).stem.lower()
        candidates = stem_map.get(stem) or basename_map.get(filename.lower())
        if not candidates or len(candidates) != 1:
            continue

        csv_path, _ = candidates[0].resolve("auto")
        if csv_path is None or not csv_path.exists():
            continue

        tonic_midi = _tonic_name_to_midi(tonic) if tonic else 60.0
        phrases = _load_notes_from_csv(csv_path, tonic_midi)
        if phrases:
            model.add_sequence(raga, phrases)
            raga_counts[raga] += 1

    # Remove ragas with < 3 recordings
    for raga in list(model._counts.keys()):
        if raga_counts.get(raga, 0) < 3:
            del model._counts[raga]
            del model._context_counts[raga]

    model.finalize()

    data = model.to_dict()
    with open(output_path, "w") as f:
        json.dump(data, f)

    return len(model.ragas())


def _as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _load_checkpoint(checkpoint_path):
    if not checkpoint_path.exists():
        return {}
    processed = {}
    with checkpoint_path.open(newline="") as f:
        for row in csv.DictReader(f):
            filename = str(row.get("filename", "")).strip()
            if not filename:
                continue
            processed[filename] = row
    return processed


def _append_checkpoint_row(checkpoint_path, row):
    write_header = not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0
    with checkpoint_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CHECKPOINT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _compute_stats(processed_rows):
    evaluated = top1 = top3 = 0
    errors = []
    for row in processed_rows.values():
        if not _as_bool(row.get("evaluated", "false")):
            continue
        evaluated += 1
        if _as_bool(row.get("top1", "false")):
            top1 += 1
        if _as_bool(row.get("top3", "false")):
            top3 += 1
        if not _as_bool(row.get("top1", "false")):
            errors.append(
                (
                    str(row.get("filename", "")).strip(),
                    str(row.get("true_raga", "")).strip(),
                    str(row.get("pred_raga", "")).strip() or "?",
                    str(row.get("rank", "")).strip() or "None",
                )
            )
    return evaluated, top1, top3, errors


def main():
    # Load GT
    rows = []
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    checkpoint_path = Path(CHECKPOINT_CSV)
    processed_rows = _load_checkpoint(checkpoint_path)
    evaluated, top1, top3, errors = _compute_stats(processed_rows)

    total = len(rows)
    pending_rows = [row for row in rows if row["Filename"].strip() not in processed_rows]
    t0 = time.time()
    print(
        f"Pipeline LOO: {total} recordings "
        f"({len(processed_rows)} checkpointed, {len(pending_rows)} pending)",
        flush=True,
    )

    if len(pending_rows) == 0:
        print("[INFO] Nothing to do; all recordings are already checkpointed.", flush=True)

    for i, row in enumerate(pending_rows, 1):
        filename = row["Filename"].strip()
        true_raga = row["Raga"].strip()
        gender = row.get("Gender", "").strip().upper()
        instrument = row.get("Instrument", "").strip().lower()

        audio_dir_name = RAGA_DIR_ALIASES.get(true_raga, true_raga)
        audio_path = Path(AUDIO_BASE) / audio_dir_name / f"{filename}.mp3"

        progress_row = {
            "filename": filename,
            "true_raga": true_raga,
            "pred_raga": "",
            "rank": "",
            "evaluated": "false",
            "top1": "false",
            "top3": "false",
            "status": "",
        }

        if not audio_path.exists():
            progress_row["status"] = "audio_missing"
            _append_checkpoint_row(checkpoint_path, progress_row)
            processed_rows[filename] = progress_row
            continue

        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(pending_rows) - i) / rate / 60 if rate > 0 else 0

        # Train LOO model
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_model = tmp.name

        try:
            train_loo_model(rows, filename, UNCORR_DIR, tmp_model)

            # Run detect with LOO model
            cmd = [
                PYTHON, "driver.py", "detect",
                "--audio", str(audio_path),
                "--output", OUTPUT,
                "--use-lm-scoring",
                "--lm-model", tmp_model,
                "--lm-skip-correction",
                "--skip-report",
            ]
            if "vocal" in instrument:
                cmd += ["--source-type", "vocal"]
                if gender == "M":
                    cmd += ["--vocalist-gender", "male"]
                elif gender == "F":
                    cmd += ["--vocalist-gender", "female"]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                progress_row["status"] = f"detect_failed_{result.returncode}"
                _append_checkpoint_row(checkpoint_path, progress_row)
                processed_rows[filename] = progress_row
                print(f"[{i}/{len(pending_rows)}] {filename} FAILED", flush=True)
                continue

            # Read lm_candidates.csv
            lm_csv = Path(OUTPUT) / "htdemucs" / filename / "lm_candidates.csv"
            if not lm_csv.exists():
                progress_row["status"] = "lm_candidates_missing"
                _append_checkpoint_row(checkpoint_path, progress_row)
                processed_rows[filename] = progress_row
                continue

            with open(lm_csv) as cf:
                candidates = list(csv.DictReader(cf))

            rank = None
            for c in candidates:
                if raga_match(true_raga, c["raga"]) and c.get("gated") == "True":
                    rank = int(c["lm_rank"])
                    break

            evaluated += 1
            if rank == 1:
                top1 += 1
            if rank and rank <= 3:
                top3 += 1

            correct = "OK" if rank == 1 else f"WRONG(rank={rank})"
            pred = candidates[0]["raga"] if candidates else "?"

            progress_row["pred_raga"] = pred
            progress_row["rank"] = "" if rank is None else str(rank)
            progress_row["evaluated"] = "true"
            progress_row["top1"] = "true" if rank == 1 else "false"
            progress_row["top3"] = "true" if rank is not None and rank <= 3 else "false"
            progress_row["status"] = "ok"
            _append_checkpoint_row(checkpoint_path, progress_row)
            processed_rows[filename] = progress_row

            if rank != 1:
                errors.append((filename, true_raga, pred, rank))

            if i % 10 == 0 or rank != 1:
                print(
                    f"[{i}/{len(pending_rows)}] {filename} {correct} "
                    f"(top1={top1}/{evaluated} ETA={eta:.0f}m)",
                    flush=True,
                )

        finally:
            os.unlink(tmp_model)

    elapsed = time.time() - t0
    print(f"\n=== Pipeline LOO Results ===")
    print(f"  Evaluated: {evaluated}")
    print(f"  Top-1: {top1}/{evaluated} ({top1/evaluated:.1%})")
    print(f"  Top-3: {top3}/{evaluated} ({top3/evaluated:.1%})")
    print(f"  Time: {elapsed/60:.1f} minutes")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for fn, true_r, pred_r, rank in errors:
            print(f"    {fn:<25} true={true_r:<20} pred={pred_r:<20} rank={rank}")


if __name__ == "__main__":
    main()
