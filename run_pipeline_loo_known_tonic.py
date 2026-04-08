"""Honest pipeline LOO evaluation WITH KNOWN TONIC.

Same as run_pipeline_loo.py but supplies --tonic from the GT CSV.
This isolates raga discrimination accuracy from tonic detection errors.

Usage: /opt/miniconda3/envs/raga/bin/python run_pipeline_loo.py
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

GT_CSV = "compmusic_gt.csv"
OUTPUT = "/Volumes/Extreme SSD/stems/separated_stems"
AUDIO_BASE = "/Volumes/Extreme SSD/RagaDataset/compmusic"
UNCORR_DIR = "/Volumes/Extreme SSD/stems/separated_stems_nocorrection"
PYTHON = sys.executable  # should be raga conda env

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


def main():
    # Load GT
    rows = []
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    total = len(rows)
    t0 = time.time()
    top1 = top3 = evaluated = 0
    errors = []

    print(f"Pipeline LOO: {total} recordings", flush=True)

    for i, row in enumerate(rows, 1):
        filename = row["Filename"].strip()
        true_raga = row["Raga"].strip()
        gender = row.get("Gender", "").strip().upper()
        instrument = row.get("Instrument", "").strip().lower()

        audio_dir_name = RAGA_DIR_ALIASES.get(true_raga, true_raga)
        audio_path = Path(AUDIO_BASE) / audio_dir_name / f"{filename}.mp3"

        if not audio_path.exists():
            continue

        elapsed = time.time() - t0
        rate = evaluated / elapsed if elapsed > 0 else 0
        eta = (total - i) / rate / 60 if rate > 0 else 0

        # Train LOO model
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_model = tmp.name

        try:
            n_ragas = train_loo_model(rows, filename, UNCORR_DIR, tmp_model)

            # Run detect with LOO model
            tonic = row["Tonic"].strip()
            cmd = [
                PYTHON, "driver.py", "detect",
                "--audio", str(audio_path),
                "--output", OUTPUT,
                "--tonic", tonic,
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
                print(f"[{i}/{total}] {filename} FAILED", flush=True)
                continue

            # Read lm_candidates.csv
            lm_csv = Path(OUTPUT) / "htdemucs" / filename / "lm_candidates.csv"
            if not lm_csv.exists():
                continue

            evaluated += 1
            with open(lm_csv) as cf:
                candidates = list(csv.DictReader(cf))

            rank = None
            for c in candidates:
                if raga_match(true_raga, c["raga"]) and c.get("gated") == "True":
                    rank = int(c["lm_rank"])
                    break

            if rank == 1:
                top1 += 1
            if rank and rank <= 3:
                top3 += 1

            correct = "OK" if rank == 1 else f"WRONG(rank={rank})"
            if rank != 1:
                pred = candidates[0]["raga"] if candidates else "?"
                errors.append((filename, true_raga, pred, rank))

            if i % 10 == 0 or rank != 1:
                print(f"[{i}/{total}] {filename} {correct} (top1={top1}/{evaluated} ETA={eta:.0f}m)", flush=True)

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
