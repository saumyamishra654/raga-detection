"""Re-run analyze on CompMusic subset with --skip-raga-correction.

Expects symlinked pitch CSVs in the output directory.
Usage: python run_uncorrected_batch.py
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

GT_CSV = "compmusic_gt.csv"
OUTPUT = "/Volumes/Extreme SSD/stems/separated_stems_nocorrection"
AUDIO_BASE = "/Volumes/Extreme SSD/RagaDataset/compmusic"

# GT raga names -> actual audio directory names (where they differ)
RAGA_DIR_ALIASES = {
    "Alhaiya Bilawal": "Alhaiya Bilaval",
    "Marwa": "Marva",
    "Puriya Dhanashree": "Puriya Dhanashri",
}

done = 0
fail = 0
skip = 0
rows = []

with open(GT_CSV) as f:
    for row in csv.DictReader(f):
        rows.append(row)

total = len(rows)
t0 = time.time()

for i, row in enumerate(rows, 1):
    raga = row["Raga"].strip()
    tonic = row["Tonic"].strip()
    filename = row["Filename"].strip()
    gender = row.get("Gender", "").strip().upper()
    instrument = row.get("Instrument", "").strip().lower()

    audio_dir_name = RAGA_DIR_ALIASES.get(raga, raga)
    audio_path = Path(AUDIO_BASE) / audio_dir_name / f"{filename}.mp3"
    trans_csv = Path(OUTPUT) / "htdemucs" / filename / "transcribed_notes.csv"

    # Skip if already processed
    if trans_csv.exists():
        skip += 1
        continue

    if not audio_path.exists():
        print(f"[{i}/{total}] SKIP {filename} -- audio not found")
        skip += 1
        continue

    cmd = [
        sys.executable, "driver.py", "analyze",
        "--audio", str(audio_path),
        "--output", OUTPUT,
        "--tonic", tonic,
        "--raga", raga,
        "--skip-raga-correction",
        "--transcription-only",
    ]
    if "vocal" in instrument:
        cmd += ["--source-type", "vocal"]
        if gender == "M":
            cmd += ["--vocalist-gender", "male"]
        elif gender == "F":
            cmd += ["--vocalist-gender", "female"]

    elapsed = time.time() - t0
    rate = (done + fail) / elapsed if elapsed > 0 else 0
    eta = (total - i) / rate / 60 if rate > 0 else 0
    print(f"[{i}/{total}] {filename} (done={done} fail={fail} skip={skip} ETA={eta:.0f}m)")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        done += 1
    else:
        fail += 1
        print(f"  FAILED: {result.stderr[-200:]}")

elapsed = time.time() - t0
print(f"\nBatch complete: {done} succeeded, {fail} failed, {skip} skipped out of {total}")
print(f"Total time: {elapsed/60:.1f} minutes")
