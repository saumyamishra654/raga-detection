"""Re-run detect on CompMusic subset WITHOUT vocal/gender metadata.
Uses --source-type mixed (no tonic bias). Stems + pitch cached via symlinks.
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

GT_CSV = "compmusic_gt.csv"
OUTPUT = "/Volumes/Extreme SSD/stems/separated_stems_nometa"
AUDIO_BASE = "/Volumes/Extreme SSD/RagaDataset/compmusic"
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
    filename = row["Filename"].strip()

    # Skip if already has candidates.csv
    cand_csv = Path(OUTPUT) / "htdemucs" / filename / "candidates.csv"
    if cand_csv.exists():
        skip += 1
        continue

    audio_dir_name = RAGA_DIR_ALIASES.get(raga, raga)
    audio_path = Path(AUDIO_BASE) / audio_dir_name / f"{filename}.mp3"

    if not audio_path.exists():
        skip += 1
        continue

    cmd = [
        sys.executable, "driver.py", "detect",
        "--audio", str(audio_path),
        "--output", OUTPUT,
        "--source-type", "mixed",  # No vocal/gender metadata
        "--skip-report",
    ]

    elapsed = time.time() - t0
    rate = (done + fail) / elapsed if elapsed > 0 else 0
    eta = (total - i) / rate / 60 if rate > 0 else 0
    print(f"[{i}/{total}] {filename} (done={done} fail={fail} skip={skip} ETA={eta:.0f}m)")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        done += 1
    else:
        fail += 1
        err = result.stderr[-200:] if result.stderr else result.stdout[-200:]
        print(f"  FAILED: {err}")

elapsed = time.time() - t0
print(f"\nBatch complete: {done} succeeded, {fail} failed, {skip} skipped out of {total}")
print(f"Total time: {elapsed/60:.1f} minutes")
