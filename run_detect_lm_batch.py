"""Run detect with --use-lm-scoring on all CompMusic recordings.

Produces lm_candidates.csv for each recording with all three signals
(histogram, LM, deletion residual). Uses cached stems + pitch data.

Usage: /opt/miniconda3/envs/raga/bin/python run_detect_lm_batch.py
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

GT_CSV = "compmusic_gt.csv"
OUTPUT = "/Volumes/Extreme SSD/stems/separated_stems"
AUDIO_BASE = "/Volumes/Extreme SSD/RagaDataset/compmusic"
LM_MODEL = "compmusic_ngram_model_v2.json"

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
print(f"Starting batch: {total} recordings, output={OUTPUT}, model={LM_MODEL}")
print(f"Audio base: {AUDIO_BASE}")
print(f"Python: {sys.executable}")
sys.stdout.flush()

for i, row in enumerate(rows, 1):
    raga = row["Raga"].strip()
    filename = row["Filename"].strip()
    gender = row.get("Gender", "").strip().upper()
    instrument = row.get("Instrument", "").strip().lower()

    # Skip if lm_candidates.csv already exists
    lm_csv = Path(OUTPUT) / "htdemucs" / filename / "lm_candidates.csv"
    if lm_csv.exists():
        skip += 1
        continue

    audio_dir_name = RAGA_DIR_ALIASES.get(raga, raga)
    audio_path = Path(AUDIO_BASE) / audio_dir_name / f"{filename}.mp3"

    if not audio_path.exists():
        print(f"[{i}/{total}] SKIP {filename} -- audio not found", flush=True)
        skip += 1
        continue

    cmd = [
        sys.executable, "driver.py", "detect",
        "--audio", str(audio_path),
        "--output", OUTPUT,
        "--use-lm-scoring",
        "--lm-model", LM_MODEL,
        "--skip-report",
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
    print(f"[{i}/{total}] {filename} (done={done} fail={fail} skip={skip} ETA={eta:.0f}m)", flush=True)

    t_start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t_dur = time.time() - t_start
    if result.returncode == 0:
        done += 1
        # Verify lm_candidates.csv was created
        lm_check = Path(OUTPUT) / "htdemucs" / filename / "lm_candidates.csv"
        status = "OK" if lm_check.exists() else "OK (no lm_csv?)"
        print(f"  -> {status} in {t_dur:.1f}s")
    else:
        fail += 1
        err = result.stderr[-300:] if result.stderr else result.stdout[-300:]
        print(f"  -> FAILED in {t_dur:.1f}s: {err}")
    sys.stdout.flush()

elapsed = time.time() - t0
print(f"\nBatch complete: {done} succeeded, {fail} failed, {skip} skipped out of {total}")
print(f"Total time: {elapsed/60:.1f} minutes")
