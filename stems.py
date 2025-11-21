#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lean batch Demucs separation script for HPC use.
- Returns exit code 99 if there are more files to process (signals Bash wrapper to resubmit).
- Returns exit code 0 if all files are finished.
"""

import argparse
import os
import sys
import json
import gc
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import torch

# ---------------------- Configuration (defaults) -------------------------
DEMUCS_MODEL = os.environ.get('DEMUCS_MODEL', 'htdemucs')
DEMUCS_SHIFTS = int(os.environ.get('DEMUCS_SHIFTS', '0'))
DEMUCS_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_FILES_PER_RUN = int(os.environ.get('MAX_FILES_PER_RUN', '10'))
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}

# Default project root — safe place on this cluster. Override with --project-root
DEFAULT_PROJECT_ROOT = '/storage/saumya.mishra_ug25/audio_project'

# ---------------------- Helper utilities --------------------------------

def setup_logging(log_path: str):
    """Configure root logger to write to file and stdout."""
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if logger.handlers:
        logger.handlers = []

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

def now_iso() -> str:
    return datetime.now().isoformat()

# ---------------------- Progress utils ----------------------------------

def load_progress(progress_file: str) -> dict:
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            archived = progress_file + '.corrupt.' + datetime.now().strftime('%Y%m%d%H%M%S')
            try:
                os.rename(progress_file, archived)
            except Exception:
                pass
            return {'processed': [], 'failed': []}
    return {'processed': [], 'failed': []}

def save_progress(progress_file: str, data: dict):
    os.makedirs(os.path.dirname(progress_file) or '.', exist_ok=True)
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# ---------------------- Audio discovery ---------------------------------

def find_audio_files(root_dir: str) -> List[str]:
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if Path(fn).suffix.lower() in AUDIO_EXTENSIONS:
                audio_files.append(os.path.join(dirpath, fn))
    audio_files.sort()
    return audio_files

# ---------------------- Demucs model management --------------------------
DEMUCS_MODEL_OBJ = None

def init_demucs_model(model_name: str = DEMUCS_MODEL, device: str = DEMUCS_DEVICE):
    global DEMUCS_MODEL_OBJ
    if DEMUCS_MODEL_OBJ is not None:
        return DEMUCS_MODEL_OBJ

    try:
        from demucs.pretrained import get_model
    except Exception as e:
        raise RuntimeError(f"Failed importing demucs: {e}")

    logging.info(f"Initializing Demucs model '{model_name}' on device {device}")
    model = get_model(model_name)
    model.to(device)
    model.eval()

    DEMUCS_MODEL_OBJ = model
    return DEMUCS_MODEL_OBJ

def free_demucs_model():
    global DEMUCS_MODEL_OBJ
    try:
        if DEMUCS_MODEL_OBJ is not None:
            del DEMUCS_MODEL_OBJ
            DEMUCS_MODEL_OBJ = None
            if DEMUCS_DEVICE == 'cuda':
                torch.cuda.empty_cache()
    except Exception:
        pass

# ---------------------- Separation functions -----------------------------

def separate_with_demucs(audio_path: str, output_dir: str, model=None, shifts: int = DEMUCS_SHIFTS, device: str = DEMUCS_DEVICE) -> Tuple[str, str]:
    """Separate audio into vocals + accompaniment using Demucs model passed in."""
    from demucs.apply import apply_model
    from demucs.audio import AudioFile, save_audio

    audio_path = Path(audio_path)
    filename = audio_path.stem
    stem_dir = Path(output_dir) / DEMUCS_MODEL / filename
    stem_dir.mkdir(parents=True, exist_ok=True)

    # Read audio
    wav = AudioFile(str(audio_path)).read(
        streams=0,
        samplerate=model.samplerate,
        channels=model.audio_channels
    )

    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None], # Add batch dimension
            device=device,
            shifts=shifts,
            split=True,
            overlap=0.25,
            progress=False
        )[0]

    stem_names = getattr(model, 'sources', None) or ['vocals']
    vocals_idx = stem_names.index('vocals') if 'vocals' in stem_names else 0

    vocals = sources[vocals_idx]
    other_stems = [sources[i] for i in range(len(sources)) if i != vocals_idx]
    accompaniment = torch.stack(other_stems).sum(0) if other_stems else vocals * 0.0

    vocals_path = stem_dir / 'vocals.mp3'
    accompaniment_path = stem_dir / 'accompaniment.mp3'

    # Save outputs
    save_audio(vocals, str(vocals_path), model.samplerate)
    save_audio(accompaniment, str(accompaniment_path), model.samplerate)

    # Cleanup local references
    del wav, sources, vocals, accompaniment
    if device == 'cuda':
        torch.cuda.empty_cache()

    return str(vocals_path), str(accompaniment_path)

# ---------------------- File processing ---------------------------------

def get_stem_directory_for(audio_path: str, project_root: str, model_name: str = DEMUCS_MODEL) -> str:
    filename = Path(audio_path).stem
    return os.path.join(project_root, 'stems', 'separated_stems', model_name, filename)

def process_audio_file(audio_path: str, project_root: str, progress: dict, demucs_model=None) -> bool:
    logger = logging.getLogger()
    filename = os.path.basename(audio_path)
    stem_dir = get_stem_directory_for(audio_path, project_root)

    if audio_path in progress.get('processed', []):
        logger.info(f"⏭️  Skipping (already processed): {filename}")
        return True

    if os.path.isdir(stem_dir):
        logger.info(f"⏭️  Skipping (stems exist): {filename}")
        progress.setdefault('processed', []).append(audio_path)
        return True

    try:
        logger.info(f"🎵 Processing: {filename}")
        vocals, accom = separate_with_demucs(audio_path, os.path.join(project_root, 'stems', 'separated_stems'), model=demucs_model)

        if os.path.exists(vocals) and os.path.exists(accom):
            logger.info(f"✅ Success: {filename}")
            progress.setdefault('processed', []).append(audio_path)
            return True
        else:
            logger.warning(f"⚠️  Stems not found after separation for {filename}")
            progress.setdefault('failed', []).append({'path': audio_path, 'error': 'stems missing', 'timestamp': now_iso()})
            return False

    except Exception as e:
        logger.error(f"❌ Error processing {filename}: {e}")
        progress.setdefault('failed', []).append({'path': audio_path, 'error': str(e), 'timestamp': now_iso()})
        gc.collect()
        if DEMUCS_DEVICE == 'cuda':
            torch.cuda.empty_cache()
        return False

# ---------------------- Main orchestration -------------------------------

def main():
    global DEMUCS_MODEL, MAX_FILES_PER_RUN
    
    parser = argparse.ArgumentParser(description='Batch Demucs stem separation (HPC-friendly)')
    parser.add_argument('--project-root', '-p', default=DEFAULT_PROJECT_ROOT, help='Project root on the cluster')
    parser.add_argument('--model', '-m', default=DEMUCS_MODEL, help='Demucs model name')
    parser.add_argument('--max-files', type=int, default=MAX_FILES_PER_RUN, help='Max files to attempt in this run')
    parser.add_argument('--test', action='store_true', help='Process only the first file and exit (smoke test)')
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    
    # Update globals based on CLI arguments
    DEMUCS_MODEL = args.model
    MAX_FILES_PER_RUN = args.max_files

    ROOT_AUDIO_DIR = os.path.join(project_root, 'RagaDataset')
    PROGRESS_FILE = os.path.join(project_root, 'stems', f'separation_progress_demucs.json')
    LOG_FILE = os.path.join(project_root, 'stems', f'separation_log_demucs.txt')

    os.makedirs(project_root, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)

    setup_logging(LOG_FILE)
    logger = logging.getLogger()

    logger.info('='.ljust(60, '='))
    logger.info(f'Starting Batch Run | Max Files: {MAX_FILES_PER_RUN}')
    logger.info('='.ljust(60, '='))

    audio_files = find_audio_files(ROOT_AUDIO_DIR)
    if not audio_files:
        logger.info('No audio files found. Exiting.')
        sys.exit(0)

    progress = load_progress(PROGRESS_FILE)
    processed_set = set(progress.get('processed', []))

    if args.test:
        audio_files = audio_files[:1]
        logger.info('TEST MODE: processing 1 file only')

    demucs_model = None
    try:
        demucs_model = init_demucs_model(model_name=DEMUCS_MODEL, device=DEMUCS_DEVICE)
    except Exception as e:
        logger.error(f'Model init failed: {e}')
        sys.exit(1)

    pending = [p for p in audio_files if p not in processed_set]
    total_pending = len(pending)
    logger.info(f'Total Pending Files: {total_pending}')

    if total_pending == 0:
        logger.info("All files processed successfully.")
        sys.exit(0)

    new_processed = 0
    for audio_path in pending:
        ok = process_audio_file(audio_path, project_root, progress, demucs_model=demucs_model)
        save_progress(PROGRESS_FILE, progress)
        if ok and audio_path not in processed_set:
            processed_set.add(audio_path)
            new_processed += 1

        if new_processed >= MAX_FILES_PER_RUN:
            logger.info(f'🛑 Limit reached ({MAX_FILES_PER_RUN} files). Stopping run.')
            break
    
    # Cleanup
    save_progress(PROGRESS_FILE, progress)
    free_demucs_model()
    
    # --- EXIT CODE LOGIC FOR BASH WRAPPER ---
    remaining = [p for p in audio_files if p not in processed_set]
    
    if args.test:
        sys.exit(0)

    if len(remaining) > 0:
        logger.info(f"⚠️  {len(remaining)} files remaining. Exiting with code 99 to trigger resubmission.")
        sys.exit(99) # Special code for "Resubmit Me"
    else:
        logger.info("✅ Job Complete. No files remaining.")
        sys.exit(0)

if __name__ == '__main__':
    main()