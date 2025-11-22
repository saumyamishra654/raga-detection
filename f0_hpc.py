#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import json
import sys
import argparse
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC
import matplotlib.pyplot as plt
from datetime import datetime

# Import your custom library
# Ensure swift_f0.py is in the same directory or PYTHONPATH
try:
    from swift_f0 import (
        SwiftF0,
        PitchResult,
        plot_pitch,
        export_to_csv,
        segment_notes,
        plot_notes,
        plot_pitch_and_notes,
        export_to_midi
    )
except ImportError:
    print("Error: swift_f0 module not found. Make sure it is in the python path.")
    sys.exit(1)

# ============================================================================
# CONFIGURATION DEFAULTS
# ============================================================================
# These will be overridden by command line arguments
DEFAULT_PROJECT_ROOT = '/storage/saumya.mishra_ug25/audio_project'
DEFAULT_MAX_FILES = 50

# Pitch detection parameters
MIN_NOTE_STR = 'G1'
MAX_NOTE_STR = 'C4'
FMIN_HZ = librosa.note_to_hz(MIN_NOTE_STR)
FMAX_HZ = librosa.note_to_hz(MAX_NOTE_STR)
VOCAL_CONFIDENCE = 0.98
ACCOMPANIMENT_CONFIDENCE = 0.8
NUM_BINS = 25

# Global paths (set in main)
STEMS_DIR = ""
PROGRESS_FILE = ""
LOG_FILE = ""

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_message(message, print_to_console=True):
    """Log message to file and optionally print to console."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    
    if print_to_console:
        print(log_entry)
    
    if LOG_FILE:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')

def load_progress():
    """Load the list of already processed stems."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {'processed': [], 'failed': []}
    return {'processed': [], 'failed': []}

def save_progress(progress_data):
    """Save the current progress to file."""
    progress_dir = os.path.dirname(PROGRESS_FILE)
    if progress_dir:
        os.makedirs(progress_dir, exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2)

def find_stem_directories(stems_root):
    """Find all directories containing vocals.mp3 and accompaniment.mp3."""
    stem_dirs = []
    for dirpath, dirnames, filenames in os.walk(stems_root):
        has_vocals = 'vocals.mp3' in filenames
        has_accompaniment = 'accompaniment.mp3' in filenames
        if has_vocals and has_accompaniment:
            stem_dirs.append(dirpath)
    stem_dirs.sort()
    return stem_dirs

def cleanup_memory():
    """Force garbage collection to free memory."""
    gc.collect()

# ============================================================================
# PITCH ANALYSIS FUNCTIONS
# ============================================================================

def load_swiftf0_csv_as_pitchresult(csv_path, silent=False):
    """Load a SwiftF0 CSV export back into a PitchResult object."""
    df = pd.read_csv(csv_path)
    
    # Accept 'time' or 'timestamp' for timing
    ts_col = 'time' if 'time' in df.columns else 'timestamp' if 'timestamp' in df.columns else None
    pitch_col = 'pitch_hz' if 'pitch_hz' in df.columns else None
    conf_col = 'confidence' if 'confidence' in df.columns else None
    voiced_col = 'voicing' if 'voicing' in df.columns else ('voiced' if 'voiced' in df.columns else None)

    if pitch_col is None or conf_col is None:
        raise ValueError(f"CSV {csv_path} missing required columns.")

    timestamps = pd.to_numeric(df[ts_col], errors='coerce').to_numpy() if ts_col else np.arange(len(df)).astype(float)
    pitch_hz = pd.to_numeric(df[pitch_col], errors='coerce').to_numpy()
    pitch_hz = np.nan_to_num(pitch_hz, nan=0.0)
    confidence = pd.to_numeric(df[conf_col], errors='coerce').to_numpy()
    confidence = np.nan_to_num(confidence, nan=0.0)

    if voiced_col:
        voiced_raw = df[voiced_col].astype(str).fillna('').str.strip().str.lower()
        def parse_bool(x):
            return x in ('1','1.0','true','t','yes','y')
        voiced = np.array([parse_bool(x) for x in voiced_raw], dtype=bool)
    else:
        voiced = (pitch_hz > 0) & (confidence > 0)

    # Frame period estimation
    frame_period = None
    if 'frame_rate' in df.columns:
        try:
            frame_period = 1.0 / float(df['frame_rate'].iloc[0])
        except: pass
    elif ts_col and len(timestamps) > 1:
        diffs = np.diff(timestamps[:min(50, len(timestamps))])
        frame_period = float(np.median(diffs)) if diffs.size else None

    pr = PitchResult(pitch_hz=pitch_hz, confidence=confidence, timestamps=timestamps, voicing=voiced)
    setattr(pr, 'frame_period', frame_period)
    
    if not silent:
        log_message(f"  Loaded CSV: {os.path.basename(csv_path)}")
    
    return pr

def analyze_or_load_with_plots(stem_path, detector, output_prefix="output",
                              num_bins=25, save_midi=True, force_recompute=False,
                              override_confidence_threshold=None):
    """Analyze audio file or load existing analysis, generate plots and stats."""
    
    stem_dir = os.path.dirname(stem_path) or "."
    csv_path = os.path.join(stem_dir, f"{output_prefix}_pitch_data.csv")
    midi_npy_path = os.path.join(stem_dir, f"{output_prefix}_midi.npy")
    midi_csv_path = os.path.join(stem_dir, f"{output_prefix}_midi.csv")

    pr = None
    sr = None

    # 1) Load CSV or run detection
    if os.path.isfile(csv_path) and not force_recompute:
        try:
            pr = load_swiftf0_csv_as_pitchresult(csv_path, silent=True)
            log_message(f"  Loaded existing pitch data from CSV")
        except Exception:
            pr = None

    if pr is None:
        # log_message(f"  Running SwiftF0 pitch detection...")
        y, sr = librosa.load(stem_path, sr=None, duration=600.0)
        pr = detector.detect_from_array(y, sr)
        try:
            export_to_csv(pr, csv_path)
            # log_message(f"  Exported pitch data to CSV")
        except Exception as e:
            log_message(f"  Warning: export_to_csv failed: {e}")

    # Apply confidence threshold override
    if override_confidence_threshold is not None:
        voiced = (pr.pitch_hz > 0) & (pr.confidence >= override_confidence_threshold)
        setattr(pr, "voicing", voiced)

    # Plot pitch
    try:
        plot_pitch(pr, show=False, output_path=os.path.join(stem_dir, f"{output_prefix}_pitch.jpg"))
    except Exception: pass

    # Note segmentation and plots
    notes = []
    try:
        notes = segment_notes(pr, split_semitone_threshold=0.8, min_note_duration=0.25)
        try:
            plot_notes(notes, output_path=os.path.join(stem_dir, f"{output_prefix}_note_segments.jpg"))
            plot_pitch_and_notes(pr, notes, output_path=os.path.join(stem_dir, f"{output_prefix}_combined_analysis.jpg"))
            export_to_midi(notes, os.path.join(stem_dir, f"{output_prefix}_notes.mid"))
        except Exception: pass
    except Exception:
        notes = []

    # Extract voiced frames for histogram
    voiced_mask = getattr(pr, 'voicing', None)
    pitch_hz_arr = getattr(pr, 'pitch_hz', np.array([], dtype=float))
    conf_arr = getattr(pr, 'confidence', None)

    if voiced_mask is not None and len(voiced_mask) == len(pitch_hz_arr):
        voiced_freqs = pitch_hz_arr[voiced_mask]
        voiced_confs = conf_arr[voiced_mask] if conf_arr is not None else np.ones_like(voiced_freqs)
    else:
        voiced_freqs = pitch_hz_arr
        voiced_confs = conf_arr if conf_arr is not None else np.ones_like(voiced_freqs)

    valid_mask = voiced_freqs > 0
    valid_frequencies = voiced_freqs[valid_mask]
    valid_confidences = voiced_confs[valid_mask] if np.size(voiced_confs) else np.array([])

    # Frame duration
    frame_period = getattr(pr, 'frame_period', None)
    if frame_period is None:
        frame_period = 512.0 / float(sr) if sr else 0.01
    weights = (valid_confidences * frame_period) if valid_confidences.size else np.array([])

    # MIDI values
    midi_vals = None
    if save_midi and os.path.isfile(midi_npy_path) and not force_recompute:
        try:
            midi_vals = np.load(midi_npy_path)
        except Exception:
            midi_vals = None

    if midi_vals is None:
        if valid_frequencies.size:
            midi_vals = librosa.hz_to_midi(valid_frequencies)
        else:
            midi_vals = np.array([], dtype=float)
        if save_midi:
            try:
                np.save(midi_npy_path, midi_vals)
                pd.DataFrame({"midi": np.round(midi_vals, 6)}).to_csv(midi_csv_path, index=False)
            except Exception: pass

    # Cent histogram
    if valid_frequencies.size:
        midi_vals_full = librosa.hz_to_midi(valid_frequencies)
        cent_values = (midi_vals_full % 12) * 100.0
        cent_hist, _ = np.histogram(cent_values, bins=num_bins, range=(0, 1200), weights=weights if weights.size else None)
    else:
        cent_hist = np.zeros(num_bins, dtype=float)

    # Save histograms
    try:
        bin_width = 1200.0 / num_bins
        plt.figure(figsize=(12, 4))
        plt.bar(np.arange(num_bins), cent_hist, width=1.0, color='firebrick', alpha=0.8)
        plt.title(f"{output_prefix}: Weighted Cent Histogram")
        plt.savefig(os.path.join(stem_dir, f"{output_prefix}_cent_histogram.jpg"))
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.hist(valid_frequencies, bins=1200, color='darkcyan', alpha=0.8, weights=weights if weights.size else None)
        plt.title(f"{output_prefix}: Weighted Frequency Histogram")
        plt.savefig(os.path.join(stem_dir, f"{output_prefix}_frequency_histogram.jpg"))
        plt.close()
    except Exception as e:
        log_message(f"  Warning: Plotting failed: {e}")

    return pr, valid_frequencies, cent_hist, midi_vals

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_stem_directory(stem_dir, detector_v, detector_a, progress_data):
    """Process vocals and accompaniment in a stem directory."""
    dir_name = os.path.basename(stem_dir)
    vocals_path = os.path.join(stem_dir, 'vocals.mp3')
    accompaniment_path = os.path.join(stem_dir, 'accompaniment.mp3')
    
    # Check if output files already exist
    vocal_csv = os.path.join(stem_dir, 'vocals_pitch_data.csv')
    accomp_csv = os.path.join(stem_dir, 'accompaniment_pitch_data.csv')
    
    if os.path.exists(vocal_csv) and os.path.exists(accomp_csv):
        log_message(f"⏭️  Skipping (analysis exists): {dir_name}")
        return True
    
    try:
        log_message(f"🎵 Processing: {dir_name}")
        
        # Analyze vocals
        analyze_or_load_with_plots(
            vocals_path, detector_v, output_prefix="vocals",
            num_bins=NUM_BINS, override_confidence_threshold=VOCAL_CONFIDENCE
        )
        
        # Analyze accompaniment
        analyze_or_load_with_plots(
            accompaniment_path, detector_a, output_prefix="accompaniment",
            num_bins=NUM_BINS
        )
        
        log_message(f"✅ Success: {dir_name}")
        progress_data['processed'].append(stem_dir)
        save_progress(progress_data)
        cleanup_memory()
        return True
        
    except Exception as e:
        log_message(f"❌ Error processing {dir_name}: {str(e)}")
        progress_data['failed'].append({
            'path': stem_dir,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })
        save_progress(progress_data)
        cleanup_memory()
        return False

def main():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description='Batch SwiftF0 Pitch Analysis')
    parser.add_argument('--project-root', type=str, default=DEFAULT_PROJECT_ROOT, help='Path to project root')
    parser.add_argument('--max-files', type=int, default=DEFAULT_MAX_FILES, help='Files per batch')
    args = parser.parse_args()

    # Update globals
    global STEMS_DIR, PROGRESS_FILE, LOG_FILE
    project_root = os.path.abspath(args.project_root)
    STEMS_DIR = os.path.join(project_root, 'stems', 'separated_stems')
    PROGRESS_FILE = os.path.join(project_root, 'stems', 'pitch_analysis_progress.json')
    LOG_FILE = os.path.join(project_root, 'stems', 'pitch_analysis_log.txt')
    MAX_FILES_PER_RUN = args.max_files

    log_message("="*60)
    log_message(f"Starting Pitch Analysis | Batch Size: {MAX_FILES_PER_RUN}")
    log_message(f"Root: {project_root}")
    log_message("="*60)
    
    # Find stems
    stem_dirs = find_stem_directories(STEMS_DIR)
    if not stem_dirs:
        log_message("No stem directories found. Exiting.")
        sys.exit(0)
    
    # Load progress
    progress_data = load_progress()
    processed_set = set(progress_data['processed'])
    pending_dirs = [path for path in stem_dirs if path not in processed_set]
    
    total_pending = len(pending_dirs)
    log_message(f"Total Pending Directories: {total_pending}")

    if total_pending == 0:
        log_message("All directories processed successfully.")
        sys.exit(0)

    # Initialize detectors (only if we have work to do)
    log_message("Initializing SwiftF0 detectors...")
    detector_v = SwiftF0(fmin=FMIN_HZ, fmax=FMAX_HZ, confidence_threshold=VOCAL_CONFIDENCE)
    detector_a = SwiftF0(fmin=FMIN_HZ, fmax=FMAX_HZ, confidence_threshold=ACCOMPANIMENT_CONFIDENCE)

    # Process batch
    count = 0
    for stem_dir in pending_dirs:
        if count >= MAX_FILES_PER_RUN:
            log_message(f"🛑 Batch limit reached ({MAX_FILES_PER_RUN}). Stopping run.")
            break
            
        success = process_stem_directory(stem_dir, detector_v, detector_a, progress_data)
        if success:
            count += 1

    # --- EXIT CODE LOGIC FOR BASH WRAPPER ---
    # Re-check remaining files after this run
    progress_data = load_progress() # Reload to be sure
    processed_now = set(progress_data['processed'])
    remaining = [p for p in stem_dirs if p not in processed_now]
    
    if len(remaining) > 0:
        log_message(f"⚠️  {len(remaining)} directories remaining. Exiting with code 99 to trigger resubmission.")
        sys.exit(99) # Signal to PBS to resubmit
    else:
        log_message("✅ Job Complete. No directories remaining.")
        sys.exit(0)

if __name__ == "__main__":
    main()