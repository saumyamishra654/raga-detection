#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from datetime import datetime

# Try to import SwiftF0 classes if available, otherwise define minimal mocks
try:
    from swift_f0 import PitchResult
except ImportError:
    # Minimal definition if module not found (we only need the class structure for loading)
    class PitchResult:
        def __init__(self, pitch_hz, confidence, timestamps, voicing):
            self.pitch_hz = pitch_hz
            self.confidence = confidence
            self.timestamps = timestamps
            self.voicing = voicing

# ============================================================================
# CONFIGURATION DEFAULTS
# ============================================================================
DEFAULT_PROJECT_ROOT = '/Volumes/Extreme SSD'
DEFAULT_RAGA_DB = '/Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-detection/main notebooks/raga_list_final.csv'
DEFAULT_MAX_FILES = 5000

# Feature Extraction Constants
EPS = 1e-12
WINDOW_CENTS = 35.0
BIN_SIZE_CENTS = 1.0
SIGMA_SMOOTHING = 0.8

# Global paths
STEMS_DIR = ""
PROGRESS_FILE = ""
LOG_FILE = ""

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_message(message, print_to_console=True):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    if print_to_console:
        print(log_entry)
    if LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {'processed': [], 'failed': []}
    return {'processed': [], 'failed': []}

def save_progress(progress_data):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2)

def find_stem_directories(stems_root):
    stem_dirs = []
    for dirpath, dirnames, filenames in os.walk(stems_root):
        # Look for the output CSVs from the previous step
        if 'vocals_pitch_data.csv' in filenames and 'accompaniment_pitch_data.csv' in filenames:
            stem_dirs.append(dirpath)
    stem_dirs.sort()
    return stem_dirs

# ============================================================================
# DATA LOADING
# ============================================================================

def load_swiftf0_csv_as_pitchresult(csv_path):
    """Load a SwiftF0 CSV export back into a PitchResult object."""
    df = pd.read_csv(csv_path)
    
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

    pr = PitchResult(pitch_hz=pitch_hz, confidence=confidence, timestamps=timestamps, voicing=voiced)
    
    # Frame period estimation
    if 'frame_rate' in df.columns:
        try:
            setattr(pr, 'frame_period', 1.0 / float(df['frame_rate'].iloc[0]))
        except: pass
    elif ts_col and len(timestamps) > 1:
        diffs = np.diff(timestamps[:min(50, len(timestamps))])
        setattr(pr, 'frame_period', float(np.median(diffs)) if diffs.size else 0.01)
    else:
        setattr(pr, 'frame_period', 0.01)
        
    return pr

def load_raga_db(db_path):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Raga DB not found at {db_path}")
    return pd.read_csv(db_path)

# ============================================================================
# FEATURE EXTRACTION LOGIC
# ============================================================================

def get_valid_frequencies(pr):
    """Extract valid frequencies and weights from PitchResult."""
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
    
    frame_period = getattr(pr, 'frame_period', 0.01)
    weights = (valid_confidences * frame_period) if valid_confidences.size else np.array([])
    
    return valid_frequencies, weights

def compute_histograms(valid_frequencies, weights):
    """Compute fine, coarse, and smoothed histograms."""
    if not valid_frequencies.size:
        return None, None, None

    midi_vals = librosa.hz_to_midi(valid_frequencies)
    cent_values = (midi_vals % 12) * 100.0
    
    # Fine histogram (100 bins)
    H_mel_100, bin_edges_100 = np.histogram(cent_values, bins=100, range=(0, 1200), weights=weights)
    
    # Coarse histogram (33 bins)
    H_mel_33, bin_edges_33 = np.histogram(cent_values, bins=33, range=(0, 1200), weights=weights)
    
    # Smoothed histogram
    smoothed_H_100 = gaussian_filter1d(H_mel_100, sigma=SIGMA_SMOOTHING, mode='wrap')
    
    return H_mel_100, bin_edges_100, H_mel_33, bin_edges_33, smoothed_H_100

def find_circular_peaks(arr, prominence=None, distance=None, height=None):
    n = len(arr)
    peaks, _ = find_peaks(arr, prominence=prominence, distance=distance, height=height)
    peaks = list(peaks)

    if n >= 2:
        if 0 not in peaks and arr[0] > arr[1] and arr[0] > arr[-1]:
            peaks.append(0)
        if (n - 1) not in peaks and arr[-1] > arr[-2] and arr[-1] > arr[0]:
            peaks.append(n - 1)

    peaks = sorted(set(peaks))
    return np.array(peaks, dtype=int)

def detect_final_peaks(H_mel_100, bin_edges_100, H_mel_33, bin_edges_33, smoothed_H_100):
    # Constants from notebook
    prom_high = max(1.0, 0.03 * float(smoothed_H_100.max()))
    dist_high = 2
    prom_low = max(0, 0.01 * float(H_mel_33.max()))
    dist_low = 1
    
    peak_tolerance_cents = 45
    tolerance_cents = 50
    note_centers = np.arange(0, 1200, 100)

    # High-res peaks
    bin_centers_100 = (bin_edges_100[:-1] + bin_edges_100[1:]) / 2.0
    smoothed_peaks_indices = find_circular_peaks(smoothed_H_100, prominence=prom_high, distance=dist_high)
    smoothed_peaks_cents = bin_centers_100[smoothed_peaks_indices]

    # Low-res peaks
    bin_centers_33 = (bin_edges_33[:-1] + bin_edges_33[1:]) / 2.0
    raw_peaks_indices = find_circular_peaks(H_mel_33, prominence=prom_low, distance=dist_low)
    raw_peaks_cents = bin_centers_33[raw_peaks_indices]

    # Validation
    final_peak_indices = []
    for i, sp_cent in zip(smoothed_peaks_indices, smoothed_peaks_cents):
        # Check 1: Near a low-res peak
        is_validated = any(abs(sp_cent - rp_cent) <= peak_tolerance_cents for rp_cent in raw_peaks_cents)
        if not is_validated:
            continue
            
        # Check 2: Near a semitone center
        diffs = np.abs((sp_cent - note_centers + 600) % 1200 - 600)
        if np.min(diffs) <= tolerance_cents:
            final_peak_indices.append(i)
            
    return final_peak_indices, bin_centers_100

def compute_gmm_vector(smoothed_H_100, bin_centers_100, peak_indices):
    # Normalize histogram for GMM
    total = smoothed_H_100.sum()
    if total <= 0:
        return np.full((12, 3), np.nan).tolist()
    
    smoothed_norm = smoothed_H_100.astype(float) / total
    
    note_centers = np.arange(0, 1200, 100)
    note_gmm_vector_12 = np.full((12, 3), np.nan) # (height, deviation, sigma)
    
    peak_cents = bin_centers_100[peak_indices]
    
    for peak_idx, peak_cent in zip(peak_indices, peak_cents):
        # Window extraction (±75 cents -> 150 width)
        window_width_cents = 150
        half_window = window_width_cents / 2
        
        distances = np.abs(bin_centers_100 - peak_cent)
        distances = np.minimum(distances, 1200 - distances)
        window_mask = distances <= half_window
        
        if np.sum(window_mask) < 3: continue
        
        window_cents = bin_centers_100[window_mask]
        window_values = smoothed_norm[window_mask]
        
        # Create samples for GMM
        scale_factor = max(1, int(1000 / max(np.sum(window_values), 1e-6)))
        samples = []
        for cent, value in zip(window_cents, window_values):
            n_samples = max(1, int(value * scale_factor))
            samples.extend([cent] * n_samples)
            
        if len(samples) < 5: continue
        
        X = np.array(samples).reshape(-1, 1)
        
        try:
            gmm = GaussianMixture(n_components=1, random_state=42)
            gmm.fit(X)
            
            mu = float(gmm.means_.ravel()[0])
            sigma = float(np.sqrt(gmm.covariances_.ravel()[0]))
            peak_height = float(smoothed_norm[peak_idx])
            
            # Map to nearest note
            diffs_to_notes = np.abs((peak_cent - note_centers + 600) % 1200 - 600)
            nearest_note_idx = int(np.argmin(diffs_to_notes))
            
            # Deviation from peak (as per notebook default 'peak' reference)
            deviation = (mu - peak_cent + 600) % 1200 - 600
            
            # Store if empty or higher peak
            existing = note_gmm_vector_12[nearest_note_idx]
            if np.isnan(existing[0]) or peak_height > existing[0]:
                note_gmm_vector_12[nearest_note_idx] = [peak_height, deviation, sigma]
                
        except:
            continue
            
    # Convert NaNs to None for JSON
    return [[None if np.isnan(x) else x for x in row] for row in note_gmm_vector_12]

def compute_pitch_distribution(valid_frequencies, weights):
    """Compute the 12-bin pitch class distribution (p_pc) using windowing."""
    if not valid_frequencies.size:
        return np.zeros(12)

    midi_vals = librosa.hz_to_midi(valid_frequencies)
    cent_vals = (midi_vals % 12) * 100.0
    
    # High-res histogram for windowing
    num_bins = int(1200 / BIN_SIZE_CENTS)
    bin_edges = np.linspace(0.0, 1200.0, num_bins + 1)
    cent_hist, _ = np.histogram(cent_vals, bins=bin_edges, range=(0.0, 1200.0), weights=weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    def mass_within_window(note_idx):
        center = (note_idx * 100.0) % 1200.0
        diff = np.abs(bin_centers - center)
        diff = np.minimum(diff, 1200.0 - diff)
        return float(np.sum(cent_hist[diff <= WINDOW_CENTS]))

    note_masses = np.array([mass_within_window(i) for i in range(12)], dtype=float)
    
    # Normalize
    total = np.sum(note_masses)
    if total > 0:
        p_pc = (note_masses + EPS) / (total + 12 * EPS)
    else:
        p_pc = np.ones(12) / 12.0
        
    return p_pc

def extract_raga_features(p_pc, raga_df):
    """Calculate regression features for all ragas across all tonics."""
    candidates = []
    
    # Pre-process ragas to avoid repeated parsing
    parsed_ragas = []
    for _, row in raga_df.iterrows():
        mask_abs = None
        # Try 'mask' column first
        if 'mask' in raga_df.columns and pd.notna(row.get('mask')):
            try:
                raw = row['mask']
                if isinstance(raw, str):
                    mask_abs = tuple(int(x) for x in raw.split(',') if x.strip())
                else:
                    mask_abs = tuple(int(x) for x in list(raw))
            except: pass
        
        # Try 0..11 columns
        if mask_abs is None:
            try:
                mask_abs = tuple(int(row[str(i)]) for i in range(12))
            except: pass
            
        if mask_abs and sum(mask_abs) >= 2:
            name = row.get('names') if 'names' in raga_df.columns else row.get('raga', 'Unknown')
            parsed_ragas.append({
                'name': name,
                'mask': np.array(mask_abs, dtype=int),
                'indices': np.where(np.array(mask_abs) == 1)[0],
                'size': len(np.where(np.array(mask_abs) == 1)[0])
            })

    # Iterate all 12 tonics
    for tonic in range(12):
        # Rotate distribution for this tonic
        p_rot = np.roll(p_pc, -tonic)
        peak_val = float(np.max(p_rot) + EPS)
        
        for raga in parsed_ragas:
            indices = raga['indices']
            raga_size = raga['size']
            
            # 1. Match Mass
            match_mass = float(np.sum(p_rot[indices]))
            
            # 2. Extra Mass
            extra_mass = float(1.0 - match_mass)
            
            # 3. Presence (Observed Note Score)
            pres = (p_rot[indices] / peak_val)
            # Using USE_PRESENCE_MEAN = True logic from notebook
            observed_note_score = float(np.mean(pres)) if raga_size > 0 else 0.0
            
            # 4. Log Likelihood
            sum_logp = float(np.sum(np.log(p_rot[indices] + EPS)))
            baseline = -np.log(12.0)
            avg_logp = sum_logp / (raga_size + EPS)
            loglike_norm = 1.0 + (avg_logp / (-baseline + EPS))
            loglike_norm = max(0.0, min(1.0, loglike_norm))
            
            # 5. Complexity Penalty
            complexity_pen = max(0.0, (raga_size - 5) / 12.0)
            
            # 6. Size Penalty (assuming detected_peak_count is roughly raga_size for now, 
            # or we can omit if we don't have peak count from this script. 
            # Let's store raga_size so we can compute penalty later if needed)
            
            candidates.append({
                'tonic': tonic,
                'raga': raga['name'],
                'features': {
                    'match_mass': match_mass,
                    'extra_mass': extra_mass,
                    'presence': observed_note_score,
                    'loglike': loglike_norm,
                    'complexity': complexity_pen,
                    'raga_size': int(raga_size)
                }
            })
            
    return candidates

def save_analysis_plots(output_dir, stem_name, acc_vector, p_pc_v, 
                       H_mel_100, bin_edges_100, 
                       H_mel_33, bin_edges_33, 
                       smoothed_H_100, 
                       final_peak_indices, bin_centers_100):
    
    # 1. Accompaniment & Melody Distributions (Side by Side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    x = np.arange(12)
    
    # Accompaniment
    ax1.bar(x, acc_vector, color='skyblue')
    ax1.set_xticks(x, labels=notes)
    ax1.set_title(f'Accompaniment Salience\n{stem_name}')
    ax1.set_ylim(0, 1.1)
    
    # Melody
    ax2.bar(x, p_pc_v, color='salmon')
    ax2.set_xticks(x, labels=notes)
    ax2.set_title(f'Melody Pitch Class Distribution\n{stem_name}')
    ax2.set_ylim(0, max(p_pc_v) * 1.1 if max(p_pc_v) > 0 else 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pitch_distributions.png'))
    plt.close()

    # 2. Histograms (Fine, Coarse, Smoothed)
    if H_mel_100 is not None:
        plt.figure(figsize=(12, 6))
        
        # Fine (100 bins)
        plt.stairs(H_mel_100, bin_edges_100, color='lightgray', fill=True, label='Fine (100 bins)', alpha=0.5)
        
        # Coarse (33 bins)
        plt.stairs(H_mel_33, bin_edges_33, color='orange', linewidth=1.5, label='Coarse (33 bins)')
        
        # Smoothed
        bin_centers = (bin_edges_100[:-1] + bin_edges_100[1:]) / 2
        plt.plot(bin_centers, smoothed_H_100, color='blue', linewidth=2, label='Smoothed')
        
        # Peaks
        if final_peak_indices is not None and len(final_peak_indices) > 0:
            peak_cents = bin_centers_100[final_peak_indices]
            peak_vals = smoothed_H_100[final_peak_indices]
            plt.scatter(peak_cents, peak_vals, color='red', zorder=5, label='Detected Peaks')
            
        plt.xlim(0, 1200)
        plt.xlabel('Cents')
        plt.ylabel('Weighted Frequency')
        plt.title(f'Melody Histograms & Peaks - {stem_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, 'histograms_analysis.png'))
        plt.close()

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_stem_directory(stem_dir, raga_df):
    dir_name = os.path.basename(stem_dir)
    output_dir = os.path.join(stem_dir, 'raga_features')
    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, 'raga_features.json')

    try:
        log_message(f"📊 Processing: {dir_name}")
        
        # Load Pitch Data
        vocal_csv = os.path.join(stem_dir, 'vocals_pitch_data.csv')
        accomp_csv = os.path.join(stem_dir, 'accompaniment_pitch_data.csv')
        
        pr_v = load_swiftf0_csv_as_pitchresult(vocal_csv)
        pr_a = load_swiftf0_csv_as_pitchresult(accomp_csv)
        
        # Process Vocals
        freqs_v, weights_v = get_valid_frequencies(pr_v)
        p_pc_v = compute_pitch_distribution(freqs_v, weights_v)
        
        # Initialize plot vars
        H_mel_100 = bin_edges_100 = H_mel_33 = bin_edges_33 = smoothed_H_100 = None
        final_peak_indices = bin_centers_100 = None
        
        # Compute GMM Vector
        gmm_vector = None
        if freqs_v.size:
            H_mel_100, bin_edges_100, H_mel_33, bin_edges_33, smoothed_H_100 = compute_histograms(freqs_v, weights_v)
            if H_mel_100 is not None:
                final_peak_indices, bin_centers_100 = detect_final_peaks(H_mel_100, bin_edges_100, H_mel_33, bin_edges_33, smoothed_H_100)
                gmm_vector = compute_gmm_vector(smoothed_H_100, bin_centers_100, final_peak_indices)
        
        # Process Accompaniment (Salience)
        freqs_a, weights_a = get_valid_frequencies(pr_a)
        if freqs_a.size:
            midi_a = librosa.hz_to_midi(freqs_a)
            pc_a = np.mod(np.round(midi_a), 12).astype(int)
            H_acc, _ = np.histogram(pc_a, bins=12, range=(0, 12), weights=weights_a)
            # Normalize accompaniment vector
            max_val = H_acc.max()
            acc_vector = (H_acc / max_val).tolist() if max_val > 0 else np.zeros(12).tolist()
        else:
            acc_vector = np.zeros(12).tolist()

        # Extract Features
        candidates = extract_raga_features(p_pc_v, raga_df)
        
        # Save Plots
        save_analysis_plots(output_dir, dir_name, acc_vector, p_pc_v, 
                          H_mel_100, bin_edges_100, H_mel_33, bin_edges_33, 
                          smoothed_H_100, final_peak_indices, bin_centers_100)
        
        # Save Result
        result_data = {
            'stem_dir': stem_dir,
            'stem_name': dir_name,
            'accompaniment_salience': acc_vector,
            'melody_distribution': p_pc_v.tolist(),
            'gmm_vector': gmm_vector,
            'candidates': candidates
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)
            
        log_message(f"✅ Success: {dir_name}")
        return True

    except Exception as e:
        log_message(f"❌ Error processing {dir_name}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract Raga Features for Regression')
    parser.add_argument('--project-root', type=str, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument('--raga-db', type=str, default=DEFAULT_RAGA_DB)
    parser.add_argument('--max-files', type=int, default=DEFAULT_MAX_FILES)
    args = parser.parse_args()

    global STEMS_DIR, PROGRESS_FILE, LOG_FILE
    project_root = os.path.abspath(args.project_root)
    STEMS_DIR = os.path.join(project_root, 'stems', 'separated_stems')
    PROGRESS_FILE = os.path.join(project_root, 'stems', 'feature_extraction_progress.json')
    LOG_FILE = os.path.join(project_root, 'stems', 'feature_extraction_log.txt')
    
    log_message("="*60)
    log_message(f"Starting Feature Extraction | Batch Size: {args.max_files}")
    log_message(f"DB: {args.raga_db}")
    log_message("="*60)

    # Load Raga DB
    try:
        raga_df = load_raga_db(args.raga_db)
        log_message(f"Loaded Raga DB: {len(raga_df)} entries")
    except Exception as e:
        log_message(f"Failed to load Raga DB: {e}")
        sys.exit(1)

    # Find stems
    stem_dirs = find_stem_directories(STEMS_DIR)
    progress_data = load_progress()
    processed_set = set(progress_data['processed'])
    pending_dirs = [p for p in stem_dirs if p not in processed_set]
    
    log_message(f"Total Pending: {len(pending_dirs)}")
    
    if not pending_dirs:
        log_message("All directories processed.")
        sys.exit(0)

    count = 0
    for stem_dir in pending_dirs:
        if count >= args.max_files:
            log_message(f"🛑 Batch limit reached ({args.max_files}).")
            break
            
        success = process_stem_directory(stem_dir, raga_df)
        if success:
            progress_data['processed'].append(stem_dir)
            save_progress(progress_data)
            count += 1
        else:
            progress_data['failed'].append({'path': stem_dir, 'time': str(datetime.now())})
            save_progress(progress_data)

    # Final status
    remaining = len([p for p in stem_dirs if p not in set(load_progress()['processed'])])
    if remaining > 0:
        log_message(f"⚠️  {remaining} remaining.")
    
    log_message("✅ Job Complete.")
    sys.exit(0)

if __name__ == "__main__":
    main()
