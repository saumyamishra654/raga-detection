import os
import json
import pandas as pd
import numpy as np
import librosa
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_PROJECT_ROOT = '/Volumes/Extreme SSD'
DEFAULT_GT_CSV = '/Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-detection/main notebooks/ground_truth.csv'
DEFAULT_RAGA_DB = '/Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-detection/main notebooks/raga_list_final.csv'
DEFAULT_STEMS_DIR = os.path.join(DEFAULT_PROJECT_ROOT, 'stems', 'separated_stems', 'htdemucs')

# Feature Extraction Constants
EPS = 1e-12
WINDOW_CENTS = 35.0
BIN_SIZE_CENTS = 1.0

# Your Manual Weights (for comparison)
OLD_WEIGHTS = {
    'match_mass': 0.40,
    'presence': 0.25,
    'loglike': 1.0,
    'extra_mass': -1.10,
    'complexity': -0.4,
    'size_penalty': -0.25,
    'tonic_salience': 0.12,
    'primary_score': 0.0
}

NOTE_MAPPING = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 
    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

# ============================================================================
# DATA PROCESSING UTILS
# ============================================================================

class PitchResult:
    def __init__(self, pitch_hz, confidence, timestamps, voicing):
        self.pitch_hz = pitch_hz
        self.confidence = confidence
        self.timestamps = timestamps
        self.voicing = voicing

def load_swiftf0_csv_as_pitchresult(csv_path):
    df = pd.read_csv(csv_path)
    ts_col = 'time' if 'time' in df.columns else 'timestamp' if 'timestamp' in df.columns else None
    pitch_col = 'pitch_hz' if 'pitch_hz' in df.columns else None
    conf_col = 'confidence' if 'confidence' in df.columns else None
    voiced_col = 'voicing' if 'voicing' in df.columns else ('voiced' if 'voiced' in df.columns else None)

    if pitch_col is None or conf_col is None:
        return None

    timestamps = pd.to_numeric(df[ts_col], errors='coerce').to_numpy() if ts_col else np.arange(len(df)).astype(float)
    pitch_hz = pd.to_numeric(df[pitch_col], errors='coerce').to_numpy()
    pitch_hz = np.nan_to_num(pitch_hz, nan=0.0)
    confidence = pd.to_numeric(df[conf_col], errors='coerce').to_numpy()
    confidence = np.nan_to_num(confidence, nan=0.0)

    if voiced_col:
        voiced_raw = df[voiced_col].astype(str).fillna('').str.strip().str.lower()
        def parse_bool(x): return x in ('1','1.0','true','t','yes','y')
        voiced = np.array([parse_bool(x) for x in voiced_raw], dtype=bool)
    else:
        voiced = (pitch_hz > 0) & (confidence > 0)

    pr = PitchResult(pitch_hz=pitch_hz, confidence=confidence, timestamps=timestamps, voicing=voiced)
    
    if 'frame_rate' in df.columns:
        try: setattr(pr, 'frame_period', 1.0 / float(df['frame_rate'].iloc[0]))
        except: pass
    elif ts_col and len(timestamps) > 1:
        diffs = np.diff(timestamps[:min(50, len(timestamps))])
        setattr(pr, 'frame_period', float(np.median(diffs)) if diffs.size else 0.01)
    else:
        setattr(pr, 'frame_period', 0.01)
    return pr

def get_valid_frequencies(pr):
    if pr is None: return np.array([]), np.array([])
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

def compute_pitch_distribution(valid_frequencies, weights):
    if not valid_frequencies.size: return np.zeros(12)
    midi_vals = librosa.hz_to_midi(valid_frequencies)
    cent_vals = (midi_vals % 12) * 100.0
    
    num_bins = int(1200 / BIN_SIZE_CENTS)
    bin_edges = np.linspace(0.0, 1200.0, num_bins + 1)
    # UNWEIGHTED to match notebook
    cent_hist, _ = np.histogram(cent_vals, bins=bin_edges, range=(0.0, 1200.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    def mass_within_window(note_idx):
        center = (note_idx * 100.0) % 1200.0
        diff = np.abs(bin_centers - center)
        diff = np.minimum(diff, 1200.0 - diff)
        return float(np.sum(cent_hist[diff <= WINDOW_CENTS]))

    note_masses = np.array([mass_within_window(i) for i in range(12)], dtype=float)
    total = np.sum(note_masses)
    if total > 0:
        p_pc = (note_masses + EPS) / (total + 12 * EPS)
    else:
        p_pc = np.ones(12) / 12.0
    return p_pc

def get_detected_peak_count(valid_frequencies, weights):
    if not valid_frequencies.size: return 7
    
    midi_vals = librosa.hz_to_midi(valid_frequencies)
    cent_vals = (midi_vals % 12) * 100.0
    
    # High Res (100 bins) - UNWEIGHTED to match notebook
    H_mel_100, bin_edges_100 = np.histogram(cent_vals, bins=100, range=(0, 1200))
    bin_centers_100 = (bin_edges_100[:-1] + bin_edges_100[1:]) / 2.0
    
    # Low Res (33 bins) - UNWEIGHTED to match notebook
    H_mel_25, bin_edges_25 = np.histogram(cent_vals, bins=33, range=(0, 1200))
    bin_centers_25 = (bin_edges_25[:-1] + bin_edges_25[1:]) / 2.0
    
    # Smoothing
    sigma = 0.8
    smoothed_H_100 = gaussian_filter1d(H_mel_100, sigma=sigma, mode='wrap')
    
    # Peak Finding Helper - Notebook's approach with edge checks
    def find_circular_peaks_simple(arr, prominence, distance):
        n = len(arr)
        peaks, _ = find_peaks(arr, prominence=prominence, distance=distance)
        peaks = list(peaks)
        # Manual wrap checks for boundaries (from notebook)
        if n >= 2:
            if 0 not in peaks and arr[0] > arr[1] and arr[0] > arr[-1]:
                peaks.append(0)
            if (n - 1) not in peaks and arr[-1] > arr[-2] and arr[-1] > arr[0]:
                peaks.append(n - 1)
        return np.array(sorted(set(peaks)), dtype=int)

    # High Res Peaks
    prom_high = max(1.0, 0.03 * float(smoothed_H_100.max()))
    smoothed_peaks_indices = find_circular_peaks_simple(smoothed_H_100, prom_high, 2)
    smoothed_peaks_cents = bin_centers_100[smoothed_peaks_indices]
    
    # Low Res Peaks
    prom_low = max(0, 0.01 * float(H_mel_25.max()))
    raw_peaks_indices = find_circular_peaks_simple(H_mel_25, prom_low, 1)
    raw_peaks_cents = bin_centers_25[raw_peaks_indices]
    
    # Validation
    peak_tolerance_cents = 45
    tolerance_cents = 35
    note_centers = np.arange(0, 1200, 100)
    
    pc_cand = set()
    for sp_cent in smoothed_peaks_cents:
        # Check against low res
        if not any(abs(sp_cent - rp_cent) <= peak_tolerance_cents for rp_cent in raw_peaks_cents):
            continue
        
        # Check against note centers
        diffs = np.abs((sp_cent - note_centers + 600) % 1200 - 600)
        nearest_idx = np.argmin(diffs)
        if diffs[nearest_idx] <= tolerance_cents:
            pc_cand.add(nearest_idx % 12)
            
    return len(pc_cand)

def parse_tonic(tonic_str):
    if pd.isna(tonic_str): return None
    tonic_str = str(tonic_str).strip()
    return NOTE_MAPPING.get(tonic_str)

def fuzzy_match_raga(gt_name, raga_masks):
    """Try to find a raga mask using fuzzy matching strategies."""
    # Direct match
    if gt_name in raga_masks:
        return raga_masks[gt_name]
    
    # Case insensitive
    if gt_name.lower() in raga_masks:
        return raga_masks[gt_name.lower()]
    
    # Common spelling variants
    variants = [gt_name]
    
    # i/ee variations: Shri/Shree, Shruti/Shreeti
    if 'i' in gt_name:
        variants.append(gt_name.replace('i', 'ee'))
    if 'ee' in gt_name:
        variants.append(gt_name.replace('ee', 'i'))
    
    # ti/i variations: Jhinjoti/Jhinjhoti
    if gt_name.endswith('ti'):
        variants.append(gt_name[:-2] + 'hoti')
    if gt_name.endswith('hoti'):
        variants.append(gt_name[:-4] + 'ti')
    
    # Check all variants
    for variant in variants:
        if variant in raga_masks:
            return raga_masks[variant]
        if variant.lower() in raga_masks:
            return raga_masks[variant.lower()]
    
    # Try removing common suffixes/prefixes
    # Common patterns: "X Kanada" -> "X", "X Kalyan" -> "X", etc.
    for suffix in [' Kanada', ' Kalyan', ' Sarang', ' Malhar', ' Bahar', ' Kafi']:
        if gt_name.endswith(suffix):
            base = gt_name[:-len(suffix)].strip()
            if base in raga_masks:
                return raga_masks[base]
            if base.lower() in raga_masks:
                return raga_masks[base.lower()]
    
    # Try partial matching - if GT name contains a DB raga name
    gt_lower = gt_name.lower()
    for db_name, mask in raga_masks.items():
        db_lower = db_name.lower()
        # Check if DB name is a significant substring of GT name
        if db_lower in gt_lower and len(db_lower) >= 4:
            return mask
    
    return None

def load_raga_masks(db_path):
    df = pd.read_csv(db_path)
    raga_data = {}
    for _, row in df.iterrows():
        raw_name = row.get('raga') or row.get('names')
        if not isinstance(raw_name, str): continue
        
        cleaned_names = []
        try:
            parsed = json.loads(raw_name)
            if isinstance(parsed, list): cleaned_names = parsed
            else: cleaned_names = [str(parsed)]
        except:
            s = raw_name.strip()
            if s.startswith('[') and s.endswith(']'): s = s[1:-1]
            parts = s.split(',')
            for p in parts:
                p = p.strip().replace('"', '').replace("'", "").replace('"', '').replace('"', '')
                if p: cleaned_names.append(p)

        mask = None
        if 'mask' in row and pd.notna(row['mask']):
            try: mask = [int(x) for x in str(row['mask']).split(',')]
            except: pass
        if mask is None:
            try: mask = [int(row[str(i)]) for i in range(12)]
            except: pass
            
        if mask and sum(mask) > 0:
            mask_arr = np.array(mask)
            for name in cleaned_names:
                # Also handle comma-separated names inside the cleaned string
                for sub_name in name.split(','):
                    raga_data[sub_name.strip()] = mask_arr
                    raga_data[sub_name.strip().lower()] = mask_arr # Add lowercase version too
                    
    return raga_data

def calculate_features(melody_dist, acc_dist, raga_mask, tonic, detected_peak_count):
    """Calculate 8 hand-crafted features."""
    p_rot = np.roll(melody_dist, -tonic)
    acc_rot = np.roll(acc_dist, -tonic)
    tonic_salience = acc_rot[0]
    
    raga_indices = np.where(raga_mask == 1)[0]
    raga_size = len(raga_indices)
    if raga_size == 0: return None

    match_mass = np.sum(p_rot[raga_indices])
    extra_mass = 1.0 - match_mass
    
    peak = np.max(p_rot) + 1e-9
    pres = p_rot[raga_indices] / peak
    presence_score = np.mean(pres)
    
    sum_logp = np.sum(np.log(p_rot[raga_indices] + 1e-9))
    avg_logp = sum_logp / raga_size
    baseline = -np.log(12.0)
    loglike_norm = 1.0 + (avg_logp / (-baseline + 1e-9))
    loglike_norm = max(0.0, min(1.0, loglike_norm))
    
    complexity_pen = max(0.0, (raga_size - 5) / 12.0)
    
    size_diff = abs(raga_size - detected_peak_count)
    size_penalty = (size_diff / 4.0)
    
    prim = p_rot[0]
    bonus_options = [0.0]
    if raga_mask[5] == 1: bonus_options.append(p_rot[5])
    if raga_mask[6] == 1: bonus_options.append(p_rot[6])
    if raga_mask[7] == 1: bonus_options.append(p_rot[7])
    primary_score = prim + max(bonus_options)
    
    return [match_mass, extra_mass, presence_score, loglike_norm, complexity_pen, size_penalty, tonic_salience, primary_score]

def calculate_combined_features(melody_dist, acc_dist, raga_mask, tonic, detected_peak_count):
    """
    Calculate combined feature vector: 
    - 12 melody pitch class features (rotated by tonic)
    - 12 accompaniment pitch class features (rotated by tonic)
    - 8 hand-crafted features
    Total: 32 features
    """
    # Rotate distributions by tonic
    p_rot = np.roll(melody_dist, -tonic)
    acc_rot = np.roll(acc_dist, -tonic)
    
    # Calculate hand-crafted features
    handcrafted = calculate_features(melody_dist, acc_dist, raga_mask, tonic, detected_peak_count)
    if handcrafted is None:
        return None
    
    # Combine: 12 (melody) + 12 (acc) + 8 (handcrafted) = 32 features
    combined = np.concatenate([p_rot, acc_rot, handcrafted])
    return combined

def main():
    print("Loading Data...")
    gt_df = pd.read_csv(DEFAULT_GT_CSV)
    raga_masks = load_raga_masks(DEFAULT_RAGA_DB)
    
    print(f"Loaded {len(raga_masks)} raga keys.")
    
    X = []
    y = []
    ranking_data = []
    processed_count = 0
    
    for _, row in gt_df.iterrows():
        filename = str(row['Filename'])
        gt_raga_name = str(row['Raga']).strip()
        gt_tonic_str = row['Tonic']
        gt_tonic = parse_tonic(gt_tonic_str)
        if gt_tonic is None: continue
        
        # Try to find raga in DB using fuzzy matching
        raga_mask = fuzzy_match_raga(gt_raga_name, raga_masks)
            
        if raga_mask is None:
            if processed_count < 10:
                print(f"Skipping {filename}: Raga '{gt_raga_name}' not found in DB.")
            continue
        
        # Load Raw Pitch Data
        stem_path = os.path.join(DEFAULT_STEMS_DIR, filename)
        vocal_csv = os.path.join(stem_path, 'vocals_pitch_data.csv')
        accomp_csv = os.path.join(stem_path, 'accompaniment_pitch_data.csv')
        
        if not os.path.exists(vocal_csv) or not os.path.exists(accomp_csv): continue
        
        # Process Vocals
        pr_v = load_swiftf0_csv_as_pitchresult(vocal_csv)
        freqs_v, weights_v = get_valid_frequencies(pr_v)
        melody_dist = compute_pitch_distribution(freqs_v, weights_v)
        detected_peak_count = get_detected_peak_count(freqs_v, weights_v)
        
        # Process Accompaniment
        pr_a = load_swiftf0_csv_as_pitchresult(accomp_csv)
        freqs_a, weights_a = get_valid_frequencies(pr_a)
        if freqs_a.size:
            midi_a = librosa.hz_to_midi(freqs_a)
            pc_a = np.mod(np.round(midi_a), 12).astype(int)
            H_acc, _ = np.histogram(pc_a, bins=12, range=(0, 12))
            max_val = H_acc.max()
            acc_dist = (H_acc / max_val) if max_val > 0 else np.zeros(12)
        else:
            acc_dist = np.zeros(12)
            
        # Determine Allowed Tonics
        mean_sal = np.mean(acc_dist)
        allowed_tonics = [t for t in range(12) if acc_dist[t] > mean_sal]
        if not allowed_tonics: allowed_tonics = list(range(12))

        # 1. Correct Pair
        feats = calculate_combined_features(melody_dist, acc_dist, raga_mask, gt_tonic, detected_peak_count)
        if feats is not None:
            X.append(feats)
            y.append(1)
            # Calculate old score using just the handcrafted features
            handcrafted = feats[-8:]
            old_score = sum(OLD_WEIGHTS[k] * v for k, v in zip(OLD_WEIGHTS.keys(), handcrafted))
            is_allowed = (gt_tonic in allowed_tonics)
            ranking_data.append({'song': filename, 'correct': True, 'old_score': old_score, 'feats': feats, 'allowed': is_allowed})

        # 2. Negatives
        unique_masks = {}
        for k, v in raga_masks.items():
            unique_masks[tuple(v)] = k
            
        for mask_tuple, cand_raga_name in unique_masks.items():
            mask = np.array(mask_tuple)
            for cand_tonic in range(12):
                is_correct = (cand_tonic == gt_tonic and np.array_equal(mask, raga_mask))
                if is_correct: continue
                
                feats = calculate_combined_features(melody_dist, acc_dist, mask, cand_tonic, detected_peak_count)
                if feats is None: continue
                
                handcrafted = feats[-8:]
                old_score = sum(OLD_WEIGHTS[k] * v for k, v in zip(OLD_WEIGHTS.keys(), handcrafted))
                is_allowed = (cand_tonic in allowed_tonics)
                ranking_data.append({'song': filename, 'correct': False, 'old_score': old_score, 'feats': feats, 'allowed': is_allowed})
                
                # Add to training data (subsample negatives)
                if np.random.rand() < 0.005: # 0.5% of negatives
                    X.append(feats)
                    y.append(0)

        processed_count += 1
        if processed_count % 10 == 0: print(f"Processed {processed_count} songs...")
        if processed_count >= 50: break

    print(f"Training on {len(X)} samples with 32 features each...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # MLP with 2 hidden layers
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True
    )
    mlp.fit(X_scaled, y)
    
    print("\n" + "="*40)
    print("MLP TRAINING COMPLETE")
    print("="*40)
    print(f"Architecture: 32 -> 64 -> 32 -> 1")
    print(f"Training samples: {len(X)}")
    print(f"Final loss: {mlp.loss_:.4f}")
    
    print("\n" + "="*40)
    print("RANKING IMPROVEMENT")
    print("="*40)
    
    rank_df = pd.DataFrame(ranking_data)
    feats_matrix = np.array(rank_df['feats'].tolist())
    feats_scaled = scaler.transform(feats_matrix)
    rank_df['new_score'] = mlp.predict_proba(feats_scaled)[:, 1]  # Probability of class 1
    
    old_ranks = []
    new_ranks = []
    
    songs = rank_df['song'].unique()
    for s in songs:
        sub = rank_df[rank_df['song'] == s]
        correct_row = sub[sub['correct'] == True]
        if len(correct_row) == 0: continue
        
        # Old Ranking (Filtered by Allowed Tonics)
        sub_old = sub[sub['allowed'] == True].sort_values('old_score', ascending=False).reset_index(drop=True)
        if not any(sub_old['correct']):
             rank_old = 1000 # Penalty: Correct answer was filtered out
        else:
             rank_old = sub_old[sub_old['correct'] == True].index[0] + 1
        old_ranks.append(rank_old)
        
        # New Ranking (Unfiltered - Model should learn to filter)
        sub_new = sub.sort_values('new_score', ascending=False).reset_index(drop=True)
        rank_new = sub_new[sub_new['correct'] == True].index[0] + 1
        new_ranks.append(rank_new)
        
    print(f"Average Rank (Old): {np.mean(old_ranks):.2f}")
    print(f"Average Rank (New): {np.mean(new_ranks):.2f}")
    print(f"Top-1 Accuracy (Old): {sum(r == 1 for r in old_ranks)/len(old_ranks)*100:.2f}%")
    print(f"Top-1 Accuracy (New): {sum(r == 1 for r in new_ranks)/len(new_ranks)*100:.2f}%")

    joblib.dump(mlp, 'raga_mlp_model.pkl')
    joblib.dump(scaler, 'raga_mlp_scaler.pkl')
    print("\nSaved MLP model and scaler.")

if __name__ == '__main__':
    main()
