import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ============================================================================
# CONFIGURATION
# ============================================================================
# Default Paths (Edit these)
DEFAULT_PROJECT_ROOT = '/Volumes/Extreme SSD'
DEFAULT_GT_CSV = '/Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-detection/main notebooks/ground_truth.csv'
DEFAULT_STEMS_DIR = '/Volumes/Extreme SSD/stems/separated_stems/htdemucs'
DEFAULT_RAGA_DB = '/Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-detection/main notebooks/raga_list_final.csv'

NOTE_MAPPING = {
    'C': 0, 'C ': 0,
    'C#': 1, 'Db': 1, 'C #': 1,
    'D': 2, 'D ': 2,
    'D#': 3, 'Eb': 3, 'D #': 3,
    'E': 4, 'E ': 4,
    'F': 5, 'F ': 5,
    'F#': 6, 'Gb': 6, 'F #': 6,
    'G': 7, 'G ': 7,
    'G#': 8, 'Ab': 8, 'G #': 8,
    'A': 9, 'A ': 9,
    'A#': 10, 'Bb': 10, 'A #': 10,
    'B': 11, 'B ': 11
}

REVERSE_NOTE_MAPPING = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 
    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
}

# Aliases to map Ground Truth names to Database names
RAGA_ALIASES = {
    'darbari kanada': 'darbari',
    'jhinjoti': 'jhinjhoti',
    'hanskinki': 'hanskinkini',
    'hari kauns': 'harikauns',
    'khambhavati': 'khambavati',
    'madhumand sarang': 'madhumad sarang',
    'shri': 'shree',
    'shuddh sarang': 'shuddha sarang',
    'yaman kalyan': 'yaman'
}

def load_raga_masks(db_path):
    """Load raga masks from the database."""
    if not os.path.exists(db_path):
        print(f"⚠️  Raga DB not found at {db_path}")
        return {}
        
    df = pd.read_csv(db_path)
    masks = {}
    for _, row in df.iterrows():
        # Parse mask
        try:
            mask = [int(row[str(i)]) for i in range(12)]
        except:
            continue
            
        # Parse names
        raw_names = str(row['names'])
        cleaned = raw_names.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
        names = [n.strip().lower() for n in cleaned.split(',')]
        
        for name in names:
            masks[name] = mask
            
    return masks

def parse_tonic(tonic_str):
    """Convert tonic string (e.g., 'C#') to integer (0-11)."""
    if pd.isna(tonic_str): return None
    clean_str = str(tonic_str).strip()
    return NOTE_MAPPING.get(clean_str)

def load_data(gt_csv, stems_dir, raga_masks):
    """Load features and labels for Raga regression."""
    df_gt = pd.read_csv(gt_csv)
    
    # Filter for Vocal instruments only
    if 'Instrument' in df_gt.columns:
        original_count = len(df_gt)
        df_gt = df_gt[df_gt['Instrument'] == 'Vocal']
        print(f"Filtered for Vocal: {len(df_gt)}/{original_count} entries remaining.")
        
    # Check for duplicates
    unique_files = df_gt['Filename'].nunique()
    if unique_files != len(df_gt):
        print(f"⚠️  Warning: CSV contains {len(df_gt) - unique_files} duplicate filenames.")
        duplicates = df_gt[df_gt.duplicated(subset=['Filename'])]
        print("Duplicate files:")
        for f in duplicates['Filename'].unique():
            print(f"  - {f}")
    
    X = []
    y = []
    meta = []
    
    skipped_counts = {
        'invalid_tonic': 0,
        'json_not_found': 0,
        'json_error': 0,
        'no_candidates': 0
    }
    
    missing_json_files = []
    
    print(f"Loading data from {len(df_gt)} ground truth entries...")
    
    for _, row in df_gt.iterrows():
        filename = str(row['Filename']).strip()
        gt_raga = str(row['Raga']).strip()
        gt_tonic_str = row['Tonic']
        gt_tonic = parse_tonic(gt_tonic_str)
        
        if gt_tonic is None:
            print(f"⚠️  Skipping {filename}: Invalid tonic '{gt_tonic_str}'")
            skipped_counts['invalid_tonic'] += 1
            continue
            
        # Construct path to JSON
        # Structure: stems_dir / filename / raga_features / raga_features.json
        json_path = os.path.join(stems_dir, filename, 'raga_features', 'raga_features.json')
        
        if not os.path.exists(json_path):
            # Try searching if folder name is slightly different? 
            # For now, assume exact match or skip
            # print(f"⚠️  Skipping {filename}: JSON not found at {json_path}")
            skipped_counts['json_not_found'] += 1
            missing_json_files.append(filename)
            continue
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading {json_path}: {e}")
            skipped_counts['json_error'] += 1
            continue

        # Calculate detected peak count from GMM vector (non-null entries)
        # gmm_vector is a list of 12 lists: [height, deviation, sigma]
        # Empty bins are [None, None, None] or None. We count how many bins have a valid peak.
        gmm_vector = data.get('gmm_vector', [])
        detected_peak_count = 0
        if gmm_vector:
            for x in gmm_vector:
                # Check if the bin has data (height is not None)
                if x is not None and isinstance(x, list) and len(x) > 0 and x[0] is not None:
                    detected_peak_count += 1
        
        if detected_peak_count == 0:
             detected_peak_count = 7 # Default if missing or empty
        
        # Get melody distribution
        melody_distribution = np.array(data.get('melody_distribution', np.zeros(12)))

        # Filter candidates for the Ground Truth Tonic
        candidates = [c for c in data['candidates'] if c['tonic'] == gt_tonic]
        
        if not candidates:
            print(f"⚠️  No candidates found for tonic {gt_tonic} in {filename}")
            skipped_counts['no_candidates'] += 1
            continue
            
        for cand in candidates:
            feat = cand['features']
            
            # Calculate size penalty feature (abs diff from detected peaks)
            raga_size = feat.get('raga_size', 7)
            size_diff = abs(raga_size - detected_peak_count)
            
            # Feature Vector
            # We use raw features. The model will learn the signs.
            # [match_mass, extra_mass, presence, loglike, complexity, size_diff]
            base_features = [
                feat['match_mass'],
                feat['extra_mass'],
                feat['presence'],
                feat['loglike'],
                feat['complexity'],
                size_diff
            ]
            
            # Add Rotated Melody Distribution
            rotated_dist = np.roll(melody_distribution, -cand['tonic']).tolist()
            
            # Add Raga Mask
            c_raw = cand['raga']
            c_cleaned = c_raw.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
            c_names = [n.strip().lower() for n in c_cleaned.split(',')]
            
            mask = [0]*12
            for n in c_names:
                if n in raga_masks:
                    mask = raga_masks[n]
                    break
            
            features = base_features + rotated_dist + mask
            
            # Label
            # The 'raga' field in JSON is a string representation of a list, e.g., '[“Jayat"]'
            cand_raga_raw = cand['raga']
            
            # Clean up the string: remove brackets, quotes (both types)
            cleaned = cand_raga_raw.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
            
            # Split by comma and normalize
            candidate_names = [name.strip().lower() for name in cleaned.split(',')]
            
            # Check for match using direct name or alias
            gt_lower = gt_raga.lower()
            alias_name = RAGA_ALIASES.get(gt_lower, gt_lower)
            
            is_match = (gt_lower in candidate_names) or (alias_name in candidate_names)
            
            X.append(features)
            y.append(1 if is_match else 0)
            meta.append((filename, cand_raga_raw, gt_raga, gt_tonic))

    print("\nData Loading Summary:")
    print(f"Total entries in CSV (Vocal): {len(df_gt)}")
    print(f"Successfully loaded files: {len(set(m[0] for m in meta))}")
    print(f"Skipped - Invalid Tonic: {skipped_counts['invalid_tonic']}")
    print(f"Skipped - JSON Not Found: {skipped_counts['json_not_found']}")
    if missing_json_files:
        print(f"All missing JSONs ({len(missing_json_files)}):")
        for f in missing_json_files:
            print(f"  - {f}")
    print(f"Skipped - JSON Error: {skipped_counts['json_error']}")
    print(f"Skipped - No Candidates for GT Tonic: {skipped_counts['no_candidates']}")

    return np.array(X), np.array(y), meta

def perform_detailed_analysis(clf, scaler, meta, X, stems_dir, raga_masks):
    """
    Generates detailed error analysis files and performs tonic rotation checks.
    """
    print("\n" + "="*40)
    print("PERFORMING DETAILED ANALYSIS")
    print("="*40)
    
    analysis_dir = 'analysis_results_mlp'
    raga_errors_dir = os.path.join(analysis_dir, 'raga_errors')
    tonic_check_dir = os.path.join(analysis_dir, 'tonic_validation')
    
    os.makedirs(raga_errors_dir, exist_ok=True)
    os.makedirs(tonic_check_dir, exist_ok=True)
    
    # 1. Group data by filename
    from collections import defaultdict
    file_groups = defaultdict(list)
    
    # Re-organize X and meta into a structure we can query by filename
    # meta is list of (filename, cand_raga_raw, gt_raga, gt_tonic)
    # X is list of feature vectors
    
    for i, (filename, cand_raga_raw, gt_raga, gt_tonic) in enumerate(meta):
        file_groups[filename].append({
            'cand_raga_raw': cand_raga_raw,
            'gt_raga': gt_raga,
            'gt_tonic': gt_tonic,
            'features': X[i]
        })

    # We also need to load the FULL JSON for tonic rotation check
    # This is expensive, so we do it per file
    
    # Store confusion data: gt_raga -> list of (predicted_raga, count)
    confusion_data = defaultdict(list)
    
    # Store tonic stats
    tonic_better_count = 0
    total_files_checked = 0
    
    print(f"Generating reports in '{analysis_dir}'...")
    
    for filename, candidates in file_groups.items():
        if not candidates: continue
        
        gt_raga = candidates[0]['gt_raga']
        gt_tonic = candidates[0]['gt_tonic']
        gt_lower = gt_raga.lower()
        alias_name = RAGA_ALIASES.get(gt_lower, gt_lower)
        
        # --- A. Prediction Analysis ---
        cand_features = [c['features'] for c in candidates]
        # Scale features
        cand_features_scaled = scaler.transform(cand_features)
        
        # Use predict_proba for MLP scores (probability of class 1)
        scores = clf.predict_proba(cand_features_scaled)[:, 1]
        
        scored_candidates = []
        for j, score in enumerate(scores):
            scored_candidates.append((score, candidates[j]['cand_raga_raw']))
            
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Identify Top-1 Prediction
        top1_score, top1_raw = scored_candidates[0]
        # Clean name
        cleaned = top1_raw.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
        top1_names = [name.strip().lower() for name in cleaned.split(',')]
        top1_name_display = top1_names[0] # Just take the first one for display
        
        # Check if correct
        is_correct = (gt_lower in top1_names) or (alias_name in top1_names)
        
        # Find rank of correct raga
        correct_rank = -1
        correct_score = -999
        for rank, (score, raw) in enumerate(scored_candidates):
            c_cleaned = raw.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
            c_names = [name.strip().lower() for name in c_cleaned.split(',')]
            if (gt_lower in c_names) or (alias_name in c_names):
                correct_rank = rank + 1 # 1-based
                correct_score = score
                break
        
        # Log for Raga Error File
        log_entry = f"File: {filename}\n"
        log_entry += f"GT Raga: {gt_raga}\n"
        log_entry += f"Top-1 Pred: {top1_name_display} (Score: {top1_score:.2f})\n"
        log_entry += f"Correct Raga Rank: {correct_rank if correct_rank != -1 else 'Not Found'}\n"
        if correct_rank != -1:
             log_entry += f"Correct Raga Score: {correct_score:.2f}\n"
        
        if not is_correct:
            log_entry += f"❌ MISCLASSIFIED (Confused with {top1_name_display})\n"
            confusion_data[gt_raga].append(top1_name_display)
        else:
            log_entry += f"✅ CORRECT\n"
            
        log_entry += "-" * 30 + "\n"
        
        # Write to raga file
        raga_file_path = os.path.join(raga_errors_dir, f"{gt_raga}.txt")
        with open(raga_file_path, 'a') as f:
            f.write(log_entry)

        # --- B. Tonic Rotation Check ---
        # We need to load the JSON again to get ALL candidates (all tonics)
        json_path = os.path.join(stems_dir, filename, 'raga_features', 'raga_features.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    full_data = json.load(f)
                
                # Find the GT Raga candidates across ALL tonics
                # We look for candidates where 'raga' matches GT Raga
                gt_candidates_all_tonics = []
                
                for cand in full_data['candidates']:
                    c_raw = cand['raga']
                    c_cleaned = c_raw.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
                    c_names = [name.strip().lower() for name in c_cleaned.split(',')]
                    
                    if (gt_lower in c_names) or (alias_name in c_names):
                        gt_candidates_all_tonics.append(cand)
                
                if gt_candidates_all_tonics:
                    # Score all of them
                    feats = []
                    # We need detected_peak_count again for size_diff
                    gmm_vector = full_data.get('gmm_vector', [])
                    detected_peak_count = 0
                    if gmm_vector:
                        for x in gmm_vector:
                            if x is not None and isinstance(x, list) and len(x) > 0 and x[0] is not None:
                                detected_peak_count += 1
                    
                    if detected_peak_count == 0:
                        detected_peak_count = 7

                    # Get melody distribution
                    melody_distribution = np.array(full_data.get('melody_distribution', np.zeros(12)))

                    for cand in gt_candidates_all_tonics:
                        f = cand['features']
                        raga_size = f.get('raga_size', 7)
                        size_diff = abs(raga_size - detected_peak_count)
                        
                        base_features = [
                            f['match_mass'], f['extra_mass'], f['presence'], 
                            f['loglike'], f['complexity'], size_diff
                        ]
                        
                        # Add Rotated Melody Distribution
                        rotated_dist = np.roll(melody_distribution, -cand['tonic']).tolist()
                        
                        # Add Raga Mask
                        c_raw = cand['raga']
                        c_cleaned = c_raw.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
                        c_names = [n.strip().lower() for n in c_cleaned.split(',')]
                        
                        mask = [0]*12
                        for n in c_names:
                            if n in raga_masks:
                                mask = raga_masks[n]
                                break
                        
                        feats.append(base_features + rotated_dist + mask)
                    
                    # Scale features
                    feats_scaled = scaler.transform(feats)
                    tonic_scores = clf.predict_proba(feats_scaled)[:, 1]
                    
                    # Find max score
                    best_idx = np.argmax(tonic_scores)
                    best_score = tonic_scores[best_idx]
                    best_tonic = gt_candidates_all_tonics[best_idx]['tonic']
                    
                    # Get the score for the GT Tonic (which we used in main loop)
                    # We have to find which one corresponds to the GT Tonic
                    # Note: In main loop we filtered by GT Tonic.
                    
                    if correct_rank != -1:
                        gt_tonic_score = correct_score
                        
                        if best_score > gt_tonic_score + 0.01: # Small epsilon
                            tonic_better_count += 1
                            
                            # Calculate rank for best_tonic
                            best_tonic_candidates = [c for c in full_data['candidates'] if c['tonic'] == best_tonic]
                            
                            # Score all candidates for this new tonic to find rank
                            bt_feats = []
                            bt_raga_names = []
                            
                            for c in best_tonic_candidates:
                                f = c['features']
                                raga_size = f.get('raga_size', 7)
                                size_diff = abs(raga_size - detected_peak_count)
                                
                                base_features = [
                                    f['match_mass'], f['extra_mass'], f['presence'], 
                                    f['loglike'], f['complexity'], size_diff
                                ]
                                
                                # Add Rotated Melody Distribution
                                rotated_dist = np.roll(melody_distribution, -c['tonic']).tolist()
                                
                                # Add Raga Mask
                                c_raw = c['raga']
                                c_cleaned = c_raw.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
                                c_names = [n.strip().lower() for n in c_cleaned.split(',')]
                                
                                mask = [0]*12
                                for n in c_names:
                                    if n in raga_masks:
                                        mask = raga_masks[n]
                                        break
                                
                                bt_feats.append(base_features + rotated_dist + mask)
                                bt_raga_names.append(c['raga'])
                                
                            bt_feats_scaled = scaler.transform(bt_feats)
                            bt_scores = clf.predict_proba(bt_feats_scaled)[:, 1]
                            
                            # Sort and find rank
                            bt_scored = sorted(zip(bt_scores, bt_raga_names), key=lambda x: x[0], reverse=True)
                            
                            best_tonic_rank = -1
                            for r, (s, name) in enumerate(bt_scored):
                                 # Clean name
                                 c_cleaned = name.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
                                 c_names = [n.strip().lower() for n in c_cleaned.split(',')]
                                 if (gt_lower in c_names) or (alias_name in c_names):
                                     best_tonic_rank = r + 1
                                     break
                            
                            # Log this
                            gt_tonic_name = REVERSE_NOTE_MAPPING.get(gt_tonic, str(gt_tonic))
                            best_tonic_name = REVERSE_NOTE_MAPPING.get(best_tonic, str(best_tonic))
                            
                            # Calculate gap
                            gap = (best_tonic - gt_tonic) % 12
                            
                            t_log = f"File: {filename}\n"
                            t_log += f"GT Raga: {gt_raga}\n"
                            t_log += f"Original Tonic: {gt_tonic_name} (Rank: {correct_rank}, Score: {gt_tonic_score:.2f})\n"
                            t_log += f"Better Tonic: {best_tonic_name} (Rank: {best_tonic_rank}, Score: {best_score:.2f})\n"
                            t_log += f"Tonic Shift: +{gap} semitones\n"
                            t_log += f"Score Improvement: {best_score - gt_tonic_score:.2f}\n"
                            t_log += "-" * 30 + "\n"
                            
                            with open(os.path.join(tonic_check_dir, 'tonic_issues.txt'), 'a') as f:
                                f.write(t_log)
                                    
                    total_files_checked += 1
                    
            except Exception as e:
                print(f"Error in detailed analysis for {filename}: {e}")

    print(f"Detailed analysis complete.")
    print(f"Raga error reports saved to: {raga_errors_dir}")
    print(f"Tonic validation saved to: {tonic_check_dir}")
    if total_files_checked > 0:
        print(f"Tonic Mismatch Detected: {tonic_better_count}/{total_files_checked} files ({tonic_better_count/total_files_checked:.1%}) could be improved by rotating tonic.")

def main():
    parser = argparse.ArgumentParser(description='Train Raga Similarity Weights (MLP)')
    parser.add_argument('--gt-csv', default=DEFAULT_GT_CSV, help='Path to Ground Truth CSV')
    parser.add_argument('--stems-dir', default=DEFAULT_STEMS_DIR, help='Path to stems directory containing feature JSONs')
    parser.add_argument('--raga-db', default='main notebooks/raga_list_final.csv', help='Path to Raga DB CSV for masks')
    args = parser.parse_args()

    print(f"GT CSV: {args.gt_csv}")
    print(f"Stems Dir: {args.stems_dir}")
    print(f"Raga DB: {args.raga_db}")

    # Load Raga Masks
    print("Loading Raga Masks...")
    raga_masks = load_raga_masks(args.raga_db)
    print(f"Loaded {len(raga_masks)} raga masks.")

    X, y, meta = load_data(args.gt_csv, args.stems_dir, raga_masks)
    
    print(f"\nLoaded {len(X)} samples.")
    print(f"Feature vector size: {X.shape[1]}")
    print(f"Positive samples: {sum(y)}")
    print(f"Negative samples: {len(y) - sum(y)}")
    
    if len(X) == 0:
        print("No data found. Check paths and CSV format.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale Features (Crucial for MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Model (MLP)
    print("\nTraining MLP Classifier...")
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', 
                        alpha=0.0001, batch_size='auto', learning_rate='constant', 
                        learning_rate_init=0.001, max_iter=500, random_state=42,
                        early_stopping=True)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred))

    # ============================================================================
    # RETRIEVAL EVALUATION (Top-N Accuracy)
    # ============================================================================
    print("\n" + "="*40)
    print("RETRIEVAL STATISTICS (Top-N Accuracy)")
    print("="*40)
    
    from collections import defaultdict
    
    # Group samples by filename to rank candidates for each file
    file_groups = defaultdict(list)
    
    for i, (filename, cand_raga_raw, gt_raga, _) in enumerate(meta):
        file_groups[filename].append({
            'cand_raga_raw': cand_raga_raw,
            'gt_raga': gt_raga,
            'features': X[i]
        })
        
    raga_stats = defaultdict(lambda: {'count': 0, 'top1': 0, 'top5': 0, 'found_any': 0})
    total_files = 0
    total_top1 = 0
    total_top5 = 0
    
    for filename, candidates in file_groups.items():
        if not candidates: continue
        
        gt_raga = candidates[0]['gt_raga']
        raga_stats[gt_raga]['count'] += 1
        total_files += 1
        
        # Get scores for all candidates using the trained model
        cand_features = [c['features'] for c in candidates]
        cand_features_scaled = scaler.transform(cand_features)
        
        # Use predict_proba for MLP scores
        scores = clf.predict_proba(cand_features_scaled)[:, 1]
        
        # Pair scores with candidate info
        scored_candidates = []
        for j, score in enumerate(scores):
            scored_candidates.append((score, candidates[j]['cand_raga_raw']))
            
        # Sort by score descending (highest score first)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Find rank of the correct raga
        found_rank = -1
        gt_lower = gt_raga.lower()
        alias_name = RAGA_ALIASES.get(gt_lower, gt_lower)
        
        for rank, (score, cand_raga_raw) in enumerate(scored_candidates):
            # Clean up string to match GT
            cleaned = cand_raga_raw.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
            candidate_names = [name.strip().lower() for name in cleaned.split(',')]
            
            if (gt_lower in candidate_names) or (alias_name in candidate_names):
                found_rank = rank
                break
        
        if found_rank != -1:
            raga_stats[gt_raga]['found_any'] += 1
            if found_rank == 0:
                raga_stats[gt_raga]['top1'] += 1
                total_top1 += 1
            if found_rank < 5:
                raga_stats[gt_raga]['top5'] += 1
                total_top5 += 1
    
    # Print Global Stats
    print(f"Total Files Evaluated: {total_files}")
    print(f"Global Top-1 Accuracy: {total_top1/total_files:.2%}")
    print(f"Global Top-5 Accuracy: {total_top5/total_files:.2%}")
    
    # Print Per-Raga Stats
    print("\n" + "-" * 75)
    print(f"{'Raga':<25} | {'Count':<5} | {'Top-1':<8} | {'Top-5':<8} | {'Found'}")
    print("-" * 75)
    
    sorted_ragas = sorted(raga_stats.keys())
    for raga in sorted_ragas:
        s = raga_stats[raga]
        count = s['count']
        if count == 0: continue
        t1 = s['top1'] / count
        t5 = s['top5'] / count
        found = s['found_any']
        print(f"{raga:<25} | {count:<5} | {t1:.1%}    | {t5:.1%}    | {found}/{count}")
    print("-" * 75)

    # ============================================================================
    # DETAILED ANALYSIS (Confusion & Tonic Rotation)
    # ============================================================================
    perform_detailed_analysis(clf, scaler, meta, X, args.stems_dir, raga_masks)

    # Save Model
    print("\nSaving model...")
    joblib.dump(clf, 'raga_mlp_model.pkl')
    joblib.dump(scaler, 'raga_mlp_scaler.pkl')
    print("Model saved to raga_mlp_model.pkl")
    print("Scaler saved to raga_mlp_scaler.pkl")

if __name__ == "__main__":
    main()
