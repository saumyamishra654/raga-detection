#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ============================================================================
# CONFIGURATION
# ============================================================================
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

def parse_tonic(tonic_str):
    """Convert tonic string (e.g., 'C#') to integer (0-11)."""
    if pd.isna(tonic_str): return None
    clean_str = str(tonic_str).strip()
    return NOTE_MAPPING.get(clean_str)

def load_data(gt_csv, stems_dir):
    """Load features and labels for Raga regression."""
    df_gt = pd.read_csv(gt_csv)
    
    X = []
    y = []
    meta = []
    
    print(f"Loading data from {len(df_gt)} ground truth entries...")
    
    for _, row in df_gt.iterrows():
        filename = str(row['Filename']).strip()
        gt_raga = str(row['Raga']).strip()
        gt_tonic_str = row['Tonic']
        gt_tonic = parse_tonic(gt_tonic_str)
        
        if gt_tonic is None:
            print(f"⚠️  Skipping {filename}: Invalid tonic '{gt_tonic_str}'")
            continue
            
        # Construct path to JSON
        # Assuming structure: stems_dir / filename / raga_features.json
        # Or stems_dir / filename_folder / raga_features.json
        # We'll try direct match first
        json_path = os.path.join(stems_dir, filename, 'raga_features.json')
        
        if not os.path.exists(json_path):
            # Try searching if folder name is slightly different? 
            # For now, assume exact match or skip
            # print(f"⚠️  Skipping {filename}: JSON not found at {json_path}")
            continue
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading {json_path}: {e}")
            continue

        # Calculate detected peak count from GMM vector (non-null entries)
        gmm_vector = data.get('gmm_vector', [])
        detected_peak_count = sum(1 for x in gmm_vector if x is not None) if gmm_vector else 7 # Default to 7 if missing
        
        # Filter candidates for the Ground Truth Tonic
        candidates = [c for c in data['candidates'] if c['tonic'] == gt_tonic]
        
        if not candidates:
            print(f"⚠️  No candidates found for tonic {gt_tonic} in {filename}")
            continue
            
        for cand in candidates:
            feat = cand['features']
            
            # Calculate size penalty feature (abs diff from detected peaks)
            raga_size = feat.get('raga_size', 7)
            size_diff = abs(raga_size - detected_peak_count)
            
            # Feature Vector
            # We use raw features. The model will learn the signs.
            # [match_mass, extra_mass, presence, loglike, complexity, size_diff]
            features = [
                feat['match_mass'],
                feat['extra_mass'],
                feat['presence'],
                feat['loglike'],
                feat['complexity'],
                size_diff
            ]
            
            # Label
            # Fuzzy match raga name? Or exact?
            # Let's try simple containment or exact match
            cand_raga = cand['raga'].strip()
            is_match = (cand_raga.lower() == gt_raga.lower())
            
            X.append(features)
            y.append(1 if is_match else 0)
            meta.append((filename, cand_raga, gt_raga))

    return np.array(X), np.array(y), meta

def main():
    parser = argparse.ArgumentParser(description='Train Raga Similarity Weights')
    parser.add_argument('--gt-csv', required=True, help='Path to Ground Truth CSV')
    parser.add_argument('--stems-dir', required=True, help='Path to stems directory containing feature JSONs')
    args = parser.parse_args()

    X, y, meta = load_data(args.gt_csv, args.stems_dir)
    
    print(f"\nLoaded {len(X)} samples.")
    print(f"Positive samples: {sum(y)}")
    print(f"Negative samples: {len(y) - sum(y)}")
    
    if len(X) == 0:
        print("No data found. Check paths and CSV format.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Model
    # We use Logistic Regression. 
    # Note: We don't use an intercept if we want a pure scoring function, 
    # but for classification, intercept helps. We can ignore it for the final formula.
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Extract Weights
    coefs = clf.coef_[0]
    feature_names = ['Match Mass', 'Extra Mass', 'Presence', 'LogLike', 'Complexity', 'Size Diff']
    
    print("\n" + "="*40)
    print("LEARNED RAGA COEFFICIENTS")
    print("="*40)
    print(f"{'Feature':<15} | {'Weight':<10} | {'Expected Sign'}")
    print("-" * 45)
    
    expectations = ['+', '-', '+', '+', '-', '-']
    
    for name, weight, exp in zip(feature_names, coefs, expectations):
        print(f"{name:<15} | {weight:+.4f}    | {exp}")
        
    print("-" * 45)
    print("\nSuggested Formula:")
    print(f"Score = ({coefs[0]:.2f} * match) + ({coefs[1]:.2f} * extra) + ({coefs[2]:.2f} * presence) + ...")
    
    # Save weights for next script
    weights_path = 'raga_weights.json'
    with open(weights_path, 'w') as f:
        json.dump({k: float(v) for k, v in zip(feature_names, coefs)}, f, indent=2)
    print(f"\nWeights saved to {weights_path}")

if __name__ == "__main__":
    main()
