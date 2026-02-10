#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
    if pd.isna(tonic_str): return None
    clean_str = str(tonic_str).strip()
    return NOTE_MAPPING.get(clean_str)

def calculate_melody_score(candidate, weights, detected_peak_count):
    """Calculate S_melody using learned weights."""
    feat = candidate['features']
    raga_size = feat.get('raga_size', 7)
    size_diff = abs(raga_size - detected_peak_count)
    
    score = (
        weights['Match Mass'] * feat['match_mass'] +
        weights['Extra Mass'] * feat['extra_mass'] +
        weights['Presence'] * feat['presence'] +
        weights['LogLike'] * feat['loglike'] +
        weights['Complexity'] * feat['complexity'] +
        weights['Size Diff'] * size_diff
    )
    return score

def load_data(gt_csv, stems_dir, raga_weights):
    """Load features for Tonic regression."""
    df_gt = pd.read_csv(gt_csv)
    
    X = []
    y = []
    
    print(f"Loading data from {len(df_gt)} ground truth entries...")
    
    for _, row in df_gt.iterrows():
        filename = str(row['Filename']).strip()
        gt_tonic_str = row['Tonic']
        gt_tonic = parse_tonic(gt_tonic_str)
        
        if gt_tonic is None: continue
            
        json_path = os.path.join(stems_dir, filename, 'raga_features', 'raga_features.json')
        if not os.path.exists(json_path): continue
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except: continue

        gmm_vector = data.get('gmm_vector', [])
        detected_peak_count = sum(1 for x in gmm_vector if x is not None) if gmm_vector else 7
        acc_salience = data.get('accompaniment_salience', [0]*12)
        
        # For each of the 12 tonics, find the BEST melody score
        for tonic in range(12):
            candidates = [c for c in data['candidates'] if c['tonic'] == tonic]
            if not candidates:
                best_melody_score = -10.0 # Penalty for no candidates
            else:
                scores = [calculate_melody_score(c, raga_weights, detected_peak_count) for c in candidates]
                best_melody_score = max(scores)
            
            salience = acc_salience[tonic]
            
            # Feature Vector: [Melody Score, Accompaniment Salience]
            X.append([best_melody_score, salience])
            
            # Label: 1 if this is the correct tonic
            y.append(1 if tonic == gt_tonic else 0)

    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser(description='Train Tonic Weight')
    parser.add_argument('--gt-csv', required=True)
    parser.add_argument('--stems-dir', required=True)
    parser.add_argument('--weights-file', default='raga_weights.json', help='Path to learned raga weights')
    args = parser.parse_args()

    # Load Raga Weights
    if not os.path.exists(args.weights_file):
        print(f"Error: Weights file {args.weights_file} not found. Run train_raga_weights.py first.")
        return
        
    with open(args.weights_file, 'r') as f:
        raga_weights = json.load(f)
        
    X, y = load_data(args.gt_csv, args.stems_dir, raga_weights)
    
    print(f"\nLoaded {len(X)} samples (12 per song).")
    
    # Train Model
    # We want to learn the balance: Score = 1.0 * Melody + W * Salience
    # Logistic Regression learns: z = w1 * Melody + w2 * Salience + b
    # The ratio w2/w1 gives us the relative importance W.
    
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, y)
    
    coefs = clf.coef_[0]
    w_melody = coefs[0]
    w_salience = coefs[1]
    
    print("\n" + "="*40)
    print("LEARNED TONIC COEFFICIENTS")
    print("="*40)
    print(f"Melody Score Weight: {w_melody:.4f}")
    print(f"Salience Weight:     {w_salience:.4f}")
    
    if w_melody != 0:
        ratio = w_salience / w_melody
        print(f"\nOptimal Ratio (W_tonic): {ratio:.4f}")
        print(f"Formula: Final_Score = S_melody + ({ratio:.4f} * S_salience)")
    else:
        print("\nWarning: Melody weight is zero. Model failed to use melody.")

    # Calculate Accuracy
    y_scores = clf.decision_function(X)
    num_songs = len(X) // 12
    correct_count = 0
    
    for i in range(num_songs):
        scores_slice = y_scores[i*12 : (i+1)*12]
        labels_slice = y[i*12 : (i+1)*12]
        
        best_idx = np.argmax(scores_slice)
        
        if labels_slice[best_idx] == 1:
            correct_count += 1
            
    print(f"\nTonic Detection Accuracy: {correct_count}/{num_songs} ({correct_count/num_songs:.2%})")

if __name__ == "__main__":
    main()
