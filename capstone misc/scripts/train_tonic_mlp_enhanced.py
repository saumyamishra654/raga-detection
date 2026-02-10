import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import librosa
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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

def get_best_raga_features(candidates, tonic):
    """Find the best raga match for this tonic and return its features."""
    tonic_candidates = [c for c in candidates if c['tonic'] == tonic]
    if not tonic_candidates:
        return [0.0] * 6
    best_cand = max(tonic_candidates, key=lambda x: x['features']['match_mass'])
    f = best_cand['features']
    return [
        f.get('match_mass', 0), f.get('extra_mass', 0), f.get('presence', 0),
        f.get('loglike', 0), f.get('complexity', 0), f.get('raga_size', 0)
    ]

def load_data(gt_csv, stems_dir):
    """Load features for Enhanced Tonic MLP."""
    df_gt = pd.read_csv(gt_csv)
    
    X = []
    y = []
    
    print(f"Loading data from {len(df_gt)} ground truth entries...")
    
    missing_gender_count = 0
    
    for _, row in df_gt.iterrows():
        filename = str(row['Filename']).strip()
        gt_tonic_str = row['Tonic']
        gt_tonic = parse_tonic(gt_tonic_str)
        
        if gt_tonic is None: continue
            
        # 1. Load Raga Features
        json_path = os.path.join(stems_dir, filename, 'raga_features', 'raga_features.json')
        if not os.path.exists(json_path): continue
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except: continue

        # 2. Load/Calculate New Features
        # Gender
        gender_raw = str(row.get('Gender', '')).strip().lower()
        if gender_raw in ['m', 'male']:
            gender_val = 0.0
        elif gender_raw in ['f', 'female']:
            gender_val = 1.0
        else:
            gender_val = -1.0 # Missing
            missing_gender_count += 1

        melody_dist = np.array(data.get('melody_distribution', np.zeros(12)))
        acc_salience = np.array(data.get('accompaniment_salience', np.zeros(12)))
        candidates = data.get('candidates', [])
        
        # For each of the 12 tonics, build a feature vector
        for tonic in range(12):
            # Base Features (34)
            rot_melody = np.roll(melody_dist, -tonic)
            rot_acc = np.roll(acc_salience, -tonic)
            tonic_acc = acc_salience[tonic]
            fifth_acc = acc_salience[(tonic + 7) % 12]
            tonic_mel = melody_dist[tonic]
            fifth_mel = melody_dist[(tonic + 7) % 12]
            raga_feats = get_best_raga_features(candidates, tonic)
            
            # New Features (1) - Gender Only
            extra_feats = [gender_val]
            
            # Vector Size: 34 + 1 = 35
            feature_vector = np.concatenate([
                rot_melody, 
                rot_acc, 
                [tonic_acc, fifth_acc, tonic_mel, fifth_mel],
                raga_feats,
                extra_feats
            ])
            
            X.append(feature_vector)
            y.append(1 if tonic == gt_tonic else 0)

    print(f"Missing Gender: {missing_gender_count} songs")
    
    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Tonic MLP')
    parser.add_argument('--gt-csv', required=True)
    parser.add_argument('--stems-dir', required=True)
    args = parser.parse_args()

    # Load Data
    X, y = load_data(args.gt_csv, args.stems_dir)
    print(f"\nLoaded {len(X)} samples (12 per song).")
    print(f"Feature Vector Size: {X.shape[1]}")
    
    # Impute missing values (Gender=-1, Pitch=0)
    # We'll use SimpleImputer to replace -1/0 with mean/median?
    # Or just let MLP handle it?
    # Let's use an Imputer for robustness.
    # Note: Gender is col -3, Vocal is -2, Accomp is -1
    
    # Split into Train/Test (Grouped by Song)
    num_songs = len(X) // 12
    indices = np.arange(num_songs)
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = []
    y_train = []
    for i in train_indices:
        X_train.extend(X[i*12 : (i+1)*12])
        y_train.extend(y[i*12 : (i+1)*12])
        
    X_test = []
    y_test = []
    for i in test_indices:
        X_test.extend(X[i*12 : (i+1)*12])
        y_test.extend(y[i*12 : (i+1)*12])
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training on {len(train_indices)} songs, Testing on {len(test_indices)} songs.")
    
    print("Training Enhanced Tonic MLP...")
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, verbose=True)
    clf.fit(X_train_scaled, y_train)
    
    # Save
    joblib.dump(clf, 'tonic_mlp_enhanced_model.pkl')
    joblib.dump(scaler, 'tonic_mlp_enhanced_scaler.pkl')
    print("\nModel saved to 'tonic_mlp_enhanced_model.pkl'")
    
    # Calculate Accuracy on Test Set
    y_test_probs = clf.predict_proba(X_test_scaled)[:, 1]
    
    correct_count = 0
    top2_correct_count = 0
    fifth_error_count = 0
    total_errors = 0
    
    total_test_songs = len(test_indices)
    
    print("\n" + "="*40)
    print("DETAILED EVALUATION")
    print("="*40)
    
    for i in range(total_test_songs):
        probs_slice = y_test_probs[i*12 : (i+1)*12]
        labels_slice = y_test[i*12 : (i+1)*12]
        
        # Ground Truth Index
        gt_idx = np.argmax(labels_slice)
        
        # Top 1 Prediction
        pred_idx = np.argmax(probs_slice)
        
        # Top 2 Prediction
        top2_indices = np.argsort(probs_slice)[-2:]
        
        # Top 1 Accuracy
        if pred_idx == gt_idx:
            correct_count += 1
        else:
            total_errors += 1
            # Check for 5th error
            # Interval is 7 (Perfect 5th) or 5 (Perfect 4th / Inverted 5th)
            interval = abs(pred_idx - gt_idx) % 12
            if interval == 7 or interval == 5:
                fifth_error_count += 1
        
        # Top 2 Accuracy
        if gt_idx in top2_indices:
            top2_correct_count += 1
            
    print(f"Top-1 Accuracy: {correct_count}/{total_test_songs} ({correct_count/total_test_songs:.2%})")
    print(f"Top-2 Accuracy: {top2_correct_count}/{total_test_songs} ({top2_correct_count/total_test_songs:.2%})")
    
    if total_errors > 0:
        print(f"Fifth Errors:   {fifth_error_count}/{total_errors} ({fifth_error_count/total_errors:.2%} of errors)")
    else:
        print("Fifth Errors:   0 (No errors found)")
    
    # Train Accuracy
    y_train_probs = clf.predict_proba(X_train_scaled)[:, 1]
    train_correct = 0
    for i in range(len(train_indices)):
        probs_slice = y_train_probs[i*12 : (i+1)*12]
        labels_slice = y_train[i*12 : (i+1)*12]
        if labels_slice[np.argmax(probs_slice)] == 1:
            train_correct += 1
    print(f"Train Set Tonic Accuracy: {train_correct}/{len(train_indices)} ({train_correct/len(train_indices):.2%})")

if __name__ == "__main__":
    main()
