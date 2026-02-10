import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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

def get_best_raga_features(candidates, tonic):
    """Find the best raga match for this tonic and return its features."""
    # Filter candidates for this tonic
    tonic_candidates = [c for c in candidates if c['tonic'] == tonic]
    
    if not tonic_candidates:
        # Return zeros if no candidates (shouldn't happen usually)
        return [0.0] * 6
    
    # Pick the 'best' candidate based on match_mass (or loglike)
    # We'll use match_mass as a proxy for "good fit"
    best_cand = max(tonic_candidates, key=lambda x: x['features']['match_mass'])
    f = best_cand['features']
    
    return [
        f.get('match_mass', 0),
        f.get('extra_mass', 0),
        f.get('presence', 0),
        f.get('loglike', 0),
        f.get('complexity', 0),
        f.get('raga_size', 0)
    ]

def load_data(gt_csv, stems_dir):
    """Load features for Tonic MLP."""
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

        melody_dist = np.array(data.get('melody_distribution', np.zeros(12)))
        acc_salience = np.array(data.get('accompaniment_salience', np.zeros(12)))
        candidates = data.get('candidates', [])
        
        # For each of the 12 tonics, build a feature vector
        for tonic in range(12):
            # 1. Rotated Melody (12)
            rot_melody = np.roll(melody_dist, -tonic)
            
            # 2. Rotated Accompaniment (12)
            rot_acc = np.roll(acc_salience, -tonic)
            
            # 3. Specific Scalar Features
            tonic_acc = acc_salience[tonic]
            fifth_acc = acc_salience[(tonic + 7) % 12]
            tonic_mel = melody_dist[tonic]
            fifth_mel = melody_dist[(tonic + 7) % 12]
            
            # 4. Best Raga Features (6)
            raga_feats = get_best_raga_features(candidates, tonic)
            
            # Combine
            # Vector Size: 12 + 12 + 4 + 6 = 34
            feature_vector = np.concatenate([
                rot_melody, 
                rot_acc, 
                [tonic_acc, fifth_acc, tonic_mel, fifth_mel],
                raga_feats
            ])
            
            X.append(feature_vector)
            y.append(1 if tonic == gt_tonic else 0)

    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser(description='Train Full Tonic MLP')
    parser.add_argument('--gt-csv', required=True)
    parser.add_argument('--stems-dir', required=True)
    args = parser.parse_args()

    # Load Data
    X, y = load_data(args.gt_csv, args.stems_dir)
    print(f"\nLoaded {len(X)} samples (12 per song).")
    print(f"Feature Vector Size: {X.shape[1]}")
    
    # Split into Train/Test (Grouped by Song)
    # We have num_songs. We'll split indices.
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training on {len(train_indices)} songs, Testing on {len(test_indices)} songs.")
    
    print("Training Tonic MLP...")
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, verbose=True)
    clf.fit(X_train_scaled, y_train)
    
    # Save
    joblib.dump(clf, 'tonic_mlp_full_model.pkl')
    joblib.dump(scaler, 'tonic_mlp_full_scaler.pkl')
    print("\nModel saved to 'tonic_mlp_full_model.pkl'")
    
    # Calculate Accuracy on Test Set
    y_test_probs = clf.predict_proba(X_test_scaled)[:, 1]
    
    correct_count = 0
    total_test_songs = len(test_indices)
    
    for i in range(total_test_songs):
        probs_slice = y_test_probs[i*12 : (i+1)*12]
        labels_slice = y_test[i*12 : (i+1)*12]
        
        best_idx = np.argmax(probs_slice)
        
        if labels_slice[best_idx] == 1:
            correct_count += 1
            
    print(f"\nTest Set Tonic Accuracy: {correct_count}/{total_test_songs} ({correct_count/total_test_songs:.2%})")
    
    # Calculate Accuracy on Train Set (for sanity check)
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
