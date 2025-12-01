import os
import json
import argparse
import joblib
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

def load_raga_masks(db_path):
    """Load raga masks from the database."""
    if not os.path.exists(db_path):
        print(f"⚠️  Raga DB not found at {db_path}")
        return {}
        
    df = pd.read_csv(db_path)
    masks = {}
    for _, row in df.iterrows():
        try:
            mask = [int(row[str(i)]) for i in range(12)]
        except:
            continue
        
        raw_names = str(row['names'])
        cleaned = raw_names.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
        names = [n.strip().lower() for n in cleaned.split(',')]
        
        for name in names:
            masks[name] = mask
            
    return masks

def get_mlp_score(candidate, clf, scaler, raga_masks, detected_peak_count, melody_distribution):
    """Get the probability score from the Raga MLP."""
    feat = candidate['features']
    raga_size = feat.get('raga_size', 7)
    size_diff = abs(raga_size - detected_peak_count)
    
    base_features = [
        feat['match_mass'],
        feat['extra_mass'],
        feat['presence'],
        feat['loglike'],
        feat['complexity'],
        size_diff
    ]
    
    # Rotated Melody Distribution
    rotated_dist = np.roll(melody_distribution, -candidate['tonic']).tolist()
    
    # Raga Mask
    c_raw = candidate['raga']
    c_cleaned = c_raw.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
    c_names = [n.strip().lower() for n in c_cleaned.split(',')]
    
    mask = [0]*12
    for n in c_names:
        if n in raga_masks:
            mask = raga_masks[n]
            break
            
    features = base_features + rotated_dist + mask
    
    # Scale and Predict
    features_scaled = scaler.transform([features])
    score = clf.predict_proba(features_scaled)[0, 1] # Probability of class 1
    return score

def load_data(gt_csv, stems_dir, clf, scaler, raga_masks):
    """Load features for Tonic regression using MLP scores."""
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
        detected_peak_count = 0
        if gmm_vector:
            for x in gmm_vector:
                if x is not None and isinstance(x, list) and len(x) > 0 and x[0] is not None:
                    detected_peak_count += 1
        if detected_peak_count == 0: detected_peak_count = 7
        
        melody_distribution = np.array(data.get('melody_distribution', np.zeros(12)))
        acc_salience = data.get('accompaniment_salience', [0]*12)
        
        # For each of the 12 tonics, find the BEST melody score from MLP
        for tonic in range(12):
            candidates = [c for c in data['candidates'] if c['tonic'] == tonic]
            
            if not candidates:
                best_mlp_score = 0.0 
            else:
                # We need to score ALL candidates for this tonic and take the max
                scores = []
                for c in candidates:
                    s = get_mlp_score(c, clf, scaler, raga_masks, detected_peak_count, melody_distribution)
                    scores.append(s)
                best_mlp_score = max(scores)
            
            salience = acc_salience[tonic]
            
            # Feature Vector: [MLP Score, Accompaniment Salience]
            X.append([best_mlp_score, salience])
            
            # Label: 1 if this is the correct tonic
            y.append(1 if tonic == gt_tonic else 0)

    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser(description='Train Tonic Weight (MLP Hybrid)')
    parser.add_argument('--gt-csv', required=True)
    parser.add_argument('--stems-dir', required=True)
    parser.add_argument('--raga-db', default='main notebooks/raga_list_final.csv')
    parser.add_argument('--model-path', default='raga_mlp_model.pkl')
    parser.add_argument('--scaler-path', default='raga_mlp_scaler.pkl')
    args = parser.parse_args()

    # Load MLP Model
    print("Loading Raga MLP Model...")
    if not os.path.exists(args.model_path) or not os.path.exists(args.scaler_path):
        print("Error: Model or Scaler not found.")
        return
        
    clf_mlp = joblib.load(args.model_path)
    scaler_mlp = joblib.load(args.scaler_path)
    
    # Load Raga Masks
    raga_masks = load_raga_masks(args.raga_db)
    
    X, y = load_data(args.gt_csv, args.stems_dir, clf_mlp, scaler_mlp, raga_masks)
    
    print(f"\nLoaded {len(X)} samples (12 per song).")
    
    # Train Logistic Regression to combine MLP Score and Salience
    print("Training Tonic Combiner Model...")
    clf_tonic = LogisticRegression(class_weight='balanced')
    clf_tonic.fit(X, y)
    
    coefs = clf_tonic.coef_[0]
    w_mlp = coefs[0]
    w_salience = coefs[1]
    
    print("\n" + "="*40)
    print("LEARNED TONIC COEFFICIENTS (MLP HYBRID)")
    print("="*40)
    print(f"MLP Score Weight:    {w_mlp:.4f}")
    print(f"Salience Weight:     {w_salience:.4f}")
    
    if w_mlp != 0:
        ratio = w_salience / w_mlp
        print(f"\nOptimal Ratio (W_tonic): {ratio:.4f}")
        print(f"Formula: Final_Score = MLP_Prob + ({ratio:.4f} * S_salience)")
    
    # Save this model too?
    joblib.dump(clf_tonic, 'tonic_combiner_model.pkl')
    print("\nTonic combiner model saved to 'tonic_combiner_model.pkl'")

    # Calculate Accuracy
    y_scores = clf_tonic.decision_function(X)
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
