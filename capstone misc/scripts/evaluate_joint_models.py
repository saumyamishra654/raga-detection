import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    if not os.path.exists(db_path): return {}
    df = pd.read_csv(db_path)
    masks = {}
    for _, row in df.iterrows():
        try:
            mask = [int(row[str(i)]) for i in range(12)]
            raw_names = str(row['names'])
            cleaned = raw_names.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
            names = [n.strip().lower() for n in cleaned.split(',')]
            for name in names:
                masks[name] = mask
        except: continue
    return masks

def get_raga_mlp_features(candidate, scaler, raga_masks, detected_peak_count, melody_distribution):
    """Prepare features for Raga MLP (single candidate)."""
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
    
    rotated_dist = np.roll(melody_distribution, -candidate['tonic']).tolist()
    
    c_raw = candidate['raga']
    c_cleaned = c_raw.replace('[', '').replace(']', '').replace('"', '').replace('“', '').replace('”', '').replace("'", "")
    c_names = [n.strip().lower() for n in c_cleaned.split(',')]
    
    mask = [0]*12
    for n in c_names:
        if n in raga_masks:
            mask = raga_masks[n]
            break
            
    features = base_features + rotated_dist + mask
    return scaler.transform([features])

def get_tonic_mlp_features(data, tonic, candidates):
    """Prepare features for Tonic Full MLP (single tonic)."""
    melody_dist = np.array(data.get('melody_distribution', np.zeros(12)))
    acc_salience = np.array(data.get('accompaniment_salience', np.zeros(12)))
    
    rot_melody = np.roll(melody_dist, -tonic)
    rot_acc = np.roll(acc_salience, -tonic)
    
    tonic_acc = acc_salience[tonic]
    fifth_acc = acc_salience[(tonic + 7) % 12]
    tonic_mel = melody_dist[tonic]
    fifth_mel = melody_dist[(tonic + 7) % 12]
    
    # Best Raga Features
    tonic_candidates = [c for c in candidates if c['tonic'] == tonic]
    if not tonic_candidates:
        raga_feats = [0.0] * 6
    else:
        best_cand = max(tonic_candidates, key=lambda x: x['features']['match_mass'])
        f = best_cand['features']
        raga_feats = [
            f.get('match_mass', 0), f.get('extra_mass', 0), f.get('presence', 0),
            f.get('loglike', 0), f.get('complexity', 0), f.get('raga_size', 0)
        ]
        
    feature_vector = np.concatenate([
        rot_melody, rot_acc, 
        [tonic_acc, fifth_acc, tonic_mel, fifth_mel],
        raga_feats
    ])
    return feature_vector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-csv', required=True)
    parser.add_argument('--stems-dir', required=True)
    parser.add_argument('--raga-db', default='main notebooks/raga_list_final.csv')
    args = parser.parse_args()

    # 1. Load Models
    print("Loading models...")
    try:
        # Raga MLP
        raga_clf = joblib.load('raga_mlp_model.pkl')
        raga_scaler = joblib.load('raga_mlp_scaler.pkl')
        raga_classes = raga_clf.classes_
        
        # Tonic Full MLP
        tonic_full_clf = joblib.load('tonic_mlp_full_model.pkl')
        tonic_full_scaler = joblib.load('tonic_mlp_full_scaler.pkl')
        
        # Tonic Combiner (Linear Weights)
        tonic_combiner = joblib.load('tonic_combiner_model.pkl')
        # Extract weights: Score = w_mlp * MLP_Prob + w_sal * Salience
        # The model was trained on [mlp_score, salience]
        weights = tonic_combiner.coef_[0]
        w_mlp = weights[0]
        w_sal = weights[1]
        print(f"Loaded Linear Weights -> MLP: {w_mlp:.4f}, Salience: {w_sal:.4f}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    raga_masks = load_raga_masks(args.raga_db)
    df_gt = pd.read_csv(args.gt_csv)
    
    # Stats
    total = 0
    
    # Approach 1: Full MLP Chain
    correct_tonic_1 = 0
    correct_raga_1 = 0
    correct_joint_1 = 0
    
    # Approach 2: Weighted Linear
    correct_tonic_2 = 0
    correct_raga_2 = 0
    correct_joint_2 = 0

    print(f"\nEvaluating on {len(df_gt)} songs...")
    
    for _, row in df_gt.iterrows():
        filename = str(row['Filename']).strip()
        gt_tonic_str = row['Tonic']
        gt_tonic = parse_tonic(gt_tonic_str)
        gt_raga = str(row['Raga']).strip().lower()
        
        if gt_tonic is None: continue
        
        json_path = os.path.join(args.stems_dir, filename, 'raga_features', 'raga_features.json')
        if not os.path.exists(json_path): continue
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except: continue
        
        total += 1
        
        melody_dist = np.array(data.get('melody_distribution', np.zeros(12)))
        acc_salience = data.get('accompaniment_salience', [0]*12)
        candidates = data.get('candidates', [])
        
        # Peak count for Raga MLP
        gmm_vector = data.get('gmm_vector', [])
        detected_peak_count = 0
        if gmm_vector:
            for x in gmm_vector:
                if x is not None and isinstance(x, list) and len(x) > 0 and x[0] is not None:
                    detected_peak_count += 1
        if detected_peak_count == 0: detected_peak_count = 7

        # ---------------------------------------------------------
        # PRE-CALCULATE TONIC PROBABILITIES (Approach 1)
        # ---------------------------------------------------------
        tonic_vectors = []
        for t in range(12):
            v = get_tonic_mlp_features(data, t, candidates)
            tonic_vectors.append(v)
        
        tonic_vectors_scaled = tonic_full_scaler.transform(tonic_vectors)
        tonic_probs_full = tonic_full_clf.predict_proba(tonic_vectors_scaled)[:, 1] # P(Tonic=1)
        
        # ---------------------------------------------------------
        # JOINT ESTIMATION LOOP
        # ---------------------------------------------------------
        best_score_1 = -1
        pred_1 = (None, None) # (Tonic, Raga)
        
        best_score_2 = -float('inf')
        pred_2 = (None, None)
        
        # We iterate over all candidates (Raga, Tonic pairs) provided by the feature extractor
        # Note: The feature extractor generates candidates for all ragas for all tonics
        
        # Optimization: Group candidates by tonic to avoid re-running Raga MLP unnecessarily?
        # Actually, Raga MLP needs to run for each candidate.
        
        for cand in candidates:
            t = cand['tonic']
            # Clean raga name
            raw_raga = str(cand['raga'])
            r_name = raw_raga.replace('["', '').replace('"]', '').replace("['", "").replace("']", "").replace('"', '').replace("'", "").strip().lower()
            
            # --- Run Raga MLP ---
            # Get P(Raga | Tonic)
            # Note: The Raga MLP is a binary classifier (Is this Raga X? Yes/No).
            # It doesn't output a probability distribution over ALL ragas.
            # It outputs "Probability that this candidate is the correct raga".
            # So we can use this directly as P(R|T).
            
            X_raga = get_raga_mlp_features(cand, raga_scaler, raga_masks, detected_peak_count, melody_dist)
            raga_prob = raga_clf.predict_proba(X_raga)[0, 1]
            
            # --- Approach 1: Full MLP Chain ---
            # Score = P(Tonic)_FullMLP * P(Raga|Tonic)_RagaMLP
            p_tonic = tonic_probs_full[t]
            joint_prob_1 = p_tonic * raga_prob
            
            if joint_prob_1 > best_score_1:
                best_score_1 = joint_prob_1
                pred_1 = (t, r_name)
                
            # --- Approach 2: Weighted Linear ---
            # Score = w_mlp * P(Raga|Tonic) + w_sal * Salience(T)
            salience = acc_salience[t]
            score_2 = (w_mlp * raga_prob) + (w_sal * salience)
            
            if score_2 > best_score_2:
                best_score_2 = score_2
                pred_2 = (t, r_name)

        # ---------------------------------------------------------
        # EVALUATION
        # ---------------------------------------------------------
        
        # Check Approach 1
        p1_tonic, p1_raga = pred_1
        if p1_tonic == gt_tonic:
            correct_tonic_1 += 1
            if p1_raga == gt_raga:
                correct_raga_1 += 1
                correct_joint_1 += 1
        
        # Check Approach 2
        p2_tonic, p2_raga = pred_2
        if p2_tonic == gt_tonic:
            correct_tonic_2 += 1
            if p2_raga == gt_raga:
                correct_joint_2 += 1

    print("\n" + "="*50)
    print("JOINT ESTIMATION RESULTS")
    print("="*50)
    print(f"Total Songs Evaluated: {total}")
    
    print("\n--- Approach 1: Full MLP Chain (P(T)*P(R|T)) ---")
    print(f"Tonic Accuracy: {correct_tonic_1}/{total} ({correct_tonic_1/total:.2%})")
    print(f"Joint Accuracy: {correct_joint_1}/{total} ({correct_joint_1/total:.2%})")
    
    print("\n--- Approach 2: Weighted Linear (w1*P(R|T) + w2*Sal) ---")
    print(f"Tonic Accuracy: {correct_tonic_2}/{total} ({correct_tonic_2/total:.2%})")
    print(f"Joint Accuracy: {correct_joint_2}/{total} ({correct_joint_2/total:.2%})")

if __name__ == "__main__":
    main()
