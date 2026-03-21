#!/usr/bin/env python3
"""
Optimize raga scoring coefficients against annotated ground truth.

Workflow:
    1. Run batch detect with --skip-report on all annotated songs
    2. Collect feature vectors from candidates.csv files:
         python tools/optimize_scoring.py collect \
             --results-dir batch_results/ \
             --gt "main notebooks/ground truth/ground_truth_v6.csv" \
             --output scoring_dataset.csv
    3. Optimize linear coefficients:
         python tools/optimize_scoring.py optimize \
             --data scoring_dataset.csv \
             --output optimized_params.json
    4. Evaluate (current defaults or custom params):
         python tools/optimize_scoring.py evaluate \
             --data scoring_dataset.csv \
             [--params optimized_params.json]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Features extracted from candidates.csv that feed into the linear score.
# Order matters: weights[i] corresponds to FEATURE_COLS[i].
FEATURE_COLS = [
    "match_mass",
    "observed_note_score",
    "loglike_norm",
    "extra_mass",
    "complexity_pen",
    "match_diff_norm",   # match_diff / 4.0 (derived)
    "primary_score",
    "salience_norm",     # salience / max_salience_per_song (derived)
]

# Extended features for MLP (includes histogram distributions + valley penalty)
MLP_FEATURE_COLS = FEATURE_COLS + [
    f"melody_dist_{i}" for i in range(12)
] + [
    f"accomp_dist_{i}" for i in range(12)
] + [
    "valley_penalty",
]

# Mapping from FEATURE_COLS index to ScoringParams field name.
# Negative sign means the current formula subtracts this term.
PARAM_MAP = [
    ("ALPHA_MATCH",           +1),
    ("BETA_PRESENCE",         +1),
    ("GAMMA_LOGLIKE",         +1),
    ("DELTA_EXTRA",           -1),   # subtracted in formula
    ("COMPLEX_PENALTY",       -1),   # subtracted
    ("MATCH_SIZE_GAMMA",      -1),   # subtracted (via size_penalty)
    ("(primary_score)",       +1),   # not in current formula; new
    ("TONIC_SALIENCE_WEIGHT", +1),
]

# Current hand-tuned defaults (from raga.py ScoringParams)
CURRENT_WEIGHTS = np.array([
    +0.40,   # ALPHA_MATCH * match_mass
    +0.25,   # BETA_PRESENCE * observed_note_score
    +1.00,   # GAMMA_LOGLIKE * loglike_norm
    -1.10,   # -DELTA_EXTRA * extra_mass
    -0.10,   # -COMPLEX_PENALTY * complexity_pen
    -0.25,   # -MATCH_SIZE_GAMMA * (match_diff / 4.0)
    +0.00,   # primary_score (currently tiebreaker only, not in score)
    +0.12,   # TONIC_SALIENCE_WEIGHT * salience_norm
], dtype=np.float64)

TONIC_NAME_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def _normalize_header(h: str) -> str:
    return "".join(ch.lower() for ch in str(h) if ch.isalnum())


def _find_col(headers: List[str], aliases: List[str]) -> Optional[str]:
    norm_map = {_normalize_header(h): h for h in headers}
    for alias in aliases:
        orig = norm_map.get(_normalize_header(alias))
        if orig is not None:
            return orig
    return None


def load_ground_truth(csv_path: str) -> Dict[str, dict]:
    """Load ground truth CSV. Returns {lowercase_filename_stem: {raga, tonic, ...}}."""
    gt: Dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return gt
        fn_col = _find_col(reader.fieldnames, ["filename", "file", "audio", "name"])
        raga_col = _find_col(reader.fieldnames, ["raga"])
        tonic_col = _find_col(reader.fieldnames, ["tonic"])
        gender_col = _find_col(reader.fieldnames, ["gender", "vocalist_gender"])
        instrument_col = _find_col(reader.fieldnames, ["instrument", "instrument_type"])
        if fn_col is None or raga_col is None or tonic_col is None:
            raise ValueError(f"GT CSV must have filename, raga, tonic columns. Found: {reader.fieldnames}")
        for row in reader:
            stem = row[fn_col].strip()
            if not stem:
                continue
            # Remove extension if present
            stem_no_ext = os.path.splitext(stem)[0]
            gt[stem_no_ext.lower()] = {
                "raga": row[raga_col].strip(),
                "tonic": row[tonic_col].strip(),
                "gender": row.get(gender_col or "", "").strip() if gender_col else "",
                "instrument": row.get(instrument_col or "", "").strip() if instrument_col else "",
            }
    return gt


# ---------------------------------------------------------------------------
# Collect: walk results dir and build feature dataset
# ---------------------------------------------------------------------------

def _raga_matches(candidate_raga_field: str, gt_raga: str) -> bool:
    """Check if gt_raga appears in the comma-separated candidate raga field."""
    candidate_ragas = [r.strip().lower() for r in str(candidate_raga_field).split(",")]
    return gt_raga.strip().lower() in candidate_ragas


def _tonic_matches(candidate_tonic_name: str, gt_tonic: str) -> bool:
    """Check if tonic names refer to the same pitch class."""
    cand_pc = TONIC_NAME_TO_PC.get(candidate_tonic_name.strip())
    gt_pc = TONIC_NAME_TO_PC.get(gt_tonic.strip())
    if cand_pc is None or gt_pc is None:
        return candidate_tonic_name.strip().lower() == gt_tonic.strip().lower()
    return cand_pc == gt_pc


def collect_features(results_dir: str, gt_csv_path: str, output_path: str) -> pd.DataFrame:
    """Walk results dir, load candidates.csv files, match to ground truth."""
    gt = load_ground_truth(gt_csv_path)
    print(f"Loaded {len(gt)} ground truth entries.")

    all_dfs: List[pd.DataFrame] = []
    matched = 0
    unmatched_stems: List[str] = []

    for candidates_csv in sorted(Path(results_dir).rglob("candidates.csv")):
        song_dir = candidates_csv.parent
        song_stem = song_dir.name

        gt_row = gt.get(song_stem.lower())
        if gt_row is None:
            unmatched_stems.append(song_stem)
            continue

        try:
            df = pd.read_csv(candidates_csv)
        except Exception as e:
            print(f"  [WARN] Failed to read {candidates_csv}: {e}")
            continue

        if df.empty or "fit_score" not in df.columns:
            continue

        # Add identifiers
        df["song_stem"] = song_stem
        df["gt_raga"] = gt_row["raga"]
        df["gt_tonic"] = gt_row["tonic"]

        # Label correct candidates
        df["is_correct"] = df.apply(
            lambda row: (
                _raga_matches(str(row.get("raga", "")), gt_row["raga"])
                and _tonic_matches(str(row.get("tonic_name", "")), gt_row["tonic"])
            ),
            axis=1,
        )

        all_dfs.append(df)
        matched += 1

    if not all_dfs:
        print("No candidates.csv files matched ground truth entries.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Derive normalized features
    combined["match_diff_norm"] = combined["match_diff"] / 4.0
    combined["salience_norm"] = combined.groupby("song_stem")["salience"].transform(
        lambda x: x / (x.max() + 1e-12)
    )

    # Save
    combined.to_csv(output_path, index=False)

    # Stats
    songs_with_correct = combined.groupby("song_stem")["is_correct"].any()
    n_with = songs_with_correct.sum()
    n_total = len(songs_with_correct)

    print(f"\nCollected {matched} songs ({len(combined)} total candidates)")
    print(f"Songs with correct candidate present: {n_with}/{n_total}")
    if unmatched_stems:
        print(f"Unmatched stems (no GT row): {len(unmatched_stems)}")
    print(f"Saved to {output_path}")

    return combined


# ---------------------------------------------------------------------------
# Optimization: pairwise ranking loss
# ---------------------------------------------------------------------------

def _build_pair_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, int, int]:
    """Build pairwise difference vectors (correct - incorrect) for all songs.

    Returns (X_pairs, n_songs_used, n_songs_skipped).
    """
    pairs: List[np.ndarray] = []
    n_used = 0
    n_skipped = 0

    for _stem, group in df.groupby("song_stem"):
        correct = group[group["is_correct"]]
        incorrect = group[~group["is_correct"]]

        if correct.empty:
            n_skipped += 1
            continue

        correct_feats = correct[FEATURE_COLS].iloc[0].values.astype(np.float64)
        n_used += 1

        for _, inc_row in incorrect.iterrows():
            inc_feats = inc_row[FEATURE_COLS].values.astype(np.float64)
            pairs.append(correct_feats - inc_feats)

    if not pairs:
        return np.empty((0, len(FEATURE_COLS))), n_used, n_skipped

    return np.array(pairs, dtype=np.float64), n_used, n_skipped


def _pairwise_loss(w: np.ndarray, X_pairs: np.ndarray, l2_lambda: float = 1e-4) -> float:
    """Pairwise logistic ranking loss: -log sigma(w^T (x+ - x-))."""
    scores = X_pairs @ w
    loss = np.mean(np.logaddexp(0, -scores))
    loss += l2_lambda * np.dot(w, w)
    return float(loss)


def _pairwise_grad(w: np.ndarray, X_pairs: np.ndarray, l2_lambda: float = 1e-4) -> np.ndarray:
    """Gradient of pairwise logistic loss."""
    scores = X_pairs @ w
    sigmoid_neg = 1.0 / (1.0 + np.exp(np.clip(scores, -500, 500)))
    grad = -X_pairs.T @ sigmoid_neg / len(X_pairs)
    grad += 2 * l2_lambda * w
    return grad


def compute_metrics(
    df: pd.DataFrame,
    w: np.ndarray,
    label: str = "",
) -> dict:
    """Compute MRR, top-k accuracy, and rank distribution for given weights."""
    reciprocal_ranks: List[float] = []
    ranks: List[int] = []

    for _stem, group in df.groupby("song_stem"):
        correct = group[group["is_correct"]]
        if correct.empty:
            continue

        features = group[FEATURE_COLS].values.astype(np.float64)
        scores = features @ w

        correct_idx = correct.index[0] - group.index[0]
        correct_score = scores[correct_idx]
        rank = 1 + int(np.sum(scores > correct_score))
        reciprocal_ranks.append(1.0 / rank)
        ranks.append(rank)

    ranks_arr = np.array(ranks)
    metrics = {
        "mrr": float(np.mean(reciprocal_ranks)),
        "top1": float(np.mean(ranks_arr == 1)),
        "top3": float(np.mean(ranks_arr <= 3)),
        "top5": float(np.mean(ranks_arr <= 5)),
        "top10": float(np.mean(ranks_arr <= 10)),
        "median_rank": int(np.median(ranks_arr)),
        "mean_rank": float(np.mean(ranks_arr)),
        "n_songs": len(ranks),
    }

    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  MRR:         {metrics['mrr']:.4f}")
    print(f"  Top-1:       {metrics['top1']:.2%}")
    print(f"  Top-3:       {metrics['top3']:.2%}")
    print(f"  Top-5:       {metrics['top5']:.2%}")
    print(f"  Top-10:      {metrics['top10']:.2%}")
    print(f"  Median rank: {metrics['median_rank']}")
    print(f"  Mean rank:   {metrics['mean_rank']:.1f}")
    print(f"  Songs:       {metrics['n_songs']}")

    return metrics


def optimize_linear(data_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, dict]:
    """Optimize scoring coefficients using pairwise ranking loss."""
    df = pd.read_csv(data_path)

    # Ensure derived features exist
    if "match_diff_norm" not in df.columns:
        df["match_diff_norm"] = df["match_diff"] / 4.0
    if "salience_norm" not in df.columns:
        df["salience_norm"] = df.groupby("song_stem")["salience"].transform(
            lambda x: x / (x.max() + 1e-12)
        )

    # Build pairs
    X_pairs, n_used, n_skipped = _build_pair_matrix(df)
    print(f"Built {len(X_pairs)} pairwise comparisons from {n_used} songs")
    if n_skipped:
        print(f"  ({n_skipped} songs skipped: correct candidate not in results)")

    if len(X_pairs) == 0:
        print("No pairs to optimize. Check that ground truth matches candidates.")
        return CURRENT_WEIGHTS, {}

    # Evaluate current weights
    compute_metrics(df, CURRENT_WEIGHTS, label="BEFORE (current hand-tuned)")

    # Optimize
    w0 = CURRENT_WEIGHTS.copy()
    result = minimize(
        _pairwise_loss,
        w0,
        args=(X_pairs,),
        jac=_pairwise_grad,
        method="L-BFGS-B",
        options={"maxiter": 2000, "disp": True},
    )
    w_opt = result.x

    # Evaluate optimized
    metrics_after = compute_metrics(df, w_opt, label="AFTER (optimized)")

    # Print coefficient comparison
    print(f"\n{'='*60}")
    print(f"  Coefficient comparison")
    print(f"{'='*60}")
    print(f"  {'Feature':<25s} {'Before':>10s} {'After':>10s} {'Delta':>10s}")
    print(f"  {'-'*55}")
    for i, name in enumerate(FEATURE_COLS):
        before = CURRENT_WEIGHTS[i]
        after = w_opt[i]
        delta = after - before
        print(f"  {name:<25s} {before:>+10.4f} {after:>+10.4f} {delta:>+10.4f}")

    # Map to ScoringParams
    scoring_params = _weights_to_scoring_params(w_opt)
    print(f"\n  ScoringParams equivalent:")
    for k, v in scoring_params.items():
        print(f"    {k}: {v}")

    # Save
    if output_path:
        out = {
            "feature_cols": FEATURE_COLS,
            "weights": w_opt.tolist(),
            "scoring_params": scoring_params,
            "metrics_before": compute_metrics(df, CURRENT_WEIGHTS),
            "metrics_after": metrics_after,
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {output_path}")

    return w_opt, scoring_params


def _weights_to_scoring_params(w: np.ndarray) -> dict:
    """Convert weight vector to ScoringParams-style dict.

    The current formula is:
        score = α*match + β*presence + γ*loglike - δ*extra - ε*complex - ζ*(diff/4) + η*salience
    So negative-convention features need sign flip.
    """
    return {
        "ALPHA_MATCH": float(w[0]),            # match_mass (positive)
        "BETA_PRESENCE": float(w[1]),          # observed_note_score (positive)
        "GAMMA_LOGLIKE": float(w[2]),          # loglike_norm (positive)
        "DELTA_EXTRA": float(-w[3]),           # extra_mass (flip: stored positive, applied negative)
        "COMPLEX_PENALTY": float(-w[4]),       # complexity_pen (flip)
        "MATCH_SIZE_GAMMA": float(-w[5]),      # match_diff_norm (flip)
        "PRIMARY_SCORE_WEIGHT": float(w[6]),   # new: not in current ScoringParams
        "TONIC_SALIENCE_WEIGHT": float(w[7]),  # salience_norm (positive)
    }


# ---------------------------------------------------------------------------
# Evaluate with custom params
# ---------------------------------------------------------------------------

def evaluate(data_path: str, params_path: Optional[str] = None, mlp_path: Optional[str] = None) -> dict:
    """Evaluate MRR and top-k accuracy with given or default weights, or an MLP model."""
    df = pd.read_csv(data_path)

    if "match_diff_norm" not in df.columns:
        df["match_diff_norm"] = df["match_diff"] / 4.0
    if "salience_norm" not in df.columns:
        df["salience_norm"] = df.groupby("song_stem")["salience"].transform(
            lambda x: x / (x.max() + 1e-12)
        )

    if mlp_path:
        return _evaluate_mlp(df, mlp_path)

    if params_path:
        with open(params_path) as f:
            params = json.load(f)
        w = np.array(params["weights"], dtype=np.float64)
        label = f"Custom params ({params_path})"
    else:
        w = CURRENT_WEIGHTS
        label = "Current defaults"

    return compute_metrics(df, w, label=label)


# ---------------------------------------------------------------------------
# MLP Training and Evaluation
# ---------------------------------------------------------------------------

def _get_mlp_features(df: pd.DataFrame) -> np.ndarray:
    """Extract MLP feature matrix from dataframe, using extended columns if available."""
    # Check if new columns are present
    has_extended = "melody_dist_0" in df.columns
    if has_extended:
        cols = MLP_FEATURE_COLS
    else:
        cols = FEATURE_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for MLP features: {missing}")
    return df[cols].values.astype(np.float64), cols


def _evaluate_mlp(df: pd.DataFrame, mlp_path: str) -> dict:
    """Evaluate MLP model using ranking metrics."""
    import joblib
    bundle = joblib.load(mlp_path)
    clf = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle.get("feature_cols", FEATURE_COLS)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset for MLP: {missing}")

    X = df[feature_cols].values.astype(np.float64)
    X_scaled = scaler.transform(X)
    probs = clf.predict_proba(X_scaled)[:, 1]

    songs = df.groupby("song_stem")
    reciprocal_ranks = []
    ranks = []

    for stem, group in songs:
        correct_mask = group["is_correct"].values.astype(bool)
        if not correct_mask.any():
            continue
        song_probs = probs[group.index]
        correct_prob = song_probs[correct_mask].max()
        rank = 1 + int(np.sum(song_probs > correct_prob))
        reciprocal_ranks.append(1.0 / rank)
        ranks.append(rank)

    ranks_arr = np.array(ranks)
    label = f"MLP ({mlp_path})"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  MRR:         {np.mean(reciprocal_ranks):.4f}")
    print(f"  Top-1:       {np.mean(ranks_arr == 1):.2%}")
    print(f"  Top-3:       {np.mean(ranks_arr <= 3):.2%}")
    print(f"  Top-5:       {np.mean(ranks_arr <= 5):.2%}")
    print(f"  Top-10:      {np.mean(ranks_arr <= 10):.2%}")
    print(f"  Median rank: {int(np.median(ranks_arr))}")
    print(f"  Mean rank:   {np.mean(ranks_arr):.1f}")
    print(f"  Songs:       {len(ranks)}")

    return {
        "mrr": float(np.mean(reciprocal_ranks)),
        "top1": float(np.mean(ranks_arr == 1)),
        "top5": float(np.mean(ranks_arr <= 5)),
        "top10": float(np.mean(ranks_arr <= 10)),
        "n_songs": len(ranks),
    }


def train_mlp(
    data_path: str,
    output_path: str = "mlp_scorer.pkl",
    hidden_layers: Tuple[int, ...] = (64, 32),
    test_size: float = 0.2,
    max_iter: int = 1000,
) -> dict:
    """Train an MLP binary classifier on the scoring dataset."""
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)

    # Ensure derived features
    if "match_diff_norm" not in df.columns:
        df["match_diff_norm"] = df["match_diff"] / 4.0
    if "salience_norm" not in df.columns:
        df["salience_norm"] = df.groupby("song_stem")["salience"].transform(
            lambda x: x / (x.max() + 1e-12)
        )

    X, feature_cols = _get_mlp_features(df)
    y = df["is_correct"].astype(int).values

    print(f"Dataset: {len(X)} candidates, {y.sum()} correct, {len(feature_cols)} features")
    print(f"Features: {feature_cols}")

    # Split by song (not by row) to avoid leakage
    song_stems = df["song_stem"].unique()
    train_songs, test_songs = train_test_split(song_stems, test_size=test_size, random_state=42)
    train_mask = df["song_stem"].isin(train_songs)
    test_mask = df["song_stem"].isin(test_songs)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train: {len(X_train)} rows ({train_mask.sum()} candidates, {len(train_songs)} songs)")
    print(f"Test:  {len(X_test)} rows ({test_mask.sum()} candidates, {len(test_songs)} songs)")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True,
    )
    clf.fit(X_train_s, y_train)

    # Classification metrics on test set
    from sklearn.metrics import classification_report
    y_pred = clf.predict(X_test_s)
    print("\nClassification Report (test set):")
    print(classification_report(y_test, y_pred, target_names=["incorrect", "correct"]))

    # Save
    bundle = {"model": clf, "scaler": scaler, "feature_cols": list(feature_cols)}
    joblib.dump(bundle, output_path)
    print(f"Saved MLP + scaler to {output_path}")

    # Ranking evaluation on test set
    df_test = df[test_mask].copy()
    df_test["_mlp_prob"] = clf.predict_proba(X_test_s)[:, 1]

    songs = df_test.groupby("song_stem")
    reciprocal_ranks = []
    ranks = []
    for stem, group in songs:
        correct_mask_g = group["is_correct"].values.astype(bool)
        if not correct_mask_g.any():
            continue
        probs_g = group["_mlp_prob"].values
        correct_prob = probs_g[correct_mask_g].max()
        rank = 1 + int(np.sum(probs_g > correct_prob))
        reciprocal_ranks.append(1.0 / rank)
        ranks.append(rank)

    ranks_arr = np.array(ranks)
    print(f"\n{'='*60}")
    print(f"  MLP Ranking (test set, {len(test_songs)} songs)")
    print(f"{'='*60}")
    print(f"  MRR:         {np.mean(reciprocal_ranks):.4f}")
    print(f"  Top-1:       {np.mean(ranks_arr == 1):.2%}")
    print(f"  Top-3:       {np.mean(ranks_arr <= 3):.2%}")
    print(f"  Top-5:       {np.mean(ranks_arr <= 5):.2%}")
    print(f"  Top-10:      {np.mean(ranks_arr <= 10):.2%}")
    print(f"  Median rank: {int(np.median(ranks_arr))}")
    print(f"  Mean rank:   {np.mean(ranks_arr):.1f}")

    # Also evaluate linear baseline on same test split for comparison
    print()
    compute_metrics(df_test, CURRENT_WEIGHTS, label="Linear baseline (same test split)")

    return {"output": output_path, "n_features": len(feature_cols)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimize raga scoring coefficients against annotated ground truth."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # collect
    p_collect = sub.add_parser("collect", help="Collect feature vectors from batch results")
    p_collect.add_argument("--results-dir", required=True, help="Batch results directory")
    p_collect.add_argument("--gt", required=True, help="Ground truth CSV path")
    p_collect.add_argument("--output", default="scoring_dataset.csv", help="Output CSV path")

    # optimize
    p_opt = sub.add_parser("optimize", help="Optimize linear scoring coefficients")
    p_opt.add_argument("--data", required=True, help="Scoring dataset CSV (from collect)")
    p_opt.add_argument("--output", default="optimized_params.json", help="Output JSON path")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate scoring metrics")
    p_eval.add_argument("--data", required=True, help="Scoring dataset CSV (from collect)")
    p_eval.add_argument("--params", default=None, help="Optimized params JSON (omit for current defaults)")
    p_eval.add_argument("--mlp", default=None, help="MLP model .pkl path (overrides --params)")

    # mlp
    p_mlp = sub.add_parser("mlp", help="Train an MLP binary classifier for scoring")
    p_mlp.add_argument("--data", required=True, help="Scoring dataset CSV (from collect)")
    p_mlp.add_argument("--output", default="mlp_scorer.pkl", help="Output .pkl path")
    p_mlp.add_argument("--hidden-layers", default="64,32", help="Comma-separated hidden layer sizes (default: 64,32)")
    p_mlp.add_argument("--test-size", type=float, default=0.2, help="Fraction of songs held out for testing (default: 0.2)")
    p_mlp.add_argument("--max-iter", type=int, default=1000, help="Max training iterations (default: 1000)")

    args = parser.parse_args()

    if args.command == "collect":
        collect_features(args.results_dir, args.gt, args.output)
    elif args.command == "optimize":
        optimize_linear(args.data, args.output)
    elif args.command == "evaluate":
        evaluate(args.data, args.params, mlp_path=args.mlp)
    elif args.command == "mlp":
        layers = tuple(int(x) for x in args.hidden_layers.split(","))
        train_mlp(args.data, args.output, hidden_layers=layers, test_size=args.test_size, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
