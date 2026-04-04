"""Learn optimal weights for the three-signal raga detection formula.

Reads lm_candidates.csv files produced by detect --use-lm-scoring,
fits a logistic regression on (histogram_score, lm_score, del_residual)
to predict the correct raga, and reports optimal weights + nested LOO accuracy.

Usage: python scripts/learn_signal_weights.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

GT_CSV = "compmusic_gt.csv"
RESULTS_BASE = Path("/Volumes/Extreme SSD/stems/separated_stems/htdemucs")


def raga_match(true_r: str, cand_r: str) -> bool:
    """Fuzzy match: true raga matches candidate (handles substring)."""
    t = true_r.lower().strip()
    c = cand_r.lower().strip()
    return t == c or t in c or c in t


def load_corpus():
    """Load all (recording, candidate) pairs with features and labels."""
    recordings = []  # list of (filename, true_raga, candidates_list)

    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            filename = row["Filename"].strip()
            true_raga = row["Raga"].strip()

            lm_csv = RESULTS_BASE / filename / "lm_candidates.csv"
            if not lm_csv.exists():
                continue

            candidates = []
            with open(lm_csv) as cf:
                for cand in csv.DictReader(cf):
                    if cand.get("gated") == "False":
                        continue  # only use gated candidates
                    try:
                        candidates.append({
                            "raga": cand["raga"],
                            "hist": float(cand["norm_histogram"]),
                            "lm": float(cand["norm_lm"]),
                            "del_resid": float(cand["del_residual"]),
                            "hist_raw": float(cand["histogram_score"]),
                            "lm_raw": float(cand["lm_score"]),
                            "correct": raga_match(true_raga, cand["raga"]),
                        })
                    except (KeyError, ValueError):
                        continue

            if candidates:
                recordings.append((filename, true_raga, candidates))

    return recordings


def evaluate_fixed_weights(recordings, alpha=1.0, beta=1.0, gamma=2.0):
    """Evaluate the fixed-weight formula: alpha*hist + beta*lm - gamma*del_resid."""
    top1 = 0
    top3 = 0
    total = len(recordings)

    for filename, true_raga, candidates in recordings:
        scored = []
        for c in candidates:
            score = alpha * c["hist"] + beta * c["lm"] - gamma * c["del_resid"]
            scored.append((score, c["raga"], c["correct"]))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored and scored[0][2]:
            top1 += 1
        if any(s[2] for s in scored[:3]):
            top3 += 1

    return top1 / total, top3 / total


def run_logistic_regression_loo(recordings):
    """Nested LOO: hold out one recording, fit logistic regression, predict."""
    from sklearn.linear_model import LogisticRegression

    top1 = 0
    top3 = 0
    total = len(recordings)
    all_coefs = []

    for held_out_idx in range(total):
        # Train on all except held-out
        X_train = []
        y_train = []
        for idx, (fn, raga, cands) in enumerate(recordings):
            if idx == held_out_idx:
                continue
            for c in cands:
                X_train.append([c["hist"], c["lm"], c["del_resid"]])
                y_train.append(1 if c["correct"] else 0)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Fit
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train, y_train)
        all_coefs.append(model.coef_[0])

        # Predict on held-out
        fn, raga, cands = recordings[held_out_idx]
        X_test = np.array([[c["hist"], c["lm"], c["del_resid"]] for c in cands])
        probs = model.predict_proba(X_test)[:, 1]

        # Rank by probability
        ranked = sorted(zip(probs, cands), key=lambda x: x[0], reverse=True)

        if ranked and ranked[0][1]["correct"]:
            top1 += 1
        if any(c["correct"] for _, c in ranked[:3]):
            top3 += 1

        if (held_out_idx + 1) % 50 == 0:
            print(f"  LOO progress: {held_out_idx + 1}/{total}")

    mean_coefs = np.mean(all_coefs, axis=0)
    std_coefs = np.std(all_coefs, axis=0)

    return top1 / total, top3 / total, mean_coefs, std_coefs


def main():
    print("Loading corpus...")
    recordings = load_corpus()
    print(f"Loaded {len(recordings)} recordings with lm_candidates.csv")

    if not recordings:
        print("ERROR: No lm_candidates.csv files found. Run detect --use-lm-scoring first.")
        sys.exit(1)

    # Count total candidates
    total_cands = sum(len(c) for _, _, c in recordings)
    correct_cands = sum(sum(1 for c in cands if c["correct"]) for _, _, cands in recordings)
    print(f"Total gated candidates: {total_cands}")
    print(f"Correct candidates present: {correct_cands}/{len(recordings)}")

    # Evaluate fixed weights
    print("\n=== Fixed-Weight Formula ===")
    for alpha, beta, gamma in [(1.0, 1.0, 2.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
        t1, t3 = evaluate_fixed_weights(recordings, alpha, beta, gamma)
        print(f"  alpha={alpha} beta={beta} gamma={gamma}: top1={t1:.1%} top3={t3:.1%}")

    # Logistic regression LOO
    print("\n=== Logistic Regression (Nested LOO) ===")
    t1, t3, coefs, coef_std = run_logistic_regression_loo(recordings)
    print(f"  Top-1: {t1:.1%}")
    print(f"  Top-3: {t3:.1%}")
    print(f"  Learned weights (mean +/- std across folds):")
    print(f"    w_histogram:    {coefs[0]:+.4f} +/- {coef_std[0]:.4f}")
    print(f"    w_lm:           {coefs[1]:+.4f} +/- {coef_std[1]:.4f}")
    print(f"    w_del_residual: {coefs[2]:+.4f} +/- {coef_std[2]:.4f}")

    # Interpret
    print("\n=== Interpretation ===")
    abs_coefs = np.abs(coefs)
    total_w = abs_coefs.sum()
    print(f"  Relative importance:")
    print(f"    histogram:    {abs_coefs[0]/total_w:.1%}")
    print(f"    lm:           {abs_coefs[1]/total_w:.1%}")
    print(f"    del_residual: {abs_coefs[2]/total_w:.1%}")


if __name__ == "__main__":
    main()
