# N-gram Language Model for Raga Detection: Experimental Results

## 1. Overview

This report documents the experiments conducted to evaluate character n-gram language models as an approach to raga detection from transcribed Hindustani classical music. The work progresses through four experimental phases, each addressing a methodological issue discovered in the previous one.

**Corpus:** 300 recordings across 30 ragas from the CompuMusic database (~10 recordings per raga, median recording length 30-40 minutes). 3 recordings have missing transcription data, leaving 297 usable recordings.

**Evaluation protocol for all experiments:** Leave-one-out cross-validation (LOO CV). For each of the 297 recordings, the model is trained on the remaining 296 and the held-out recording is scored against all 30 raga models. No recording is ever scored by a model that saw it during training.

---

## 2. Experiment 1: Baseline N-gram Model (Flat Tokens, Corrected Transcriptions)

### 2.1 Method

**Tokenization:** Each recording's `transcribed_notes.csv` is converted to a sequence of sargam tokens with octave markers:
- 12 base tokens: `Sa, re, Re, ga, Ga, ma, Ma, Pa, dha, Dha, ni, Ni` (case encodes komal/shuddha)
- Octave markers: bare = middle octave (containing tonic), `'` = one below, `''` = one above
- `<BOS>` token inserted at phrase boundaries (silence gaps > 0.25s between consecutive notes)
- The reference octave is determined by the tonic MIDI note, making the representation tonic-relative

**Model:** Per-raga n-gram tables (orders 1 through N) with add-k smoothing (k=0.01) and equal-weight interpolation across orders. The model stores raw counts; probabilities are computed at scoring time.

**Scoring:** For each held-out recording, the average log-probability per token is computed under each raga's model. The raga with the highest average log-probability (lowest perplexity) is the prediction.

**Transcription source:** The transcriptions were produced by the pipeline's `analyze` mode, which applies **raga correction** after chromatic transcription -- discarding or re-snapping notes that don't fit the specified raga.

**Token stream:** A flat list of tokens with `<BOS>` markers interspersed. N-grams were extracted by sliding a window over this flat list, meaning n-grams could span across phrase boundaries.

### 2.2 Results

| Order | Top-1 Accuracy | Top-3 Accuracy | MRR |
|-------|---------------|----------------|------|
| 2 | 93.6% | 99.0% | 0.959 |
| 3 | 95.6% | 99.0% | 0.971 |
| 4 | 96.3% | 99.0% | 0.975 |
| 5 | 96.3% | 99.0% | 0.976 |
| 6 | **97.0%** | **99.0%** | **0.979** |

9 misclassified recordings at order 6:
- 5 in the Alhaiya Bilawal / Desh / Gaud Malhar cluster (same thaat, near-identical scores)
- 2 Puriya Dhanashree errors (confused with Basant, Kedar)
- 1 Marwa -> Puriya Dhanashree (rank 20, likely transcription issue)
- 1 Malkauns -> Yaman Kalyan (rank 29, almost certainly bad transcription)

25 of 30 ragas achieved 100% accuracy. Accuracy improved monotonically from order 2 to 6.

### 2.3 Issues Discovered

**Circularity in transcriptions:** The transcriptions were produced with raga correction enabled, meaning notes outside the specified raga's scale were removed before the language model ever saw them. The model was partly learning "what does a transcription look like after filtering to raga X" rather than "what does raga X sound like."

**Cross-boundary n-grams:** Analysis revealed that 49% of n-grams at order 5 spanned phrase boundaries (e.g., the last note of one phrase combined with the first note of the next). These are musically meaningless sequences.

---

## 3. Experiment 2: Uncorrected Transcriptions (Flat Tokens)

### 3.1 Method

Identical to Experiment 1, except the transcriptions were regenerated with `--skip-raga-correction` -- the post-transcription raga correction step was disabled. The transcription itself is unchanged (chromatic snapping), but notes outside the raga's scale are no longer removed.

**Impact on transcription size:** Uncorrected transcriptions contain 15-71% more notes than corrected ones (varies by raga scale size):

| Recording | Corrected Notes | Uncorrected Notes | Increase |
|-----------|----------------|-------------------|----------|
| Abhogi_1 | 887 | 1,312 | +48% |
| Yaman Kalyan_1 | 3,552 | 5,291 | +49% |
| Bhairav_1 | 3,761 | 4,448 | +18% |
| Hansadhwani_1 | 3,542 | 6,057 | +71% |

Pentatonic ragas (Hansadhwani, Bhupali) show the largest increase because correction is most aggressive for small scales.

### 3.2 Results

| Order | Top-1 (Corrected) | Top-1 (Uncorrected) | Drop |
|-------|-------------------|---------------------|------|
| 2 | 93.6% | 81.5% | -12.1 |
| 3 | 95.6% | 85.9% | -9.7 |
| 4 | 96.3% | **88.6%** | -7.7 |
| 5 | 96.3% | 88.6% | -7.7 |
| 6 | **97.0%** | 88.2% | -8.8 |

| Order | Top-3 (Corrected) | Top-3 (Uncorrected) | Drop |
|-------|-------------------|---------------------|------|
| 2 | 99.0% | 93.9% | -5.1 |
| 3 | 99.0% | 94.6% | -4.4 |
| 4 | 99.0% | **95.6%** | -3.4 |
| 5 | 99.0% | 95.3% | -3.7 |
| 6 | **99.0%** | 94.9% | -4.1 |

35 misclassified recordings at order 4. Ragas most affected:
- Hansadhwani: 60% (was 100%) -- pentatonic, correction removed the most noise
- Bhupali: 70% (was 100%) -- also pentatonic
- Shuddh Sarang: 70% (was 100%)
- Desh: 70% (was 90%)

### 3.3 Key Findings

- Raga correction inflated accuracy by **8-12 percentage points** across all orders
- The gap is consistent, confirming it's the correction bias rather than a single-order artifact
- Optimal order shifted: on uncorrected data, order 4 peaks and orders 5-6 slightly hurt (noise sensitivity)
- **88.6% top-1** is the honest unbiased baseline for this approach
- Pentatonic ragas are disproportionately affected because their corrected transcriptions were the most artificially cleaned

---

## 4. Experiment 3: Phrase-Aware N-grams

### 4.1 Method

Modified the tokenizer and n-gram counting to respect phrase boundaries:
- The tokenizer now returns a `List[List[str]]` (list of per-phrase token sequences) instead of a flat list
- N-grams are counted within each phrase only; no n-gram ever spans a phrase boundary
- Scoring is also phrase-aware: each phrase is scored independently and the per-token log-likelihoods are averaged across all phrases

This eliminates the 49% of cross-boundary n-grams that were present in Experiments 1-2.

### 4.2 Results -- Corrected Transcriptions

| Order | Flat (Exp 1) | Phrase-Aware | Change |
|-------|-------------|--------------|--------|
| 2 | 93.6% | 92.9% | -0.7 |
| 3 | 95.6% | 95.3% | -0.3 |
| 4 | 96.3% | 96.0% | -0.3 |
| 5 | 96.3% | 96.0% | -0.3 |
| 6 | **97.0%** | **96.6%** | -0.4 |

Top-3 accuracy: unchanged at 99.0% across all orders.

### 4.3 Results -- Uncorrected Transcriptions

| Order | Flat (Exp 2) | Phrase-Aware | Change |
|-------|-------------|--------------|--------|
| 2 | 81.5% | 81.1% | -0.4 |
| 3 | 85.9% | 83.8% | -2.1 |
| 4 | **88.6%** | 86.2% | -2.4 |
| 5 | 88.6% | **87.2%** | -1.4 |
| 6 | 88.2% | 87.2% | -1.0 |

### 4.4 Key Findings

- **Phrase-awareness has minimal impact on corrected data** (-0.3 to -0.7 pp): cross-boundary n-grams were mostly harmless noise on clean transcriptions
- **Phrase-awareness slightly hurts uncorrected data** (-1.0 to -2.4 pp): surprising at first, but uncorrected transcriptions have more notes and less reliable phrase boundaries (noise notes create false gaps), so removing cross-boundary patterns also removes some accidentally useful signal
- **Optimal order for uncorrected shifts to 5** with phrase-aware scoring (was 4 with flat)
- The phrase fix is architecturally correct (cross-boundary n-grams are musically meaningless) even though it doesn't improve raw accuracy

---

## 5. Experiment 4: Multi-Hypothesis LM-Enhanced Detect

### 5.1 Method

Integrated LM scoring into the existing detect pipeline as an optional step (`--use-lm-scoring`). The flow:

1. **Detect phase runs as normal** -- stem separation, pitch extraction, histogram scoring produce ranked (tonic, raga) candidates
2. **One chromatic transcription** -- the raw notes are produced without any raga assumption (20-30 seconds)
3. **Per-hypothesis scoring** -- for each (tonic, raga) candidate from the histogram:
   - Apply that raga's correction to the chromatic notes (discard out-of-scale notes)
   - Record the **deletion rate** (fraction of notes removed)
   - Tokenize the corrected notes with that tonic
   - Score against that raga's trained LM
4. **Output** -- `lm_candidates.csv` with three independent signals: `histogram_score`, `lm_score`, `deletion_rate`

The model is trained on corrected transcriptions (matching the per-hypothesis correction at inference).

### 5.2 Case Study: Parveen Sultana, Raga Puriya Dhanashree

Test recording: a YouTube performance not in the training corpus. Histogram detector identified tonic A# with top candidates Basant, Puriya Dhanashree, Shri (all tied).

| Signal | Puriya Dhanashree (A#) | Hansadhwani (B) | Kedar (A#) |
|--------|----------------------|-----------------|------------|
| Histogram score | **699.99** (rank 1) | -574.09 (poor) | 106.88 |
| LM score | -2.77 (rank 14) | **-2.67** (rank 1) | -2.68 (rank 2) |
| Deletion rate | **3.5%** (excellent) | 23.7% (high) | 9.2% |

- **LM score alone** ranks Puriya Dhanashree 14th. Hansadhwani (pentatonic) ranks 1st because after deleting 24% of notes, the surviving 5-note sequence is simple and scores well.
- **Histogram alone** correctly identifies the raga family but can't distinguish Basant/Puriya Dhanashree/Shri.
- **Deletion rate** strongly supports Puriya Dhanashree (only 3.5% of notes removed) and penalizes Hansadhwani (23.7% removed).
- **All three signals together** clearly point to Puriya Dhanashree.

### 5.3 Key Finding

No single signal is sufficient. The LM is biased toward small-scale ragas (fewer notes to model = lower perplexity). The histogram can't distinguish ragas with the same scale. The deletion rate captures how well a raga hypothesis explains the full chromatic content. A combined scoring function is needed.

---

## 6. Summary of All Results

| Experiment | Condition | Best Top-1 | Best Top-3 | Best Order |
|-----------|-----------|-----------|-----------|------------|
| 1 | Flat, corrected | 97.0% | 99.0% | 6 |
| 2 | Flat, uncorrected | 88.6% | 95.6% | 4 |
| 3a | Phrase-aware, corrected | 96.6% | 99.0% | 6 |
| 3b | Phrase-aware, uncorrected | 87.2% | 94.3% | 5 |

The honest, unbiased accuracy of the n-gram language model on chromatic transcriptions is **87-89% top-1** and **94-96% top-3** depending on phrase handling. Raga correction inflates accuracy by 8-12 percentage points.

---

## 7. Signal Combination Strategies

The multi-hypothesis experiment (Section 5) produces three independent signals per candidate. Here are four approaches to combining them, ordered from simplest to most powerful.

### 7.1 Approach A: Rank Fusion (Simplest)

Assign each candidate a rank under each signal (histogram rank, LM rank, deletion-rate rank). The combined score is the sum or average of ranks.

```
combined_rank = histogram_rank + lm_rank + deletion_rank
```

**Pros:** No parameters to tune. No assumptions about score distributions. Robust to outliers.

**Cons:** Ignores score magnitudes. A candidate that's rank 1 by a huge margin is treated the same as rank 1 by a tiny margin.

**Variant:** Reciprocal rank fusion: `combined = 1/(k + hist_rank) + 1/(k + lm_rank) + 1/(k + del_rank)` with k=60 (standard value from information retrieval). Weights top-ranked candidates more heavily.

### 7.2 Approach B: Normalized Weighted Sum

Normalize each signal to [0, 1] across candidates for a given recording, then combine with tunable weights:

```
norm(x) = (x - min) / (max - min)    per signal, per recording

combined = alpha * norm(histogram_score)
         + beta  * norm(lm_score)
         + gamma * norm(1 - deletion_rate)
```

Note: `1 - deletion_rate` is used so that higher = better for all signals.

**Tuning:** Grid search over (alpha, beta, gamma) on LOO CV. Evaluate top-1 accuracy at each parameter setting.

**Pros:** Interpretable weights. Easy to adjust per domain knowledge (e.g., increase gamma for pentatonic ragas).

**Cons:** Requires tuning. Linear combination may not capture interactions.

**The deletion rate deserves special attention.** As seen in Section 5.2, the deletion rate is the strongest single discriminator for ruling out wrong hypotheses -- a 3.5% rate vs 23.7% is a much larger signal gap than the LM score difference (-2.77 vs -2.67). Consider starting with high gamma.

### 7.3 Approach C: Log-Linear Combination

More principled for probability-like scores:

```
combined = alpha * histogram_score
         + beta  * lm_score
         + gamma * log(1 - deletion_rate + epsilon)
```

The `log(1 - deletion_rate)` term penalizes high deletion rates exponentially -- deleting 50% of notes is much worse than deleting 25%, which is much worse than deleting 5%. This matches the intuition that the deletion rate is most informative at the extremes.

**Pros:** Better-calibrated than linear combination. The log term naturally handles the fact that deletion rate differences matter more at high values.

**Cons:** Slightly harder to interpret. Still requires tuning alpha, beta, gamma.

### 7.4 Approach D: Two-Stage Filtering

Use the signals sequentially rather than combining them:

1. **Stage 1 (Histogram):** Keep top-K candidates by histogram score (e.g., K=20). This eliminates tonics/ragas with no pitch evidence.
2. **Stage 2 (Deletion rate):** Among survivors, discard candidates with deletion rate > threshold (e.g., 0.30). This eliminates ragas that don't explain the chromatic content.
3. **Stage 3 (LM score):** Among survivors, rank by LM score. This selects the raga whose grammar best matches the sequence.

**Pros:** Each signal handles what it's best at. The histogram eliminates impossible tonics. The deletion rate eliminates ragas that require too much surgery. The LM disambiguates the survivors. No parameter tuning for the filtering stages (just thresholds).

**Cons:** Hard cutoffs lose information (a candidate at deletion rate 0.31 is discarded, 0.29 is kept). The threshold values need to be chosen.

**Variant:** Soft filtering -- instead of hard cutoffs, weight the LM score by a sigmoid-gated deletion penalty:

```
gate = sigmoid((threshold - deletion_rate) / temperature)
combined = gate * lm_score
```

### 7.5 Initial Recommendation (Revised -- see Section 9 for updated analysis)

Start with **Approach D (two-stage filtering)** as the default pipeline behavior -- it's the most interpretable and requires the least tuning. Then implement **Approach B (normalized weighted sum)** as an evaluation tool to empirically determine optimal weights via grid search on the CompMusic LOO CV. The grid search results inform what the thresholds for Approach D should be.

The deletion rate is the strongest novel signal. In the Parveen Sultana case, a simple rule of "discard candidates with >20% deletion" would have eliminated the LM's false top-1 (Hansadhwani at 23.7%) while keeping the correct answer (Puriya Dhanashree at 3.5%). However, as shown in Section 8, raw deletion rate is heavily confounded with raga scale size and needs normalization.

---

## 8. Deletion Rate Analysis: Scale-Size Confound

### 8.1 Per-Raga Deletion Rate Distributions

Analysis of the correct-raga deletion rates across all 297 CompMusic recordings reveals that the deletion rate is strongly correlated with raga scale size:

| Scale Size | Ragas | Mean Deletion Rate |
|------------|-------|-------------------|
| 5 notes (pentatonic) | Bhupali, Malkauns, Hansadhwani, Abhogi, Bairagi, Madhukauns | 30-42% |
| 6 notes (hexatonic) | Marwa, Jog, Rageshri | 18-26% |
| 7 notes (heptatonic) | Bhairav, Todi, Basant, Puriya Dhanashree, etc. | 13-22% |
| 8 notes (octotonic) | Bihag, Maru Bihag, Desh, Gaud Malhar, etc. | 10-20% |

This means a raw deletion rate threshold would be heavily biased against pentatonic ragas: a 20% cutoff eliminates almost every correct pentatonic hypothesis, while most heptatonic ragas pass easily.

### 8.2 Linear Regression: Deletion Rate vs Scale Size

Fitting a linear model across the 28 ragas with known scale sizes:

```
expected_deletion = -0.0684 * scale_size + 0.6640
```

- **R-squared = 0.77** -- scale size explains 77% of the variance in deletion rate
- **Slope = -6.84% per note** -- each additional note in the raga's scale reduces the expected deletion rate by ~7 percentage points
- Actual deletion rates are always lower than naive random expectation (performers emphasize in-scale notes)

### 8.3 Residual as Normalized Signal

The **residual** (actual deletion - expected deletion from regression) removes the scale-size confound:

```
del_residual = actual_del - (-0.0684 * scale_size + 0.6640)
```

Per-raga residuals are mostly within +/-5%:
- **Bhupali (+10.2%) and Gaud Malhar (+8.1%)** are outliers -- their performers use more out-of-scale notes than scale size predicts
- **Rageshri (-7.1%) and Basant (-5.7%)** are unusually clean
- Most ragas cluster near 0, confirming the regression captures the dominant effect

### 8.4 Distribution Shape Concerns

With only ~10 recordings per raga, per-raga distributions are noisy:
- Most ragas are roughly symmetric (skew near 0)
- **Malkauns (skew 1.39)** and **Bilaskhani Todi (skew 1.16)** have long right tails
- **Gaud Malhar** appears bimodal (8-12% cluster and 23-32% cluster)
- Nearly every raga has at least one outlier recording

Z-score normalization per raga would be fragile with 10 samples. The regression-based residual is more robust because it uses a known quantity (scale size) rather than estimated per-raga statistics.

### 8.5 Implication for Signal Combination

The raw deletion rate should **not** be used directly in any combination formula. Instead, use the scale-normalized residual. This ensures that a pentatonic raga at 35% deletion (normal for its scale) is treated equivalently to a heptatonic raga at 15% deletion (also normal for its scale).

---

## 9. Updated Signal Combination Strategies

Based on the deletion rate analysis and the Parveen Sultana case study, here is the full set of combination strategies under consideration. The three signals are:
- **Histogram score** (`hist`): pitch class distribution match, strongest at tonic detection and eliminating impossible ragas
- **LM score** (`lm`): n-gram sequence likelihood, captures raga grammar but biased toward small-scale ragas
- **Deletion residual** (`del_resid`): scale-normalized correction rate, captures how well the raw chromatic content fits the hypothesis

### 9.1 Rank Fusion (Simplest)

Sum ranks across signals. No parameters. Ignores score magnitudes.

```
combined_rank = hist_rank + lm_rank + del_resid_rank
```

**Variant:** Reciprocal rank fusion: `combined = 1/(k + hist_rank) + 1/(k + lm_rank) + 1/(k + del_rank)` with k=60.

**Pros:** Zero parameters, robust to outliers.
**Cons:** A candidate that's rank 1 by a huge margin is treated identically to rank 1 by a tiny margin.

### 9.2 Normalized Weighted Sum

Min-max normalize each signal to [0,1] per recording, then combine with tunable weights:

```
norm(x) = (x - min) / (max - min)    per signal, per recording
combined = alpha * norm(hist) + beta * norm(lm) - gamma * norm(del_resid)
```

Negative sign on `del_resid` because positive residual = bad fit.

**Pros:** Interpretable. Grid-searchable.
**Cons:** Linear assumption. Requires tuning.

### 9.3 Log-Linear

```
combined = alpha * hist + beta * lm + gamma * log(1 - del_resid_clipped + eps)
```

Where `del_resid_clipped = max(0, del_resid)` to avoid log of values > 1. Exponential penalty for high deletion residual.

**Pros:** Better-calibrated than linear for probability-like scores.
**Cons:** Slightly harder to interpret.

### 9.4 Two-Stage Filtering

Use signals sequentially:
1. **Stage 1 (Histogram):** Keep candidates where `hist_score > 0` or top-K
2. **Stage 2 (Deletion residual):** Discard candidates with `del_resid > threshold`
3. **Stage 3 (LM):** Rank survivors by LM score

**Pros:** Each signal handles what it's best at. Interpretable.
**Cons:** Hard cutoffs lose information.

### 9.5 Residual-Based Deletion + Weighted Sum

Use the regression residual directly:

```
combined = alpha * norm(hist) + beta * norm(lm) - gamma * del_resid
```

No normalization needed on the residual since it's already centered at 0 across the corpus.

**Pros:** Scale-fair. Simple.
**Cons:** The residual has different variance across scale sizes (pentatonic std ~6%, heptatonic std ~4%).

### 9.6 Deletion Residual as Bayesian Prior

Treat the deletion residual as a prior and the LM score as the likelihood:

```
log_posterior = lm_score + lambda * log(1 - del_resid_normalized)
```

Interpretation: the LM says "how likely is this sequence under raga X's grammar" and the deletion residual says "how likely is it that raga X generated this chromatic content." Multiplying them gives a joint probability.

**Pros:** Principled probabilistic interpretation.
**Cons:** Requires calibrating the lambda to put both terms on comparable scales.

### 9.7 Histogram as Gate, LM + Deletion as Final Score

The histogram is fundamentally different evidence (pitch class distribution, no sequence info). Use it purely as a filter:

- Keep candidates where `hist_score > 0` or top-K by histogram
- Among survivors, rank by `lm_score - penalty * del_resid`

This acknowledges that the histogram is good at eliminating impossible tonics but bad at ranking similar ragas.

**Pros:** Matches observed signal strengths (histogram for tonic, LM+deletion for raga).
**Cons:** The hard gate may discard the correct answer if the histogram is wrong about tonic.

### 9.8 Logistic Regression (Learn Weights from Data)

Treat it as classification. For each (recording, candidate) pair in LOO CV, features are (hist, lm, del_resid) and label is correct/incorrect. Fit:

```
P(correct) = sigmoid(w1 * hist + w2 * lm + w3 * del_resid + b)
```

~297 recordings x ~200 candidates = ~60K training examples. Nested CV needed for honest evaluation.

**Pros:** Empirically optimal weights. No manual tuning.
**Cons:** Risk of overfitting. Requires careful cross-validation setup.

### 9.9 Ensemble Voting

Each signal produces its own top-K ranking. Candidates are scored by how many top-K lists they appear in:

```
votes = (1 if in hist_top10 else 0) + (1 if in lm_top10 else 0) + (1 if in del_top10 else 0)
```

Tie-break by average rank across the lists.

**Pros:** Simple, interpretable, no parameters except K.
**Cons:** Discards all information beyond "top-K or not."

### 9.10 Cascade with Confidence Thresholds

Use each signal only when the previous one isn't confident:

- If histogram's #1 is far ahead of #2 (margin > threshold): accept, done
- If histogram is ambiguous: use deletion residual to filter, then LM to rank
- If everything is ambiguous: report top-3 with uncertainty flag

Mirrors how a human musicologist reasons: "if the scale is obvious, I don't need to analyze the phrases."

**Pros:** Fast for easy cases. Only invokes expensive LM scoring when needed.
**Cons:** Multiple thresholds to tune. Harder to evaluate systematically.

### 9.11 Assessment and Recommendation

From the experiments, the three signals have distinct strengths:

| Signal | Good at | Bad at |
|--------|---------|--------|
| Histogram | Tonic detection, eliminating impossible ragas | Distinguishing ragas that share a scale |
| LM score | Sequential grammar discrimination | Biased toward small-scale ragas (simpler sequences = lower perplexity) |
| Deletion residual | Measuring chromatic fit independent of scale size | Noisy with small sample sizes |

The histogram and deletion residual are partially redundant (both measure pitch class fit). The LM is the only signal using sequential information.

**Recommended path:**
1. Start with **9.7 (histogram gate + LM/deletion)** as the pipeline default -- simple, matches signal strengths
2. Implement **9.8 (logistic regression)** as an analysis tool to empirically validate whether the manual weights are near optimal
3. Use the logistic regression coefficients to inform thresholds for the gated approach

---

## 10. Open Questions

1. **Optimal n-gram order for multi-hypothesis pipeline:** The pipeline applies per-hypothesis correction before LM scoring, so the corrected-data optimal (order 5-6) likely applies. Needs validation.

2. **Training corpus size and artist confounding:** With ~10 recordings per raga, the model may overfit to artist/composition fingerprints. Cross-artist evaluation would quantify this.

3. **Interaction between histogram and LM errors:** Do they fail on the same recordings? If errors are independent, combination helps more. If correlated, less.

4. **Deletion residual regression stability:** The regression coefficients (-0.0684, 0.6640) were fit on 28 ragas. How stable are they with different corpus sizes or different pitch extraction settings?

5. **Computational cost at scale:** The multi-hypothesis pipeline scores ~200-500 (tonic, raga) candidates per recording, each requiring raga correction + tokenization + LM scoring. Currently ~25 seconds total. Acceptable for offline use; may need optimization for real-time applications.
