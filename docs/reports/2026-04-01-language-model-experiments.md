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

### Terminology: Scoring Modes

| Name | Formula | Available in | Notes |
|------|---------|-------------|-------|
| **LM-only** | `model.rank_ragas(phrases)` | LOO CV, pipeline | Pure n-gram perplexity |
| **LM+deletion** | `lm_score - lambda * del_residual` | LOO CV, pipeline | No histogram; testable in LOO |
| **Three-signal pipeline** | `alpha * norm(hist) + beta * norm(lm) - gamma * del_residual` | Pipeline only | Requires histogram from detect; cannot be tested in LOO |

### Limitations

- All LOO accuracies reported are **within-corpus** estimates on the 300-song CompMusic subset. Artist/composition overlap between folds likely inflates these numbers relative to true generalization accuracy.
- The three-signal pipeline formula has been validated on a single unseen recording (Parveen Sultana). Corpus-level evaluation requires running full detect with `--use-lm-scoring` on all recordings.

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

## 10. Experiment 5: Combined LM + Deletion Residual in Pipeline (Parveen Sultana Case Study)

### 10.1 Method

Integrated the three-signal combined scoring into the detect pipeline's `--use-lm-scoring` mode. For each (tonic, raga) candidate from the histogram scorer:

1. Apply raga correction to the chromatic transcription
2. Compute scale-normalized deletion residual: `del_residual = actual_del - (-0.0684 * scale_size + 0.6640)`
3. Score corrected tokens against the raga's LM
4. Apply histogram gate: only candidates with `histogram_score > 0` survive
5. Compute combined score: `alpha * norm(histogram) + beta * norm(lm) - gamma * del_residual`

Where `norm()` is min-max normalization to [0,1] across gated candidates. Default weights: alpha=1.0, beta=1.0, gamma=2.0.

### 10.2 Evolution of the Formula

Three iterations were tested on the Parveen Sultana recording (Puriya Dhanashree, unseen by the model):

**Iteration 1: LM score only (pre-existing)**

| Rank | Raga | LM Score |
|------|------|----------|
| 1 | Hansadhwani (B) | -2.674 |
| 14 | Puriya Dhanashree (A#) | -2.771 |

Puriya Dhanashree ranked 14th. Hansadhwani (pentatonic) dominates because simpler sequences = lower perplexity.

**Iteration 2: LM + deletion residual with histogram gate**

```
combined = lm_score - lambda * del_residual
```

| Rank | Raga | Combined | LM | Del Residual |
|------|------|----------|-----|-------------|
| 1 | Hansadhwani (A#) | -2.449 | -2.798 | -0.174 |
| 2 | Puriya Dhanashree (A#) | -2.472 | -2.771 | -0.150 |

Puriya Dhanashree improved to rank 2, but Hansadhwani still wins because it has both a good LM score AND a good deletion residual (its notes fit well for a pentatonic raga). The histogram gate passes it (score 101 > 0) even though the histogram strongly prefers Puriya Dhanashree (score 700).

**Iteration 3: Three-way combined with normalized histogram**

```
combined = alpha * norm(hist) + beta * norm(lm) - gamma * del_residual
```

| Rank | Raga | Combined | Norm Hist | Norm LM | Del Residual | Histogram |
|------|------|----------|-----------|---------|-------------|-----------|
| 1 | Puriya Dhanashree (A#) | 2.206 | 1.000 | 0.907 | -0.150 | 699.99 |
| 2 | Basant (A#) | 2.118 | 1.000 | 0.818 | -0.150 | 699.99 |
| 3 | Shri (A#) | 2.112 | 1.000 | 0.812 | -0.150 | 699.99 |

Puriya Dhanashree correctly ranked #1. The normalized histogram score (1.0 vs 0.14 for Hansadhwani) overwhelms the small LM advantage, while the deletion residual provides fine-grained discrimination within the histogram's top cluster.

### 10.3 Key Finding

In this case study, all three signals contribute complementary information. The histogram identifies the tonic and eliminates impossible ragas. The LM captures sequential grammar but is biased toward small scales. The deletion residual measures chromatic fit but doesn't distinguish ragas of the same scale size. Corpus-level validation of the three-signal formula is pending (requires running detect with LM scoring on all 297 recordings).

---

## 11. Experiment 6: Combined Scoring in LOO CV (Without Histogram)

### 11.1 Method

Extended the LOO evaluation to use combined LM + deletion residual scoring. For each held-out recording, the model trains on the remaining 296, then for each candidate raga:

1. Applies raga correction to the held-out recording's raw notes
2. Computes scale-normalized deletion residual
3. Tokenizes corrected notes
4. Scores with LM
5. Combined score: `lm_score - lambda * del_residual`

**Critical difference from the pipeline test:** The LOO evaluation does NOT have histogram scores (the histogram comes from the detect pipeline's pitch analysis, not from transcription data). So the histogram gate is absent -- every raga is scored for every recording.

### 11.2 Results

Order 5, lambda=2.0, corrected transcriptions:

| Metric | Pure LM (Exp 3a) | Combined LM+Deletion | Change |
|--------|-------------------|---------------------|--------|
| Top-1 | 96.0% | 79.5% | **-16.5** |
| Top-3 | 99.0% | 92.6% | -6.4 |
| MRR | 0.974 | 0.865 | -0.109 |

**The combined scoring is significantly worse than pure LM without the histogram gate.**

### 11.3 Analysis

The deletion residual, without histogram pre-filtering, introduces more noise than signal. The problem: in LOO, every raga is a candidate for every recording. A recording in raga Yaman (7 notes) gets scored against all 30 ragas including Hansadhwani (5 notes), Bhupali (5 notes), etc. The pentatonic ragas have *negative* deletion residuals (better than expected for their scale size) on many recordings because any recording with strong Sa-Re-Ga-Pa-Dha content will have low deletion for these ragas.

The deletion residual rewards ragas that "fit" the chromatic content, but "fit" is not the same as "correct." A pentatonic raga trivially fits any heptatonic recording (just delete the 2 extra notes). The regression normalization accounts for *average* deletion by scale size but doesn't prevent pentatonic ragas from scoring well on individual recordings where their notes happen to be prominent.

**This validates the gated approach**: the histogram must filter first to eliminate candidates that don't match the tonic/scale, THEN the LM + deletion residual can discriminate among the survivors. Lambda sweep results (running) will confirm the optimal weight.

### 11.4 Implication for Pipeline Design

The three-signal formula used in the pipeline (`alpha * norm(hist) + beta * norm(lm) - gamma * del_residual`) is the correct architecture. The histogram is not just a gate -- it's a first-class scoring signal whose normalized value must be part of the combined score. The LOO evaluation without histogram access cannot fully test this formula; full evaluation requires running detect with `--use-lm-scoring` on all recordings.

---

## 12. Experiment 7: Histogram-Only Baseline

### 12.1 Method

Evaluated the existing detect pipeline's histogram scorer (8-coefficient `RagaScorer`) alone on all 298 CompMusic recordings with existing `candidates.csv` files. Checked whether the correct raga appeared at rank 1 and top-3 in the histogram ranking. Also tested with and without vocal/gender metadata, and with full 78-raga DB vs our 30-raga subset.

### 12.2 Results

| Condition | Top-1 | Top-3 | Tonic Top-1 | Tonic Top-3 |
|-----------|-------|-------|-------------|-------------|
| With metadata, full DB | 48.0% | 65.8% | 62.1% | 91.3% |
| With metadata, 30-raga filter | 50.7% | 67.4% | -- | -- |
| Without metadata, full DB | 48.0% | 65.8% | 62.1% | 91.3% |
| Without metadata, 30-raga filter | 50.3% | 66.1% | -- | -- |

### 12.3 Key Findings

- **Vocal/gender metadata makes no difference** on this all-vocal corpus. The tonic bias from metadata is redundant.
- **Tonic detection is strong** (62% top-1, 91% top-3) but **raga discrimination is weak** (48-51% top-1). The histogram identifies the scale but cannot distinguish ragas that share one.
- Ragas at 0% histogram accuracy: Bageshri, Bhairav, Bihag, Shri -- all share their scale with other ragas in the DB.
- The 30-raga filter gives only a ~2pp improvement, meaning the main confusion is within our subset, not with ragas outside it.

---

## 13. Updated Summary of All Results

| # | Experiment | Condition | Best Top-1 | Best Top-3 | Notes |
|---|-----------|-----------|-----------|-----------|-------|
| 1 | Flat n-grams (LM-only) | Corrected transcriptions | 97.0% | 99.0% | Order 6 |
| 2 | Flat n-grams (LM-only) | Uncorrected transcriptions | 88.6% | 95.6% | Order 4 |
| 3a | Phrase-aware (LM-only) | Corrected transcriptions | 96.6% | 99.0% | Order 6 |
| 3b | Phrase-aware (LM-only) | Uncorrected transcriptions | 87.2% | 94.3% | Order 5 |
| 5 | Three-signal pipeline | Parveen Sultana (unseen) | Rank 1 | -- | Single case study |
| 6 | LM+deletion (LOO, no histogram) | Corrected, lambda=2.0 | 79.5% | 92.6% | Order 5 |
| 6b | LM+deletion (LOO, lambda=0) | Per-hypothesis correction | 88.9% | -- | Order 5; lambda sweep running |
| 7 | Histogram only | With/without metadata | 48-51% | 66-67% | 30-raga filter |

Within-corpus LOO accuracy; artist/composition overlap likely inflates generalization estimates.

---

## 14. Experiment 8: Lambda Sweep for LM+Deletion Mode (Partial)

### 14.1 Method

Swept the deletion-residual weight (lambda) in the LM+deletion LOO evaluation to determine whether any lambda value improves over pure LM scoring without histogram gating. All runs at order 5, corrected transcriptions, min_recordings=3.

In LM+deletion mode, each held-out recording is scored against every candidate raga by: applying that raga's correction to the raw notes, computing the scale-normalized deletion residual, tokenizing, scoring with the LM, and combining as `lm_score - lambda * del_residual`.

Note: even at lambda=0, this mode differs from pure LM (Experiment 3a) because it re-corrects per hypothesis rather than using the pre-corrected CSV. This means the "baseline" for this mode is 88.9%, not 96.6%.

### 14.2 Results

| Lambda | Top-1 Accuracy | Change from lambda=0 |
|--------|---------------|---------------------|
| 0.0 | 88.9% (264/297) | baseline |
| 0.5 | 87.5% (260/297) | -1.4 |
| 1.0 | 85.5% (254/297) | -3.4 |
| 2.0 | 79.5% (236/297) | -9.4 |
| 5.0 | running... | -- |

### 14.3 Key Finding

**Increasing lambda monotonically decreases accuracy.** The deletion residual hurts rather than helps when used without histogram pre-filtering. This is because the deletion residual rewards ragas whose scale happens to cover the recording's prominent pitch classes -- pentatonic ragas trivially "fit" any recording by deleting the extra notes, and their deletion residuals are often better-than-expected (negative), which inflates their combined score.

This confirms that the histogram gate is essential: it must eliminate impossible tonic/raga candidates before deletion residual scoring is applied. The deletion residual is only useful for fine-grained discrimination within a set of candidates that already have reasonable histogram evidence.

---

## 15. Experiment 9: Tonic Sensitivity Analysis

### 15.1 Method

For the 3 recordings with the worst LOO rankings (Malkauns_3 at rank 29, Marva_6 at rank 20, Puriya Dhanashri_7 at rank 9), tested alternative tonics by scoring each recording individually against the trained model with different tonic values.

### 15.2 Results

**Malkauns_3 (GT tonic: A#):**
- Tonic A# with full model (not LOO): rank 1 (score -1.35, runner-up -1.89)
- Tonic D: rank 5 (much worse)

**Marva_6 (GT tonic: D#):**
- Tonic D# with full model: rank 1 (score -1.56)
- Tonic D: rank 1 (score -1.70, slightly worse but still correct)
- Tonic E: rank >5

**Puriya Dhanashri_7 (GT tonic: D):**
- Tonic D with full model: rank 1 (score -1.37)
- Tonic C#: rank 1 (score -1.49, slightly worse)
- Tonic Eb: rank >5

### 15.3 Key Finding

All 3 recordings classify correctly when scored against the full (non-LOO) model with their GT tonics, revealing that these failures are **LOO sensitivity** issues rather than tonic problems. The model's representation of these ragas depends heavily on specific recordings -- removing one tips the balance. This also exposed a concern about memorization vs generalization (the model scoring well on its own training data is not evidence of correctness).

### 15.4 Modified Tonic LOO Evaluation

Re-ran full LOO at order 6 with modified tonics (Malkauns_3: A#->D, Puriya Dhanashri_7: D->C#):

| Metric | Original tonics | Modified tonics |
|--------|----------------|-----------------|
| Top-1 | 97.0% | 97.0% (same) |
| Top-3 | 99.0% | 99.3% (slightly better) |
| MRR | 0.979 | 0.981 |

The tonic changes fixed 2 Puriya Dhanashree errors but broke 2 Basant recordings (Basant and Puriya Dhanashree are extremely similar ragas -- a zero-sum trade).

---

## 16. Development Timeline and Code Changes

### 16.1 Chronological Summary

**2026-03-30 (Saturday): Design and initial implementation**

- Brainstormed the language modelling approach for raga detection (inspired by language identification from romanized text)
- Designed three approaches: (1) character n-gram LMs, (2) variable-length Markov (PPM), (3) discriminative reweighting
- Wrote design spec (`docs/superpowers/specs/2026-03-30-ngram-language-model-raga-detection.md`)
- Wrote implementation plan with 9 tasks
- Implemented all 9 tasks via subagent-driven development:
  - Task 1: Shared tokenizer `tokenize_notes_for_lm` in `sequence.py`
  - Task 2: `NgramModel` class with add-k smoothing and interpolated scoring
  - Task 3: Model serialization round-trip (JSON with counts)
  - Task 4: `train_model` function (corpus training from GT CSV + transcription CSVs)
  - Task 5: `score_transcription` function with segment-level confidence
  - Task 6: `evaluate_model` with LOO CV and order sweep
  - Task 7: CLI entry point (`python -m raga_pipeline.language_model train|score|evaluate`)
  - Task 8: Documentation updates
  - Task 9: Full test suite verification
- Code review found critical octave-marking bug (MIDI octave boundaries vs tonic-relative)
- Fixed octave bug, added non-C tonic tests

**2026-03-31 (Sunday): Corpus evaluation**

- Created filtered GT CSV (`compmusic_gt.csv`: 300 songs, 30 ragas from CompMusic DB)
- Trained model on CompMusic corpus: 297 recordings, 30 ragas, order 5
- Ran LOO evaluation with order sweep (2-6): **97.0% top-1 at order 6**
- Analyzed 9 misclassifications: Bilawal/Desh/Malhar cluster, Puriya Dhanashree confusion, 2 likely bad transcriptions
- Tested alternative tonics on worst recordings
- Discovered raga correction bias: transcriptions were pre-filtered to match the correct raga, inflating accuracy

**2026-04-01 (Monday): Unbiased evaluation and multi-hypothesis pipeline**

- Added `--skip-raga-correction` flag to analyze mode
- Created symlinked output directory for uncorrected transcriptions
- Re-ran analyze on all 297 recordings without raga correction (note counts increased 15-71%)
- Re-evaluated on uncorrected transcriptions: **88.6% top-1** (8-12pp drop from corrected)
- Discovered 49% of n-grams crossed phrase boundaries -- fixed tokenizer to return `List[List[str]]` (phrase-separated)
- Re-evaluated with phrase-aware n-grams: 96.6% corrected, 87.2% uncorrected
- Designed multi-hypothesis detect pipeline (`--use-lm-scoring`)
- Implemented STEP 5.5 in detect: chromatic transcription -> per-candidate raga correction -> tokenize -> LM score
- Tested on Parveen Sultana (YouTube, unseen): LM alone ranked Puriya Dhanashree 14th
- Analyzed deletion rates: strong linear relationship with scale size (R^2=0.77)
- Developed scale-normalized deletion residual: `del_residual = actual_del - (-0.0684 * scale_size + 0.6640)`
- Implemented histogram gate + LM + deletion scoring in pipeline
- Tested three-signal formula: Puriya Dhanashree correctly ranked #1
- Implemented LOO evaluation with LM+deletion mode: 79.5% (worse than pure LM without histogram)
- Wrote comprehensive experiment report

**2026-04-02-04 (Tuesday-Thursday): Audit, fixes, baselines**

- Evaluated histogram-only baseline: **48% top-1, 66% top-3**
- Compared with/without vocal/gender metadata: no difference on all-vocal corpus
- Tonic detection: 62% top-1, 91% top-3
- Received detailed technical audit identifying 7 issues
- Fixed lambda order inconsistency (CLI highest-order-first vs model unigram-first)
- Removed unimplemented kneser-ney from CLI choices
- Renamed LOO "combined" to "lm-deletion" to distinguish from pipeline three-signal scoring
- Fixed train-infer mismatch (added merge_consecutive_notes to STEP 5.5)
- Added early validation for `--lm-model` path
- Cleaned stale docstrings and help text
- Fixed report section numbering and deduplicated open questions
- Added 5 new tests (lambda parsing reversal, lm-deletion smoke test, lambda variation)
- Started lambda sweep (0.0-5.0): increasing lambda monotonically decreases accuracy without histogram gating

### 16.2 Complete Commit Log

| SHA | Date | Description |
|-----|------|-------------|
| `dddb18b` | Mar 30 | docs: n-gram LM design spec |
| `b1af57b` | Mar 30 | docs: implementation plan (9 tasks) |
| `ae3892e` | Mar 30 | feat: tokenize_notes_for_lm shared tokenizer |
| `1c4acdb` | Mar 30 | feat: NgramModel class with counting/smoothing/scoring |
| `a9ff01e` | Mar 30 | feat: train_model for corpus-level training |
| `f28312d` | Mar 30 | feat: score_transcription with segment-level scoring |
| `87d0540` | Mar 30 | feat: evaluate_model with LOO CV and order sweep |
| `ca10bbc` | Mar 30 | feat: CLI entry point (train/score/evaluate) |
| `09f6891` | Mar 30 | fix: mypy type errors in _find_col signature |
| `2251b3a` | Mar 31 | fix: tonic-relative octave boundaries in tokenizer |
| `7c77f19` | Apr 1 | feat: phrase-aware n-grams + --skip-raga-correction flag |
| `fc131f2` | Apr 1 | feat: --use-lm-scoring detect mode with multi-hypothesis re-ranking |
| `66980c1` | Apr 1 | docs: comprehensive experiment report |
| `1539d19` | Apr 1 | docs: deletion rate analysis and 10 combination strategies |
| `2171b23` | Apr 3 | feat: histogram gate + combined LM/deletion scoring |
| `d6269ec` | Apr 3 | feat: normalized histogram score in combined formula |
| `5373872` | Apr 3 | feat: combined scoring mode in LM evaluation (LOO CV) |
| `ad57959` | Apr 3 | docs: pipeline combined scoring and LOO combined results |
| `42272d7` | Apr 4 | fix: audit findings (lambda order, kneser-ney, naming, train-infer) |
| `d932e51` | Apr 4 | docs: histogram-only baseline and metadata comparison |
| `786e224` | Apr 4 | fix: stale references, test coverage for lm-deletion and lambda |
| `8ac5ac1` | Apr 4 | docs: final reference cleanup and open question dedup |

### 16.3 Files Created

| File | Purpose |
|------|---------|
| `raga_pipeline/language_model/__init__.py` | NgramModel, train_model, score_transcription, evaluate_model, LOO functions |
| `raga_pipeline/language_model/__main__.py` | CLI: `python -m raga_pipeline.language_model train\|score\|evaluate` |
| `tests/test_lm_tokenizer.py` | 11 tests for tokenize_notes_for_lm |
| `tests/test_ngram_model.py` | 9 tests for NgramModel class |
| `tests/test_lm_train.py` | 2 tests for train_model |
| `tests/test_lm_score.py` | 2 tests for score_transcription |
| `tests/test_lm_evaluate.py` | 4 tests for evaluate_model (including lm-deletion) |
| `tests/test_lm_cli.py` | 5 tests for CLI (including lambda parsing) |
| `compmusic_gt.csv` | Filtered 300-song CompMusic ground truth |
| `compmusic_ngram_model.json` | v1 trained model (flat n-grams, order 5) |
| `compmusic_ngram_model_v2.json` | v2 trained model (phrase-aware, order 5) |
| `docs/reports/2026-04-01-language-model-experiments.md` | This report |
| `docs/language-model-raga-detection-approaches.md` | Overview of 3 LM approaches |
| `docs/superpowers/specs/2026-03-30-ngram-language-model-raga-detection.md` | Design spec |
| `docs/superpowers/specs/2026-04-01-lm-enhanced-detect-design.md` | LM detect integration spec |
| `docs/superpowers/specs/2026-04-01-multi-hypothesis-detect-design.md` | Multi-hypothesis pipeline spec |
| `run_uncorrected_batch.py` | Batch script for uncorrected transcription runs |
| `run_detect_nometa_batch.py` | Batch script for no-metadata detect runs |

### 16.4 Files Modified

| File | Changes |
|------|---------|
| `raga_pipeline/sequence.py` | Added `tokenize_notes_for_lm` (phrase-separated, octave-marked, tonic-relative) |
| `raga_pipeline/config.py` | Added `skip_raga_correction`, `use_lm_scoring`, `lm_model_path`, `lm_deletion_lambda/slope/intercept`, early validation |
| `driver.py` | Added STEP 5.5 (LM re-ranking with histogram gate + three-signal combined scoring), conditional raga correction skip, `get_raga_notes` import, merge step in STEP 5.5 |
| `CLAUDE.md` | Added language_model module references |

### 16.5 Test Coverage Summary

33 new tests across 6 test files:
- Tokenizer: 11 tests (middle/lower/upper octave, phrase boundaries, empty, komal/shuddha, clipping, non-C tonic, multiple phrases)
- NgramModel: 9 tests (counting, accumulation, smoothing, scoring, serialization round-trip, metadata)
- Training: 2 tests (basic training, min_recordings filter)
- Scoring: 2 tests (correct raga ranks first, segment-level output)
- Evaluation: 4 tests (basic LOO, order sweep, lm-deletion smoke, lambda variation)
- CLI: 5 tests (train+score smoke, evaluate smoke, lambda reversal, lambda None, lambda wrong count)

---

## 17. Updated Summary of All Results

| # | Experiment | Condition | Best Top-1 | Best Top-3 | MRR | Notes |
|---|-----------|-----------|-----------|-----------|-----|-------|
| 1 | Flat n-grams (LM-only) | Corrected transcriptions | 97.0% | 99.0% | 0.979 | Order 6; biased by raga correction |
| 2 | Flat n-grams (LM-only) | Uncorrected transcriptions | 88.6% | 95.6% | 0.924 | Order 4; honest unbiased baseline |
| 3a | Phrase-aware (LM-only) | Corrected transcriptions | 96.6% | 99.0% | 0.977 | Order 6 |
| 3b | Phrase-aware (LM-only) | Uncorrected transcriptions | 87.2% | 94.3% | 0.913 | Order 5 |
| 5 | Three-signal pipeline | Parveen Sultana (unseen) | Rank 1 | -- | -- | Single case study |
| 6 | LM+deletion (LOO, no hist) | Corrected, lambda=2.0 | 79.5% | 92.6% | 0.865 | Deletion hurts without histogram |
| 7 | Histogram only | With/without metadata | 48-51% | 66-67% | 0.61 | Tonic: 62% top-1, 91% top-3 |
| 8 | Lambda sweep (LM+deletion) | lambda=0 to 5.0 | 88.9% (lam=0) | -- | -- | Monotonically decreasing with lambda |

Within-corpus LOO accuracy; artist/composition overlap likely inflates generalization estimates.

---

## 18. Open Questions

1. **Lambda=5.0 result (running):** Final lambda sweep value still computing. Expected to continue the decreasing trend.

2. **Full pipeline evaluation:** The three-signal formula (`alpha * norm(hist) + beta * norm(lm) - gamma * del_residual`) needs corpus-wide testing by running detect with `--use-lm-scoring` on all 297 recordings.

3. **Artist/composition confounding:** Cross-artist grouped CV would quantify how much the 87-97% accuracy reflects genuine raga grammar learning vs artist/composition memorization.

4. **Logistic regression weight optimization:** Fit `P(correct) = sigmoid(w1*hist + w2*lm + w3*del_resid + b)` on three-signal pipeline output to determine optimal weights empirically.

5. **Optimal n-gram order for multi-hypothesis pipeline:** Per-hypothesis correction may favor higher orders (5-6). Needs validation.

6. **Histogram-LM error independence:** Do the histogram and LM fail on the same recordings? If errors are independent, their combination should improve substantially over either alone.

7. **Deletion residual regression stability:** Coefficients (-0.0684, 0.6640) were fit on 28 ragas. Stability across different corpora/settings unknown.

8. **Richer tokenization:** Duration bins, direction markers, ornament proxies could improve LM discriminative power.

9. **Soft histogram gating:** Replace the hard `hist_score > 0` gate with a sigmoid-weighted soft gate to avoid over-pruning when the histogram is uncertain.

1. **Lambda sweep results (running):** Testing lambda values 0, 0.5, 1.0, 2.0, 5.0 in LOO LM+deletion mode. Lambda=0 gives the per-hypothesis-correction baseline (88.9%). Results will determine if there's any lambda value where deletion residual helps without histogram gating.

2. **Full pipeline evaluation:** The three-signal formula needs to be tested on the full corpus by running detect with `--use-lm-scoring` on all 297 recordings. This would provide the histogram scores needed for the three-signal combined formula.

3. **Artist/composition confounding:** With ~10 recordings per raga, the model may overfit to artist/composition fingerprints. Cross-artist grouped CV would quantify this.

4. **Logistic regression weight optimization:** Running Strategy 9.8 (logistic regression on the three signals from the full pipeline) would empirically determine optimal alpha/beta/gamma weights.

5. **Optimal n-gram order for multi-hypothesis pipeline:** The pipeline applies per-hypothesis correction before LM scoring, so the corrected-data optimal (order 5-6) likely applies. Needs validation.

6. **Interaction between histogram and LM errors:** Do they fail on the same recordings? If errors are independent, combination helps more. If correlated, less.

7. **Deletion residual regression stability:** The coefficients (-0.0684, 0.6640) were fit on 28 ragas. How stable are they across different corpus sizes or pitch extraction settings?

8. **Computational cost at scale:** The multi-hypothesis pipeline scores ~200-500 candidates per recording. Currently ~25 seconds total. Acceptable offline; may need optimization for real-time use.
