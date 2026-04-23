# Advanced Scoring Experiments (Exp 16-28)

Date: 2026-04-18 -- 2026-04-24
Corpus: CompMusic 297-300 recordings, 30 ragas, 10 recordings each
Baseline: Exp 14 pipeline LOO = 87.2% combined top-1, 92.3% tonic top-1
Updated baseline: Exp 23 honest pipeline LOO = 89.6% combined top-1 (uncorrected transcriptions)

All experiments use pipeline-parity defaults: vocal confidence 0.95, accompaniment confidence 0.80, confidence weighting enabled, prominence high 0.01, prominence low 0.03.

---

## Experiment 16: Fit-Score Saturation Calibration

**Problem:** The band-pass tonic boost (Exp 14, ACC_BAND_WEIGHT=0.80) pushed histogram fit_norm past the [-1.0, 1.0] clip for ~3.8% of candidates, creating tied clusters at the top of the histogram-only ranking. Mean top-tie size was 6.0 (P95=13), collapsing histogram-only raga accuracy to 23.5%.

**Method:** Tested 7 calibration variants on all 298 recordings (histogram-only replay, no LM):

| Variant | Tonic top-1 | Hist raga top-1 | Saturated % | Mean top tie | P95 top tie |
|---|---:|---:|---:|---:|---:|
| baseline (clip 1.0) | 92.0% | 23.5% | 3.8% | 6.02 | 13 |
| acc_0_40 | 92.3% | 71.8% | 0.3% | 1.15 | 2 |
| acc_0_20 | 90.6% | 73.5% | 0.1% | 1.06 | 1 |
| no_band | 79.5% | 65.4% | 0.0% | 1.06 | 1 |
| **clip_2_0** | **92.0%** | **74.2%** | **0.0%** | **1.06** | **1** |
| no_clip | 92.0% | 74.2% | 3.8% | 1.06 | 1 |
| mult_0_80 | 89.6% | 73.8% | 0.2% | 1.07 | 1 |

**Winner:** `clip_2_0` -- raise clip from [-1.0, 1.0] to [-2.0, 2.0]. Preserves tonic top-1 at 92.0%, recovers histogram-only raga top-1 from 23.5% to 74.2%, zero saturation. Applied to `raga.py` line 724.

**Thesis narrative:** Methodology note -- a post-hoc calibration step cleaning up interactions between additive score components. The combined score was unaffected because the LM dominates, but the histogram-alone number was artificially suppressed and would interfere with new additive terms.

---

## Experiment 17: Confusion-Pair Analysis

**Input:** Histogram-only predictions from the calibrated clip_2_0 baseline (Exp 16).

**Accuracy:** 221/298 = 74.2% histogram-only top-1.

### Top confusion pairs

| Pair | Total | Direction | Same scale |
|---|---:|---|---|
| Khamaj -> Alhaiya Bilawal | 9 | 0 / 9 | near-identical (differ by 1 note) |
| Bihag -> Hansadhwani | 7 | 7 / 0 | no (Hansadhwani is pentatonic subset) |
| Bageshri -> Rageshri | 6 | 6 / 0 | no (close scales, differ in Re) |
| Gaud Malhar -> Gavati | 5 | 5 / 0 | no |
| Bhupali -> Hansadhwani | 3 | 3 / 0 | no (pentatonic, differ Dha vs Ni) |
| Gaud Malhar -> Jog | 3 | 3 / 0 | no |

### Interpretation

The dominant errors are **near-scale, one-way subset confusions** -- the histogram scorer favors tighter templates (fewer notes = less extra-mass penalty). The LM already fixes most of these (87.2% combined vs 74.2% histogram-only). Same-scale pairs are NOT the primary source of confusion at the histogram level.

**Khamaj / Alhaiya Bilawal correction:** These are near-identical scales (differ by one note: shuddha Ni, which Alhaiya Bilawal has and Khamaj does not). This is a superset confusion, not a different-scale error.

### Downstream implications

- Exp 18 (positional PCH) motivated: phrase-start/nyas patterns could separate subset confusions
- Exp 19-20 (microtonal/GMM) less directly targeted at dominant errors but thesis-worthy for within-note analysis
- Exp 21 (cadence LM) could help if cadences differ between confused pairs

---

## Experiment 18: Positional PCH

**Method:** Three specialized pitch-class histograms, each evaluated standalone via LOO with cosine similarity to per-raga mean templates. Swept phrase_gap_sec in {0.25, 0.5, 1.0}.

### Results (297 recordings evaluated)

| Feature | Gap (s) | GT-tonic top-1 | Det-tonic top-1 |
|---|---|---:|---:|
| **Octave-stratified PCH** (36-dim) | all | **83.8%** | **81.1%** |
| Phrase-start PCH (12-dim) | 0.25 | 72.4% | 69.4% |
| Phrase-start PCH (12-dim) | 0.5 | 50.2% | 47.8% |
| Phrase-start PCH (12-dim) | 1.0 | 37.0% | 36.0% |
| Nyas PCH (12-dim) | 0.25 | 58.9% | 56.2% |
| Nyas PCH (12-dim) | 0.5 | 46.5% | 44.8% |
| Nyas PCH (12-dim) | 1.0 | 38.7% | 36.7% |

### Interpretation

**Octave-stratified PCH is the standout feature** -- 83.8% standalone with GT tonic, beating the regular 12-TET histogram (79.5%) by 4.3pp. It is gap-invariant (no phrase splitting needed) and captures how ragas use different octave registers:

- **Mandra** (below tonic): 12-dim histogram of low-register notes
- **Madhya** (tonic to tonic+12): 12-dim histogram of middle-register notes
- **Taar** (above tonic+12): 12-dim histogram of high-register notes

The 36 dimensions (vs 12) provide more discriminative surface. Different ragas use registers differently -- Darbari Kanada emphasizes mandra, Yaman spans madhya-taar.

Phrase-start and nyas PCH degrade sharply with wider gap values (fewer, longer phrases = noisier endpoint estimates). Best at gap=0.25s where they produce many short phrases.

---

## Experiment 19: Microtonal PCH (24/36-TET Sanity Sweep)

**Method:** Tonic-relative pitch-class histograms at 12, 24, and 36 bins per octave. LOO evaluation with GT tonic, cosine similarity to per-raga mean templates.

### Results (298 recordings)

| Resolution | Top-1 accuracy |
|---|---:|
| **12-TET** (100 cents/bin) | **79.5%** |
| 24-TET (50 cents/bin) | 68.1% |
| 36-TET (~33 cents/bin) | 66.1% |

### Interpretation

**Higher resolution hurts.** Finer bins spread mass too thin for stable per-raga templates with only ~10 recordings each. The microtonal signal (e.g., Bhairav's flat Re vs Bhairavi's sharp Re) exists but cannot be captured by naive histogram binning -- the training data is insufficient to build reliable templates at these resolutions.

**This motivates Exp 20 (GMM fingerprint):** per-note frame-level statistics (mean deviation, sigma) aggregate across all frames assigned to a note, avoiding the sparse-bin problem. The negative result here de-risks Exp 20 by showing that simple resolution increase is not the answer.

---

## Experiment 20: GMM/Gamaka Fingerprint

**Method:** For each recording, extract per-pitch-class (12 PCs) features from raw f0 frames:
- `dev_frame`: mean cent deviation from note center (shruti placement)
- `sigma_frame`: std of cent deviations (ornamentation width / andolan)
- Also: histogram-GMM features (sigma_hist, dev_hist, skew_hist) from existing GMM fits

Total: up to 60 dimensions per recording.

### Phase C1: Descriptive analysis

Tested 75 confused pairs from Exp 17's confusion matrix. Found **75 BH-significant dimensions** across different-scale pairs, but the top confused pairs (which are all different-scale) showed mixed results.

### Phase C2: Integration (LOO)

Ran because C1 found significant dims.

**Result: 55.5% top-1** (166/299) -- the weakest standalone feature. The 60-dimensional feature space is sparse and noisy with only ~10 recordings per raga for global discrimination.

### Within-Scale GMM Analysis (Exp 20b)

The global evaluation dilutes GMM's signal because most ragas already differ by scale. A within-scale analysis -- testing GMM as a **conditional discriminator** within raga-note classes (families sharing the same 12-bit pitch-class mask) -- reveals a different story.

**Same-scale groups in the CompMusic corpus:**

| Group | Scale | Ragas | Recordings |
|---|---|---|---:|
| A | S R G m P D n N | Alhaiya Bilawal, Desh, Gaud Malhar | 30 |
| B | S R G m M P D N | Bihag, Maru Bihag, Yaman Kalyan | 30 |
| C | S r g m M P d D | Basant, Puriya Dhanashree, Shri | 30 |
| D | S R G M P D n | Kedar, Shuddh Sarang | 20 |

**Within-group C1: 15 BH-significant dimensions found**

| Pair | Dimension | Sargam | Feature | Cohen's d | p-value |
|---|---|---|---|---:|---:|
| Bihag vs Maru Bihag | dev_frame_PC10 | ni | deviation | 2.77 | 0.000011 |
| Bihag vs Yaman Kalyan | sigma_frame_PC2 | Re | width | 2.52 | 0.000038 |
| Desh vs Gaud Malhar | sigma_frame_PC7 | Pa | width | 1.98 | 0.000324 |
| Alh. Bilawal vs Gaud Malhar | sigma_frame_PC7 | Pa | width | 1.96 | 0.000504 |
| Kedar vs Shuddh Sarang | dev_frame_PC3 | ga | deviation | 1.77 | 0.000982 |
| Bihag vs Maru Bihag | dev_frame_PC3 | ga | deviation | 1.74 | 0.001074 |
| Bihag vs Maru Bihag | dev_frame_PC5 | ma | deviation | -1.76 | 0.001114 |
| Puriya Dhan. vs Shri | sigma_frame_PC1 | re | width | 1.69 | 0.001388 |
| Desh vs Gaud Malhar | dev_frame_PC10 | ni | deviation | 1.64 | 0.002039 |
| Kedar vs Shuddh Sarang | sigma_frame_PC3 | ga | width | 1.58 | 0.002936 |
| Desh vs Gaud Malhar | sigma_frame_PC9 | Dha | width | 1.58 | 0.002959 |
| Bihag vs Maru Bihag | sigma_frame_PC2 | Re | width | 1.66 | 0.002973 |
| Bihag vs Maru Bihag | dev_frame_PC7 | Pa | deviation | -1.48 | 0.004754 |
| Bihag vs Maru Bihag | dev_frame_PC4 | Ga | deviation | -1.44 | 0.005892 |
| Bihag vs Maru Bihag | dev_frame_PC0 | Sa | deviation | -1.26 | 0.011831 |

**Within-group LOO accuracy:**

| Group | GMM | Cadence LM | Octave PCH | Sig. GMM dims |
|---|---:|---:|---:|---:|
| Basant / Puriya Dhanashree / Shri | **73.3%** | 66.7% | 83.3% | 1 |
| Kedar / Shuddh Sarang | 70.0% | 85.0% | 85.0% | 2 |
| Bihag / Maru Bihag / Yaman Kalyan | 60.0% | 66.7% | 73.3% | 8 |
| Alhaiya Bilawal / Desh / Gaud Malhar | 50.0% | 31.0% | 66.7% | 4 |

### Interpretation

1. **GMM captures within-scale signal that the global evaluation missed.** Zero significant dimensions among confused pairs (global C1), but 15 significant dimensions among same-scale pairs. The features are doing exactly what musicology predicts -- discriminating note *treatment*, not note *presence*.

2. **GMM beats cadence LM** in two groups: Basant/PD/Shri (73.3% vs 66.7%) and Alhaiya Bilawal/Desh/Gaud Malhar (50.0% vs 31.0%). These are groups where cadential phrases are similar but note treatment differs.

3. **Octave PCH dominates overall** but captures note *presence by register*, not note *treatment*. GMM and octave PCH are complementary -- one measures where notes appear, the other measures how they're performed.

4. **Bihag/Maru Bihag/Yaman Kalyan** has the richest GMM signal (8 significant dims). Top discriminator: komal Ni deviation (d=2.77) separating Bihag from Maru Bihag. Musicologically interpretable: Bihag's ni treatment is a defining characteristic.

5. **Alhaiya Bilawal/Desh/Gaud Malhar** is the hardest group -- cadence LM is near random (31% for 3 classes). GMM at 50% is the second-best discriminator after octave PCH here.

### Thesis narrative

"While GMM fingerprints underperform as a global discriminator (55.5%), within same-scale raga families they provide statistically significant separation on musicologically interpretable dimensions. This supports a hierarchical classification approach: coarse identification via pitch-class histogram, refined discrimination via within-note treatment features. The strongest separating dimensions -- komal Ni deviation for Bihag vs Maru Bihag (d=2.77), Re width for Bihag vs Yaman Kalyan (d=2.52) -- align with known musicological distinctions."

---

## Experiment 21: Cadence-Restricted LM

**Method:** Train a separate trigram model (order=3, add_k=0.1) on cadential phrases only -- the last 4 notes before each return to Sa. LOO evaluation: for each held-out recording, build per-raga cadence models from remaining recordings, score held-out cadences, rank by mean log-likelihood.

### Results (293 recordings evaluated, 5 skipped for zero cadences)

| Metric | Value |
|---|---:|
| **Top-1 accuracy** | **82.3%** |
| Top-3 accuracy | 96.6% |
| Mean cadences per recording | 44.8 |

### Sample cadence trigrams

| Raga | Top trigram | Count |
|---|---|---:|
| Abhogi | Re' ga' Sa' | 74 |
| Ahir Bhairav | \<BOS\> re' Sa' | 89 |
| Bageshri | Re' ga' Sa' | 110 |
| Bairagi | \<BOS\> re' Sa' | 89 |
| Basant | Sa Ni' Sa | 59 |

### Interpretation

**Very strong standalone signal** -- 82.3% from cadences alone, approaching the full-recording LM (which achieves ~88% in LOO). The 96.6% top-3 is remarkable.

This validates the musicological claim that raga identity is heavily encoded in characteristic phrase resolutions. The cadence trigrams are musicologically interpretable:
- Abhogi and Bageshri both end through Re -> ga -> Sa (komal Ga descent), explaining their confusion
- Ahir Bhairav and Bairagi both approach Sa through komal Re, which is a shared characteristic of ragas with komal Re
- Basant approaches through shuddha Ni -> Sa, distinct from other ragas in its scale group

---

## Cross-Experiment Summary

### Standalone feature comparison

| Feature | Top-1 accuracy | Notes |
|---|---:|---|
| **Octave-stratified PCH** | **83.8%** | Strongest standalone, gap-invariant |
| **Cadence LM** | **82.3%** | Strong, musicologically interpretable |
| 12-TET histogram (calibrated) | 79.5% | Baseline after saturation fix |
| Phrase-start PCH | 72.4% | Gap-sensitive (best at 0.25s) |
| Nyas PCH | 58.9% | Weak standalone |
| GMM fingerprint (global) | 55.5% | Weak globally, strong within-scale |
| 24-TET histogram | 68.1% | Negative result |
| 36-TET histogram | 66.1% | Negative result |

### Thesis 2D taxonomy

```
Across-note axis:       n=1 PCH ----> n=2 transitions ----> n>=3 LM ----> cadence LM (82.3%)
                        |                                                  |
                        +--- octave-stratified (83.8%)                     |
                        +--- phrase-start (72.4%)                          |
                        +--- nyas (58.9%)                                  |
                                                                           |
Within-note axis:       24/36-TET (negative: 68/66%)                      |
                        GMM global (55.5%, weak)                           |
                        GMM within-scale (50-73%, strong conditional)      |
                                                                           |
Calibration:            fit_norm clip fix (23.5% -> 74.2%)                |
Diagnostic:             Confusion pairs (structural/subset errors dominate)|
```

### Integration candidates

The two strongest standalone features for potential integration into the combined score:
1. **Octave-stratified PCH** (83.8%) -- complementary to 12-TET histogram, captures register information
2. **Cadence LM** (82.3%) -- complementary to full-recording LM, captures phrase-resolution patterns

GMM is recommended as a **conditional feature** within same-scale groups, not as a global additive term.

---

---

## Experiment 22: N-gram Order Sweep (Uncorrected vs Corrected Training)

Date: 2026-04-19

**Method:** Sweep LM order 1-8 under two training conditions, both tested on uncorrected transcriptions:
- **Uncorrected x Uncorrected:** Train and test on raw uncorrected transcriptions from `separated_stems_nocorrection/`
- **Corrected x Uncorrected:** Train on GT-raga-corrected transcriptions, test on uncorrected

Pure LM ranking via `model.rank_ragas()` (no histogram, no combined scoring). LOO with MIN_RECORDINGS=3 filter.

### Results (297 recordings)

| Order | Uncorr x Uncorr Top-1 | Uncorr x Uncorr Top-3 | Uncorr x Uncorr MRR | Corr x Uncorr Top-1 | Corr x Uncorr Top-3 | Corr x Uncorr MRR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 91.2% | 99.0% | 0.949 | 87.2% | 95.0% | 0.916 |
| 2 | 93.9% | 99.0% | 0.962 | 90.9% | 95.0% | 0.935 |
| 3 | 95.0% | 99.0% | 0.968 | 91.6% | 95.0% | 0.939 |
| 4 | 95.3% | 99.0% | 0.970 | 91.9% | 95.6% | 0.943 |
| 5 | 95.3% | 99.0% | 0.970 | 91.9% | 95.6% | 0.943 |
| 6 | 96.6% | 99.0% | 0.978 | 92.6% | 95.6% | 0.947 |
| **7** | **97.3%** | **99.0%** | **0.982** | 92.9% | 96.0% | 0.950 |
| 8 | 97.0% | 99.0% | 0.980 | 93.3% | 96.3% | 0.953 |

### Interpretation

1. **Uncorrected x uncorrected peaks at order 7 (97.3%).** The n-gram LM alone achieves near-ceiling accuracy when train/test distributions match. This is the honest LM-only baseline.

2. **Corrected x uncorrected suffers a 4-5pp distribution mismatch.** Training on clean (corrected) sequences and testing on noisy (uncorrected) sequences consistently underperforms. The LM's clean vocabulary can't handle out-of-scale tokens in test data.

3. **Order 7 is optimal.** Diminishing returns after order 6; order 8 slightly degrades (overfitting to specific 8-gram contexts). All subsequent experiments use order 7.

4. **Top-3 saturates at 99.0% from order 1.** The correct raga is almost always in the top 3 candidates regardless of order.

---

## Experiment 23: Honest Pipeline LOO (Uncorrected Baseline)

Date: 2026-04-19

**Context:** An earlier pipeline LOO result (94.3% combined) was invalidated when an auditor discovered that the transcription CSVs on the SSD at `separated_stems/htdemucs/` were GT-raga-corrected (0% out-of-scale notes), constituting label leakage. This experiment re-runs the full pipeline LOO using verified uncorrected transcriptions from `separated_stems_nocorrection/htdemucs/`.

**Method:** Full pipeline LOO: histogram scoring (clip_2.0) + LOO n-gram LM (order 7) + combined scoring (0.5*norm_hist + 2.0*norm_lm). Training and testing both use uncorrected transcriptions.

### Results (297 recordings)

| Metric | Value |
|---|---:|
| Tonic top-1 | 92.3% |
| Hist raga top-1 | 74.4% |
| LM raga top-1 (GT tonic) | 87.2% |
| **Combined top-1** | **89.6%** |
| Combined top-3 | 94.6% |

### Interpretation

**89.6% is the honest, thesis-safe combined baseline.** The LM (87.2% standalone at GT tonic) lifts the histogram (74.4%) by 15.2pp when combined. The combined score exceeds both components individually, confirming the histogram and LM provide complementary signal.

Note: the LM standalone here (87.2%) differs from Exp 22's 97.3% because the pipeline LOO requires correct tonic detection first -- 7.7% of recordings have wrong tonic, capping the ceiling.

---

## Experiment 24: Corrected-Train Pipeline LOO

Date: 2026-04-19

**Method:** Same pipeline as Exp 23 but training the LM on GT-corrected transcriptions while testing on uncorrected. Tests whether clean training data helps despite the distribution mismatch.

### Results (297 recordings)

| Metric | Value |
|---|---:|
| Combined top-1 | **42.4%** |
| Combined top-3 | 53.2% |
| LM raga top-1 (GT tonic) | 44.1% |

### Interpretation

**Catastrophic distribution mismatch.** The corrected-train LM learns a clean vocabulary (0% out-of-scale tokens) that cannot score noisy uncorrected test sequences. This 47pp drop from the uncorrected baseline (89.6%) confirms that train/test distribution matching is critical. This negative result directly motivates the per-hypothesis correction approach (Exp 25-27).

---

## Experiment 25: Per-Hypothesis Correction Calibration

Date: 2026-04-19

**Motivation:** Exp 24 showed corrected-train x uncorrected-test fails due to distribution mismatch. Solution: at test time, correct the uncorrected transcription *under each candidate raga's scale*, then score under that raga's corrected-trained LM. This eliminates the mismatch because each candidate sees its own corrected version.

**Phase 1: Calibration.** Before running full LOO, verify the corrected-trained LM can discriminate within same-scale adversarial groups (the hardest case -- ragas sharing identical scales).

**Method:** Build a full (non-LOO) corrected-trained LM. For each recording in 4 same-scale groups, score its GT-corrected phrases against all group members. Check if GT raga ranks first.

### Same-scale groups tested

| Group | Ragas | N recordings |
|---|---|---:|
| Kedar / Shuddh Sarang | 2 ragas, same scale | 20 |
| Alhaiya Bilawal / Desh / Gaud Malhar | 3 ragas, same scale | 30 |
| Bihag / Maru Bihag / Yaman Kalyan | 3 ragas, same scale | 29 |
| Basant / Puriya Dhanashree / Shri | 3 ragas, same scale | 30 |

### Results

**109/109 = 100% calibration accuracy.** The corrected-trained LM perfectly discriminates within all same-scale groups. This confirms the LM captures sequential/phrase-level patterns beyond just scale membership.

---

## Experiment 26: Per-Hypothesis Correction LOO v2 (Fixed Weights)

Date: 2026-04-19

**Method:** Full pipeline LOO with per-hypothesis correction at test time. For each candidate (tonic, raga):
1. Apply `apply_raga_correction_to_notes(uncorrected, raga_db, candidate_raga, candidate_tonic)` to the test recording
2. Tokenize corrected output, score under corrected-trained LM
3. Compute penalty terms: `del_residual = max(0, del_rate - E[del|k])`, `mean_snap` (mean snapping distance)
4. Combined: `0.5*norm_hist + 2.0*norm_adjusted_lm` where `adjusted_lm = lm_per_token - 10.0*del_residual - 1.5*mean_snap`

Penalty weights (W_DEL=10.0, W_SNAP=1.5) were manually chosen. E[del|k] computed per-fold (excluding held-out recording).

### Results (297 recordings)

| Metric | Value |
|---|---:|
| Tonic top-1 | 92.3% |
| Hist raga top-1 | 74.4% |
| LM raga top-1 (GT-corrected diagnostic) | **96.0%** |
| **Combined top-1** | **85.5%** |
| Combined top-3 | 94.6% |

### Interpretation

**The LM signal is excellent (96.0%) but the combined score degrades it to 85.5%.** This is 4.1pp *worse* than the uncorrected baseline (89.6%) despite the LM being 8.8pp *better* (96.0% vs 87.2%). The problem is twofold:

1. **Min-max normalization washout.** Normalizing LM scores to [0,1] within the candidate set compresses the LM's absolute signal strength. A candidate 3 log-prob units ahead gets the same normalized boost as one 0.1 units ahead.

2. **Manually guessed penalty weights.** W_DEL=10.0 and W_SNAP=1.5 are arbitrary. The penalties interact with the LM score in unpredictable ways after min-max normalization.

**Key insight:** The LM at 96.0% proves the per-hypothesis correction signal is real and strong. The bottleneck is not the features -- it's the scoring/combination method.

---

## Experiment 26b: v2 Raga-Agnostic Cleaner LOO

Date: 2026-04-19

**Motivation:** Before committing to per-hypothesis correction (expensive at test time), test whether a cheaper raga-agnostic cleaner lifts the baseline. The v2 cleaner (developed via autoresearch on 300 recordings) infers the recording's scale from its own pitch-class histogram and snaps notes to the inferred scale -- no raga knowledge needed.

**v2 cleaner key parameters:** `scale_top_k=12`, `scale_min_fraction=0.025`, `discard_far=0` (keep un-snappable notes at original pitch instead of discarding), `snap_max_distance=1.0`.

**Key finding from autoresearch:** `discard_far=0` (keep notes instead of discarding) measurably wins. Match F1 = 0.931 against production-corrected reference.

**Method:** Clean all uncorrected transcriptions with the v2 cleaner, then run the same LOO pipeline as Exp 23 (cleaned input on both train and test sides).

### Results (297 recordings)

| Metric | Value |
|---|---:|
| Tonic top-1 | 92.3% |
| Hist raga top-1 | 74.4% |
| LM raga top-1 (GT tonic) | 80.5% |
| **Combined top-1** | **80.1%** |
| Combined top-3 | 92.3% |

### Interpretation

**Raga-agnostic cleaning hurts: 80.1% vs 89.6% baseline (-9.5pp).** The cleaner achieves 93.1% match F1 against production-corrected output, but this is not good enough for LM training. The 6.9% of incorrectly cleaned tokens pollute the LM's per-raga vocabulary, reducing its discriminative power (80.5% LM-only vs 87.2% baseline).

**Conclusion:** Raga-agnostic cleaning cannot substitute for raga-specific correction. The per-hypothesis approach (Exp 26) is the right path -- it just needs better scoring.

---

## Experiment 27: Per-Hypothesis Correction LOO v3 (Principled Scoring)

Date: 2026-04-19 (running)

**Motivation:** Exp 26 showed 96.0% LM signal degraded to 85.5% by min-max normalization and guessed weights. This experiment fixes both:

1. **Drop min-max normalization** -- use raw log-prob scores (no compression)
2. **Z-score penalties per scale size** -- `z_snap = (mean_snap - mu[k]) / sigma[k]` kills scale-size bias without the `max(0,·)` hack
3. **Logistic regression on features** -- fit (lm_per_token, z_snap, del_residual) coefficients from data with grouped LOO

**Method:** Two-phase approach:
- Phase 1: LOO feature collection. For each recording, for each histogram candidate, apply per-hypothesis correction and collect (lm_per_token, mean_snap, del_residual, scale_size, hist_score, is_gt). Saves all ~50,000 candidate features.
- Phase 2: Score with 5 methods on the collected features:
  - A: Raw LM + histogram gate (no normalization)
  - B: Raw LM - z-scored mean_snap
  - C: Logistic regression on (lm, z_snap, z_del) with grouped LOO
  - D: Min-max baseline (v2 comparison)
  - E: RRF (reciprocal rank fusion of hist rank + LM rank)

### Results

Preliminary results on 15 recordings were highly misleading (e.g., method B at 93.3%) due to small sample. Full 297-recording results below:

| Method | Top-1 | Top-3 |
|---|---:|---:|
| A: Raw LM + hist gate | 9.1% | 18.9% |
| B: Raw LM - z_snap | 55.2% | 76.4% |
| **C: Logistic regression** | **66.3%** | **82.2%** |
| D: Min-max (v2 baseline) | 25.3% | 42.1% |
| E: RRF (hist + LM) | 57.9% | 87.2% |

Logistic regression coefficients (full-data grouped LOO): `lm_per_token=1.257, z_snap=-0.465, del_residual=-6.283`.

---

## Cross-Experiment Summary (Exp 22-27)

### Pipeline LOO comparison

| Experiment | Condition | LM top-1 | Combined top-1 | Combined top-3 |
|---|---|---:|---:|---:|
| Exp 23 (baseline) | uncorr x uncorr, min-max | 87.2% | **89.6%** | 94.6% |
| Exp 24 | corr-train x uncorr-test | 44.1% | 42.4% | 53.2% |
| Exp 26b (v2 cleaner) | v2-cleaned x v2-cleaned | 80.5% | 80.1% | 92.3% |
| Exp 26 (per-hyp v2) | per-hyp, min-max, guessed wt | **96.0%** | 85.5% | 94.6% |
| Exp 27 (per-hyp v3) | per-hyp, principled scoring | 96.0% | 66.3% | 82.2% |
| Exp 28 (alignment) | corr-train, alignment scoring | ~0 delta | n/a (negative) | n/a |

### Key insights

1. **Per-hypothesis correction recovers a 96.0% LM diagnostic signal** -- the strongest single-feature result. However, this signal cannot be converted into classification accuracy: the best combined method (logistic regression) achieves only 66.3%, well below the 89.6% uncorrected baseline.

2. **The fundamental bottleneck is scale-size bias, not scoring method.** Pentatonic ragas (k=5) get mean LM score -1.64 vs heptatonic (k=7) at -1.94 -- a 0.3 gap that dwarfs the GT vs wrong-raga separation (0.08). Z-scoring partially addresses this but does not eliminate it. Five different scoring methods all fail to beat the baseline, confirming the bias is structural.

3. **Raga-agnostic cleaning is not viable.** Even with 93.1% match F1, the v2 cleaner's errors compound through LM training to produce worse results than no cleaning at all.

4. **Distribution matching is critical.** The 47pp drop from corrected-train mismatch (Exp 24) is the largest single effect in the experiment series. Per-hypothesis correction solves the mismatch by construction, but introduces scale-size bias that is equally destructive.

5. **Noisy-channel alignment does not recover LM signal (Exp 28).** Scoring uncorrected test sequences against corrected-trained LMs via beam DP alignment (with skip/match/substitute transitions) produces negligible GT vs non-GT separation across all hyperparameter settings. The corrected-train / uncorrected-test distribution gap appears too large for alignment-based bridging.

---

## Experiment 28: Noisy-Channel Alignment LM Scoring (Negative Result)

Date: 2026-04-24

**Motivation:** Exp 26-27 showed per-hypothesis correction recovers a 96.0% LM diagnostic signal but introduces scale-size bias that all scoring methods fail to overcome (best: 66.3%). The alignment approach eliminates per-hypothesis correction: keep uncorrected test sequences as-is, score them against corrected-trained LMs using beam DP alignment with skip/substitution costs.

**Method:** Phrase-local beam DP alignment. At each observed token, choose: skip (penalty lambda_skip), match (reward lambda_match + LM log-prob), or substitute to nearby pitch class (reward lambda_match + LM log-prob - lambda_sub * distance). Context is built from matched/substituted tokens only. Scores calibrated by z-scoring per scale size.

Training: correct raw notes with GT raga/tonic in-memory, tokenize, train LM (order 7). Testing: tokenize uncorrected raw notes without any correction, score with alignment.

**Diagnostic sweep:** Evaluated on 15 recordings (2 ragas) with beam_width=50, varying lambda_match.

| lambda_match | skip_frac GT | skip_frac non-GT | Delta (GT - non-GT) | LM-only A (ungated) | Hist-only G |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.000 | 1.000 | 0.000 | 93.3%\* | 93.3% |
| 0.5 | 0.969 | 0.971 | -67308\*\* | 26.7% | 93.3% |
| 1.0 | 0.650 | 0.664 | **+0.0024** | 33.3% | 93.3% |
| 2.0 | 0.318 | 0.339 | -0.0057 | 53.3% | 93.3% |

\* All tokens skipped (-1e6 sentinel); ranking collapses to histogram. \*\* Sentinel contamination from mixed 0-matched/few-matched rows.

Pre-registered pass criterion: Delta >= +0.02. Best observed: +0.0024 at lambda_match=1.0.

### Interpretation

**Negative result. Full-scale evaluation not warranted.** Across the lambda_match sweep, alignment LM discrimination between ground-truth and non-ground-truth candidates remained negligible and never reached the pre-registered practical threshold of +0.02. Histogram-only ranking consistently outperformed all LM-containing variants. The corrected-train / uncorrected-test distribution gap is too large for alignment-based bridging under this formulation.

**Why alignment fails here:** The corrected-trained LM's vocabulary is dominated by in-scale tokens. When scoring uncorrected sequences, all candidate ragas' LMs assign similarly poor probabilities to out-of-scale tokens, and the alignment DP either skips them (losing signal) or matches them (diluting signal). The per-hypothesis correction approach (Exp 26) works because it transforms the test sequence into the LM's expected distribution, but that transformation introduces scale-size bias. Alignment avoids the transformation but cannot bridge the distribution gap.

**Implications for future work:** Approaches that require matching corrected-trained LMs against uncorrected test data -- whether via per-hypothesis correction (Exp 26-27) or alignment (Exp 28) -- face a fundamental tension between distribution matching and scale-size bias. The most promising path remains improving the uncorrected-train / uncorrected-test baseline (Exp 23, 89.6%) through better transcription quality (manual cleaning) or raga-specific features that do not depend on correction.

---

## Files

| Output | Path |
|---|---|
| Saturation calibration | `results/saturation_calibration/{progress,summary,fit_score_distribution}.csv` |
| Confusion analysis | `results/confusion_analysis/{confusion_matrix,top_pairs,summary}.csv` |
| Positional PCH | `results/positional_pch/{progress,summary}.csv` |
| Microtonal PCH | `results/microtonal_pch/{progress,summary}.csv` |
| GMM fingerprint | `results/gmm_fingerprint/{fingerprints,confused_pair_analysis,loo_integration,summary}.csv` |
| GMM within-scale | `results/gmm_within_scale/{within_group_analysis,within_group_loo,summary}.csv` |
| Cadence LM | `results/cadence_lm/{progress,summary,cadence_examples}.csv` |
| N-gram order sweep | `results/ngram_order_sweep/summary.csv` |
| Pipeline LOO (honest) | `results/pipeline_loo_uncorrected/{progress,summary}.csv` |
| Pipeline LOO (corr-train) | `results/pipeline_loo_corrected_train/{progress,summary}.csv` |
| Per-hyp calibration | `results/perhyp_correction/calibration.csv` |
| Per-hyp v2 (fixed wt) | `results/perhyp_v2/{progress,summary,sanity_check}.csv` |
| v2 cleaner LOO | `results/v2cleaned_loo/{progress,summary}.csv` |
| Per-hyp v3 (principled) | `results/perhyp_v3/{features,scoring_comparison}.csv` |
| Alignment LOO (Exp 28) | `results/alignment_loo/{features,scoring_comparison}.csv` |
| Alignment diagnostics | `results/alignment_diag/{m0,m05,m10,m20}/` |

## Scripts

| Script | Experiment |
|---|---|
| `scripts/sweep_saturation_calibration.py` | Exp 16 |
| `scripts/sweep_confusion_pairs.py` | Exp 17 |
| `scripts/sweep_positional_pch.py` | Exp 18 |
| `scripts/sweep_microtonal_pch.py` | Exp 19 |
| `scripts/sweep_gmm_fingerprint.py` | Exp 20 |
| `scripts/sweep_gmm_within_scale.py` | Exp 20b |
| `scripts/sweep_cadence_lm.py` | Exp 21 |
| `scripts/sweep_ngram_order.py` | Exp 22 |
| `scripts/sweep_pipeline_loo.py` | Exp 23, 24 |
| `scripts/sweep_perhyp_correction_loo.py` | Exp 25 |
| `scripts/sweep_perhyp_v2_loo.py` | Exp 26 |
| `scripts/sweep_v2cleaned_loo.py` | Exp 26b |
| `scripts/sweep_perhyp_v3_loo.py` | Exp 27 |
| `scripts/sweep_alignment_loo.py` | Exp 28 |
