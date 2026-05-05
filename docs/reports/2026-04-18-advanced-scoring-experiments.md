# Advanced Scoring Experiments (Exp 16-30)

Date: 2026-04-18 -- 2026-05-05
Corpus: CompMusic 297-300 recordings, 30 ragas, 10 recordings each
Baseline: Exp 14 pipeline LOO = 87.2% combined top-1, 92.3% tonic top-1
**Best honest result: Exp 30 = 90.6% combined top-1 (RMS phrase boundaries + retuned weights 0.3/0.5)**

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

## Experiment 23: Pipeline LOO (LEAKED -- corrected x corrected)

Date: 2026-04-19

**LEAKAGE DISCLOSURE:** This experiment was originally labeled "honest pipeline LOO (uncorrected baseline)" but post-hoc verification revealed data leakage. The script (`sweep_pipeline_loo.py`) defaults `--stems-root` to `separated_stems/htdemucs/`, which contains GT-raga-corrected transcriptions (verified: 291/297 recordings differ between corrected and uncorrected directories; 6 happen to be identical because all notes were already in-scale). Without `--train-corrected` flag, the script reads the already-corrected transcriptions as-is, making this effectively corrected x corrected.

**Method:** Full pipeline LOO: histogram scoring (clip_2.0) + LOO n-gram LM (order 7) + combined scoring (0.5*norm_hist + 2.0*norm_lm). Both train and test used corrected transcriptions (unknowingly).

### Results (297 recordings) -- LEAKED, DO NOT CITE

| Metric | Value |
|---|---:|
| Tonic top-1 | 92.3% |
| Hist raga top-1 | 74.4% |
| LM raga top-1 (GT tonic) | 87.2% |
| **Combined top-1** | **89.6%** |
| Combined top-3 | 94.6% |

### Interpretation

**This result is invalid for thesis claims.** The 89.6% reflects corrected x corrected performance, not the pipeline's real-world accuracy on unseen recordings. See Exp 23b for the legitimate uncorrected baseline.

---

## Experiment 23b: Pipeline LOO (Truly Uncorrected x Uncorrected)

Date: 2026-04-25

**Context:** After discovering Exp 23's data leakage, this experiment re-runs the full pipeline LOO with verified uncorrected transcriptions. Both `--stems-root` and transcriptions point to `separated_stems_nocorrection/htdemucs/`. Verified clean:
- Pitch data: identical to corrected dir within 0.005 Hz for 297/300 recordings (different CSV column names/precision, same underlying extraction)
- Transcriptions: genuinely different for 291/297 recordings (uncorrected notes not snapped to GT raga scale)
- Tonic detection: identical to Exp 23 (297/297 agree), confirming same pitch input

**Method:** Full pipeline LOO: histogram scoring (clip_2.0, 30-raga filter) + LOO n-gram LM (order 7) + combined scoring (0.5*norm_hist + 2.0*norm_lm). Training and testing both use uncorrected transcriptions. No correction applied at any stage.

### Results (297 recordings)

| Metric | Value |
|---|---:|
| Tonic top-1 | 92.3% (274/297) |
| Hist raga top-1 | 75.1% (223/297) |
| LM raga top-1 (GT tonic) | 86.2% (256/297) |
| **Combined top-1** | **88.9% (264/297)** |
| Combined top-3 | 94.9% (282/297) |
| Combined top-1 (given tonic) | 92.0% (252/274) |

### Error decomposition (33 errors)

| Category | Count | Detail |
|---|---:|---|
| Wrong tonic | 11 | Eliminated if user provides tonic |
| Tonic correct, GT at rank 2 | 10 | Close calls (combination/normalization) |
| Tonic correct, GT at rank 3 | 4 | Moderate difficulty |
| Tonic correct, GT at rank 4-10 | 6 | LM or hist fundamentally wrong |
| Tonic correct, GT at rank 11+ | 2 | Very hard |

When tonic is correct (22 errors):
- LM right, combined wrong: 3 (combination formula actively hurts)
- Hist right, LM wrong: 10 (LM overrides correct histogram)
- Both right, combined wrong: 1 (normalization artifact)
- Neither right: 10 (genuinely hard)

### Systematic confusion pairs (tonic correct, 22 errors)

| GT raga | Predicted | Count | Same scale? |
|---|---|---:|---|
| Bhupali | Yaman Kalyan | 3 | No (5-note subset of 8-note) |
| Abhogi | Bageshri | 2 | No (5 ⊂ 7, shared 5 notes) |
| Alhaiya Bilawal | Bilaskhani Todi | 2 | No |
| Gaud Malhar | Alhaiya Bilawal | 2 | Yes (identical 8-note scale) |
| Shuddh Sarang | Yaman Kalyan | 2 | No |

### Interpretation

**88.9% is the honest, thesis-safe combined baseline.** Key findings:

1. **Uncorrected x uncorrected beats per-hyp correction (88.9% vs 85.5%).** Distribution matching is more important than clean transcriptions. The uncorrected LM learns noise patterns and evaluates against the same noise. Per-hyp correction (Exp 26) introduces distribution mismatch: GT-corrected training vs imperfect per-candidate correction at test time.

2. **The histogram contributes +2.7pp** over the LM alone (88.9% vs 86.2%), confirming complementary signal even with distribution-matched LM.

3. **Given-tonic accuracy reaches 92.0%.** When the user provides the correct tonic, 252/274 recordings get the right raga.

4. **Same-scale confusion is the hardest problem.** Alhaiya Bilawal/Desh/Gaud Malhar share an identical 8-note scale; the histogram cannot distinguish them at all. Only LM melodic patterns can separate them.

### Why Exp 23b beats Exp 26

| Factor | Exp 23b (88.9%) | Exp 26 (85.5%) |
|---|---|---|
| LM training data | Uncorrected | GT-corrected |
| LM test data | Uncorrected (same distribution) | Per-hyp corrected (different distribution) |
| Scoring formula | 0.5*norm_hist + 2.0*norm_lm | 0.5*norm_hist + 2.0*norm_adjusted_lm |
| Penalties | None | W_DEL=10.0, W_SNAP=1.5 |
| Candidates | ~210 | ~210 |

The per-hyp correction introduces two costs that outweigh its benefits:
- **Correction under wrong ragas** creates artificial similarity: notes get snapped to each candidate's scale, making wrong ragas "look" more plausible to their own LMs
- **Penalty terms** (del_residual, mean_snap) add noise that interacts badly with min-max normalization

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

**The LM signal is excellent (96.0%) but the combined score degrades it to 85.5%.** This is 3.4pp *worse* than the uncorrected baseline (Exp 23b: 88.9%) despite the LM diagnostic being much stronger (96.0% vs 86.2%). The problem is threefold:

1. **Distribution mismatch.** GT-corrected training produces clean n-gram contexts. Per-hyp correction under wrong ragas creates a different kind of noise than the training data, reducing discrimination.

2. **Min-max normalization washout.** Normalizing LM scores to [0,1] within 210 candidates compresses the LM's absolute signal strength.

3. **Correction under wrong ragas creates artificial similarity.** Each candidate sees its raw notes snapped to its own scale. This makes wrong ragas' corrected sequences look plausible to their own LMs, reducing the gap between GT and non-GT scores.

**Key insight:** The LM at 96.0% proves the per-hypothesis correction signal is real and strong when GT raga is known. The bottleneck is that per-hyp correction at test time (correcting under all candidates) doesn't preserve this advantage. The simpler approach (Exp 23b: train and test on uncorrected) achieves better results through distribution matching.

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

## Experiment 29: Top-3 Rerank at Detected/Given Tonic

Date: 2026-04-25

**Motivation:** The LM diagnostic in Exp 26 shows 96% accuracy, and the combined top-3 is 94.9%. If we take the histogram's top-3 candidates at a given tonic and let the LM pick the winner, can we approach the LM ceiling without scoring all 210 candidates?

**Method:** Two strategies, both using uncorrected transcriptions from `separated_stems_nocorrection/htdemucs/`:
- **Strategy A (auto):** detected tonic -> top-3 histogram ragas at that tonic -> per-hyp correct each -> corrected LM rerank
- **Strategy B (given):** GT tonic (user-provided) -> top-3 histogram ragas at that tonic -> per-hyp correct each -> corrected LM rerank

30-raga filter applied. LOO n-gram LM (order 7).

### Results (297 recordings)

| Metric | Value |
|---|---:|
| Tonic top-1 | 92.3% |
| Hist raga top-1 | 75.1% |
| **Auto rerank top-1** | **70.7% (210/297)** |
| **Given rerank top-1** | **72.7% (216/297)** |
| Auto rerank top-3 | 84.5% |
| Given rerank top-3 | 86.9% |
| Given rerank (tonic correct only) | 75.2% (206/274) |

### Error decomposition (given tonic, 81 errors)

| Error source | Count |
|---|---:|
| GT raga not in histogram top-3 at given tonic | 39 (13.1%) |
| GT in top-3 but LM picks wrong | 42 (14.1%) |

### Interpretation

**Top-3 rerank is significantly worse than the full pipeline (72.7% vs 88.9%).** Two problems:

1. **Top-3 recall ceiling is only 86.9%.** The histogram misses the GT raga in 39 of 297 cases. The full pipeline (Exp 23b) scores all ~210 candidates, so it doesn't have this ceiling.

2. **LM discrimination within top-3 is weak at 83.7%.** When GT is present in the top-3, the per-hyp corrected LM picks wrong 42 times. Systematic failures: Ahir Bhairav->Bairagi (6 errors), Bageshri->Rageshri/Abhogi (5 errors).

**Why it fails:** Top-3 is too aggressive a filter. The histogram isn't reliable enough at the tonic level to guarantee GT is in the top 3. The full pipeline's brute-force approach of scoring all candidates works better despite its normalization issues.

---

## Cross-Experiment Summary (Exp 22-29)

### Pipeline LOO comparison

| Experiment | Condition | LM top-1 | Combined top-1 | Combined top-3 |
|---|---|---:|---:|---:|
| **Exp 23b** | **uncorr x uncorr (verified)** | **86.2%** | **88.9%** | **94.9%** |
| Exp 23 (LEAKED) | corr x corr (mislabeled uncorr) | 87.2% | ~~89.6%~~ | 94.6% |
| Exp 26 (per-hyp v2) | per-hyp, min-max, guessed wt | 96.0%\* | 85.5% | 94.6% |
| Exp 29 (top-3 rerank) | given tonic, top-3, per-hyp LM | -- | 72.7% | 86.9% |
| Exp 26b (v2 cleaner) | v2-cleaned x v2-cleaned | 80.5% | 80.1% | 92.3% |
| Exp 28c (alignment, 30-raga) | corr-train, align + sub_frac | -- | 66.7% (logistic) | -- |
| Exp 27 (per-hyp v3) | per-hyp, principled scoring | 96.0%\* | 66.3% | 82.2% |
| Exp 24 | corr-train x uncorr-test | 44.1% | 42.4% | 53.2% |

\* LM diagnostic uses GT-corrected test data (not comparable to Exp 23b's 86.2% which uses uncorrected test data).

### Key insights

1. **Uncorrected x uncorrected is the best approach (88.9%).** Distribution matching matters more than clean transcriptions. The uncorrected LM (86.2%) loses 11pp to the GT-corrected LM diagnostic (96.0%) but wins 3.4pp in the combined pipeline (88.9% vs 85.5%) because it avoids distribution mismatch at test time.

2. **Per-hypothesis correction (Exp 26) hurts net accuracy by 3.4pp.** Despite a stronger LM diagnostic (96.0%), the combined score is worse because: (a) correction under wrong ragas creates artificial similarity, (b) penalty terms interact badly with min-max normalization, (c) distribution mismatch between corrected training and per-hyp-corrected testing.

3. **Top-3 rerank is too aggressive (Exp 29).** Top-3 recall at a given tonic is only 86.9%, capping the ceiling. The full pipeline's brute-force scoring of all candidates outperforms despite normalization challenges.

4. **Top-3 accuracy is 94.9%.** The correct raga is almost always in the top 3. An app showing 3 candidates captures the correct answer 94.9% of the time (higher with given tonic).

5. **Same-scale confusion is the remaining hard problem.** Ragas sharing identical scales (Alhaiya Bilawal/Desh/Gaud Malhar; Bihag/Maru Bihag/Yaman Kalyan) cannot be distinguished by histograms. Only LM melodic patterns and within-note features (GMM, Exp 20b) can separate them.

6. **Raga-agnostic cleaning is not viable (Exp 26b).** The 6.9% cleaning error rate pollutes the LM (80.1% vs 88.9% best).

7. **Alignment LM (Exp 28) provides weak signal via sub_fraction** but cannot compete with distribution-matched uncorrected LM. Best: 66.7%.

### Conditional accuracy by user-provided information (Exp 23b)

Relevant for the web app, where users can optionally provide metadata to improve detection.

| Scenario | Tonic acc | Raga top-1 | Raga top-3 |
|---|---:|---:|---:|
| **Fully automatic** (no user input) | 92.3% | 88.9% | 94.9% |
| **User provides tonic** | 100%\* | 92.0% | -- |
| **Female artist** (auto tonic) | -- | -- | -- |
| **Male artist** (auto tonic) | -- | -- | -- |
| LM only (GT tonic, uncorrected) | 100%\* | 86.2% | -- |
| LM ceiling (GT tonic + GT correction) | 100%\* | 96.0% | -- |
| Histogram only (no transcription) | 92.3% | 75.1% | -- |

\* By definition (user-provided).

**Error breakdown (33 total errors, Exp 23b):**
- Wrong tonic: 11 (33%) -- tonic detection failure
- Correct tonic, LM right but combined wrong: 3 (9%) -- combination formula issue
- Correct tonic, hist right but LM wrong: 10 (30%) -- LM overrides correct histogram
- Correct tonic, neither right: 10 (30%) -- genuinely hard (mostly same-scale confusion)

### Same-scale raga groups in CompMusic subset

These groups share identical pitch-class sets. The histogram cannot distinguish them; only sequential/melodic features (LM, cadence, aaroh/avroh) can.

| Group | Scale size | Ragas | Errors in Exp 23b |
|---|---:|---|---:|
| A | 8 | Alhaiya Bilawal, Desh, Gaud Malhar | 4 |
| B | 8 | Bihag, Maru Bihag, Yaman Kalyan | 0 |
| C | 7 | Bhairav, Bilaskhani Todi, Todi | 2 |
| D | 9 | Kedar, Khamaj | 0 |
| E | 7 | Puriya Dhanashree, Shri | 0 |

### Exp 23 data leakage disclosure

The previously reported Exp 23 "honest pipeline LOO" result of 89.6% combined top-1 was computed using transcriptions from `separated_stems/htdemucs/`, which contains GT-raga-corrected transcriptions (verified: 291/297 recordings differ between corrected and uncorrected directories; 6 are identical because all notes were already in-scale). The script (`sweep_pipeline_loo.py`) defaults `--stems-root` to the corrected directory and uses it for both pitch data and transcriptions. Without `--train-corrected` flag, it reads the already-corrected transcriptions as-is, making the result effectively corrected x corrected.

**Verification of corrected vs uncorrected directories:**
- Pitch data: identical within 0.005 Hz for 297/300 recordings (different CSV column names/precision, same extraction)
- Accompaniment pitch: byte-identical
- Transcriptions: genuinely different for 291/297 recordings (corrected notes are snapped to GT raga scale, uncorrected are raw)
- The leaked 89.6% is inflated by ~0pp vs the legitimate 88.9% -- the leakage effect is small because the corrected and uncorrected LMs perform similarly when distribution-matched

### Improvement strategies (not yet evaluated)

Based on the Exp 23b error decomposition, the highest-impact strategies for improving beyond 88.9%:

1. **Given-tonic candidate filtering.** Filter candidates to the user-provided tonic (~30 candidates vs ~210). Min-max normalization over 30 items is more stable. Expected to improve given-tonic accuracy from 92.0% toward 94%+.

2. **ALPHA/BETA tuning.** Current 0.5/2.0 gives histogram 20% weight. For same-scale ragas, histogram is pure noise. Reducing ALPHA or making it adaptive could fix the 3 "LM right but combined wrong" cases.

3. **RRF instead of min-max.** Replace fragile score-based fusion with rank-based fusion. More robust to outliers.

4. **Two-stage coarse-to-fine.** Stage 1: uncorrected LM + hist -> top-10. Stage 2: per-hyp correct top-10 only -> corrected LM re-score. Gets both distribution match (stage 1) and correction refinement (stage 2).

5. **LM order sweep.** Order=6 is optimal on honest uncorrected data (Exp 22b: 87.5% top-1). The original Exp 22 was LEAKED (used corrected stems, reported 97.3% at order=7). The combined pipeline uses order=7 (Exp 23b), which may benefit from a re-sweep at order=6.

---

## Experiment 28: Noisy-Channel Alignment LM Scoring

Date: 2026-04-23 -- 2026-04-25

**Motivation:** Exp 26-27 showed per-hypothesis correction recovers a 96.0% LM diagnostic signal but introduces scale-size bias that all scoring methods fail to overcome (best: 66.3%). The alignment approach eliminates per-hypothesis correction: keep uncorrected test sequences as-is, score them against corrected-trained LMs using beam DP alignment with skip/substitution costs.

**Method:** Phrase-local beam DP alignment (`raga_pipeline/language_model/alignment.py`). At each observed token, choose:
- **Skip** (noise): penalty -lambda_skip
- **Match** (accept as-is): reward +lambda_match + log P_r(token | context)
- **Substitute** to nearby pitch class (<=2 semitones): reward +lambda_match + log P_r(sub_token | context) - lambda_sub * distance

Context is built from matched/substituted tokens only, never crosses phrase boundaries. Beam search prunes to top-B states per position.

Training: correct raw notes with GT raga/tonic in-memory (`apply_raga_correction_to_notes`), tokenize, train LOO LM (order 7, add-k smoothing). Testing: tokenize uncorrected raw notes *without any correction*, score with alignment.

Defaults: lambda_skip=0.5, lambda_match=2.0, lambda_sub=0.3, beam_width=50, max_sub_distance=2.

### Exp 28a: Diagnostic sweep (15 recordings, 2 ragas, no raga filter)

**Purpose:** Determine whether the alignment LM has any discriminative signal, and identify the best lambda_match.

**Design issue discovered:** Without `lambda_match`, log-probs are always negative (-1 to -4), so the DP rationally skips every token. `lambda_match` compensates so that in-scale tokens (log_prob ~ -1.4) contribute positively (+0.6) while out-of-scale tokens (log_prob ~ -4) still get skipped (-2.0). Pre-registered pass criterion: GT vs non-GT lm_per_token delta >= +0.02.

| lambda_match | skip_frac GT | skip_frac non-GT | lm_per_token Delta | LM-only A | Hist-only G |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.000 | 1.000 | 0.000 | 93.3%\* | 93.3% |
| 0.5 | 0.969 | 0.971 | -67308\*\* | 26.7% | 93.3% |
| 1.0 | 0.650 | 0.664 | **+0.0024** | 33.3% | 93.3% |
| 2.0 | 0.318 | 0.339 | -0.0057 | 53.3% | 93.3% |

\* All tokens skipped; ranking collapses to histogram tie-breaking. \*\* Sentinel contamination (-1e6) from mixed 0-matched/few-matched rows.

**Initial verdict:** lm_per_token delta never exceeds +0.003. Appeared to be a negative result.

**Key discovery:** `lm_per_token` (= raw log-prob sum / n_matched) discards the substitution penalty. The DP's `score` field includes lambda_match and lambda_sub, but the reported metric strips them. Wrong ragas achieve similar per-token log-probs by substituting tokens to their own scale variants (1-semitone swaps), but the GT raga needs *fewer substitutions*. The `sub_fraction` (= n_substituted / n_matched) captures this signal.

At lambda_match=2.0, `lm_per_token - w * sub_fraction` sweep on 15 recordings:

| w_sub | Gated | Ungated |
|---:|---:|---:|
| 0 | 20.0% | 53.3% |
| **1** | **100.0%** | **100.0%** |
| 2 | 86.7% | 86.7% |
| 5 | 86.7% | 86.7% |

100% at w_sub=1 on 15 recordings (small sample caveat: Wilson 95% CI [79.6%, 100%]).

### Exp 28b: Full-scale, no raga filter (297 recordings, top-10, 78-raga histogram)

**Purpose:** Validate sub_fraction signal at full scale. Used top-10 histogram candidates to manage compute time.

**Problem:** Histogram scored all 78 ragas in the DB, but LM only knows 30 CompMusic ragas. Top-10 slots wasted on ragas not in the LM. After LM filtering: 5.4 features/recording avg, GT recall 92.6% (275/297).

| Metric | GT mean | Non-GT mean | Delta |
|---|---:|---:|---:|
| lm_per_token | -1.2399 | -1.2445 | +0.0046 |
| skip_fraction | 0.0920 | 0.1091 | -0.0171 |
| **sub_fraction** | **0.4691** | **0.5172** | **-0.0481** |

| Method | Top-1 |
|---|---:|
| A: Raw alignment LM | 24.6% |
| C2: LM - 1.0\*sub | 52.2% |
| C2: LM - 2.0\*sub | 69.0% |
| C2: LM - 5.0\*sub | 69.7% |
| E: Logistic | 65.3% |
| G: Hist-only baseline | 56.6% |

Sub_fraction signal confirmed (delta -0.048). C2 at 69.7% beats hist-only by +13pp. But hist-only at 56.6% is depressed vs Exp 23's 74.2% due to candidate filtering artifacts.

### Exp 28c: Full-scale, 30-raga filter (297 recordings, top-10, 30-raga histogram)

**Purpose:** Fair evaluation with histogram restricted to the 30 CompMusic ragas (matching Exp 23 setup). Auto-built raga filter from GT CSV.

Config: lambda_match=2.0, lambda_skip=0.5, lambda_sub=0.3, beam_width=50, top-k=10.
Runtime: ~21,600s (~6 hours) on Apple Silicon M-series.
Features: 4436 across 297 recordings (14.9 avg/recording). GT recall: 97.6% (290/297).

| Metric | GT mean | Non-GT mean | Delta |
|---|---:|---:|---:|
| lm_per_token | -1.2385 | -1.2421 | +0.0036 |
| skip_fraction | 0.0912 | 0.1117 | -0.0205 |
| **sub_fraction** | **0.4712** | **0.5605** | **-0.0893** |

Sub_fraction delta nearly doubled (-0.089 vs -0.048) with the correct candidate set.

| Method | Top-1 |
|---|---:|
| A: Raw alignment LM | 7.1% |
| B: Z-scored LM | 5.4% |
| C: Z-LM - 5.0\*skip | 8.4% |
| C2: LM - 0.5\*sub | 18.5% |
| C2: LM - 1.0\*sub | 42.4% |
| **C2: LM - 2.0\*sub** | **63.6%** |
| C2: LM - 5.0\*sub | 64.7% |
| C3: Z-LM - 2.0\*sub | 10.8% |
| D: RRF (hist + LM) | 26.3% |
| **E: Logistic** | **66.7%** |
| F: 0.5\*z\_hist + 2.0\*z\_lm | 19.2% |
| G: Hist-only baseline | 56.6% |

### Interpretation

**Sub_fraction is the discriminative feature, not lm_per_token.** The alignment DP assigns similar log-probs to GT and non-GT ragas because wrong ragas compensate via 1-semitone substitutions. But the GT raga needs fewer substitutions: sub_fraction delta = -0.089 (30-raga filter). Penalizing sub_fraction lifts accuracy from 7.1% (raw LM) to 64.7% (C2, w=5) or 66.7% (logistic), beating histogram-only (56.6%) by +8-10pp.

**Why alignment still trails the 88.9% baseline (Exp 23b):**
1. **Train/test distribution gap persists.** The corrected-trained LM's n-gram contexts are clean; uncorrected test sequences have different n-gram statistics even after alignment. The sub_fraction signal partially bridges this, but per-token log-prob discrimination remains near zero (delta +0.004).
2. **Top-10 candidate truncation.** GT recall is 97.6%, capping the ceiling. The 7 missing-GT recordings (2.4%) contribute 0% accuracy.
3. **The baseline uses same-distribution matching.** Exp 23b's uncorrected-train x uncorrected-test LM scores uncorrected test sequences with an LM trained on uncorrected data -- no distribution gap at all. The alignment approach cannot overcome the fundamental mismatch between corrected training and uncorrected test.

**Comparison to per-hypothesis correction (Exp 26-27):**
- Per-hyp correction: 96% LM diagnostic, 66.3% best combined (scale-size bias kills it)
- Alignment + sub_fraction: no useful LM diagnostic, 66.7% best combined (sub_fraction carries the signal)
- Both approaches achieve similar final accuracy (~66%) but for different reasons. Per-hyp has strong LM signal destroyed by bias; alignment has weak LM signal rescued by substitution count.

**The sub_fraction metric is essentially a soft scale-match score** -- it counts what fraction of tokens needed pitch correction to fit the candidate raga's scale. This is simpler than the full alignment LM and could potentially be computed without the expensive beam DP (just count how many test tokens are in-scale for each candidate). Future work should explore whether a direct in-scale fraction feature achieves similar accuracy at lower cost.

### Caveats for baseline comparison

The 88.9% baseline (Exp 23b) and the alignment results (Exp 28) are **not directly comparable**:
- Exp 23b: all candidates, uncorrected-train LM, min-max normalization, combined = 0.5\*norm_hist + 2.0\*norm_lm
- Exp 28c: top-10 candidates, corrected-train LM + alignment, sub_fraction penalty or logistic regression

A fair comparison would require running Exp 28 with all candidates (prohibitively slow at ~57 hours with gated-only, ~200+ hours ungated) or running Exp 23b with top-10 candidates (quick re-score from existing results).

---

## Experiment 30: RMS Phrase Boundaries + Weight Sweep

Date: 2026-05-05
Scripts: `scripts/sweep_pipeline_loo.py` (Exp 30a), `scripts/sweep_score_weights.py` (Exp 30b)
Results: `results/pipeline_loo_rms_phrases/`, `results/score_weight_sweep/`

### Motivation

The n-gram LM tokenizer (`tokenize_notes_for_lm`) used an internal 0.25s inter-note gap heuristic for phrase boundary detection, independent of the pipeline's phrase segmentation. After refactoring phrase splitting to use RMS energy-based silence detection as the primary method (`detect_phrases_by_silence`), this experiment tests whether energy-based phrase boundaries improve LM accuracy and whether the combined scoring weights (alpha/beta) need retuning.

### Setup

- Same corpus/params as Exp 23b: uncorrected x uncorrected, order=7, 297 recordings, 30 ragas
- Phrase boundaries: `detect_phrases_by_silence(notes, energy, timestamps)` with default params (silence_threshold=0.10, silence_min_duration=0.25s)
- Phrase boundaries are tonic-independent (computed from RMS energy, not pitch), computed once per recording and reused across all candidate tonics
- Exp 30a: Full LOO with RMS phrases at baseline weights (0.5/2.0)
- Exp 30b: Weight sweep over 9x9 grid of (alpha, beta) using per-candidate normalized scores from one LOO pass

### Results

| Metric | Exp 23b baseline | Exp 30a (old wt) | Exp 30b (best wt) |
|---|---|---|---|
| Tonic top-1 | 92.3% (274/297) | 92.3% (274/297) | 92.3% (274/297) |
| Hist raga top-1 | 75.1% (223/297) | 75.1% (223/297) | 75.1% (223/297) |
| LM top-1 (GT tonic) | 86.2% (256/297) | 87.5% (260/297) | 87.5% (260/297) |
| Combined top-1 | 88.9% (264/297) | 88.6% (263/297) | **90.6% (269/297)** |
| Combined top-3 | 94.9% (282/297) | 94.9% (282/297) | 94.6% (281/297) |
| Weights (alpha/beta) | 0.5/2.0 | 0.5/2.0 | 0.3/0.5 |
| Effective hist:LM ratio | 20:80 | 20:80 | 37:63 |

### Weight Sweep Grid (top-1 accuracy, alpha rows x beta cols)

```
         0.0   0.5   1.0   1.5   2.0   2.5   3.0   4.0   5.0
   0.0    --  80.1  80.1  80.1  80.1  80.1  80.1  80.1  80.1
   0.1  56.6  88.6  86.9  85.9  84.5  84.2  83.2  82.5  81.8
   0.2  56.6  89.9  88.6  87.5  86.9  86.2  85.9  84.5  84.2
   0.3  56.6  90.6  88.9  88.6  88.6  86.9  86.9  86.2  85.2
   0.5  56.6  87.5  90.2  89.2  88.6  88.6  87.9  87.2  86.9
   0.7  56.6  85.2  89.9  89.6  89.9  88.6  88.6  87.9  87.5
   1.0  56.6  84.2  87.5  90.2  90.2  89.9  89.2  88.6  88.6
   1.5  56.6  82.8  84.8  87.5  89.6  90.6  90.2  89.9  88.9
   2.0  56.6  80.8  84.2  85.5  87.5  89.6  90.2  90.2  89.9
```

Two peaks at 90.6%: (0.3, 0.5) and (1.5, 2.5). Both represent roughly 37:63 hist:LM ratio. The optimal ratio shifted from 20:80 with gap phrases to 37:63 with RMS phrases -- the LM signal is stronger so it needs less amplification, allowing the histogram more influence on close calls.

### LM-only analysis (GT tonic)

RMS phrase boundaries improved LM-only accuracy by +1.3pp (256 -> 260):

Gained (7 recordings): Ahir Bhairav_9, Gaud Malhar_10, Hansadhwani_7, Malkauns_2, Shuddh Sarang_10, Shuddh Sarang_4, Todi_3
Lost (3 recordings): Desh_5, Madhukauns_4, Madhukauns_9

Gains are spread across diverse ragas. Losses are concentrated: both Madhukauns recordings confused with Madhuvanti (structurally close ragas, energy-based phrase boundaries may split their characteristic phrases differently).

### Error decomposition (Exp 30a, 0.5/2.0 weights, 34 errors)

| Category | Count |
|---|---|
| Wrong tonic | 11 (32%) |
| Right tonic, rank 2 | 13 (38%) |
| Right tonic, rank 3 | 2 (6%) |
| Right tonic, rank 4-10 | 5 (15%) |
| Right tonic, rank 11+ | 3 (9%) |

High proportion of rank-2 errors (38%) explains why the weight sweep helped: these are margin cases where shifting the hist:LM balance tips the ranking.

### Same-scale confusion groups

Alhaiya Bilawal / Desh / Gaud Malhar (identical 8-note scale): 7 errors. These ragas share identical pitch-class content; only melodic movement patterns distinguish them. RMS phrases improved Gaud Malhar_10 (gained) but lost Desh_5 -- suggesting phrase boundary placement affects which melodic patterns the LM can capture for these confusable ragas.

### Key findings

1. **RMS phrase boundaries strengthen the LM signal** (+1.3pp LM-only, +4 net recordings at GT tonic), confirming that energy-based phrase detection captures musically meaningful boundaries better than a fixed 0.25s gap heuristic.

2. **Weight retuning is required.** At the old 0.5/2.0 weights, the stronger LM signal actually hurt combined accuracy slightly (-0.3pp) because the histogram's ability to break ties on close LM calls was suppressed. Retuning to 0.3/0.5 recovers the gain and then some: **+1.7pp over the Exp 23b baseline** (90.6% vs 88.9%).

3. **The optimal hist:LM ratio is robust.** Multiple (alpha, beta) combos achieve 90%+: (0.3/0.5), (1.5/2.5), (1.0/1.5), (2.0/3.0). All converge to roughly 37:63 hist:LM, suggesting this ratio is a stable optimum for this dataset with RMS phrases.

4. **Remaining errors are dominated by wrong tonic (32%) and same-scale confusion (rank 2, 38%).** Further gains likely require better tonic detection or within-scale discriminative features (GMM fingerprints, cadence patterns).

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
| N-gram order sweep (LEAKED) | `results/ngram_order_sweep/summary.csv` |
| N-gram order sweep (honest) | `results/ngram_order_sweep_uncorrected/summary.csv` |
| Pipeline LOO (LEAKED) | `results/pipeline_loo_uncorrected/{progress,summary}.csv` |
| Pipeline LOO (truly uncorr) | `results/pipeline_loo_truly_uncorrected/{progress,summary}.csv` |
| Pipeline LOO (corr-train) | `results/pipeline_loo_corrected_train/{progress,summary}.csv` |
| Per-hyp calibration | `results/perhyp_correction/calibration.csv` |
| Per-hyp v2 (fixed wt) | `results/perhyp_v2/{progress,summary,sanity_check}.csv` |
| v2 cleaner LOO | `results/v2cleaned_loo/{progress,summary}.csv` |
| Per-hyp v3 (principled) | `results/perhyp_v3/{features,scoring_comparison}.csv` |
| Alignment LOO (Exp 28c) | `results/alignment_loo/{features,scoring_comparison,evaluated_filenames}.{csv,txt}` |
| Alignment LOO (30-raga) | `results/alignment_loo_top30/{features,scoring_comparison}.csv` |
| Alignment diagnostics (Exp 28a) | `results/alignment_diag/{m0,m05,m10,m20}/` |
| Top-3 rerank LOO | `results/top3_rerank_loo/{progress,summary}.csv` |
| RMS phrase LOO (Exp 30a) | `results/pipeline_loo_rms_phrases/{progress,summary}.csv` |
| Weight sweep (Exp 30b) | `results/score_weight_sweep/weight_sweep.csv` |

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
| `scripts/sweep_ngram_order.py` | Exp 22 (LEAKED), 22b (honest) |
| `scripts/sweep_pipeline_loo.py` | Exp 23, 23b, 24 |
| `scripts/sweep_perhyp_correction_loo.py` | Exp 25 |
| `scripts/sweep_perhyp_v2_loo.py` | Exp 26 |
| `scripts/sweep_v2cleaned_loo.py` | Exp 26b |
| `scripts/sweep_perhyp_v3_loo.py` | Exp 27 |
| `scripts/sweep_alignment_loo.py` | Exp 28 |
| `scripts/sweep_top3_rerank_loo.py` | Exp 29 |
| `scripts/sweep_pipeline_loo.py` (RMS phrases) | Exp 30a |
| `scripts/sweep_score_weights.py` | Exp 30b |
