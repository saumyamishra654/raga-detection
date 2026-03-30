# Language Model Approaches to Raga Detection

## Context

The "language modelling" approach to raga detection draws from **language identification on romanized text** -- the problem of telling Hindi from Malay (or other languages) when all are written using the Latin alphabet. The key insight: two languages can share the same alphabet but differ dramatically in which character sequences are common. Similarly, two ragas can share the same scale (swar set) but differ in which note transitions and phrases are idiomatic.

In our pipeline, transcribed sargam sequences are the "text", ragas are the "languages", and sargam tokens are the "characters". The goal is to build statistical models of each raga's sequential grammar and score new transcriptions against them.

### Corpus

- 1100 songs across 110 ragas (full set)
- 300 songs across 30 ragas (thesis subset, CompuMusic database)
- Median recording length: 30-40 minutes
- Estimated token yield: 10,000-50,000+ tokens per raga

---

## Approach 1: Character N-gram Language Models (start here)

The direct analogue to the language detection idea. For each raga, estimate a smoothed n-gram probability distribution over sargam tokens from its corpus. Score a new transcription by its per-token log-likelihood (or perplexity) under each raga's model.

**How it works:**
- Build bigram/trigram tables per raga: `P(Ga | Sa, Re)` for Yaman vs Bhairav
- Smooth with Kneser-Kney or simple add-k to handle unseen sequences
- Score = average log-probability of the transcription under each raga model
- Lowest perplexity wins

**Pros:** Directly inspectable tables. You can literally print "top 20 most probable trigrams in Yaman" and a musicologist can verify them. Segment-level scoring is trivial (sliding window of perplexity). Proven technique in NLP -- well-understood theory. Very little code.

**Cons:** Fixed context window. A trigram model can't capture a 7-note characteristic phrase as a unit. May struggle with very similar ragas that differ only in longer-range patterns.

---

## Approach 2: Variable-length Markov Models (PPM)

An extension of Approach 1 that adaptively uses longer contexts where the data supports it. Prediction by Partial Matching (PPM) -- used in text compression and music analysis research (Pearce & Wiggins' IDyOM system uses exactly this for melody modelling).

**How it works:**
- Like Approach 1, but the model automatically backs off from longer to shorter contexts. If it's seen enough examples of `Sa Re Ga ma Pa` in Yaman to be confident, it uses the 5-gram; otherwise it falls back to the trigram.
- Escape mechanism handles unseen contexts gracefully

**Pros:** Captures both short idioms and longer characteristic phrases without you choosing a fixed n. Still fully inspectable -- you can ask "what context length did the model use for this prediction?" Better theoretical compression = better modelling.

**Cons:** Slightly more complex to implement (but well-documented algorithms exist). Harder to make a clean side-by-side table compared to fixed n-grams.

---

## Approach 3: Smoothed N-gram + Discriminative Reweighting

Approach 1 as the base, but add a second pass that reweights n-grams by how much they discriminate between ragas (using mutual information or log-likelihood ratio). This bridges toward the existing motif mining work -- the language model provides the probability backbone, and the discriminative pass highlights the raga-specific patterns.

**How it works:**
- Train per-raga n-gram models (same as Approach 1)
- Compute per-n-gram discriminative score: `log P(ngram | raga_r) - log P(ngram | all_ragas)`
- Final score blends generative likelihood with discriminative bonus
- The discriminative n-grams become your "signature motifs" with statistical backing

**Pros:** Best of both worlds -- proper probability model plus a ranked list of "this trigram is 15x more likely in Yaman than the background." Directly comparable to motif mining results. Good thesis narrative.

**Cons:** Two-stage complexity. The discriminative reweighting adds a hyperparameter (blend weight). Slightly harder to present cleanly.

---

## Implementation Order

1. **Approach 1** first -- cleanest experiment, directly maps to the advisor's suggestion, clear baseline
2. **Approach 3** next -- natural extension connecting language model back to motif mining work; the comparison (generative LM vs. frequency-based motif mining vs. combined) makes an excellent thesis chapter
3. **Approach 2** (PPM) if fixed n-grams clearly fail due to context length limitations; otherwise note in related work
