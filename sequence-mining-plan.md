# Sequence Mining for Raga Identification: Implementation Plan

## Overview

**Goal:** Build a sequence mining system to automatically discover and match characteristic melodic patterns (motifs, pakads) across raga recordings, with special focus on discriminating between ragas that share identical aaroh/avroh patterns.

**Approach:** Overcomplete motif dictionary with entropy-based weighting, designed for both accurate classification and educational value (showing the full musical vocabulary of each raga).

**Key Insight:** With ~80 unique aaroh/avroh patterns and up to 10-12 ragas sharing the same pattern, scale-based features alone are insufficient. Sequence-level patterns capture the characteristic phrases that define ragas beyond their note sets.

---

## Dataset Characteristics

### Current Status
- **Source:** `ground_truth_v6.csv` from `main notebooks/`
- **Size:** ~1096 recordings (~10 per raga, roughly equal distribution)
- **Coverage:** ~110 ragas total
- **Recording Length:** 1-1.5 hours per recording (full performances)
- **Challenge:** ~80 unique aaroh/avroh patterns → many ragas share the same scale structure

### Raga Clustering Challenge
With 110 ragas but only 80 unique patterns:
- Some patterns have 10-12 ragas mapped to them
- Most patterns have fewer matches, but a "sizeable number" have multiple ragas
- These clusters are where sequence mining becomes essential

**Example Clusters (needs verification):**
- Bilawal family: Bilawal, Alhaiya Bilawal, Devgiri Bilawal
- Kafi family: Kafi, Bageshri, Bhimpalasi (though Bhimpalasi has directional differences)
- Kalyan family: Yaman, Bhoopali Todi, Kedar

---

## Pattern Representation Strategy

### Multi-Resolution Approach
Mine patterns at multiple abstraction levels to balance precision and robustness.

#### Level 1: Exact Sargam Sequences
```python
pattern = ["Ni", "Re", "Ga", "Re"]  # Yaman pakad
```
**Properties:**
- Most precise representation
- Captures komal/shuddh distinctions
- Brittle to transcription errors
- **Use case:** High-confidence matches for well-transcribed passages

#### Level 2: Interval Sequences (Relative Pitch)
```python
pattern = [+4, +2, -2]  # semitone differences
```
**Properties:**
- Captures melodic contour
- Transposition-invariant
- Loses komal/shuddh information
- **Use case:** Matching similar contours across octaves

#### Level 3: Contour Abstraction
```python
pattern = ["UP", "UP", "DOWN"]
```
**Properties:**
- Very robust to transcription errors
- Highly generalized
- Loses most specific information
- **Use case:** Coarse-grained phrase shape matching

#### Level 4: Hybrid with Context
```python
pattern = [
    ("Ni", 0.5, "up", "strong"),    # (sargam, duration, direction, energy)
    ("Re", 0.3, "up", "medium"),
    ("Ga", 0.4, "down", "strong")
]
```
**Properties:**
- Rich multi-dimensional representation
- Includes timing and emphasis
- Requires fuzzy matching
- **Use case:** High-fidelity pattern matching for clean recordings

#### Level 5: Subsequences with Gaps
```python
pattern = ["Ni", GAP, "Ga", "Re"]  # Allow intermediate notes
```
**Properties:**
- Captures non-contiguous patterns
- Handles ornamentation (gamakas between structural notes)
- More flexible matching
- **Use case:** Finding core melodic skeleton despite variations

**Implementation Note:** Experiment with smoothing to reduce noise before gap-based mining. Test multiple gap tolerance levels (0, 1, 2 intervening notes).

**Decision Point:** Mine patterns at ALL levels simultaneously and combine scores, or start with Level 1 (exact sargam) + Level 5 (with gaps) as primary approach?

---

## Overcomplete Motif Dictionary Strategy

### Philosophy
Instead of mining only discriminative patterns, build a comprehensive vocabulary of ALL musically meaningful patterns, then use entropy/specificity scoring to weight their importance.

**Advantages:**
1. **Robustness:** Multiple patterns can match same raga (redundancy helps with transcription errors)
2. **Educational Value:** Full representation of raga's melodic vocabulary for later visualization
3. **Graceful Degradation:** If one pattern fails to match, others provide backup evidence
4. **Shared Cultural Patterns:** Captures common phrases (e.g., "Sa Pa Sa") with appropriate low weights

### Pattern Mining Algorithm

#### Step 1: Frequent Subsequence Mining
Use **PrefixSpan** or **SPADE** algorithm to extract all frequent patterns.

```
Input: All note sequences from ground_truth_v6 recordings
Parameters:
  - min_support: Minimum occurrences across dataset (e.g., 3-5 times)
  - min_length: Minimum pattern length (e.g., 3 notes)
  - max_length: Maximum pattern length (e.g., 12 notes)
  - gap_constraint: Maximum gap between notes (0, 1, or 2)

Output: Set of patterns P = {p1, p2, ..., pN} where N is very large
```

**Multi-scale Mining:**
- Mine separately for lengths 3-4 (short motifs)
- Mine for lengths 5-8 (medium phrases)
- Mine for lengths 9-12 (long phrases)
- Different minimum support thresholds per scale (longer patterns can be rarer)

#### Step 2: Compute Entropy/Discriminative Scores

For each pattern P, compute its raga distribution:

```python
def compute_pattern_entropy(pattern, recordings_database):
    """
    Compute Shannon entropy of pattern across ragas.
    Low entropy = pattern is specific to few ragas (high discriminative power)
    High entropy = pattern is spread across many ragas (generic)
    """
    # Count occurrences per raga
    raga_counts = defaultdict(int)
    total_occurrences = 0
    
    for recording in recordings_database:
        if pattern in recording.note_sequences:
            raga_counts[recording.raga] += 1
            total_occurrences += 1
    
    # Compute probability distribution
    raga_probs = {raga: count/total_occurrences 
                  for raga, count in raga_counts.items()}
    
    # Shannon entropy
    entropy = -sum(p * log2(p) for p in raga_probs.values() if p > 0)
    
    # Discriminative power (inverse entropy)
    discriminative_power = 1.0 / (1.0 + entropy)
    
    return {
        'entropy': entropy,
        'discriminative_power': discriminative_power,
        'raga_distribution': raga_probs,
        'total_occurrences': total_occurrences
    }
```

**Alternative Metric: Mutual Information**
```python
MI(Pattern, Raga) = H(Raga) - H(Raga | Pattern)
```
Measures how much knowing the pattern reduces uncertainty about the raga.

#### Step 3: Negative Evidence Scoring

Track not just where patterns appear, but where they DON'T appear.

```python
def compute_negative_evidence(pattern, raga_target, recordings_database):
    """
    Compute strength of negative evidence for a raga.
    
    Strong negative signal: Pattern appears in other ragas but NEVER in target raga
    Weak negative signal: Pattern appears everywhere
    """
    appears_in_target = count_occurrences(pattern, raga_target)
    appears_in_others = count_occurrences(pattern, all_ragas - {raga_target})
    
    if appears_in_target == 0 and appears_in_others > 5:
        # Strong negative indicator
        return -1.0 * log(appears_in_others)
    else:
        return 0.0
```

**Usage:** When scoring a test recording, patterns that appear should boost their associated ragas, while patterns that appear but are "forbidden" for certain ragas should penalize those ragas.

---

## Handling Same Aaroh/Avroh Ragas: Dissimilarity Constraints

### Problem Statement
For raga pairs/clusters that share identical aaroh/avroh patterns (e.g., Bilawal family with 10-12 members), we must ensure the mined pattern sets are sufficiently different to enable discrimination.

### Constraint-Based Optimization

#### Phase 1: Identify Confused Clusters
```python
def find_confused_clusters(raga_database):
    """
    Group ragas by aaroh/avroh pattern.
    Returns clusters where disambiguation is needed.
    """
    pattern_to_ragas = defaultdict(list)
    
    for raga in raga_database:
        pattern_key = (tuple(raga.aaroh_pattern), tuple(raga.avroh_pattern))
        pattern_to_ragas[pattern_key].append(raga.name)
    
    # Return clusters with 2+ ragas
    confused_clusters = [ragas for ragas in pattern_to_ragas.values() 
                         if len(ragas) >= 2]
    
    return confused_clusters
```

#### Phase 2: Compute Pairwise Dissimilarity
For each pair of ragas (A, B) in a confused cluster:

```python
def compute_pattern_set_dissimilarity(raga_A_patterns, raga_B_patterns):
    """
    Measure how different two ragas' pattern sets are.
    
    Uses Jensen-Shannon divergence of pattern distributions.
    """
    # Convert pattern sets to probability distributions
    all_patterns = set(raga_A_patterns.keys()) | set(raga_B_patterns.keys())
    
    dist_A = [raga_A_patterns.get(p, 0) for p in all_patterns]
    dist_B = [raga_B_patterns.get(p, 0) for p in all_patterns]
    
    # Normalize
    dist_A = normalize(dist_A)
    dist_B = normalize(dist_B)
    
    # Jensen-Shannon divergence
    M = 0.5 * (dist_A + dist_B)
    js_div = 0.5 * KL(dist_A, M) + 0.5 * KL(dist_B, M)
    
    return js_div
```

**Constraint:**
```
For ragas A, B in same aaroh/avroh cluster:
Require: JS_divergence(patterns_A, patterns_B) > threshold_min

If constraint violated:
  → Identify discriminative patterns (high in A, low in B or vice versa)
  → Boost weights of these patterns
  → Iteratively refine until threshold met
```

#### Phase 3: Active Discriminative Mining
For clusters where dissimilarity is too low:

```python
def mine_discriminative_patterns_for_cluster(cluster_ragas, min_dissimilarity):
    """
    Targeted mining to find patterns that separate ragas in a confused cluster.
    """
    for raga_A, raga_B in combinations(cluster_ragas, 2):
        if dissimilarity(raga_A, raga_B) < min_dissimilarity:
            # Find patterns unique to A
            unique_to_A = find_patterns(
                present_in=raga_A, 
                absent_in=raga_B,
                min_support=2  # Lower threshold for discriminative patterns
            )
            
            # Boost their weights
            for pattern in unique_to_A:
                pattern.discriminative_power *= 2.0
            
            # Repeat for B
            unique_to_B = find_patterns(present_in=raga_B, absent_in=raga_A)
            # ... boost weights
    
    return updated_pattern_database
```

**Evaluation Metric:**
After constraint optimization, test on held-out recordings from confused clusters. Success = correct classification within the cluster.

---

## Temporal Context Handling

Raga performances have structural sections with different characteristics. Three strategies for handling this:

### Option A: Mixed (Aggregate All)
- Mine patterns from full recordings without segmentation
- Pros: Simple, no segmentation errors, patterns naturally weighted by frequency
- Cons: Loses temporal structure information (alap vs. tan patterns mixed)

### Option B: Segmented Mining
- Automatically segment recordings by tempo/energy
- Mine patterns separately for each section type
- Pros: Section-specific patterns (e.g., "alap signatures" vs. "tan signatures")
- Cons: Requires robust segmentation algorithm

**Segmentation Strategy:**
```python
def segment_by_tempo_energy(recording):
    """
    Use energy envelope and note density to segment.
    
    Sections:
    - Alap: Low energy variation, sparse notes, no rhythm
    - Jor: Medium energy, increasing density, emerging pulse
    - Jhala: High energy, dense notes, strong pulse
    - Vilambit: Structured, slow-medium tempo
    - Drut: Fast tempo, rapid note sequences
    """
    note_density = compute_notes_per_second(recording)
    energy_variance = compute_energy_variance(recording)
    
    # Use change point detection or sliding window classification
    segments = detect_change_points(note_density, energy_variance)
    
    return segments
```

### Option C: Natural Clustering
- Mine patterns without pre-segmentation
- Let patterns naturally cluster by tempo/density characteristics
- Post-hoc analysis reveals which patterns belong to which section types

**Clustering Approach:**
```python
def cluster_patterns_by_temporal_context(patterns):
    """
    Use k-means or hierarchical clustering on pattern features:
    - Average note duration
    - Note density (notes per second)
    - Energy variation
    """
    features = []
    for pattern in patterns:
        features.append([
            pattern.avg_note_duration,
            pattern.note_density,
            pattern.energy_variance
        ])
    
    clusters = kmeans(features, n_clusters=3)  # Alap, Medium, Fast
    
    return clusters
```

**Decision Point:** Start with Option A (mixed) for initial prototype. If results are poor, try Option B (segmented). Option C for exploratory analysis.

---

## Pattern Matching and Similarity

### Two-Stage Matching Strategy
To balance accuracy and computational efficiency:

#### Stage 1: Low-Dimensional Screening (Fast)
Use compact representations for initial filtering:

```python
def quick_match(test_sequence, pattern_database):
    """
    Fast screening using low-dimensional representations.
    
    Methods:
    1. n-gram Jaccard similarity
    2. LSH (Locality Sensitive Hashing)
    3. Embedding similarity (if using learned representations)
    """
    # Convert to n-gram set
    test_ngrams = set(get_ngrams(test_sequence, n=3))
    
    candidates = []
    for pattern in pattern_database:
        pattern_ngrams = set(get_ngrams(pattern, n=3))
        
        # Jaccard similarity
        jaccard = len(test_ngrams & pattern_ngrams) / len(test_ngrams | pattern_ngrams)
        
        if jaccard > threshold:  # e.g., 0.3
            candidates.append(pattern)
    
    return candidates  # Only patterns worth expensive DTW comparison
```

#### Stage 2: DTW Refinement (Expensive)
For patterns that pass screening, compute precise similarity:

```python
def dtw_match(test_subsequence, pattern, window=5):
    """
    Dynamic Time Warping for flexible temporal alignment.
    
    Handles:
    - Tempo variations (same phrase played faster/slower)
    - Slight transcription errors
    - Ornamentation differences
    """
    from dtaidistance import dtw
    
    # Convert sargam to numeric (using offset_from_tonic)
    test_numeric = [sargam_to_semitone(note) for note in test_subsequence]
    pattern_numeric = [sargam_to_semitone(note) for note in pattern]
    
    # DTW distance with Sakoe-Chiba band (window constraint)
    distance = dtw.distance(test_numeric, pattern_numeric, window=window)
    
    # Convert to similarity score (0-1)
    max_distance = max(len(test_numeric), len(pattern_numeric)) * 12  # Max semitones
    similarity = 1.0 - (distance / max_distance)
    
    return similarity
```

**Optimization:**
- Precompute pattern n-grams and store in index
- Use parallel processing for DTW (embarrassingly parallel)
- Cache DTW results for repeated patterns

### Fuzzy Matching Parameters

```python
MATCHING_CONFIG = {
    'exact_match_bonus': 1.0,          # Perfect sargam match
    'dtw_threshold': 0.7,              # Minimum DTW similarity
    'max_gap_tolerance': 2,            # Max notes between pattern elements
    'duration_weight': 0.3,            # How much duration matters (0-1)
    'allow_octave_shift': True,        # Match across octaves
    'edit_distance_threshold': 2,      # Max Levenshtein distance
}
```

---

## Aggregation Methods: From Patterns to Raga Score

Given a test recording with detected patterns `[P1, P2, P3, ..., Pk]`, compute scores for each candidate raga. Three methods to implement and compare:

### Method 1: Weighted Voting
```python
def score_raga_weighted_voting(test_patterns, raga_name, pattern_database):
    """
    Simple weighted sum of matched patterns.
    
    Formula:
    score(Raga_X) = Σ count(P_i) × DP(P_i) × match_quality(P_i) × I(P_i ∈ Raga_X)
    
    Where:
    - count(P_i): Frequency of pattern in test recording
    - DP(P_i): Discriminative power (inverse entropy)
    - match_quality(P_i): DTW similarity (0-1)
    - I(...): Indicator function (1 if pattern known for raga, 0 otherwise)
    """
    score = 0.0
    
    for pattern, match_info in test_patterns.items():
        if pattern in pattern_database[raga_name]:
            freq = match_info['count']
            disc_power = pattern_database[pattern]['discriminative_power']
            quality = match_info['dtw_similarity']
            
            score += freq * disc_power * quality
    
    # Add negative evidence
    for pattern in test_patterns:
        neg_score = compute_negative_evidence(pattern, raga_name, pattern_database)
        score += neg_score
    
    return score
```

**Characteristics:**
- Intuitive and interpretable
- Frequency-based (more occurrences = higher score)
- Simple linear combination

### Method 2: TF-IDF Style
```python
def score_raga_tfidf(test_patterns, raga_name, pattern_database):
    """
    Information retrieval inspired scoring.
    
    Formula:
    score(Raga_X) = Σ tf(P_i) × idf(P_i) × match(P_i, Raga_X)
    
    Where:
    - tf(P_i): Term frequency (normalized by test length)
    - idf(P_i): Inverse document frequency = log(N_ragas / ragas_using_P_i)
    - match: Similarity score (DTW or exact)
    """
    test_length = sum(pattern['count'] for pattern in test_patterns.values())
    score = 0.0
    
    for pattern, match_info in test_patterns.items():
        # Term frequency (normalized)
        tf = match_info['count'] / test_length
        
        # Inverse document frequency
        ragas_with_pattern = count_ragas_using_pattern(pattern, pattern_database)
        idf = log(NUM_RAGAS / ragas_with_pattern)
        
        # Match component
        if pattern in pattern_database[raga_name]:
            match_score = match_info['dtw_similarity']
        else:
            match_score = 0.0
        
        score += tf * idf * match_score
    
    return score
```

**Characteristics:**
- Emphasizes rare patterns (high IDF)
- Normalized by recording length
- De-emphasizes overly common patterns

### Method 3: Probabilistic (Naive Bayes)
```python
def score_raga_probabilistic(test_patterns, raga_name, pattern_database):
    """
    Bayesian approach treating patterns as independent evidence.
    
    Formula:
    P(Raga_X | patterns) ∝ P(Raga_X) × Π P(P_i | Raga_X)
    
    Where:
    - P(Raga_X): Prior probability (uniform or based on prevalence)
    - P(P_i | Raga_X): Likelihood of pattern given raga
    """
    # Prior (uniform)
    prior = 1.0 / NUM_RAGAS
    
    # Likelihood
    log_likelihood = log(prior)
    
    for pattern, match_info in test_patterns.items():
        # P(pattern | raga) = occurrences in raga / total occurrences in raga recordings
        occurrences_in_raga = count_pattern_in_raga(pattern, raga_name)
        total_patterns_in_raga = count_all_patterns_in_raga(raga_name)
        
        if total_patterns_in_raga > 0:
            likelihood = (occurrences_in_raga + 1) / (total_patterns_in_raga + NUM_UNIQUE_PATTERNS)  # Laplace smoothing
        else:
            likelihood = 1.0 / NUM_UNIQUE_PATTERNS
        
        # Multiply in log space
        log_likelihood += log(likelihood) * match_info['count']
    
    # Convert back from log
    return exp(log_likelihood)
```

**Characteristics:**
- Probabilistic interpretation
- Handles absence of patterns naturally
- Requires more training data for reliable probability estimates

**Implementation Note:** Implement all three methods and compare on validation set. May also try ensemble (weighted combination of all three).

**Open Question:** In Yaman performances, does the pakad "Ni Re Ga Re" appear many times throughout (favoring Method 1/2) or just once to establish the raga (favoring presence/absence)? Need to analyze actual recordings.

---

## Integration with Existing Pipeline

### Current Pipeline Components
Your existing system provides:

1. **Histogram-based scoring** (`analysis.py`): Global pitch distribution
2. **Directional pattern matching** (planned): Aaroh/avroh usage per note
3. **Transition matrices** (`sequence.py`): Note-to-note transition probabilities
4. **Phrase segmentation** (`sequence.py`): Automatic phrase boundary detection
5. **Note transcription** (`transcription.py`): Unified stationary + inflection points

### New Sequence Mining Component

#### Architecture Integration
```python
# New module: raga_pipeline/mining.py

class SequenceMiner:
    def __init__(self, pattern_database_path):
        self.patterns = load_pattern_database(pattern_database_path)
    
    def mine_patterns_from_recording(self, notes: List[Note]) -> Dict:
        """Extract and match patterns from a recording."""
        pass
    
    def score_ragas(self, matched_patterns: Dict) -> Dict[str, float]:
        """Compute raga scores based on matched patterns."""
        pass

# Integration in driver.py
def run_full_analysis(config):
    # ... existing code ...
    
    # Phase 1: Histogram analysis
    histogram_scores = score_ragas_by_histogram(...)
    
    # Phase 2: Directional patterns
    directional_scores = score_ragas_by_direction(...)
    
    # Phase 3: Transition matrices
    transition_scores = score_ragas_by_transitions(...)
    
    # Phase 4: Sequence mining (NEW)
    from raga_pipeline.mining import SequenceMiner
    miner = SequenceMiner("pretrained_models/pattern_database.pkl")
    matched_patterns = miner.mine_patterns_from_recording(notes)
    sequence_scores = miner.score_ragas(matched_patterns)
    
    # Phase 5: Ensemble
    final_scores = ensemble_scorer(
        histogram_scores,
        directional_scores,
        transition_scores,
        sequence_scores,
        weights=config.ensemble_weights
    )
    
    return final_scores
```

### Ensemble Scoring Strategy

```python
def ensemble_scorer(hist_scores, dir_scores, trans_scores, seq_scores, weights):
    """
    Combine multiple scoring methods with learned or fixed weights.
    
    Weights can be:
    - Global: Same for all ragas
    - Per-raga: Optimized for each raga individually
    - Adaptive: Based on confidence/agreement between methods
    """
    final_scores = {}
    
    for raga in all_ragas:
        if weights.type == 'global':
            final_scores[raga] = (
                weights.w1 * hist_scores.get(raga, 0) +
                weights.w2 * dir_scores.get(raga, 0) +
                weights.w3 * trans_scores.get(raga, 0) +
                weights.w4 * seq_scores.get(raga, 0)
            )
        
        elif weights.type == 'per-raga':
            w = weights.raga_specific[raga]
            final_scores[raga] = (
                w[0] * hist_scores.get(raga, 0) +
                w[1] * dir_scores.get(raga, 0) +
                w[2] * trans_scores.get(raga, 0) +
                w[3] * seq_scores.get(raga, 0)
            )
        
        elif weights.type == 'adaptive':
            # Use method agreement as confidence
            agreement = compute_agreement(hist, dir, trans, seq, raga)
            adaptive_w = adjust_weights_by_agreement(weights.base, agreement)
            final_scores[raga] = sum(...)
    
    return normalize_scores(final_scores)
```

**Open Question - FLAG FOR LATER:**
- Should ensemble weights be **global** (same for all ragas)?
- Or **per-raga** (e.g., some ragas better identified by histograms, others by sequences)?
- How to learn these weights? (Grid search, gradient descent, genetic algorithm?)

### Output Integration

Add sequence mining results to analysis report:

```html
<section id="sequence-patterns">
  <h2>Matched Characteristic Patterns</h2>
  
  <div class="top-patterns">
    <h3>Top 10 Matched Patterns</h3>
    <table>
      <tr>
        <th>Pattern</th>
        <th>Occurrences</th>
        <th>Discriminative Power</th>
        <th>Associated Ragas</th>
      </tr>
      <tr>
        <td><span class="sargam">Ni Re Ga Re</span></td>
        <td>17</td>
        <td>0.92</td>
        <td>Yaman (0.85), Kedar (0.10), Bhoopali Todi (0.05)</td>
      </tr>
      <!-- ... -->
    </table>
  </div>
  
  <div class="pattern-timeline">
    <h3>Pattern Occurrences Over Time</h3>
    <!-- Plotly timeline showing when patterns appear in recording -->
  </div>
  
  <div class="discriminative-patterns">
    <h3>Why Not Other Ragas?</h3>
    <p>Patterns found that are <em>incompatible</em> with other candidates:</p>
    <ul>
      <li><strong>Bihag:</strong> Pattern "Ga ma Dha ni" appears 5 times (forbidden in Bihag)</li>
    </ul>
  </div>
</section>
```

---

## Implementation Phases

### Phase 1: Data Preparation and Analysis (Week 1-2)

**Goal:** Understand the data landscape before building mining algorithms.

#### Step 1.1: Batch Process All Ground Truth Recordings
```bash
# Run analyze mode on all recordings in ground_truth_v6.csv
python -m raga_pipeline.batch_analyze --ground-truth main_notebooks/ground_truth_v6.csv
```

**Output:**
- All recordings processed with note transcription
- Transcribed notes saved as JSON/CSV per recording
- Central database of all note sequences

**Data Structure:**
```json
{
  "recording_id": "yaman_artist1_001",
  "raga": "Yaman",
  "artist": "Artist Name",
  "tonic": "C",
  "duration": 5400,
  "notes": [
    {"sargam": "Sa", "start": 0.0, "end": 0.5, "pitch_hz": 261.6, "offset": 0},
    {"sargam": "Re", "start": 0.5, "end": 0.8, "pitch_hz": 293.7, "offset": 2},
    ...
  ],
  "phrases": [
    {"phrase_id": 0, "start": 0.0, "end": 5.2, "notes": [...]},
    ...
  ]
}
```

#### Step 1.2: Analyze Raga Clustering
```python
# Scan raga_list_final.csv to find aaroh/avroh clusters
python scripts/analyze_raga_clusters.py

# Output: Report showing:
# - How many unique aaroh/avroh patterns exist (~80)
# - Which ragas share each pattern
# - Cluster sizes (how many ragas per pattern)
```

**Deliverable:** `raga_clusters_report.md` with:
- List of all confused clusters
- Priority clusters (most ragas, lowest current accuracy)
- Example recordings for validation

#### Step 1.3: Manual Pakad Verification
Randomly sample 10-20 recordings and manually verify:
- Can you spot known pakads in the transcribed notes?
- Example: Does Yaman transcription show "Ni Re Ga Re"?
- Are transcription errors preventing pattern detection?

**Deliverable:** Transcription quality report with recommendations for preprocessing (smoothing, filtering, etc.)

### Phase 2: Pattern Mining Prototype (Week 3-4)

**Goal:** Build basic mining pipeline and visualize discovered patterns.

#### Step 2.1: Implement Frequent Subsequence Mining
```python
# New file: raga_pipeline/mining.py

from prefixspan import PrefixSpan

def mine_frequent_patterns(note_sequences, min_support=5, max_pattern_length=12):
    """
    Mine frequent subsequences across all recordings.
    """
    # Prepare data for PrefixSpan
    sequences = []
    for recording in note_sequences:
        sequences.append([note.sargam for note in recording.notes])
    
    # Run PrefixSpan
    ps = PrefixSpan(sequences)
    patterns = ps.frequent(min_support, closed=False)
    
    # Filter by length
    patterns = [p for p in patterns if 3 <= len(p[1]) <= max_pattern_length]
    
    return patterns
```

**Parameters to Experiment With:**
- `min_support`: 3, 5, 10
- `max_pattern_length`: 8, 10, 12
- `gap_constraint`: 0, 1, 2

#### Step 2.2: Compute Entropy and Discriminative Scores
```python
def compute_pattern_statistics(patterns, recordings_database):
    """
    For each pattern, compute:
    - Raga distribution
    - Shannon entropy
    - Discriminative power
    - Negative evidence
    """
    pattern_stats = {}
    
    for pattern in patterns:
        stats = compute_pattern_entropy(pattern, recordings_database)
        pattern_stats[pattern] = stats
    
    return pattern_stats
```

#### Step 2.3: Visualization and Exploration
```python
# Top 50 most discriminative patterns
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_pattern_heatmap(pattern_stats, top_k=50):
    """
    Heatmap: Patterns (rows) × Ragas (columns)
    Cell color = occurrence frequency
    """
    # Sort patterns by discriminative power
    top_patterns = sorted(pattern_stats.items(), 
                         key=lambda x: x[1]['discriminative_power'], 
                         reverse=True)[:top_k]
    
    # Build matrix
    matrix = build_pattern_raga_matrix(top_patterns)
    
    # Plot
    plt.figure(figsize=(20, 30))
    sns.heatmap(matrix, xticklabels=raga_names, yticklabels=pattern_labels)
    plt.title("Top Discriminative Patterns per Raga")
    plt.savefig("pattern_heatmap.png")
```

**Deliverables:**
- Pattern database (JSON/pickle)
- Visualization of top patterns
- Analysis report: Which patterns are most discriminative?

### Phase 3: Matching and Scoring (Week 5-6)

**Goal:** Implement pattern matching on test recordings and score ragas.

#### Step 3.1: Implement Two-Stage Matching
```python
def match_patterns_two_stage(test_sequence, pattern_database):
    """
    Stage 1: Fast n-gram filtering
    Stage 2: DTW refinement for candidates
    """
    # Stage 1
    candidates = quick_match(test_sequence, pattern_database)
    
    # Stage 2
    matched_patterns = {}
    for candidate in candidates:
        for start_pos in find_candidate_positions(test_sequence, candidate):
            subsequence = test_sequence[start_pos:start_pos+len(candidate)]
            dtw_score = dtw_match(subsequence, candidate)
            
            if dtw_score > DTW_THRESHOLD:
                matched_patterns[candidate] = {
                    'count': matched_patterns.get(candidate, {}).get('count', 0) + 1,
                    'dtw_similarity': dtw_score,
                    'positions': [start_pos]
                }
    
    return matched_patterns
```

#### Step 3.2: Implement All Three Aggregation Methods
- Weighted Voting (Method 1)
- TF-IDF Style (Method 2)
- Probabilistic (Method 3)

Test each on validation set to compare.

#### Step 3.3: Handle Same Aaroh/Avroh Clusters
```python
def optimize_for_confused_clusters(pattern_database, confused_clusters, validation_set):
    """
    For each confused cluster:
    1. Test current accuracy
    2. If below threshold, mine discriminative patterns
    3. Boost discriminative pattern weights
    4. Re-test
    """
    for cluster in confused_clusters:
        accuracy = test_cluster_accuracy(cluster, validation_set)
        
        if accuracy < MINIMUM_ACCURACY:
            discriminative_patterns = mine_discriminative_patterns_for_cluster(cluster)
            update_pattern_weights(pattern_database, discriminative_patterns)
    
    return pattern_database
```

**Deliverable:** Pattern database with optimized weights for confused clusters.

### Phase 4: Evaluation and Validation (Week 7)

**Goal:** Rigorous testing on held-out data.

#### Step 4.1: Cross-Validation
```python
def cross_validate_sequence_mining(recordings_database, n_folds=5):
    """
    K-fold cross-validation per raga.
    
    For each raga:
    - Split recordings into 5 folds
    - Train on 4 folds, test on 1
    - Repeat 5 times
    """
    results = {}
    
    for raga in all_ragas:
        raga_recordings = [r for r in recordings_database if r.raga == raga]
        folds = create_folds(raga_recordings, n_folds)
        
        fold_accuracies = []
        for test_fold in folds:
            train_folds = folds - test_fold
            
            # Mine patterns from training data
            patterns = mine_frequent_patterns(train_folds)
            
            # Test on test fold
            accuracy = evaluate(patterns, test_fold)
            fold_accuracies.append(accuracy)
        
        results[raga] = {
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies)
        }
    
    return results
```

#### Step 4.2: Confusion Matrix Analysis
```python
def generate_confusion_matrix(test_results):
    """
    Which ragas get confused with each other?
    Focus on confused clusters (same aaroh/avroh).
    """
    matrix = np.zeros((NUM_RAGAS, NUM_RAGAS))
    
    for true_raga, predicted_raga in test_results:
        matrix[true_raga, predicted_raga] += 1
    
    # Visualize
    plot_confusion_matrix(matrix)
    
    # Identify worst confusions
    worst_pairs = find_top_confusions(matrix, top_k=20)
    
    return matrix, worst_pairs
```

#### Step 4.3: Ablation Studies
Test impact of various design choices:

```python
experiments = [
    {'name': 'baseline', 'pattern_length': [3,8], 'gap': 0, 'method': 'weighted_voting'},
    {'name': 'with_gaps', 'pattern_length': [3,8], 'gap': 2, 'method': 'weighted_voting'},
    {'name': 'longer_patterns', 'pattern_length': [5,12], 'gap': 0, 'method': 'weighted_voting'},
    {'name': 'tfidf_method', 'pattern_length': [3,8], 'gap': 0, 'method': 'tfidf'},
    {'name': 'probabilistic', 'pattern_length': [3,8], 'gap': 0, 'method': 'naive_bayes'},
]

for exp in experiments:
    results = run_experiment(exp)
    report_results(exp['name'], results)
```

**Deliverables:**
- Cross-validation results per raga
- Confusion matrix
- Ablation study report
- Recommendations for best configuration

### Phase 5: Integration and Ensemble (Week 8)

**Goal:** Integrate sequence mining with existing pipeline components.

#### Step 5.1: Implement Ensemble Scoring
```python
def create_ensemble_scorer(config):
    """
    Combine histogram, directional, transition, and sequence scores.
    """
    weights = config.ensemble_weights
    
    def ensemble_score(recording):
        scores = {
            'histogram': histogram_scorer(recording),
            'directional': directional_scorer(recording),
            'transition': transition_scorer(recording),
            'sequence': sequence_scorer(recording)
        }
        
        final = {}
        for raga in all_ragas:
            final[raga] = sum(
                weights[method] * scores[method].get(raga, 0)
                for method in weights
            )
        
        return final
    
    return ensemble_score
```

#### Step 5.2: Learn Optimal Ensemble Weights
```python
def optimize_ensemble_weights(validation_set):
    """
    Grid search or gradient-based optimization to find best weights.
    """
    from sklearn.model_selection import GridSearchCV
    
    # Define search space
    param_grid = {
        'w_histogram': [0.1, 0.2, 0.3, 0.4],
        'w_directional': [0.1, 0.2, 0.3, 0.4],
        'w_transition': [0.1, 0.2, 0.3, 0.4],
        'w_sequence': [0.1, 0.2, 0.3, 0.4]
    }
    
    # Constraint: weights sum to 1.0
    valid_combinations = [
        combo for combo in product(*param_grid.values())
        if abs(sum(combo) - 1.0) < 0.01
    ]
    
    best_accuracy = 0
    best_weights = None
    
    for weights in valid_combinations:
        accuracy = evaluate_ensemble(validation_set, weights)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = weights
    
    return best_weights
```

#### Step 5.3: Update HTML Reports
Add sequence mining visualizations to analysis reports (see Integration section above).

**Deliverable:** Fully integrated pipeline with ensemble scoring.

### Phase 6: Refinement and Advanced Features (Week 9-10)

**Goal:** Polish and add advanced capabilities.

#### Step 6.1: Multi-Resolution Pattern Mining
- Experiment with smoothing before mining
- Try different gap tolerances
- Combine exact + fuzzy patterns

#### Step 6.2: Temporal Context Exploration
- Test segmented mining (Option B)
- Analyze which patterns are alap-specific vs. fast passages
- Visualize patterns on timeline

#### Step 6.3: Educational Features
```python
def generate_raga_pattern_summary(raga_name, pattern_database):
    """
    For educational tool: show all characteristic patterns for a raga.
    
    Output:
    - Core pakads (high discriminative power)
    - Common phrases (high frequency)
    - Unique patterns (appear in this raga only)
    - Shared patterns (appears in related ragas too)
    """
    patterns = pattern_database.get_patterns_for_raga(raga_name)
    
    return {
        'pakads': [p for p in patterns if p.discriminative_power > 0.8],
        'common': [p for p in patterns if p.total_occurrences > 20],
        'unique': [p for p in patterns if len(p.raga_distribution) == 1],
        'shared': [p for p in patterns if len(p.raga_distribution) > 1]
    }
```

**Deliverable:** Enhanced pattern database with educational metadata.

---

## Open Questions and Future Work

### Questions Flagged for Later Discussion

#### 1. Ensemble Weight Strategy
**Question:** Should ensemble weights be global (same for all ragas) or per-raga (optimized individually)?

**Considerations:**
- Global: Simpler, more robust, easier to explain
- Per-raga: More flexible, could capture that some ragas are "melodic" (sequence-based) vs. "harmonic" (histogram-based)
- Hybrid: Global base + per-raga adjustments

**Next Steps:** 
- Start with global weights
- If accuracy varies significantly across ragas, try per-raga
- Use validation set to measure benefit

#### 2. Temporal Context Approach
**Question:** Use mixed (Option A), segmented (Option B), or natural clustering (Option C)?

**Needs testing:** Does segmentation improve accuracy or just add complexity?

#### 3. Cross-Artist/Gharana Validation
**FLAG FOR LATER:** Test if patterns generalize across different artists/gharanas.

**Implementation:**
```python
def cross_artist_validation(recordings_database):
    """
    Train on Artists 1-8, test on Artists 9-10.
    Measures generalization beyond specific gayaki styles.
    """
    pass
```

**Consideration:** Gharana differences may require:
- Artist-specific pattern sets
- Shared "core" patterns + artist "style" patterns
- Weighted by artist prevalence in test scenarios

#### 4. Pattern Length Distribution
**Question:** What is the natural distribution of pattern lengths in actual performances?

**Analysis needed:**
- Mine without length constraints
- Plot histogram of pattern lengths
- Does distribution differ by raga? By section (alap vs. tan)?

#### 5. Computational Efficiency
**Question:** For 1-1.5 hour recordings with ~5000 notes each, how long does matching take?

**Optimization strategies:**
- Precompute pattern n-grams (done)
- Parallel DTW (easy to parallelize)
- Approximate DTW (FastDTW)
- GPU acceleration for batch processing

**Target:** < 30 seconds per recording for real-time application

---

## Expected Insights and Outcomes

### Automatic Discovery of Musical Knowledge

This system should automatically learn:

1. **Pakads (Signature Phrases):**
   - Yaman: Ni Re Ga Re, Ni Dha Pa Ma Ga Re
   - Bhairavi: Dha re Sa, Ni Dha Pa ga ma ga Re
   - Bhupali: Sa Re Ga Pa Dha Sa

2. **Characteristic Movements:**
   - Which ragas favor ascending motion (arohi pradhaan)
   - Which use descending emphasis (avarohi pradhaan)
   - Interval preferences (komal vs. shuddh)

3. **Structural Patterns:**
   - Alap-specific patterns (slow, expansive)
   - Vilambit patterns (structured, rhythmic)
   - Tan patterns (fast sequences)

4. **Confusability Clusters:**
   - Which ragas share melodic vocabulary (jati relationships)
   - Which phrase separates Raga A from Raga B
   - Hierarchical raga families

### Performance Metrics

**Target Accuracy:**
- Overall: > 85% top-1 accuracy
- Within confused clusters: > 70% top-1 accuracy
- Top-3 accuracy: > 95%

**Computational:**
- Pattern mining: < 1 hour for full database (one-time)
- Matching: < 30 seconds per test recording
- Memory: Pattern database < 100MB

### Educational Value

**Interactive Exploration:**
- "Show me all Yaman patterns" → Display ranked list with examples
- "Why is this Yaman and not Bihag?" → Show discriminative patterns
- "Find this pattern in other ragas" → Cross-raga pattern search

**Pattern Comparison:**
- Visual diff of Bhimpalasi vs. Kafi patterns
- Timeline view of pattern occurrences
- Audio playback of pattern examples

---

## Reference Papers

### Paper 1: Gulati et al. - Phrase-Based Raga Recognition
**File:** `/Users/saumyamishra/Desktop/intern/summer25/RagaDetection/non code/papers/gulati2016phraseRaga_icassp.pdf`

**Key contributions to extract:**
- How do they define "phrases"?
- What pattern matching algorithm do they use?
- How do they handle tempo variations?
- Performance metrics on their dataset

### Paper 2: Mining Melodic Patterns
**File:** `/Users/saumyamishra/Desktop/intern/summer25/RagaDetection/non code/papers/miningmelodicpatterns.pdf`

**Key contributions to extract:**
- Specific pattern mining algorithm (PrefixSpan? SPADE? Custom?)
- How do they represent melodic sequences?
- Discriminative vs. frequent pattern strategies
- Evaluation methodology

### Paper 3: Knowledge Graph Approach (KKG-PR-JNMR)
**File:** `/Users/saumyamishra/Desktop/intern/summer25/RagaDetection/non code/papers/kkg-pr-jnmr2021.pdf`

**Key contributions to extract:**
- Knowledge graph structure (nodes, edges)
- How is musical hierarchy represented?
- Query mechanisms for pattern matching
- Integration with other features

**Next step:** Read papers in detail and extract specific technical approaches to incorporate into implementation.

---

## Repository Structure Updates

### New Files to Create
```
raga_pipeline/
├── mining.py                    # Pattern mining algorithms
├── pattern_matching.py          # Two-stage matching + DTW
├── ensemble.py                  # Ensemble scoring strategies
└── pattern_database.pkl         # Serialized pattern database

scripts/
├── mine_patterns.py             # CLI for pattern mining
├── analyze_raga_clusters.py     # Cluster analysis
├── batch_analyze.py             # Batch processing for ground truth
└── evaluate_patterns.py         # Cross-validation and evaluation

data/
└── ground_truth_v6_processed/   # Processed note sequences
    └── [recording_id].json      # One file per recording

results/
└── pattern_mining/
    ├── pattern_database.json
    ├── pattern_heatmap.png
    ├── confusion_matrix.png
    └── evaluation_report.md
```

### Documentation Updates
- `DOCUMENTATION.md`: Add pattern mining section
- `LLM_REFERENCE.md`: Add mining.py module description
- `CHANGELOG.md`: Track mining implementation milestones
- `sequence-mining-plan.md`: This document (keep updated)

---

## Success Criteria

### Phase 1 Complete:
- ✓ All ground_truth_v6 recordings processed
- ✓ Note sequences exported to structured format
- ✓ Raga clusters identified and documented
- ✓ Manual verification shows good transcription quality

### Phase 2 Complete:
- ✓ Pattern mining pipeline functional
- ✓ Entropy/discriminative scores computed
- ✓ Pattern visualization generated
- ✓ Top 50 patterns identified and interpretable

### Phase 3 Complete:
- ✓ Two-stage matching implemented
- ✓ All three aggregation methods tested
- ✓ Confused cluster optimization working
- ✓ Initial accuracy results > 70%

### Phase 4 Complete:
- ✓ Cross-validation results documented
- ✓ Confusion matrix analyzed
- ✓ Ablation studies complete
- ✓ Best configuration identified

### Phase 5 Complete:
- ✓ Ensemble integrated with existing pipeline
- ✓ Optimal weights learned
- ✓ HTML reports include pattern analysis
- ✓ Overall accuracy improved by ≥10% over baseline

### Phase 6 Complete:
- ✓ Advanced features implemented
- ✓ Educational metadata available
- ✓ Pattern database ready for deployment
- ✓ System ready for user testing

---

## Timeline Summary

| Phase | Duration | Description |
|-------|----------|-------------|
| 1. Data Prep | 2 weeks | Batch analysis, cluster identification, verification |
| 2. Mining Prototype | 2 weeks | Pattern extraction, entropy scoring, visualization |
| 3. Matching & Scoring | 2 weeks | Two-stage matching, aggregation methods, cluster optimization |
| 4. Evaluation | 1 week | Cross-validation, confusion analysis, ablations |
| 5. Integration | 1 week | Ensemble scoring, weight optimization, reporting |
| 6. Refinement | 2 weeks | Advanced features, educational components, polish |

**Total: 10 weeks** (~2.5 months)

---

## Next Immediate Actions

1. **Read papers in detail** to extract specific algorithms and parameters
2. **Run batch analysis** on all ground_truth_v6 recordings
3. **Analyze raga clusters** to quantify the confused cluster problem
4. **Manual pattern verification** on sample recordings
5. **Implement PrefixSpan** for initial pattern mining prototype

**Which would you like to start with?**
