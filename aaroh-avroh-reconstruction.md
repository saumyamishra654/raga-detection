# Aaroh/Avroh Directional Pattern Matching

## Overview

**Goal:** Create a directional signature for each detected recording by analyzing which notes appear in ascending vs descending contexts, then match this signature against known raga patterns to distinguish between ragas with the same note set but different directional usage.

**Core Insight:** The key differentiator between ragas like **Bhimpalasi** (no Re/Dha in aaroh) and **Kafi** (all notes bidirectional) is NOT which notes exist, but **which directions each note is used in**. Most notes appear in both directions — the diagnostic feature is when a note is *missing* from one direction or appears as "alp" (weak/occasional).

**Critical Distinction:**
- **Bhimpalasi:** Sa ga ma Pa ni Sa' (no Re, no Dha ascending) | Sa' ni Dha Pa ma ga Re Sa (Re and Dha only descending)
- **Kafi:** Sa Re ga ma Pa Dha ni Sa' | Sa' ni Dha Pa ma ga Re Sa (all notes bidirectional)
- Same notes, different directional patterns → This is what we detect and match

---

## Musical Background

### Traditional Aaroh/Avroh
- **Aaroh:** The ascending scale pattern specific to a raga
- **Avroh:** The descending scale pattern
- **Key Property:** Notes may be **omitted** in one direction, creating asymmetric patterns
- **Alp Swar:** Weak/occasional notes that appear infrequently (represented as 0.5)
- **Diagnostic Value:** Directional omissions distinguish ragas with identical note sets

### Critical Examples: Same Notes, Different Patterns

#### Bhimpalasi vs Kafi
Both use: Sa, Re, ga, ma, Pa, Dha, ni

| Note | Bhimpalasi Pattern | Kafi Pattern | Encoding |
|------|-------------------|--------------|----------|
| Sa   | Both              | Both         | Both: (1, 1) |
| Re   | **Avroh only** ❌ | Both ✓       | Bhim: (0, 1), Kafi: (1, 1) |
| ga   | Both              | Both         | Both: (1, 1) |
| ma   | Both              | Both         | Both: (1, 1) |
| Pa   | Both              | Both         | Both: (1, 1) |
| Dha  | **Avroh only** ❌ | Both ✓       | Bhim: (0, 1), Kafi: (1, 1) |
| ni   | Both              | Both         | Both: (1, 1) |

**Pattern Vectors:**
- Bhimpalasi: `[(1,1), (0,1), (1,1), (1,1), (1,1), (0,1), (1,1)]` ← Re and Dha missing from aaroh
- Kafi: `[(1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1)]` ← All bidirectional

#### Deshkar with Alp Swar
Deshkar occasionally uses Re as an alp (weak) note.

| Note | Deshkar Pattern | Encoding |
|------|----------------|----------|
| Sa   | Both           | (1, 1) |
| Re   | **Alp (weak)** | (0.5, 0.5) |
| Ga   | Both           | (1, 1) |
| ma   | Both           | (1, 1) |
| Pa   | Both           | (1, 1) |

**Pattern Vector:** `[(1,1), (0.5,0.5), (1,1), (1,1), (1,1)]`

### Other Examples
- **Marwa:** ma omitted in aaroh (only Ma used ascending)
- **Puriya Dhanashree:** Re omitted in aaroh
- **Bageshri:** Ga omitted in aaroh (uses ga only)

---

## Algorithm Design

### Input Data
From existing pipeline ([sequence.py](raga_pipeline/sequence.py)):
- **`phrases: List[Phrase]`** — Each phrase contains a list of `Note` objects
- **`Note`** attributes:
  - `sargam: str` — Note name (e.g., "Sa", "Re", "ga")
  - `start_time: float`
  - `end_time: float`
  - `pitch_hz: float`
  - `offset_from_tonic: int` — Semitone offset (0-11)

### Step 1: Compute Directional Edge Counts
For each note, track the **edges** (transitions) leading TO that note from below (ascending edge) or from above (descending edge).

**Key Insight:** We count *incoming transitions*, not just note occurrences. A note that appears frequently but is only approached from below has strong "aaroh" evidence.

```python
def compute_directional_edges(phrases: List[Phrase]) -> Dict[str, Tuple[int, int]]:
    """
    Returns {sargam: (ascending_edges, descending_edges)}.
    
    An ascending edge = a transition TO this note FROM a lower note.
    A descending edge = a transition TO this note FROM a higher note.
    """
    edges = defaultdict(lambda: [0, 0])  # [asc_count, desc_count]
    
    for phrase in phrases:
        notes = phrase.notes
        
        for i in range(1, len(notes)):  # Start from second note
            current = notes[i]
            previous = notes[i - 1]
            
            # Skip repeated notes (same pitch class)
            if current.sargam == previous.sargam:
                continue
            
            pitch_diff = current.pitch_hz - previous.pitch_hz
            
            if pitch_diff > 0:
                edges[current.sargam][0] += 1  # Ascending edge TO current
            elif pitch_diff < 0:
                edges[current.sargam][1] += 1  # Descending edge TO current
    
    return {sargam: (asc, desc) for sargam, (asc, desc) in edges.items()}
```

**Example Output:**
```python
{
    'Sa': (45, 42),   # Approached from both directions (balanced)
    'Re': (5, 38),    # Mostly approached from above → Avroh only (like Bhimpalasi)
    'ga': (32, 28),   # Balanced → Both directions
    'ma': (29, 27),   # Balanced → Both directions
    'Pa': (35, 33),   # Balanced → Both directions
    'Dha': (3, 41),   # Mostly approached from above → Avroh only (like Bhimpalasi)
    'ni': (30, 29),   # Balanced → Both directions
}
```

### Step 2: Compute Directional Scores (Continuous 0-1 Range)
Convert edge counts to a continuous score representing "how much this note belongs in each direction."

**Formula:**
```python
def compute_directional_scores(edges: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[float, float]]:
    """
    Returns {sargam: (aaroh_score, avroh_score)}.
    
    Scores range from 0 (never used in this direction) to 1 (always used).
    
    Score calculation:
    - aaroh_score = ascending_edges / (ascending_edges + descending_edges)
    - avroh_score = descending_edges / (ascending_edges + descending_edges)
    
    Notes:
    - Scores sum to 1.0 for each note
    - 0.5 indicates balanced bidirectional usage
    - Close to 0 or 1 indicates strong directional preference
    """
    scores = {}
    
    for sargam, (asc, desc) in edges.items():
        total = asc + desc
        
        if total == 0:
            # Note never appears with directional context (e.g., only at phrase boundaries)
            scores[sargam] = (0.0, 0.0)
            continue
        
        aaroh_score = asc / total
        avroh_score = desc / total
        
        scores[sargam] = (aaroh_score, avroh_score)
    
    return scores
```

**Example Output (Bhimpalasi-like pattern):**
```python
{
    'Sa': (0.52, 0.48),   # ~0.5 = bidirectional
    'Re': (0.12, 0.88),   # High avroh score → avroh-only like Bhimpalasi
    'ga': (0.53, 0.47),   # Bidirectional
    'ma': (0.52, 0.48),   # Bidirectional
    'Pa': (0.51, 0.49),   # Bidirectional
    'Dha': (0.07, 0.93),  # High avroh score → avroh-only like Bhimpalasi
    'ni': (0.51, 0.49),   # Bidirectional
}
```

### Step 3: Build Raga Pattern Database
Encode known ragas as directional pattern vectors.

**Pattern Encoding:**
- **1**: Note is fully present in this direction
- **0**: Note is omitted in this direction
- **0.5**: Note is alp (weak/occasional) in this direction
- **Format:** List of `(aaroh_presence, avroh_presence)` tuples, one per chromatic note

```python
# Chromatic order (12 semitones)
CHROMATIC_SARGAM = ["Sa", "re", "Re", "ga", "Ga", "ma", "Ma", "Pa", "dha", "Dha", "ni", "Ni"]

# Raga pattern database
RAGA_PATTERNS = {
    "Bhimpalasi": {
        "notes": ["Sa", "Re", "ga", "ma", "Pa", "Dha", "ni"],
        "pattern": [
            (1, 1),    # Sa: both
            (0, 0),    # re: not used
            (0, 1),    # Re: avroh only ← KEY DIAGNOSTIC
            (0, 0),    # ga: wait, komal ga IS used...
            (1, 1),    # ga: both (komal)
            (0, 0),    # Ga: not used
            (1, 1),    # ma: both (komal)
            (0, 0),    # Ma: not used
            (1, 1),    # Pa: both
            (0, 0),    # dha: not used
            (0, 1),    # Dha: avroh only ← KEY DIAGNOSTIC
            (1, 1),    # ni: both (komal)
            (0, 0),    # Ni: not used
        ]
    },
    
    "Kafi": {
        "notes": ["Sa", "Re", "ga", "ma", "Pa", "Dha", "ni"],
        "pattern": [
            (1, 1),    # Sa: both
            (0, 0),    # re: not used
            (1, 1),    # Re: both ← DIFFERENCE from Bhimpalasi
            (1, 1),    # ga: both (komal)
            (0, 0),    # Ga: not used
            (1, 1),    # ma: both (komal)
            (0, 0),    # Ma: not used
            (1, 1),    # Pa: both
            (0, 0),    # dha: not used
            (1, 1),    # Dha: both ← DIFFERENCE from Bhimpalasi
            (1, 1),    # ni: both (komal)
            (0, 0),    # Ni: not used
        ]
    },
    
    "Deshkar": {
        "notes": ["Sa", "Re", "Ga", "ma", "Pa", "Dha", "Ni"],
        "pattern": [
            (1, 1),      # Sa: both
            (0.5, 0.5),  # Re: alp (weak) ← Uses 0.5 for occasional notes
            (1, 1),      # Ga: both (shuddh)
            (1, 1),      # ma: both (komal)
            (0, 0),      # Ma: not used
            (1, 1),      # Pa: both
            (1, 1),      # Dha: both (shuddh)
            (1, 1),      # Ni: both (shuddh)
        ]
    },
    
    "Yaman": {
        "notes": ["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"],
        "pattern": [
            (1, 1),    # Sa: both
            (1, 1),    # Re: both (shuddh)
            (1, 1),    # Ga: both (shuddh)
            (0, 0),    # ma: not used
            (1, 1),    # Ma: both (teevra) ← Only Ma, not ma
            (1, 1),    # Pa: both
            (1, 1),    # Dha: both (shuddh)
            (1, 1),    # Ni: both (shuddh)
        ]
    }
}
```

### Step 4: Compute Pattern Similarity Score
Match detected directional scores against raga pattern database using a distance metric.

**Approach: Weighted Euclidean Distance**

```python
def compute_pattern_distance(
    detected_scores: Dict[str, Tuple[float, float]],
    raga_pattern: List[Tuple[float, float]],
    chromatic_order: List[str] = CHROMATIC_SARGAM
) -> float:
    """
    Compute weighted Euclidean distance between detected and expected patterns.
    
    Lower distance = better match.
    
    Weighting strategy:
    - Missing notes (0,0) in raga pattern: skip comparison (no penalty)
    - Present notes: compare both aaroh and avroh scores
    - Alp notes (0.5, 0.5): partial comparisons expected
    """
    total_distance = 0.0
    num_compared = 0
    
    for sargam, (expected_aaroh, expected_avroh) in zip(chromatic_order, raga_pattern):
        # Skip notes not used in this raga
        if expected_aaroh == 0 and expected_avroh == 0:
            continue
        
        # Get detected scores (default to 0,0 if note never appeared)
        detected_aaroh, detected_avroh = detected_scores.get(sargam, (0.0, 0.0))
        
        # Compute squared differences
        aaroh_diff = (detected_aaroh - expected_aaroh) ** 2
        avroh_diff = (detected_avroh - expected_avroh) ** 2
        
        total_distance += aaroh_diff + avroh_diff
        num_compared += 2  # Two dimensions per note
    
    if num_compared == 0:
        return float('inf')  # No overlap
    
    # Return normalized RMSE
    return (total_distance / num_compared) ** 0.5


def rank_ragas_by_pattern_match(
    detected_scores: Dict[str, Tuple[float, float]],
    raga_database: Dict[str, dict]
) -> List[Tuple[str, float]]:
    """
    Rank ragas by pattern similarity.
    
    Returns list of (raga_name, distance) tuples sorted by distance (ascending).
    """
    results = []
    
    for raga_name, raga_info in raga_database.items():
        distance = compute_pattern_distance(
            detected_scores,
            raga_info["pattern"]
        )
        results.append((raga_name, distance))
    
    # Sort by distance (smaller = better match)
    results.sort(key=lambda x: x[1])
    
    return results
```

**Example Matching:**
```python
detected = {
    'Sa': (0.52, 0.48),
    'Re': (0.12, 0.88),   # Avroh-dominant
    'ga': (0.53, 0.47),
    'ma': (0.52, 0.48),
    'Pa': (0.51, 0.49),
    'Dha': (0.07, 0.93),  # Avroh-dominant
    'ni': (0.51, 0.49),
}

results = rank_ragas_by_pattern_match(detected, RAGA_PATTERNS)
# Results:
# [
#   ("Bhimpalasi", 0.08),  ← Best match (Re and Dha are avroh-only)
#   ("Kafi", 0.35),        ← Worse match (expects Re and Dha bidirectional)
#   ("Yaman", 0.62),       ← Poor match (wrong notes)
#   ...
# ]
```

### Step 5: Handle Ambiguities and Edge Cases

#### 5.1 Minimum Sample Threshold
Ignore notes with very few edges to avoid noise.

```python
MIN_EDGE_COUNT = 5  # Require at least 5 total edges per note

def filter_low_confidence_notes(edges: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    """Remove notes with insufficient evidence."""
    return {
        sargam: (asc, desc)
        for sargam, (asc, desc) in edges.items()
        if (asc + desc) >= MIN_EDGE_COUNT
    }
```

#### 5.2 Duration Weighting
Weight edges by note duration instead of just counting transitions.

```python
def compute_duration_weighted_edges(phrases: List[Phrase]) -> Dict[str, Tuple[float, float]]:
    """
    Weight edge contributions by note duration.
    Longer notes have more influence on pattern detection.
    """
    edges = defaultdict(lambda: [0.0, 0.0])
    
    for phrase in phrases:
        notes = phrase.notes
        
        for i in range(1, len(notes)):
            current = notes[i]
            previous = notes[i - 1]
            
            if current.sargam == previous.sargam:
                continue
            
            duration_weight = current.end_time - current.start_time
            pitch_diff = current.pitch_hz - previous.pitch_hz
            
            if pitch_diff > 0:
                edges[current.sargam][0] += duration_weight
            elif pitch_diff < 0:
                edges[current.sargam][1] += duration_weight
    
    return {sargam: (asc, desc) for sargam, (asc, desc) in edges.items()}
```

#### 5.3 Confidence Score
Compute overall confidence based on sample size and score clarity.

```python
def compute_pattern_confidence(
    edges: Dict[str, Tuple[int, int]],
    scores: Dict[str, Tuple[float, float]]
) -> float:
    """
    Compute confidence in pattern detection (0-1).
    
    High confidence when:
    - Large sample size (many edges)
    - Clear directional preferences (scores near 0 or 1, not 0.5)
    """
    total_edges = sum(asc + desc for asc, desc in edges.values())
    
    # Sample size factor (sigmoid)
    sample_factor = min(1.0, total_edges / 200)  # Saturate at 200 edges
    
    # Clarity factor (how far scores deviate from ambiguous 0.5)
    clarity_scores = []
    for aaroh_score, avroh_score in scores.values():
        # Distance from ambiguous 0.5 value
        aaroh_clarity = abs(aaroh_score - 0.5)
        avroh_clarity = abs(avroh_score - 0.5)
        clarity_scores.append(max(aaroh_clarity, avroh_clarity))
    
    clarity_factor = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0
    
    return sample_factor * clarity_factor
```

---

## Refinements and Edge Cases

### 1. Handling Octave Transitions
- Track **absolute pitch** ranges but normalize to pitch class for pattern matching
- Octave-spanning phrases provide the most reliable directional evidence
- Weight transitions that cross Pa (perfect fifth) more heavily

### 2. Phrase Boundary Handling
- First note of each phrase has weak directional context (no incoming edge)
- Consider creating synthetic edges from phrase-final to phrase-initial notes if phrases are closely spaced
- Alternative: Skip first note of each phrase entirely

### 3. Gamakas and Ornamentations
- Rapid oscillations (gamakas, murkis) create many micro-edges
- Filter by minimum note duration (e.g., >100ms) before computing edges
- Alternative: Smooth pitch contours before edge detection

### 4. Statistical Robustness
- Bootstrap resampling: compute pattern multiple times with random phrase subsets
- Report confidence intervals for directional scores
- Flag notes with high variance across bootstrap samples

---

## Integration Points

### Existing Pipeline Modules

#### `raga_pipeline/sequence.py`
Add new functions for directional pattern analysis:

```python
@dataclass
class DirectionalPattern:
    """Container for directional pattern analysis results."""
    edge_counts: Dict[str, Tuple[int, int]]        # Raw edge counts
    directional_scores: Dict[str, Tuple[float, float]]  # Normalized (0-1) scores
    confidence: float                              # Overall detection confidence
    matched_ragas: List[Tuple[str, float]]        # Ranked raga matches with distances
    
def analyze_directional_patterns(phrases: List[Phrase]) -> DirectionalPattern:
    """
    Analyze phrase transitions to detect directional usage patterns.
    
    Returns DirectionalPattern with edge statistics and raga matches.
    """
    pass
```

Call from `analyze_raga_patterns()` or as a standalone post-processing step after phrase detection.

#### `raga_pipeline/raga.py`
Extend raga database to include directional pattern encodings:

```python
@dataclass
class RagaInfo:
    name: str
    notes: List[str]
    aroha: str                    # Traditional text representation
    avroha: str                   # Traditional text representation
    pattern_vector: List[Tuple[float, float]]  # NEW: Directional encoding
    vadi: Optional[str]
    samvadi: Optional[str]
    
def load_raga_patterns(csv_path: str) -> Dict[str, RagaInfo]:
    """
    Parse CSV with new columns: 'aaroh_pattern', 'avroh_pattern'.
    
    Format example:
    - aaroh_pattern: "1,0,1,1,1,0,1,1,0,1,1,0"  (12 values, one per chromatic note)
    - avroh_pattern: "1,0,1,1,1,0,1,1,0,1,1,0"
    - alp notes: use 0.5
    """
    pass
```

#### `raga_pipeline/output.py`
Add directional pattern visualization to analysis report:

```html
<section id="directional-patterns">
  <h2>Directional Pattern Analysis</h2>
  
  <!-- Heatmap showing aaroh/avroh scores per note -->
  <div id="pattern-heatmap"></div>
  
  <!-- Top 5 raga matches -->
  <table class="pattern-matches">
    <thead>
      <tr>
        <th>Raga</th>
        <th>Pattern Distance</th>
        <th>Expected Pattern</th>
        <th>Detected Pattern</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Bhimpalasi</td>
        <td>0.08 ⭐</td>
        <td>
          <span class="note aaroh-only">Sa</span>
          <span class="note avroh-only">Re</span>
          <span class="note both">ga ma Pa</span>
          <span class="note avroh-only">Dha</span>
          <span class="note both">ni</span>
        </td>
        <td>
          <!-- Color-coded detected pattern -->
        </td>
      </tr>
      <!-- ... -->
    </tbody>
  </table>
  
  <!-- Edge statistics table -->
  <table class="edge-stats">
    <thead>
      <tr>
        <th>Note</th>
        <th>Ascending Edges</th>
        <th>Descending Edges</th>
        <th>Aaroh Score</th>
        <th>Avroh Score</th>
        <th>Pattern</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Re</td>
        <td>5</td>
        <td>38</td>
        <td>0.12</td>
        <td>0.88</td>
        <td><span class="badge avroh-only">Avroh Only</span></td>
      </tr>
      <!-- ... -->
    </tbody>
  </table>
</section>
```

### CLI Integration
```bash
./run_pipeline.sh analyze \
  --audio "song.mp3" \
  --tonic "C#" \
  --raga "Bhimpalasi, Kafi" \
  --enable-pattern-matching  # NEW FLAG
```

Add to `config.py`:
```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    enable_pattern_matching: bool = False  # Directional pattern analysis
    pattern_min_edges: int = 5            # Minimum edges per note
    pattern_duration_weighted: bool = True # Weight by duration
```

---

## Output Format

### Data Structure
```python
@dataclass
class DirectionalPattern:
    """Results of directional pattern analysis."""
    edge_counts: Dict[str, Tuple[int, int]]
    # Raw edge counts: {sargam: (ascending_edges, descending_edges)}
    # Example: {'Re': (5, 38), 'ga': (32, 28)}
    
    directional_scores: Dict[str, Tuple[float, float]]
    # Normalized scores (0-1): {sargam: (aaroh_score, avroh_score)}
    # Example: {'Re': (0.12, 0.88), 'ga': (0.53, 0.47)}
    
    matched_ragas: List[Tuple[str, float]]
    # Ranked raga matches: [(raga_name, pattern_distance)]
    # Example: [('Bhimpalasi', 0.08), ('Kafi', 0.35), ...]
    
    confidence: float
    # Overall detection confidence (0-1)
    # Based on sample size and score clarity
```

### JSON Export
```json
{
  "directional_pattern": {
    "edge_counts": {
      "Sa": [45, 42],
      "Re": [5, 38],
      "ga": [32, 28],
      "ma": [29, 27],
      "Pa": [35, 33],
      "Dha": [3, 41],
      "ni": [30, 29]
    },
    "directional_scores": {
      "Sa": [0.52, 0.48],
      "Re": [0.12, 0.88],
      "ga": [0.53, 0.47],
      "ma": [0.52, 0.48],
      "Pa": [0.51, 0.49],
      "Dha": [0.07, 0.93],
      "ni": [0.51, 0.49]
    },
    "matched_ragas": [
      {"name": "Bhimpalasi", "distance": 0.08, "rank": 1},
      {"name": "Kafi", "distance": 0.35, "rank": 2},
      {"name": "Deshkar", "distance": 0.52, "rank": 3}
    ],
    "confidence": 0.82
  }
}
```

### CSV Export (for training data)
```csv
filename,raga_detected,raga_ground_truth,confidence,Re_aaroh,Re_avroh,Dha_aaroh,Dha_avroh
bhimpalasi_001.mp3,Bhimpalasi,Bhimpalasi,0.82,0.12,0.88,0.07,0.93
kafi_002.mp3,Kafi,Kafi,0.76,0.51,0.49,0.53,0.47
```

---

## Evaluation Metrics

### Pattern Distance as Primary Metric
- **Lower distance = better match** (values typically 0.0 - 1.0)
- Distance combines errors across all chromatic notes weighted by presence in raga
- **Threshold for good match:** < 0.15 (tunable)

### Diagnostic Power: Bhimpalasi vs Kafi Test Case

**Ground Truth:**
- Both ragas use: Sa, Re, ga, ma, Pa, Dha, ni
- **Bhimpalasi:** Re and Dha are **avroh-only**
- **Kafi:** All notes are **bidirectional**

**Expected Distances for Bhimpalasi Recording:**
```python
# Perfect Bhimpalasi performance:
Bhimpalasi_distance ≈ 0.0   # Re: (0.0, 1.0), Dha: (0.0, 1.0) matches exactly
Kafi_distance ≈ 0.40        # Re: (0.5, 0.5), Dha: (0.5, 0.5) expected, but we see (0.0, 1.0)
                            # √[(0.5-0)² + (0.5-1)² + (0.5-0)² + (0.5-1)²] / 4 ≈ 0.354

# Imperfect Bhimpalasi (occasional Re in aaroh):
Bhimpalasi_distance ≈ 0.08  # Re: (0.12, 0.88) slightly off
Kafi_distance ≈ 0.35        # Still larger distance
```

### Cross-Validation Metrics

#### Precision (if candidate set provided)
```python
precision = correct_raga_in_top_k / k
```
Example: If correct raga is rank 1 out of 3 candidates → Precision@3 = 100%

#### Separation Score
Measure how well the algorithm distinguishes between similar ragas:
```python
separation_score = (distance_to_2nd_best - distance_to_1st_best) / distance_to_1st_best
```
Higher separation = more confident discrimination

**Example:**
- Bhimpalasi distance: 0.08
- Kafi distance: 0.35
- Separation: (0.35 - 0.08) / 0.08 = 3.375 → Strong separation

### Validation Tests

#### Test Suite
1. **Bhimpalasi vs Kafi:** Re and Dha directional difference
2. **Yaman vs Bihag:** Both use Ma, but Yaman has stronger Ma emphasis
3. **Deshkar Alp Detection:** Can we detect 0.5 usage of Re?
4. **Marwa:** ma missing in aaroh (only Ma used ascending)
5. **Symmetric Ragas:** Bhupali should show all notes ~0.5 (bidirectional)

---

## Implementation Checklist

- [ ] Add `compute_directional_edges()` to `sequence.py` (count incoming transitions per note)
- [ ] Add `compute_directional_scores()` to normalize edge counts to (0-1) range
- [ ] Create `RAGA_PATTERNS` database with directional encodings for 20+ common ragas
- [ ] Implement `compute_pattern_distance()` using weighted Euclidean distance
- [ ] Implement `rank_ragas_by_pattern_match()` to sort candidates
- [ ] Add `compute_pattern_confidence()` based on sample size and score clarity
- [ ] Add duration weighting option: `compute_duration_weighted_edges()`
- [ ] Add minimum edge threshold filtering: `filter_low_confidence_notes()`
- [ ] Extend `raga_list_final.csv` with `aaroh_pattern` and `avroh_pattern` columns (12 comma-separated floats each)
- [ ] Extend `RagaInfo` dataclass to parse and store pattern vectors
- [ ] Create HTML visualization template with heatmap and raga match table in `output.py`
- [ ] Add `--enable-pattern-matching` CLI flag in `config.py`
- [ ] Add JSON export: `DirectionalPattern.to_dict()`
- [ ] Write unit tests for edge cases (repeated notes, single-note phrases, phrase boundaries)
- [ ] Validate on Bhimpalasi vs Kafi test recordings (must distinguish correctly)
- [ ] Validate on Deshkar recordings (detect 0.5 usage of Re)
- [ ] Validate on symmetric ragas (Bhupali) → all notes should score ~0.5
- [ ] Document in `DOCUMENTATION.md` and `LLM_REFERENCE.md`
- [ ] Update `CHANGELOG.md` with implementation milestone

---

## Future Enhancements

### Machine Learning Approach
- Train a **pattern classifier** using labeled directional edge data
- Learn optimal thresholds for binary vs alp (0.5) classification per raga
- Use neural nets to capture complex vakra (crooked) patterns

### Multi-Raga Detection
- Apply sliding window to detect **raga changes** mid-performance
- Compute separate directional patterns for each detected section
- Identify modulations (e.g., Yaman → Yaman Kalyan)

### Advanced Pattern Features
- **Transition probabilities:** P(Re → ga) in aaroh vs avroh
- **Melodic contour matching:** Beyond note-level, capture phrase shapes
- **Temporal weighting:** Recent phrases weight more than early exploration

### Interactive Editing in Web UI
- Allow users to manually adjust directional scores
- Export corrected patterns to retrain classifier
- Community-sourced pattern database

---

## Database Schema: `raga_list_final.csv` Extensions

Add these columns to the existing CSV:

| Column | Format | Example (Bhimpalasi) |
|--------|--------|----------------------|
| `aaroh_pattern` | 12 comma-separated floats (0, 0.5, or 1) | `1,0,0,1,0,1,0,1,0,0,1,0` |
| `avroh_pattern` | 12 comma-separated floats | `1,0,1,1,0,1,0,1,0,1,1,0` |

**Mapping to chromatic positions:**
```
Index: 0   1   2   3   4   5   6   7   8   9  10  11
Note:  Sa  re  Re  ga  Ga  ma  Ma  Pa dha Dha  ni  Ni

Bhimpalasi aaroh:  [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
                    Sa      ga     ma     Pa         ni
                    
Bhimpalasi avroh:  [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
                    Sa     Re ga     ma     Pa    Dha ni
```

---

## References

- **Notation System:** `OFFSET_TO_SARGAM` in [raga_pipeline/sequence.py](raga_pipeline/sequence.py)
- **Phrase Detection:** `detect_phrases()` and `cluster_notes_into_phrases()`
- **Raga Database:** `raga_list_final.csv` (extend with pattern columns)
- **HTML Reports:** `generate_analysis_report()` in [raga_pipeline/output.py](raga_pipeline/output.py)

---

## Questions to Resolve

1. **Minimum Sample Threshold:** What is the minimum number of edges needed per note for reliable classification? (Proposed: 5)
2. **Alp Detection Threshold:** At what score range (e.g., 0.3-0.7?) should we classify notes as "alp" (0.5)? Or should alp be manually encoded in the database?
3. **Duration vs Count Weighting:** Should long-held notes dominate the pattern, or should we use raw transition counts?
4. **Phrase Boundary Strategy:** Should we skip first notes entirely, or create synthetic edges between phrases?
5. **Bootstrap Validation:** How many bootstrap iterations for confidence intervals? (Proposed: 100)
6. **Pattern Database Coverage:** How many ragas need manual pattern encoding before relying on the matcher? (Target: 50+ ragas)
