# Pitch Matching Game: Implementation Plan

**Concept:** Interactive music learning tool where users record themselves singing/playing and receive detailed feedback on how closely they match a reference pitch contour.

**Target Users:** Music students, classical music practitioners, casual learners - anyone wanting to improve their pitch accuracy and musicality.

**Think:** "Rocksmith for Hindustani Classical Music" but universal - works for any musical tradition.

---

## Core Concept

### User Flow:

1. **Select Reference Track**
   - User uploads any audio file OR selects from a library
   - System processes reference (stem separation, pitch extraction, transcription)
   - User can select specific segments to practice (4-8 sec phrases, challenging gamakas, aaroh/avroh patterns, etc.)

2. **Choose Difficulty Level**
   - Beginner: Focus on pitch accuracy only
   - Intermediate: Pitch + timing
   - Advanced: Pitch + timing + gamakas + note duration
   - Expert: All aspects + strict tolerances

3. **Practice Mode Selection**
   - **Real-time Mode:** Live feedback while singing (like Guitar Hero)
   - **Recording Mode:** Record full phrase, get detailed post-analysis
   - **Both:** Real-time visual guidance + detailed post-analysis

4. **Record & Match**
   - User sings/plays along with reference (optional playback)
   - Real-time pitch tracking and visual feedback
   - System captures user audio

5. **Receive Feedback**
   - Multi-dimensional scores (pitch: 92%, timing: 85%, gamakas: 78%, etc.)
   - Overlaid pitch contours (reference vs user)
   - Color-coded accuracy heatmap
   - DTW distance metric
   - Note-by-note breakdown
   - Suggestions for improvement

6. **Iterate & Improve**
   - Try again with feedback in mind
   - Track progress over time
   - Unlock harder segments/songs

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    WEB APPLICATION                          │
│                                                              │
│  Pages:                                                      │
│  - Library (browse/upload references)                       │
│  - Segment Selector (choose phrase to practice)             │
│  - Practice Mode (real-time + recording)                    │
│  - Feedback Dashboard (detailed analysis)                   │
│  - Progress Tracker (history, stats)                        │
└─────────────────────────────────────────────────────────────┘
                         ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│                  REAL-TIME PROCESSING                       │
│                                                              │
│  - Web Audio API (browser-based pitch tracking)             │
│  - Low-latency pitch extraction (~10-50ms)                  │
│  - Simplified algorithm for speed                           │
│  - Visual feedback generator (scrolling notation)           │
│  - Live comparison with reference contour                   │
└─────────────────────────────────────────────────────────────┘
                         ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│                POST-PROCESSING ANALYSIS                     │
│                                                              │
│  - Full pipeline (Demucs for user recording if needed)      │
│  - High-quality pitch extraction (SwiftF0)                  │
│  - Dynamic Time Warping (DTW) alignment                     │
│  - Gamaka detection & comparison                            │
│  - Note segmentation & duration analysis                    │
│  - Multi-dimensional scoring                                │
└─────────────────────────────────────────────────────────────┘
                         ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│                   REFERENCE PROCESSING                      │
│                                                              │
│  - One-time processing per reference track                  │
│  - Demucs stem separation (extract vocals)                  │
│  - SwiftF0 pitch extraction                                 │
│  - Note transcription                                       │
│  - Gamaka/inflection point detection                        │
│  - Store processed data (pitch contour, notes, timing)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Pipeline

### Phase 1: Reference Track Processing (One-time, ~5-8 min)

**Input:** User-uploaded audio file (any format)

**Processing Steps:**

1. **Stem Separation (Optional)**
   - Use Demucs if reference has accompaniment
   - Extract vocal/lead instrument track
   - Store separated stems

2. **High-Quality Pitch Extraction**
   - SwiftF0 (10ms resolution)
   - Confidence-weighted pitch contour
   - Store raw F0 curve

3. **Note Transcription**
   - Segment pitch contour into notes
   - Detect note boundaries, durations
   - Classify stationary notes vs transitions

4. **Gamaka Detection**
   - Identify inflection points (peaks, valleys)
   - Classify ornament types (meends, andolans, etc.)
   - Extract gamaka templates

5. **Metadata Extraction**
   - Tempo/rhythm analysis
   - Phrase boundaries
   - Difficulty estimation (based on contour complexity)

**Output:** Structured JSON with:
- Raw pitch contour (timestamped F0 values)
- Segmented notes (start, end, pitch, duration)
- Gamaka markers (type, location, intensity)
- Metadata (tempo, key, difficulty)

**Storage:** Pre-computed reference data in database/file system

---

### Phase 2: Real-Time User Input Processing (<50ms latency)

**Input:** Live microphone audio stream

**Processing Steps (Browser-based with Web Audio API):**

1. **Audio Capture**
   - `navigator.mediaDevices.getUserMedia()` for mic access
   - Buffer audio in small chunks (512-1024 samples)

2. **Fast Pitch Tracking**
   - **Algorithm:** Autocorrelation or YIN (faster than SwiftF0)
   - **Implementation:** Web Audio API + custom JS or WASM
   - **Options:**
     - [pitchy](https://github.com/ianramzy/pitchy) - lightweight JS library
     - [crepe.js](https://github.com/marl/crepe) - WASM-based, more accurate
     - Custom autocorrelation in AudioWorklet
   - Extract F0 every 20-50ms
   - Smooth with simple moving average

3. **Real-Time Comparison**
   - Align user pitch with current position in reference
   - Calculate instant deviation (cents)
   - Determine color code (green: ±20 cents, yellow: ±50, red: >50)

4. **Visual Rendering**
   - Update scrolling notation (like Guitar Hero)
   - Overlay user contour on reference
   - Display current note, target pitch
   - Show score/accuracy meter

**Output:** 
- Visual feedback (WebGL/Canvas rendering)
- Real-time score updates
- Low-latency audio monitoring (user hears themselves)

**Key Challenge: Latency**
- Total acceptable latency: <50ms
- Breakdown:
  - Audio capture: ~10ms (buffer size)
  - Pitch detection: ~20ms (algorithm)
  - Rendering: ~10ms (60fps)
  - Audio monitoring: ~10ms (playthrough)

**Solution Approach:**
- Use Web Audio API's low-latency mode
- Optimize pitch detection (autocorrelation in WASM)
- Pre-render visual assets
- Test on target hardware (laptops, mobile)

---

### Phase 3: Post-Analysis (After Recording, ~30-60 sec)

**Input:** Recorded user audio (full phrase/segment)

**Processing Steps (Server-side or high-quality browser processing):**

1. **High-Quality Pitch Extraction**
   - Use SwiftF0 (same as reference processing)
   - 10ms resolution, confidence-weighted
   - Full audio file (not streaming)

2. **Dynamic Time Warping (DTW)**
   - Align user pitch contour with reference
   - Handle timing variations (user may sing faster/slower)
   - **Algorithm:** FastDTW or librosa implementation
   - **Output:** Optimal alignment path, DTW distance

3. **Multi-Dimensional Scoring**

   **a) Pitch Accuracy (40% weight)**
   - Calculate RMS error in cents for aligned segments
   - Penalize large deviations (quadratic)
   - Score = 100 * exp(-error²/threshold²)

   **b) Timing/Rhythm (25% weight)**
   - Compare note onset times (after DTW)
   - Measure deviation from expected timing
   - Tempo consistency check

   **c) Gamakas/Ornamentations (20% weight)**
   - Detect user's gamakas (inflection points)
   - Match with reference gamaka templates
   - Compare shape, intensity, duration
   - Partial credit for attempted gamakas

   **d) Note Duration (10% weight)**
   - Compare held note durations
   - Check if user sustains notes correctly

   **e) Overall Contour Shape (5% weight)**
   - Visual similarity metric (2D correlation)
   - Captures "feel" of the phrase

4. **Detailed Feedback Generation**
   - Identify specific problem areas (color-coded heatmap)
   - Generate text suggestions ("Hold the second note longer", "Meend from Sa to Ga was too fast")
   - Extract best/worst segments for review

5. **Visualization Creation**
   - Overlaid pitch contours (interactive plot)
   - DTW alignment visualization
   - Note-by-note comparison table
   - Spectrograms (optional, for advanced users)

**Output:**
- Overall score (0-100)
- Multi-dimensional scores with explanations
- Annotated pitch contour visualization
- Actionable feedback text
- Saved recording + analysis for progress tracking

---

## Scoring Algorithm (Detailed)

### 1. Pitch Accuracy Score

```python
def pitch_accuracy_score(user_f0, ref_f0, aligned_indices):
    """
    Calculate pitch accuracy after DTW alignment.
    
    Args:
        user_f0: User pitch contour (Hz)
        ref_f0: Reference pitch contour (Hz)
        aligned_indices: DTW alignment path
    
    Returns:
        score: 0-100 (100 = perfect match)
        error_cents: RMS error in cents
    """
    errors_cents = []
    
    for user_idx, ref_idx in aligned_indices:
        if user_f0[user_idx] > 0 and ref_f0[ref_idx] > 0:  # Valid pitches
            # Convert to cents difference
            cents_diff = 1200 * np.log2(user_f0[user_idx] / ref_f0[ref_idx])
            errors_cents.append(abs(cents_diff))
    
    # RMS error
    rms_error = np.sqrt(np.mean(np.array(errors_cents) ** 2))
    
    # Score with exponential decay (20 cents = 90%, 50 cents = 60%)
    score = 100 * np.exp(-0.5 * (rms_error / 25) ** 2)
    
    return score, rms_error
```

### 2. Timing/Rhythm Score

```python
def timing_score(user_notes, ref_notes, user_duration, ref_duration):
    """
    Compare timing/rhythm after note segmentation.
    
    Args:
        user_notes: List of (onset, offset, pitch) tuples
        ref_notes: Reference note timings
        user_duration: Total user recording duration
        ref_duration: Total reference duration
    
    Returns:
        score: 0-100
    """
    # Tempo ratio
    tempo_ratio = user_duration / ref_duration
    
    # Align notes (match pitches, compare onsets)
    timing_errors = []
    
    for u_note, r_note in zip(user_notes, ref_notes):
        # Adjust user onset by tempo ratio
        adjusted_onset = u_note[0] / tempo_ratio
        timing_error_ms = abs(adjusted_onset - r_note[0]) * 1000
        timing_errors.append(timing_error_ms)
    
    # Average timing error
    avg_error_ms = np.mean(timing_errors)
    
    # Score (50ms = 95%, 200ms = 60%)
    score = 100 * np.exp(-0.5 * (avg_error_ms / 100) ** 2)
    
    return score
```

### 3. Gamaka Matching Score

```python
def gamaka_score(user_inflections, ref_inflections, aligned_indices):
    """
    Compare ornamentations/gamakas.
    
    Args:
        user_inflections: User's detected gamakas (peaks/valleys)
        ref_inflections: Reference gamakas
        aligned_indices: DTW alignment path
    
    Returns:
        score: 0-100
    """
    matched_gamakas = 0
    total_ref_gamakas = len(ref_inflections)
    
    for ref_gamaka in ref_inflections:
        # Find user gamaka at similar aligned time
        for user_gamaka in user_inflections:
            if is_gamaka_match(user_gamaka, ref_gamaka, aligned_indices):
                matched_gamakas += 1
                break
    
    # Score based on recall (did user attempt all gamakas?)
    recall = matched_gamakas / total_ref_gamakas
    
    # Bonus for precision (no extra gamakas)
    precision = matched_gamakas / len(user_inflections) if user_inflections else 0
    
    score = 100 * (0.7 * recall + 0.3 * precision)
    
    return score
```

### 4. Combined Score

```python
def calculate_final_score(user_audio, ref_data, difficulty_level):
    """
    Calculate overall performance score.
    
    Args:
        user_audio: User's recorded audio
        ref_data: Pre-processed reference data
        difficulty_level: 'beginner', 'intermediate', 'advanced', 'expert'
    
    Returns:
        scores: Dict with overall + multi-dimensional scores
        feedback: Detailed feedback text
    """
    # Extract user features
    user_f0 = extract_pitch(user_audio)
    user_notes = segment_notes(user_f0)
    user_gamakas = detect_gamakas(user_f0)
    
    # Align with DTW
    aligned_indices, dtw_distance = dtw_align(user_f0, ref_data['f0'])
    
    # Calculate component scores
    pitch_score, _ = pitch_accuracy_score(user_f0, ref_data['f0'], aligned_indices)
    time_score = timing_score(user_notes, ref_data['notes'], len(user_f0), len(ref_data['f0']))
    gamaka_score_val = gamaka_score(user_gamakas, ref_data['gamakas'], aligned_indices)
    duration_score = note_duration_score(user_notes, ref_data['notes'])
    contour_score = visual_similarity(user_f0, ref_data['f0'])
    
    # Weight based on difficulty
    weights = get_weights(difficulty_level)
    
    overall_score = (
        weights['pitch'] * pitch_score +
        weights['timing'] * time_score +
        weights['gamaka'] * gamaka_score_val +
        weights['duration'] * duration_score +
        weights['contour'] * contour_score
    )
    
    return {
        'overall': overall_score,
        'pitch': pitch_score,
        'timing': time_score,
        'gamaka': gamaka_score_val,
        'duration': duration_score,
        'contour': contour_score,
        'dtw_distance': dtw_distance
    }, generate_feedback(...)
```

---

## Visual Feedback System

### 1. Real-Time Scrolling Notation (Main Practice View)

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  Score: 87%        Current Note: Sa (C)        Time: 0:05    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Reference ──────●────────●─────────●──────────             │
│  (gray line)     │        │         │                        │
│                  │        │         │                        │
│  Your Voice ─────●────────●────────●────────── (green)      │
│  (colored)       ✓        ✓        ✗                        │
│                good      good     too high                   │
│                                                              │
│  ◄──────────────  Scrolling left  ─────────────────►        │
│                                                              │
│  Upcoming: Re (D) → Ga (E) → Ma (F)                         │
└─────────────────────────────────────────────────────────────┘
│ Controls: [Pause] [Restart] [Hear Reference] [Record Only] │
└─────────────────────────────────────────────────────────────┘
```

**Technical Implementation:**
- **Canvas API** or **WebGL** for smooth rendering at 60fps
- Reference contour drawn as gray scrolling curve
- User pitch overlaid in color (green/yellow/red)
- Notes marked as dots with pitch names
- Visual indicator for current target note
- Deviation displayed in cents (optional, for advanced users)

**Libraries:**
- [p5.js](https://p5js.org/) for simpler Canvas rendering
- [PixiJS](https://pixijs.com/) for performant WebGL rendering
- Custom shaders for smooth curves and color gradients

---

### 2. Post-Analysis Detail View

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  Overall Score: 87/100  ★★★★☆                               │
├─────────────────────────────────────────────────────────────┤
│  Pitch Accuracy:    92/100 ████████████░░░░ Excellent!      │
│  Timing/Rhythm:     85/100 ██████████░░░░░░ Good            │
│  Gamakas:           78/100 █████████░░░░░░░ Needs work      │
│  Note Duration:     90/100 ███████████░░░░░ Very good       │
│  Contour Shape:     88/100 ██████████░░░░░░ Great           │
├─────────────────────────────────────────────────────────────┤
│  Detailed Pitch Contour Comparison:                         │
│                                                              │
│  [Interactive Plotly Chart]                                 │
│  - Reference contour (blue solid line)                      │
│  - Your contour (orange dashed line)                        │
│  - DTW alignment markers                                    │
│  - Color-coded accuracy zones (green/yellow/red background) │
│  - Hover to see pitch values at each timestamp              │
│  - Click segments to hear reference/user audio              │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  Problem Areas (click to review):                           │
│  🔴 0:03-0:04 - Meend from Sa to Ga too fast                │
│  🟡 0:07-0:08 - Held Re slightly flat (15 cents)            │
│  🔴 0:11-0:12 - Missed gamaka on Pa                         │
├─────────────────────────────────────────────────────────────┤
│  [Play Reference] [Play Your Recording] [Try Again]         │
│  [Download Analysis PDF] [Share Progress]                   │
└─────────────────────────────────────────────────────────────┘
```

**Technical Implementation:**
- **Plotly.js** for interactive pitch contour charts
- **D3.js** for custom visualizations (DTW path, heatmaps)
- **Color Gradient:** Green (±0-20 cents) → Yellow (±20-50 cents) → Red (>50 cents)
- **Audio Playback:** Web Audio API with highlighted segment playback
- **Export:** Generate PDF report with charts using [jsPDF](https://github.com/parallax/jsPDF)

---

### 3. DTW Alignment Visualization

```
Reference:  Sa ─┬─ Re ──── Ga ──── Ma ─┬─ Pa
                │                       │
User:       Sa ─┘ Re ── Ga ──── Ma ───┘ Pa
            (aligned - user sang slower)
```

Show the DTW path to explain timing differences.

---

## Technology Stack

### Frontend (Real-Time + UI)

**Framework:** React + TypeScript (consistent with thesis demo)

**Audio Processing:**
- **Web Audio API** - core audio capture and processing
- **Pitch Detection Library:**
  - Option 1: [pitchy](https://github.com/ianramzy/pitchy) - lightweight, fast
  - Option 2: [crepe.js](https://github.com/marl/crepe) - WASM-based, more accurate (~50ms latency)
  - Option 3: Custom autocorrelation in AudioWorklet

**Visualization:**
- **Canvas API** or **PixiJS** (WebGL) for real-time scrolling notation
- **Plotly.js** for detailed post-analysis charts
- **D3.js** for custom visualizations (DTW, heatmaps)

**UI Components:**
- **Tailwind CSS** for styling
- **Framer Motion** for smooth animations
- **Lucide React** for icons
- **React Query** for data fetching

**Audio Playback:**
- **Tone.js** - Web Audio framework for playback with effects
- **Wavesurfer.js** - waveform visualization and playback

---

### Backend (Post-Processing + Storage)

**Framework:** FastAPI (Python) - same as thesis demo

**Audio Processing:**
- **Demucs** - stem separation for reference tracks
- **SwiftF0** - high-quality pitch extraction
- **librosa** - audio analysis, note segmentation
- **dtw-python** or **fastdtw** - Dynamic Time Warping
- **scipy** - signal processing, peak detection

**Storage:**
- **PostgreSQL** - user accounts, progress tracking, metadata
- **S3/R2/MinIO** - reference tracks, user recordings, analysis results
- **Redis** - caching processed reference data

**API Endpoints:**
- `POST /api/references/upload` - Upload reference track
- `POST /api/references/{id}/process` - Trigger reference processing
- `GET /api/references/{id}/segments` - Get selectable segments
- `POST /api/analyze` - Submit user recording for post-analysis
- `GET /api/users/{id}/progress` - Get practice history

---

### Deployment

**Frontend:**
- **Vercel** or **Netlify** (free tier, CDN)
- **Cost:** $0 (free tier sufficient for MVP)

**Backend:**
- **Railway** (~$10-20/month for 2GB RAM, sufficient for processing)
- **Alternative:** Fly.io, Render

**Storage:**
- **Cloudflare R2** - $0.015/GB/month (first 10GB free)
- **Alternative:** AWS S3, Backblaze B2

**Total Cost Estimate:** $10-20/month for MVP (100-500 users)

---

## Implementation Approach

### Phase 1: Proof of Concept (2-3 weeks)

**Goal:** Validate core concept with minimal viable product

**Features:**
- Simple web app with microphone access
- Fast pitch detection (pitchy.js)
- Real-time pitch display (basic Canvas visualization)
- Upload reference track (no stem separation yet)
- Basic pitch comparison (simple RMS error)
- Minimal UI (no fancy graphics)

**Tech Stack:**
- Single HTML page + vanilla JS
- Web Audio API for capture
- Simple Canvas rendering
- No backend (browser-only processing)

**Deliverable:** Working prototype that proves real-time feedback is feasible

---

### Phase 2: Enhanced Real-Time Experience (3-4 weeks)

**Goal:** Build production-quality real-time practice mode

**Features:**
- React + TypeScript frontend
- Improved pitch detection (crepe.js or custom AudioWorklet)
- Scrolling notation (Guitar Hero style)
- Color-coded accuracy feedback
- Multiple reference tracks (pre-processed)
- Difficulty levels
- Session recording

**Tech Stack:**
- React + Vite + TypeScript
- PixiJS for rendering
- FastAPI backend (minimal - just serving reference data)

**Deliverable:** Polished real-time practice interface

---

### Phase 3: Post-Analysis & Scoring (3-4 weeks)

**Goal:** Add detailed feedback system

**Features:**
- Server-side post-processing
- DTW alignment
- Multi-dimensional scoring
- Detailed visualizations (Plotly charts)
- Text feedback generation
- Progress tracking

**Tech Stack:**
- Full FastAPI backend
- PostgreSQL database
- Python audio processing pipeline (Demucs, SwiftF0, DTW)

**Deliverable:** Complete feedback and analysis system

---

### Phase 4: Reference Processing Pipeline (2-3 weeks)

**Goal:** Allow users to upload their own reference tracks

**Features:**
- Reference upload interface
- Background processing (Demucs + SwiftF0)
- Segment selection tool
- Difficulty auto-detection
- Reference library with search

**Deliverable:** Self-service reference creation

---

### Phase 5: Polish & Gamification (2-3 weeks)

**Goal:** Make it engaging and fun

**Features:**
- Achievement system (badges for milestones)
- Leaderboards (optional)
- Practice streaks
- Progress charts over time
- Social sharing (share scores)
- Mobile-responsive design

**Deliverable:** Engaging, polished user experience

---

**Total Timeline:** 12-17 weeks (~3-4 months) for full-featured MVP

---

## Key Challenges & Solutions

### Challenge 1: Real-Time Latency

**Problem:** Need <50ms total latency for real-time feedback to feel natural

**Solutions:**
1. **Use Web Audio API's low-latency mode:**
   ```javascript
   const audioContext = new AudioContext({ latencyHint: 'interactive' });
   ```

2. **Optimize pitch detection:**
   - Use WASM-compiled algorithm (crepe.js)
   - Or implement fast autocorrelation in AudioWorklet
   - Process in 1024-sample chunks (21ms at 48kHz)

3. **Minimize rendering overhead:**
   - Use WebGL (PixiJS) instead of Canvas for 60fps rendering
   - Pre-compute reference contour positions
   - Use requestAnimationFrame efficiently

4. **Test on target hardware:**
   - Profile on mid-range laptops, mobile devices
   - Add quality presets (low/medium/high latency)

---

### Challenge 2: Handling Accompaniment in Reference Tracks

**Problem:** Most reference recordings have tanpura, tabla, etc. - need clean vocal/lead track

**Solutions:**
1. **Use Demucs for stem separation** (already in your pipeline)
   - Works well for vocals vs accompaniment
   - Can run as preprocessing step (not real-time)

2. **Prompt user to provide clean recordings when possible**
   - Offer bonus points for "clean reference uploads"
   - Provide sample clean references

3. **Robust pitch tracking:**
   - SwiftF0 is confidence-weighted (handles some background noise)
   - Post-process pitch contour to remove spurious values

---

### Challenge 3: Microphone Quality Variations

**Problem:** Users have different mic quality (laptop mic, phone, USB, etc.) - affects pitch detection

**Solutions:**
1. **Audio preprocessing:**
   - High-pass filter (remove low-frequency rumble)
   - Noise gate (ignore below threshold)
   - Automatic gain control (normalize volume)

2. **Calibration step:**
   - At first use, ask user to sing a reference note (Sa)
   - Detect mic quality, adjust thresholds accordingly

3. **Quality indicator:**
   - Show audio level meter during recording
   - Warn if input is too quiet/noisy
   - Suggest better mic positioning

---

### Challenge 4: Defining 'Correctness' for Gamakas

**Problem:** Gamakas (ornaments) can vary in execution - no single "correct" way

**Solutions:**
1. **Gamaka Templates:**
   - Define acceptable ranges for each gamaka type
   - Allow variance in speed, intensity
   - Focus on "did user attempt the gamaka?" rather than exact match

2. **Partial Credit:**
   - Award points for detecting a gamaka in the right location
   - Bonus for matching shape/intensity

3. **Difficulty Levels:**
   - Beginner: Ignore gamakas, focus on pitch
   - Intermediate: Detect presence of gamakas
   - Advanced: Compare gamaka execution quality
   - Expert: Strict gamaka matching

---

### Challenge 5: Scalability & Hosting Costs

**Problem:** Audio processing (especially Demucs) is CPU-intensive - can get expensive at scale

**Solutions:**
1. **Preprocessing Strategy:**
   - Process reference tracks once, cache results
   - Most processing is one-time per reference (not per user)

2. **User Recording Analysis:**
   - Offload to user's device when possible (browser-based SwiftF0)
   - Only send to server for advanced analysis (DTW, scoring)

3. **Efficient Architecture:**
   - Use background job queue (Celery) for long processing tasks
   - Scale workers horizontally as needed
   - CDN for serving pre-processed reference data

4. **Cost Estimates (at scale):**
   - **100 users:** ~$10-20/month (Railway + R2)
   - **1,000 users:** ~$50-100/month (more workers, storage)
   - **10,000 users:** ~$500-1,000/month (need dedicated infra)

5. **Monetization Options (Future):**
   - Freemium model (limited references free, unlimited paid)
   - Premium features (export reports, advanced analytics)
   - B2B licensing for music schools

---

## Sample User Stories

### Story 1: Music Student Learning Raga Yaman

**User:** Priya, 18, learning Hindustani classical vocals

**Goal:** Master the aaroh (ascending scale) of Raga Yaman

**Flow:**
1. Priya uploads her guru's recording of Yaman aaroh
2. System processes, extracts vocal track, generates pitch contour
3. Priya selects "Difficulty: Intermediate" (pitch + timing)
4. She practices in real-time mode:
   - Sees scrolling notation with reference contour
   - Sings along, gets immediate color-coded feedback
   - Green for accurate notes, yellow/red when off
5. After 3 attempts, she records her best take
6. System provides detailed analysis:
   - Overall: 83%
   - Pitch: 90% (excellent!)
   - Timing: 78% (a bit rushed on Ga → Ma)
   - Feedback: "Hold Ma longer, you're cutting it short"
7. Priya tries again with feedback in mind, improves to 89%
8. Progress tracked - she can see improvement over days

**Outcome:** Priya gets personalized, quantitative feedback that helps her improve faster than traditional practice alone.

---

### Story 2: Self-Taught Guitarist Learning a Solo

**User:** Alex, 25, learning blues guitar by ear

**Goal:** Nail the solo from his favorite blues song

**Flow:**
1. Alex uploads the studio recording (with full band)
2. System uses Demucs to separate guitar from backing track
3. Alex selects just the solo section (30 seconds)
4. Difficulty: Advanced (pitch + timing + bends/vibrato)
5. He practices in "Recording Mode" (no real-time, just record & analyze):
   - Records himself playing the solo
   - System aligns his playing with reference using DTW
   - Gets detailed feedback on each bend and vibrato
6. Visual heatmap shows exactly which notes were sharp/flat
7. Alex can click problem areas to hear reference vs his playing
8. After 10 attempts over a week, he masters the solo (95%+ score)

**Outcome:** Alex learns the solo more accurately and faster than just playing along with the track.

---

### Story 3: Casual User Playing Musical Simon Says

**User:** Jordan, 32, not a musician, just curious

**Goal:** Have fun trying to match random melodies

**Flow:**
1. Jordan opens the app, selects "Random Challenge Mode"
2. System generates or selects a simple 4-note melody
3. Jordan sings along in real-time (like Simon Says)
4. Gets instant feedback, scores 65%
5. Tries again, scores 78%
6. Shares score on social media: "I matched this melody 78%! Can you beat it?"
7. Friends try the same challenge

**Outcome:** Fun, engaging experience for non-musicians - gamification makes music learning accessible.

---

## Future Enhancements (Beyond MVP)

### 1. Multi-Track Challenges
- Practice harmonizing with reference (sing harmony part)
- Duet mode (two users singing together)

### 2. AI-Powered Feedback
- Use LLM to generate detailed, conversational feedback
- "Your meend from Sa to Ga sounds hesitant - try to flow more smoothly. Listen to this example..."

### 3. Adaptive Difficulty
- System automatically adjusts difficulty based on user's improving skill
- Unlock harder segments as you master easier ones

### 4. Mobile App
- React Native + Expo for iOS/Android
- Offline mode (download reference tracks)
- Use device ML accelerators for faster processing

### 5. Live Multiplayer
- Compete with friends in real-time pitch matching
- Leaderboards, tournaments

### 6. Integration with Thesis Demo
- After detecting raga, offer "Practice Mode" for that raga
- Seamless flow: detect → learn → practice

### 7. Music Theory Integration
- Teach sargam notation alongside practice
- Explain raga rules as user practices
- Quiz mode: identify notes by ear

### 8. Teacher Dashboard
- Music teachers can upload assignments
- Track student progress
- Provide personalized feedback

---

## Development Priorities

### MVP Must-Haves (Launch in 3 months):
1. ✅ Real-time pitch tracking and visual feedback
2. ✅ Upload reference track + segment selection
3. ✅ Post-analysis with DTW alignment
4. ✅ Multi-dimensional scoring (pitch, timing, basic gamakas)
5. ✅ Overlaid pitch contour visualization
6. ✅ Difficulty levels
7. ✅ User accounts + progress tracking

### Nice-to-Haves (Add later):
- Gamaka detection refinement
- Advanced visual effects (particle effects for perfect notes)
- Social sharing
- Mobile app
- AI feedback

### Can Wait:
- Monetization features
- Teacher dashboard
- Multiplayer
- Integration with thesis demo

---

## Success Metrics

### Technical Metrics:
- Real-time latency: <50ms (p95)
- Pitch detection accuracy: >95% for clean recordings
- User can complete full practice session without crashes
- Mobile responsiveness: works on phone browsers

### User Engagement Metrics:
- Average practice session length: >5 minutes
- Return rate: >30% of users return within 7 days
- Recording completions: >50% of started practices completed
- Score improvement: Users improve by avg 10+ points over 10 attempts

### Qualitative Feedback:
- User interviews: "Does this help you learn music?"
- Net Promoter Score (NPS): Target >40
- Feature requests: Track most requested features

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Real-time latency too high | Medium | High | Extensive testing, fallback to recording-only mode |
| Pitch detection inaccurate | Low | High | Use proven algorithms (crepe.js, SwiftF0), add calibration |
| High hosting costs | Medium | Medium | Smart caching, optimize processing, consider freemium |
| Low user adoption | Medium | High | Focus on UX, gamification, marketing to music schools |
| Microphone issues | High | Medium | Clear setup instructions, quality indicators, preprocessing |
| Scope creep | High | Medium | Strict MVP definition, resist feature additions pre-launch |

---

## Next Steps to Get Started

1. **Week 1: Proof of Concept**
   - Build single-page app with mic access
   - Implement basic pitch detection (pitchy.js)
   - Display pitch in real-time (simple number or bar)
   - Validate latency is acceptable

2. **Week 2-3: Reference Processing**
   - Integrate Demucs + SwiftF0 from thesis pipeline
   - Build reference upload endpoint
   - Store processed pitch contours
   - Create simple segment selector

3. **Week 4-5: Real-Time Visualization**
   - Implement scrolling notation (Canvas or PixiJS)
   - Overlay user pitch on reference
   - Add color-coded accuracy feedback

4. **Week 6-7: Post-Analysis**
   - Implement DTW alignment
   - Build scoring algorithm
   - Create detailed feedback visualizations (Plotly)

5. **Week 8-9: Full Stack Integration**
   - Connect frontend + backend
   - User accounts + progress tracking
   - Difficulty levels

6. **Week 10-12: Polish + Launch**
   - UI/UX refinement
   - Mobile responsive
   - Deploy to production
   - Beta testing with real users

---

**Ready to start prototyping? Begin with the Proof of Concept in Week 1 to validate the core real-time feedback loop!**
