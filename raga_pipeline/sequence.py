"""
Sequence module: note detection, sargam conversion, phrase detection, and transcription.

This module is designed to be self-contained so that updates to the stationary
point detection methods don't cascade to other modules. The Note dataclass
output format remains stable as the contract between modules.

Provides:
- Note, Phrase: Data containers
- detect_notes: Main note detection entry point
- detect_phrases: Group notes into phrases
- cluster_phrases: Group similar phrases (motifs)
- compute_transition_matrix: Sargam bigram statistics
- midi_to_sargam: MIDI to Indian classical notation
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional, Union
import numpy as np
import librosa
from scipy import ndimage
from scipy.signal import find_peaks

from .config import PipelineConfig
from .audio import PitchData


# =============================================================================
# SARGAM CONVERSION
# =============================================================================

# Mapping from semitone offset (0-11) to sargam notation
OFFSET_TO_SARGAM = {
    0: "Sa",
    1: "re",      # komal Re
    2: "Re",      # shuddha Re
    3: "ga",      # komal Ga
    4: "Ga",      # shuddha Ga
    5: "ma",      # shuddha Ma
    6: "Ma",      # tivra Ma
    7: "Pa",
    8: "dha",     # komal Dha
    9: "Dha",     # shuddha Dha
    10: "ni",     # komal Ni
    11: "Ni",     # shuddha Ni
}

# Extended mapping with variant info
OFFSET_TO_SARGAM_FULL = {
    0:  ("Sa",  0, ""),
    1:  ("Re",  1, "komal"),
    2:  ("Re",  1, "shuddha"),
    3:  ("Ga",  2, "komal"),
    4:  ("Ga",  2, "shuddha"),
    5:  ("Ma",  3, "shuddha"),
    6:  ("Ma",  3, "tivra"),
    7:  ("Pa",  4, ""),
    8:  ("Dha", 5, "komal"),
    9:  ("Dha", 5, "shuddha"),
    10: ("Ni",  6, "komal"),
    11: ("Ni",  6, "shuddha"),
}


def tonic_to_midi_class(tonic: Union[int, float, str]) -> int:
    """
    Convert tonic to MIDI pitch class (0-11).
    
    Args:
        tonic: MIDI number, note name ('C#4', 'D'), or Hz frequency
        
    Returns:
        Pitch class 0-11
    """
    if tonic is None:
        raise ValueError("tonic must be provided")
    
    # Integer: treat as MIDI or pitch class
    if isinstance(tonic, (int, np.integer)):
        return int(tonic) % 12
    
    # Float: could be MIDI or Hz
    if isinstance(tonic, (float, np.floating)):
        if 0 <= tonic < 128 and abs(tonic - round(tonic)) < 0.01:
            return int(round(tonic)) % 12
        else:
            # Treat as Hz
            midi = librosa.hz_to_midi(float(tonic))
            return int(round(midi)) % 12
    
    # String: note name
    if isinstance(tonic, str):
        try:
            midi = librosa.note_to_midi(tonic)
            return int(midi) % 12
        except Exception:
            try:
                # Try appending octave 4
                midi = librosa.note_to_midi(tonic + "4")
                return int(midi) % 12
            except Exception:
                pass
    
    raise ValueError(f"Cannot convert tonic: {tonic}")


def midi_to_sargam(
    midi_note: float,
    tonic: Union[int, str],
    include_octave: bool = True,
) -> str:
    """
    Convert MIDI note to sargam notation relative to tonic.
    
    Args:
        midi_note: MIDI note number
        tonic: Tonic pitch class or note name
        include_octave: Include octave markers (', ·)
        
    Returns:
        Sargam notation string (e.g., "Sa", "Ga·", "ni'")
    """
    tonic_midi = tonic_to_midi_class(tonic)
    
    # Get offset from tonic
    offset = int(round(midi_note - tonic_midi)) % 12
    
    # Get base sargam
    base_sargam = OFFSET_TO_SARGAM.get(offset, f"?{offset}")
    
    if not include_octave:
        return base_sargam
    
    # Add octave markers
    # Assume tonic is in octave 4 (MIDI 60 = C4)
    tonic_midi_base = tonic_midi + 48  # Octave 4
    octave_diff = int((midi_note - tonic_midi_base) // 12)
    
    if octave_diff < 0:
        return base_sargam + "'" * abs(octave_diff)  # Lower octave
    elif octave_diff > 0:
        return base_sargam + "·" * octave_diff  # Upper octave
    
    return base_sargam


def sargam_to_midi(sargam: str, tonic: Union[int, str], octave: int = 4) -> int:
    """
    Convert sargam notation to MIDI note.
    
    Args:
        sargam: Sargam string (e.g., "Sa", "Ga·", "ni'")
        tonic: Tonic pitch class or note name
        octave: Base octave
        
    Returns:
        MIDI note number
    """
    tonic_midi = tonic_to_midi_class(tonic)
    
    # Count octave markers
    lower_octaves = sargam.count("'")
    upper_octaves = sargam.count("·")
    base_sargam = sargam.replace("'", "").replace("·", "")
    
    # Find offset
    offset = None
    for off, name in OFFSET_TO_SARGAM.items():
        if name.lower() == base_sargam.lower():
            offset = off
            break
    
    if offset is None:
        raise ValueError(f"Unknown sargam: {sargam}")
    
    midi = tonic_midi + offset + (octave * 12) + (upper_octaves - lower_octaves) * 12
    return midi


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Note:
    """A detected musical note."""
    
    start: float          # Start time (seconds)
    end: float            # End time (seconds)
    pitch_midi: float     # MIDI note number
    pitch_hz: float       # Frequency (Hz)
    confidence: float     # Detection confidence
    
    # Populated after tonic is known
    sargam: str = ""
    pitch_class: int = -1
    
    @property
    def duration(self) -> float:
        """Note duration in seconds."""
        return self.end - self.start
    
    def with_sargam(self, tonic: Union[int, str]) -> "Note":
        """Return copy with sargam label populated."""
        return Note(
            start=self.start,
            end=self.end,
            pitch_midi=self.pitch_midi,
            pitch_hz=self.pitch_hz,
            confidence=self.confidence,
            sargam=midi_to_sargam(self.pitch_midi, tonic),
            pitch_class=int(round(self.pitch_midi)) % 12,
        )


@dataclass
class Phrase:
    """A group of consecutive notes forming a musical phrase."""
    
    notes: List[Note]
    
    @property
    def start(self) -> float:
        """Phrase start time."""
        return self.notes[0].start if self.notes else 0.0
    
    @property
    def end(self) -> float:
        """Phrase end time."""
        return self.notes[-1].end if self.notes else 0.0
    
    @property
    def duration(self) -> float:
        """Phrase duration."""
        return self.end - self.start
    
    @property
    def sargam_string(self) -> str:
        """Space-separated sargam notation."""
        return " ".join(n.sargam for n in self.notes if n.sargam)
    
    def interval_sequence(self) -> List[int]:
        """Get intervals between successive notes."""
        if len(self.notes) < 2:
            return []
        return [
            int(round(self.notes[i].pitch_midi - self.notes[i-1].pitch_midi))
            for i in range(1, len(self.notes))
        ]


# =============================================================================
# NOTE DETECTION
# =============================================================================

def detect_notes(
    pitch_data: PitchData,
    config: PipelineConfig,
) -> List[Note]:
    """
    Main note detection entry point.
    
    Currently uses stationary point detection. This implementation
    will be updated with new methods, but the output format remains stable.
    
    Args:
        pitch_data: PitchData from pitch extraction
        config: Pipeline configuration
        
    Returns:
        List of Note objects
    """
    # Smooth pitch contour
    smoothed = smooth_pitch_contour(
        pitch_data,
        method=config.smoothing_method,
        sigma=config.smoothing_note_sigma,
        snap_to_semitones=config.snap_to_semitones,
    )
    
    # Detect notes using stationary point method
    notes = detect_stationary_points(
        smoothed,
        min_duration=config.note_min_duration,
        pitch_threshold=config.pitch_change_threshold,
        derivative_threshold=config.derivative_threshold,
    )
    
    return notes


def smooth_pitch_contour(
    pitch_data: PitchData,
    method: str = "gaussian",
    sigma: float = 1.5,
    snap_to_semitones: bool = False,
) -> PitchData:
    """
    Smooth pitch contour using Gaussian or median filtering.
    
    Args:
        pitch_data: Input pitch data
        method: 'gaussian' or 'median'
        sigma: Kernel width (samples)
        snap_to_semitones: Round to nearest semitone after smoothing
        
    Returns:
        New PitchData with smoothed values
    """
    pitch_hz = pitch_data.pitch_hz.copy()
    voicing = pitch_data.voicing.copy()
    voiced_mask = (pitch_hz > 0) & voicing
    
    if not np.any(voiced_mask):
        return pitch_data
    
    # Convert to MIDI for smoothing
    pitch_midi = np.zeros_like(pitch_hz)
    pitch_midi[voiced_mask] = librosa.hz_to_midi(pitch_hz[voiced_mask])
    
    if method == "gaussian":
        # Apply Gaussian filter to voiced regions only
        smoothed_midi = pitch_midi.copy()
        voiced_indices = np.where(voiced_mask)[0]
        
        if len(voiced_indices) > 0:
            voiced_midi = pitch_midi[voiced_indices]
            smoothed_voiced = ndimage.gaussian_filter1d(voiced_midi, sigma=sigma)
            smoothed_midi[voiced_indices] = smoothed_voiced
    
    elif method == "median":
        # Median filter - snap to most common value in window
        smoothed_midi = _quantile_snap(pitch_midi, voiced_mask, window_size=int(sigma))
    
    else:
        smoothed_midi = pitch_midi
    
    # Snap to semitones
    if snap_to_semitones:
        smoothed_midi[voiced_mask] = np.round(smoothed_midi[voiced_mask])
    
    # Convert back to Hz
    smoothed_hz = np.zeros_like(pitch_hz)
    smoothed_hz[voiced_mask] = librosa.midi_to_hz(smoothed_midi[voiced_mask])
    
    # Update derived values
    valid_freqs = smoothed_hz[voiced_mask]
    midi_vals = smoothed_midi[voiced_mask]
    
    return PitchData(
        timestamps=pitch_data.timestamps,
        pitch_hz=smoothed_hz,
        confidence=pitch_data.confidence,
        voicing=voicing,
        valid_freqs=valid_freqs,
        midi_vals=midi_vals,
        frame_period=pitch_data.frame_period,
        audio_path=pitch_data.audio_path,
    )


def _quantile_snap(
    pitch_midi: np.ndarray,
    voiced_mask: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """Snap to most common frequency in local window."""
    result = pitch_midi.copy()
    voiced_indices = np.where(voiced_mask)[0]
    
    for i, idx in enumerate(voiced_indices):
        start = max(0, i - window_size)
        end = min(len(voiced_indices), i + window_size + 1)
        window_indices = voiced_indices[start:end]
        window_freqs = pitch_midi[window_indices]
        
        # Find mode (most common rounded value)
        rounded = np.round(window_freqs)
        unique, counts = np.unique(rounded, return_counts=True)
        
        if len(unique) > 0:
            mode_idx = np.argmax(counts)
            result[idx] = unique[mode_idx]
    
    return result


def detect_stationary_points(
    pitch_data: PitchData,
    min_duration: float = 0.1,
    pitch_threshold: float = 0.3,
    derivative_threshold: float = 0.15,
) -> List[Note]:
    """
    Detect notes by finding stationary points in pitch contour.
    
    A stationary point is where the pitch derivative is below threshold,
    indicating a stable pitch (held note).
    
    Args:
        pitch_data: Smoothed pitch data
        min_duration: Minimum note duration (seconds)
        pitch_threshold: Semitone threshold for merging notes
        derivative_threshold: Max derivative for stationary region
        
    Returns:
        List of Note objects
    """
    timestamps = pitch_data.timestamps
    pitch_midi = np.zeros_like(pitch_data.pitch_hz)
    voiced_mask = pitch_data.voiced_mask
    
    if not np.any(voiced_mask):
        return []
    
    pitch_midi[voiced_mask] = librosa.hz_to_midi(pitch_data.pitch_hz[voiced_mask])
    
    # Get voiced times and pitches
    voiced_indices = np.where(voiced_mask)[0]
    voiced_times = timestamps[voiced_indices]
    voiced_pitches = pitch_midi[voiced_indices]
    
    if len(voiced_pitches) < 2:
        return []
    
    # Compute derivative (semitones per second)
    dt = np.diff(voiced_times)
    dt[dt == 0] = pitch_data.frame_period  # Avoid division by zero
    derivative = np.abs(np.diff(voiced_pitches) / dt)
    
    # Pad derivative to match length
    derivative = np.concatenate([[0], derivative])
    
    # Find stationary regions
    stationary_mask = derivative < derivative_threshold
    regions = _find_continuous_regions(stationary_mask)
    
    notes = []
    
    for start_idx, end_idx in regions:
        if end_idx <= start_idx:
            continue
        
        # Get timing
        start_time = float(voiced_times[start_idx])
        end_time = float(voiced_times[min(end_idx, len(voiced_times) - 1)])
        
        if end_time - start_time < min_duration:
            continue
        
        # Get representative pitch (median)
        region_pitches = voiced_pitches[start_idx:end_idx]
        median_midi = float(np.median(region_pitches))
        
        # Check if we should merge with previous note
        if notes and abs(median_midi - notes[-1].pitch_midi) < pitch_threshold:
            # Merge: extend previous note
            notes[-1] = Note(
                start=notes[-1].start,
                end=end_time,
                pitch_midi=np.median([notes[-1].pitch_midi, median_midi]),
                pitch_hz=librosa.midi_to_hz(median_midi),
                confidence=float(np.mean(pitch_data.confidence[voiced_indices[start_idx:end_idx]])),
            )
        else:
            # Create new note
            confidence_vals = pitch_data.confidence[voiced_indices[start_idx:end_idx]]
            notes.append(Note(
                start=start_time,
                end=end_time,
                pitch_midi=median_midi,
                pitch_hz=librosa.midi_to_hz(median_midi),
                confidence=float(np.mean(confidence_vals)) if len(confidence_vals) > 0 else 0.0,
            ))
    
    return notes


def detect_melodic_peaks(
    pitch_data: PitchData,
    prominence: float = 1.0,
    min_duration: float = 0.05,
) -> List[Note]:
    """
    Alternative detection using peaks and valleys.
    
    Good for ornament-rich music where notes are defined by
    local maxima/minima.
    
    Args:
        pitch_data: Smoothed pitch data
        prominence: Minimum peak prominence (semitones)
        min_duration: Minimum note duration
        
    Returns:
        List of Note objects
    """
    voiced_mask = pitch_data.voiced_mask
    
    if not np.any(voiced_mask):
        return []
    
    voiced_indices = np.where(voiced_mask)[0]
    voiced_times = pitch_data.timestamps[voiced_indices]
    voiced_midi = librosa.hz_to_midi(pitch_data.pitch_hz[voiced_indices])
    
    # Smooth for peak detection
    smoothed = ndimage.gaussian_filter1d(voiced_midi, sigma=2)
    
    # Find peaks and valleys
    peaks, _ = find_peaks(smoothed, prominence=prominence)
    valleys, _ = find_peaks(-smoothed, prominence=prominence)
    
    # Combine and sort
    all_points = sorted(np.concatenate([peaks, valleys]))
    
    notes = []
    for i, point_idx in enumerate(all_points):
        # Determine note boundaries
        if i == 0:
            start_idx = 0
        else:
            start_idx = (all_points[i-1] + point_idx) // 2
        
        if i == len(all_points) - 1:
            end_idx = len(voiced_times) - 1
        else:
            end_idx = (point_idx + all_points[i+1]) // 2
        
        start_time = float(voiced_times[start_idx])
        end_time = float(voiced_times[end_idx])
        
        if end_time - start_time < min_duration:
            continue
        
        pitch_midi = float(smoothed[point_idx])
        
        notes.append(Note(
            start=start_time,
            end=end_time,
            pitch_midi=pitch_midi,
            pitch_hz=librosa.midi_to_hz(pitch_midi),
            confidence=1.0,  # Peak-based detection doesn't have confidence
        ))
    
    return notes


def _find_continuous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find continuous True regions in boolean mask."""
    diff = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


# =============================================================================
# PHRASE DETECTION & CLUSTERING
# =============================================================================

def detect_phrases(
    notes: List[Note],
    max_gap: float = 2.0,
    min_length: int = 3,
    method: str = "auto",
    min_phrase_duration: float = 0.1,
    merge_gap_threshold: float = 0.1,
) -> List[Phrase]:
    """
    Group notes into phrases using adaptive gap detection (KMeans).
    
    Matches logic from note_sequence_playground.ipynb 'cluster_notes_into_phrases'.
    
    Args:
        notes: List of detected notes
        max_gap: Fallback max gap if adaptive method fails (seconds)
        min_length: Minimum number of notes per phrase
        method: 'auto' (KMeans), 'threshold' (fixed max_gap), or 'dbscan'
        min_phrase_duration: Minimum duration for a phrase (seconds)
        merge_gap_threshold: Gap threshold to merge adjacent phrases after splitting
        
    Returns:
        List of Phrase objects
    """
    if not notes:
        return []
        
    # Sort by start time just in case
    notes_sorted = sorted(notes, key=lambda n: n.start)
    
    # 1. Compute gaps
    if len(notes_sorted) <= 1:
        return [Phrase(notes=notes_sorted)] if len(notes_sorted) >= min_length else []
        
    starts = np.array([n.start for n in notes_sorted])
    ends = np.array([n.end for n in notes_sorted])
    gaps = np.maximum(starts[1:] - ends[:-1], 0.0)
    
    # 2. Determine threshold
    threshold_used = max_gap
    
    if method == "threshold":
        threshold_used = max_gap
        
    elif method == "auto" and len(gaps) > 2:
        try:
            from sklearn.cluster import KMeans
            # Try KMeans splitting into 2 clusters (short vs long gaps)
            X = gaps.reshape(-1, 1)
            # Use fixed random state for reproducibility
            km = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(X)
            centers = km.cluster_centers_.reshape(-1)
            
            # Cluster with smaller center = within-phrase short gaps
            short_cluster_idx = np.argmin(centers)
            long_cluster_idx = np.argmax(centers)
            
            # Threshold = midpoint between the two cluster centers
            auto_threshold = float((centers[short_cluster_idx] + centers[long_cluster_idx]) / 2.0)
            
            # Safety check: if threshold is too small, use robust fallback
            if auto_threshold < 0.2:
                raise ValueError("Auto threshold too small")
                
            threshold_used = auto_threshold
            
        except Exception:
            # Fallback to robust IQR threshold
            q1, q3 = np.percentile(gaps, [25, 75])
            iqr = max(q3 - q1, 1e-9)
            # 1.5 * IQR is standard for outlier detection
            threshold_used = float(np.median(gaps) + 1.5 * iqr)
            
            # Clamp to reasonable range [0.5, 3.0] if purely statistical
            threshold_used = max(0.5, min(threshold_used, 3.0))

    # 3. Split into phrases
    break_indices = np.where(gaps > threshold_used)[0]
    
    phrases_list = []
    start_idx = 0
    for b in break_indices:
        end_idx = b + 1
        phrases_list.append(notes_sorted[start_idx:end_idx])
        start_idx = end_idx
    phrases_list.append(notes_sorted[start_idx:])
    
    # 4. Post-processing: Merge tiny gaps & filter short phrases
    
    # Merge adjacent phrases with tiny gaps (e.g. broken notes)
    if len(phrases_list) > 1:
        merged = []
        i = 0
        while i < len(phrases_list):
            cur = phrases_list[i]
            j = i + 1
            while j < len(phrases_list):
                # Gap between end of cur and start of next
                gap_between = phrases_list[j][0].start - cur[-1].end
                if gap_between <= merge_gap_threshold:
                    cur = cur + phrases_list[j]
                    j += 1
                else:
                    break
            merged.append(cur)
            i = j
        phrases_list = merged
        
    # Filter by duration and note count
    final_phrases = []
    i = 0
    while i < len(phrases_list):
        ph_notes = phrases_list[i]
        dur = ph_notes[-1].end - ph_notes[0].start
        
        if (dur < min_phrase_duration) or (len(ph_notes) < min_length):
            # Try merging with neighbor (prefer forward, then backward)
            merged_here = False
            
            # Merge forward
            if i + 1 < len(phrases_list):
                phrases_list[i+1] = ph_notes + phrases_list[i+1]
                merged_here = True
            
            # Merge backward
            elif final_phrases:
                 final_phrases[-1] = final_phrases[-1] + ph_notes
                 merged_here = True
                 
            if merged_here:
                i += 1
                continue
                
            # Drop it
            i += 1
            continue
        else:
            final_phrases.append(ph_notes)
            i += 1
            
    # Convert to Phrase objects
    result_phrases = [Phrase(notes=ns) for ns in final_phrases]
    return result_phrases


def cluster_phrases(
    phrases: List[Phrase],
    similarity_threshold: float = 0.8,
) -> Dict[int, List[Phrase]]:
    """
    Group similar phrases together (repeated motifs).
    
    Uses interval-based similarity - phrases with similar interval
    sequences are grouped together.
    
    Args:
        phrases: List of phrases
        similarity_threshold: Minimum similarity for clustering
        
    Returns:
        Dictionary mapping cluster ID to list of phrases
    """
    if not phrases:
        return {}
    
    # Extract interval sequences
    interval_seqs = [tuple(p.interval_sequence()) for p in phrases]
    
    # Simple clustering: group exact matches first
    clusters: Dict[Tuple, List[int]] = {}
    
    for i, seq in enumerate(interval_seqs):
        if seq in clusters:
            clusters[seq].append(i)
        else:
            clusters[seq] = [i]
    
    # Convert to result format
    result = {}
    for cluster_id, (seq, indices) in enumerate(clusters.items()):
        result[cluster_id] = [phrases[i] for i in indices]
    
    return result


# =============================================================================
# TRANSITION MATRIX
# =============================================================================

# =============================================================================
# TRANSITION MATRIX
# =============================================================================

def compute_transition_matrix(
    notes: List[Note],
    tonic: Union[int, str],
) -> np.ndarray:
    """
    Compute 12x12 transition probability matrix between sargam notes.
    Legacy implementation (chromatic).
    """
    tonic_midi = tonic_to_midi_class(tonic)
    counts = np.zeros((12, 12))
    
    for i in range(len(notes) - 1):
        from_pc = int(round(notes[i].pitch_midi - tonic_midi)) % 12
        to_pc = int(round(notes[i+1].pitch_midi - tonic_midi)) % 12
        counts[from_pc, to_pc] += 1
    
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


def extract_corrected_sargam_info(note: Note, tonic: Union[int, str]) -> Tuple[Optional[int], Optional[int], str]:
    """Extract sargam information from corrected notes."""
    # Convert tonic to MIDI class
    tonic_midi = tonic_to_midi_class(tonic)
    
    # Calculate offset
    pitch_midi = note.pitch_midi
    offset = int(round(pitch_midi - tonic_midi)) % 12
    
    # Get sargam label from offset
    sargam_label = OFFSET_TO_SARGAM.get(offset, f"?{offset}")
    
    return offset, int(round(pitch_midi)), sargam_label


def build_transition_matrix_corrected(
    phrases: List[Phrase],
    tonic: Union[int, str],
    min_transition_gap: float = 0.5, # Relaxed from user snippet's 0.1*5 since our gaps are explicit
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Build transition matrix specifically for corrected raga notes.
    """
    from collections import defaultdict
    
    # Collect all unique sargam labels present
    unique_sargams = set()
    for phrase in phrases:
        for note in phrase.notes:
            _, _, sargam = extract_corrected_sargam_info(note, tonic)
            unique_sargams.add(sargam)
            
    # Define sort order (Sa -> Re -> ... -> Ni)
    sargam_order_map = {
        'Sa': 0, 're': 1, 'Re': 2, 'ga': 3, 'Ga': 4,
        'ma': 5, 'Ma': 6, 'Pa': 7, 'dha': 8, 'Dha': 9,
        'ni': 10, 'Ni': 11
    }
    
    def get_sort_key(s):
        # Handle variants if any (though OFFSET_TO_SARGAM keys are clean)
        base = s.replace("'", "").replace("·", "")
        return sargam_order_map.get(base, 99)
        
    sorted_sargams = sorted(list(unique_sargams), key=get_sort_key)
    sargam_to_idx = {s: i for i, s in enumerate(sorted_sargams)}
    
    n_notes = len(sorted_sargams)
    matrix = np.zeros((n_notes, n_notes), dtype=int)
    
    aaroh_patterns = defaultdict(int)
    avroh_patterns = defaultdict(int)
    
    for phrase in phrases:
        notes = phrase.notes
        if len(notes) < 2:
            continue
            
        for i in range(len(notes) - 1):
            curr_n = notes[i]
            next_n = notes[i+1]
            
            _, curr_midi, curr_sargam = extract_corrected_sargam_info(curr_n, tonic)
            _, next_midi, next_sargam = extract_corrected_sargam_info(next_n, tonic)
            
            if curr_sargam not in sargam_to_idx or next_sargam not in sargam_to_idx:
                continue
                
            # Check gap (though phrases shouldn't have large gaps by definition)
            if next_n.start - curr_n.end > min_transition_gap:
                continue
                
            idx1 = sargam_to_idx[curr_sargam]
            idx2 = sargam_to_idx[next_sargam]
            
            matrix[idx1, idx2] += 1
            
            # Direction
            pattern = f"{curr_sargam} → {next_sargam}"
            if next_midi > curr_midi:
                aaroh_patterns[pattern] += 1
            elif next_midi < curr_midi:
                avroh_patterns[pattern] += 1
                
    stats = {
        'aaroh': dict(aaroh_patterns),
        'avroh': dict(avroh_patterns),
        'labels': sorted_sargams
    }
    
    return matrix, sorted_sargams, stats


def extract_melodic_sequences(phrases: List[Phrase], tonic: Union[int, str], max_length: int = 8) -> List[dict]:
    """Extract melodic sequences for pattern analysis."""
    sequences = []
    
    for p_idx, phrase in enumerate(phrases):
        notes = phrase.notes
        if len(notes) < 3:
            continue
            
        sargam_seq = []
        midi_seq = []
        
        for n in notes:
            _, midi, sargam = extract_corrected_sargam_info(n, tonic)
            sargam_seq.append(sargam)
            midi_seq.append(midi)
            
        # Sliding window
        for i in range(len(sargam_seq) - 2):
            end_idx = min(i + max_length, len(sargam_seq))
            seq = sargam_seq[i:end_idx]
            
            if len(seq) >= 3:
                sequences.append({
                    'phrase_idx': p_idx,
                    'sequence': seq,
                    'length': len(seq)
                })
                
    return sequences


def find_common_patterns(sequences: List[dict], min_length: int = 3, min_frequency: int = 2) -> List[Tuple[Tuple[str], int]]:
    """Find commonly occurring melodic patterns."""
    from collections import defaultdict
    
    counts = defaultdict(int)
    
    for item in sequences:
        seq = item['sequence']
        # Subsequences
        for length in range(min_length, len(seq) + 1):
            for start in range(len(seq) - length + 1):
                pattern = tuple(seq[start:start+length])
                counts[pattern] += 1
                
    # Filter
    common = [(p, c) for p, c in counts.items() if c >= min_frequency]
    
    # Sort by frequency then length
    common.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    
    return common


# =============================================================================
# TRANSCRIPTION
# =============================================================================

def notes_to_intervals(notes: List[Note]) -> List[int]:
    """
    Convert note sequence to interval sequence (tonic-independent).
    
    Args:
        notes: List of notes
        
    Returns:
        List of intervals (semitones) between successive notes
    """
    if len(notes) < 2:
        return []
    
    return [
        int(round(notes[i].pitch_midi - notes[i-1].pitch_midi))
        for i in range(1, len(notes))
    ]


def transcribe_notes(
    notes: List[Note],
    tonic: Union[int, str],
) -> str:
    """
    Convert notes to sargam string.
    
    Args:
        notes: List of notes
        tonic: Tonic for sargam conversion
        
    Returns:
        Space-separated sargam notation
    """
    return " ".join(midi_to_sargam(n.pitch_midi, tonic) for n in notes)


def transcribe_phrase(phrase: Phrase, tonic: Union[int, str]) -> str:
    """Transcribe a phrase to sargam string."""
    return transcribe_notes(phrase.notes, tonic)
