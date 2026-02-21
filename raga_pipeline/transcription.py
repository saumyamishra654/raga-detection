"""
Transcription Module
===================

This module implements "Stationary Point Transcription" logic.
It identifies musical notes by finding regions where the pitch contour is stable
(derivative approx 0). This is designed to capture sustained notes more naturally
than simple quantization.

Key Concepts:
- Pre-smoothing: Smoothing the pitch curve before differentiation to handle vibrato.
- Stable Regions: Continuous segments where |dp/dt| < threshold.
- Snapping: Aligning partial stable pitches to the nearest semitone or raga note.
- Inflection Points: Points where derivative changes sign (peaks/valleys).
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import math
import numpy as np
import librosa
from scipy import ndimage

from .sequence import Note, tonic_to_midi_class

@dataclass
class TranscriptionEvent:
    """A transcribed musical event (note)."""
    start: float
    end: float
    pitch_midi: float      # The actual median pitch of the stable region
    snapped_midi: float    # The quantized pitch (chromatic or raga-aligned)
    sargam: str = ""       # Sargam label relative to tonic
    error_cents: float = 0.0 # Deviation from the snapped target
    is_stable: bool = True # False if this is a transient/glide region (future use)
    energy: float = 0.0    # Mean energy of the event

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_stationary_events(
    pitch_hz: np.ndarray,
    timestamps: np.ndarray,
    voicing_mask: np.ndarray,
    tonic: Union[int, str, float],
    # Energy parameters
    energy: Optional[np.ndarray] = None,
    energy_threshold: float = 0.0,
    # Configurable parameters
    smoothing_sigma_ms: float = 70.0,  # 70ms smoothing for vibrato filtering
    frame_period_s: float = 0.01,      # 10ms frame period
    derivative_threshold: float = 1.6, # Semitones/sec threshold for stability
    min_event_duration: float = 0.1,   # Minimum note length
    snap_mode: Literal["chromatic", "raga"] = "chromatic",
    allowed_raga_notes: Optional[List[int]] = None,
    snap_tolerance_cents: float = 35.0,
) -> List[TranscriptionEvent]:
    """
    Main entry point for stationary point transcription.
    
    Args:
        pitch_hz: Array of pitch frequencies in Hz.
        timestamps: Array of timestamps in seconds.
        voicing_mask: Boolean mask of voiced frames.
        tonic: Tonic pitch (Hz, MIDI, or note name).
        smoothing_sigma_ms: Gaussian kernel sigma in milliseconds.
        frame_period_s: Time between frames in seconds.
        derivative_threshold: Max pitch change (semitones/sec) to consider stable.
        min_event_duration: Minimum seconds for a region to be a note.
        snap_mode: 'chromatic' (12-tone) or 'raga' (restricted set).
        allowed_raga_notes: List of allowed pitch classes (0-11) if snap_mode='raga'.
        snap_tolerance_cents: Max deviation to allow snapping (legacy).
        
    Returns:
        List of TranscriptionEvent objects.
    """
    if len(pitch_hz) == 0 or not np.any(voicing_mask):
        return []

    # 0. Apply Energy Filter (if provided)
    # If energy is below threshold, treat as unvoiced.
    # This effectively filters phantom sounds AND splits phrases on energy dips.
    if energy is not None and energy_threshold > 0:
        # Assuming energy is normalized 0-1
        energy_mask = energy >= energy_threshold
        # Combine with existing voicing (pitch confidence)
        voicing_mask = voicing_mask & energy_mask
        
    # Check again after filtering
    if not np.any(voicing_mask):
        return []


    # 1. Convert to MIDI (semitones)
    # We use a mutable copy for smoothing
    valid_pitch_mask = np.isfinite(pitch_hz) & (pitch_hz > 0)
    voicing_mask = voicing_mask & valid_pitch_mask
    if not np.any(voicing_mask):
        return []
    pitch_midi = np.zeros_like(pitch_hz)
    pitch_midi[voicing_mask] = librosa.hz_to_midi(pitch_hz[voicing_mask])
    
    # 2. Pre-smoothing (Vibrato Handling)
    if smoothing_sigma_ms <= 0:
        smoothed_midi = pitch_midi.copy()
    else:
        # Convert sigma from ms to frames
        sigma_frames = (smoothing_sigma_ms / 1000.0) / frame_period_s
        smoothed_midi = _smooth_pitch_contour(pitch_midi, voicing_mask, sigma_frames)
    
    # 3. Calculate Derivative
    # d(pitch) / dt (semitones per second)
    # np.gradient uses central differences to keep the derivative smooth.
    d_pitch = np.gradient(smoothed_midi, timestamps)
    
    # 4. Find Stable Regions
    # Condition: Voiced AND |Derivative| < Threshold
    is_stable = (np.abs(d_pitch) < derivative_threshold) & voicing_mask
    
    # 5. Extract Regions
    # Standard run-length encoding logic
    events = []
    
    diff = np.diff(np.concatenate(([0], is_stable.astype(int), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    tonic_midi_val = _resolve_tonic(tonic)
    
    for start_idx, end_idx in zip(starts, ends):
        duration = timestamps[min(end_idx, len(timestamps)-1)] - timestamps[start_idx]
        if duration < min_event_duration:
            continue
            
        # Extract segment data
        segment_pitches = smoothed_midi[start_idx:end_idx]
        segment_pitches = segment_pitches[np.isfinite(segment_pitches)]
        if segment_pitches.size == 0:
            continue
        median_pitch = np.median(segment_pitches)
        if not np.isfinite(median_pitch):
            continue
        
        # Calculate mean energy for the segment
        mean_energy = 0.0
        if energy is not None and len(energy) > 0:
            # Handle potential length mismatch if energy array is shorter/longer
            seg_start = start_idx
            seg_end = min(end_idx, len(energy))
            if seg_start < seg_end:
                mean_energy = float(np.mean(energy[seg_start:seg_end]))
        
        # 6. Snapping
        snapped_pitch, error, label, keep_note = _snap_pitch(
            float(median_pitch),
            tonic_midi_val, 
            snap_mode, 
            allowed_raga_notes, 
            snap_tolerance_cents
        )
        if not keep_note:
            continue
        if snapped_pitch is None:
            continue
        
        event = TranscriptionEvent(
            start=timestamps[start_idx],
            end=timestamps[min(end_idx, len(timestamps)-1)],
            pitch_midi=median_pitch,
            snapped_midi=snapped_pitch,
            sargam=label,
            error_cents=error,
            is_stable=True,
            energy=mean_energy
        )
        events.append(event)
        
    return events


def detect_pitch_inflection_points(
    pitch_hz: np.ndarray,
    timestamps: np.ndarray,
    voicing_mask: np.ndarray,
    smoothing_sigma_ms: float = 0.0,
    frame_period_s: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect points where the pitch derivative crosses zero (local peaks/valleys).
    
    Returns:
        (times, values) arrays of inflection points
    """
    if len(pitch_hz) == 0 or not np.any(voicing_mask):
        return np.array([]), np.array([])

    pitch_midi = np.zeros_like(pitch_hz)
    pitch_midi[voicing_mask] = librosa.hz_to_midi(pitch_hz[voicing_mask])
    
    # Optional smoothing before inflection detection for vibrato reduction
    if smoothing_sigma_ms > 0:
        sigma_frames = (smoothing_sigma_ms / 1000.0) / frame_period_s
        pitch_curve = _smooth_pitch_contour(pitch_midi, voicing_mask, sigma_frames)
    else:
        pitch_curve = pitch_midi
        
    # Calculate Gradient
    grad = np.gradient(pitch_curve, timestamps)
    
    # Find zero crossings of gradient (sign change)
    # Only consider voiced regions
    grad[~voicing_mask] = np.nan # Avoid crossings at voiced/unvoiced boundaries
    
    # Sign of gradient
    signs = np.sign(grad)
    
    # Diff of signs != 0 means change
    # Note: sign(0) = 0. sign(-1) = -1. sign(1) = 1.
    # Crossings: -1->1, 1->-1, -1->0, 0->1 etc.
    
    # We look for sign changes in valid regions
    sign_diff = np.diff(signs)
    
    # Indices where sign changes and both neighbors are voiced (not NaN)
    # Preserve NaNs so derivative crossings can be ignored for unvoiced gaps.
    # So valid indices have finite diff
    
    crossing_indices = np.where(np.isfinite(sign_diff) & (sign_diff != 0))[0]
    
    return timestamps[crossing_indices], pitch_midi[crossing_indices]


def _smooth_pitch_contour(
    pitch_midi: np.ndarray, 
    mask: np.ndarray, 
    sigma: float
) -> np.ndarray:
    """Smooth pitch contour, handling unvoiced gaps by simple interpolation."""
    # Copy to avoid modifying original
    arr = pitch_midi.copy()
    
    # Indices
    x = np.arange(len(arr))
    
    # If no voiced frames, return
    if not np.any(mask):
        return arr
    
    # Identify valid points
    valid_x = x[mask]
    valid_y = arr[mask]
    
    # Linear interpolation for everything (fill gaps)
    interp_y = np.interp(x, valid_x, valid_y)
    
    # Gaussian smooth
    smoothed = ndimage.gaussian_filter1d(interp_y, sigma=sigma)
    
    # We still only care about the originally voiced parts' derivatives (mostly)
    # But returning the full smoothed array lets gradient work everywhere.
    return smoothed


def _snap_pitch(
    pitch: float,
    tonic_midi: float,
    mode: str,
    allowed_pcs: Optional[List[int]],
    tolerance_cents: float,
) -> Tuple[Optional[float], float, str, bool]:
    """
    Snap a raw MIDI pitch to a target scale.
    Returns: (snapped_pitch, error_in_cents, label, keep_note)
    """
    # Reserved for backwards-compatible call sites.
    _ = tolerance_cents

    # 1. Normalize to 0-11 relative to tonic
    # Note: tonic_midi might be e.g. 61.5 (if microtonal tonic)
    # But usually standard MIDI integers.
    
    semitone_offset = pitch - tonic_midi
    
    # Determine nearest and second-nearest chromatic semitones
    lower = int(math.floor(semitone_offset))
    upper = int(math.ceil(semitone_offset))
    candidate_offsets = [lower] if lower == upper else [lower, upper]
    candidate_offsets.sort(key=lambda x: abs(semitone_offset - x))

    closest_offset = candidate_offsets[0]
    closest_pitch = tonic_midi + closest_offset
    closest_pc = int(closest_offset % 12)
    error_closest = (pitch - closest_pitch) * 100

    if mode == "raga" and allowed_pcs:
        if closest_pc in allowed_pcs:
            return closest_pitch, error_closest, _get_sargam_label(closest_pitch, tonic_midi), True

        if len(candidate_offsets) > 1:
            second_offset = candidate_offsets[1]
            second_pitch = tonic_midi + second_offset
            second_pc = int(second_offset % 12)
            error_second = (pitch - second_pitch) * 100
            if second_pc in allowed_pcs:
                return second_pitch, error_second, _get_sargam_label(second_pitch, tonic_midi), True

        return None, 0.0, "", False

    return closest_pitch, error_closest, _get_sargam_label(closest_pitch, tonic_midi), True


def _get_sargam_label(midi_val: float, tonic_midi: float) -> str:
    from .sequence import OFFSET_TO_SARGAM
    offset = int(round(midi_val - tonic_midi)) % 12
    base = OFFSET_TO_SARGAM.get(offset, "?")
    
    octave_shift = int(round((midi_val - tonic_midi) / 12))
    if octave_shift > 0:
        return base + "·" * octave_shift
    elif octave_shift < 0:
        return base + "'" * abs(octave_shift)
    return base


def _resolve_tonic(tonic: Union[int, str, float]) -> float:
    """Resolve tonic to an absolute MIDI value."""
    if isinstance(tonic, (float, np.floating)) and tonic > 200:
        # Treat large float values as Hz and map them to MIDI.
        return float(librosa.hz_to_midi(float(tonic)))

    if isinstance(tonic, (int, float, np.integer, np.floating)):
        return float(tonic)

    if isinstance(tonic, str):
        try:
            return float(librosa.note_to_midi(tonic))
        except Exception:
            # Fall back to pitch-class parsing and anchor at octave 4.
            return float(60 + tonic_to_midi_class(tonic))

    return 60.0


def transcribe_to_notes(
    pitch_hz: np.ndarray,
    timestamps: np.ndarray,
    voicing_mask: np.ndarray,
    tonic: Union[int, str, float],
    # Energy parameters
    energy: Optional[np.ndarray] = None,
    energy_threshold: float = 0.0,
    # Configurable parameters
    smoothing_sigma_ms: float = 70.0,
    frame_period_s: float = 0.01,
    derivative_threshold: float = 2.0,
    min_event_duration: float = 0.04,
    snap_mode: Literal["chromatic", "raga"] = "chromatic",
    allowed_raga_notes: Optional[List[int]] = None,
    snap_tolerance_cents: float = 35.0,
    transcription_min_duration: float = 0.0,  # Alias for min_event_duration if passed via explicit config
) -> List[Note]:
    """
    Unified entry point: Combines Stationary Events + Inflection Points.
    
    1. Detect Stationary Events (Orange Bars).
    2. Detect Inflection Points (Red Points).
    3. Filter Inflection Points overlapping Stationary Events.
    4. Convert all to Note objects.
    """
    if transcription_min_duration > 0:
        min_event_duration = transcription_min_duration
    
    # 1. Stationary Events
    events = detect_stationary_events(
        pitch_hz=pitch_hz,
        timestamps=timestamps,
        voicing_mask=voicing_mask,
        tonic=tonic,
        energy=energy,
        energy_threshold=energy_threshold,
        smoothing_sigma_ms=smoothing_sigma_ms,
        frame_period_s=frame_period_s,
        derivative_threshold=derivative_threshold,
        min_event_duration=min_event_duration,
        snap_mode=snap_mode,
        allowed_raga_notes=allowed_raga_notes,
        snap_tolerance_cents=snap_tolerance_cents,
    )
    
    # 2. Inflection Points
    inf_times, inf_pitches = detect_pitch_inflection_points(
        pitch_hz=pitch_hz,
        timestamps=timestamps,
        voicing_mask=voicing_mask,
        smoothing_sigma_ms=smoothing_sigma_ms,
        frame_period_s=frame_period_s,
    )
    
    # 3. Filter Inflections
    if len(inf_times) > 0 and events:
        keep_mask = np.ones(len(inf_times), dtype=bool)
        for evt in events:
            # Overlap check
            overlap = (inf_times >= evt.start) & (inf_times <= evt.end)
            keep_mask[overlap] = False
        
        inf_times = inf_times[keep_mask]
        inf_pitches = inf_pitches[keep_mask]
        
    # 4. Convert to Notes
    final_notes = []
    
    # Add Stationary Notes
    for evt in events:
        n = Note(
            start=evt.start,
            end=evt.end,
            pitch_midi=evt.pitch_midi,
            pitch_hz=float(librosa.midi_to_hz(evt.pitch_midi)),
            confidence=1.0, 
            sargam=evt.sargam, # Pre-calculated
            energy=evt.energy
        )
        # Keep pitch class handy for downstream analysis.
        n.pitch_class = int(round(evt.snapped_midi)) % 12
        final_notes.append(n)
        
    # Add Inflection Notes
    # These are instant points. We give them a tiny duration (e.g. 10ms)
    # Point notes get a small fixed duration for visualization.
    point_duration = 0.01 
    
    tonic_midi_val = _resolve_tonic(tonic)
    
    for t, p in zip(inf_times, inf_pitches):
        # Still snap inflection points so they can show sargam labels.
        snapped, _, sargam, keep_note = _snap_pitch(
            p, tonic_midi_val, snap_mode, allowed_raga_notes, snap_tolerance_cents
        )
        if not keep_note or snapped is None:
            continue
        
        n = Note(
            start=t,
            end=t + point_duration,
            pitch_midi=p,
            pitch_hz=float(librosa.midi_to_hz(p)),
            confidence=0.8, # Lower confidence for transient points
            sargam=sargam,
            pitch_class=int(round(snapped)) % 12
        )
        final_notes.append(n)
        
    # 5. Sort by start time
    final_notes.sort(key=lambda x: x.start)
    
    return final_notes
