"""
Raga module: database, candidate matching, feature extraction, and scoring.

This module implements the exact scoring algorithm from ssje_tweaked_wit_peaks.ipynb.

Provides:
- RagaDatabase: Load and query raga definitions
- generate_candidates: Generate (raga, tonic) candidates from pitch classes
- score_candidates: Score using the 8-coefficient algorithm from the notebook
- RagaScorer: Orchestrates the full scoring pipeline
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Set, Dict, Tuple, Optional, Union
import os
import numpy as np
import pandas as pd
import librosa
from .sequence import Note

from .config import PipelineConfig
from .audio import PitchData
from .analysis import HistogramData, PeakData


# =============================================================================
# SCORING PARAMETERS (from notebook)
# =============================================================================

@dataclass
class ScoringParams:
    """Scoring coefficients from the notebook."""
    
    EPS: float = 1e-12
    ALPHA_MATCH: float = 0.40          # match mass coefficient
    BETA_PRESENCE: float = 0.25        # presence coefficient
    GAMMA_LOGLIKE: float = 1.0         # log-likelihood coefficient
    DELTA_EXTRA: float = 1.10          # extra mass penalty (-ve)
    COMPLEX_PENALTY: float = 0.4       # complexity penalty (-ve)
    MATCH_SIZE_GAMMA: float = 0.25     # size mismatch coefficient
    TONIC_SALIENCE_WEIGHT: float = 0.12  # tonic salience weight
    SCALE: float = 1000.0
    USE_PRESENCE_MEAN: bool = True     # mean vs sum/sqrt
    USE_NORM_PRIMARY: bool = True
    WINDOW_CENTS: float = 35.0         # window for note mass computation


DEFAULT_SCORING_PARAMS = ScoringParams()


# =============================================================================
# TONIC BIAS CONSTANTS (for instrumental/vocal source types)
# =============================================================================

# Tonic bias ranges (pitch class 0-11, where 0=C)
# These restrict candidate tonics based on source type and instrument/vocalist
TONIC_BIAS = {
    # Vocalist gender (biases toward typical vocal ranges)
    "vocal_female": [7, 8, 9, 10, 11, 0],     # G, G#, A, A#, B, C (bias around A-A#)
    "vocal_male": [11, 0, 1, 2, 3, 4, 5],     # B, C, C#, D, D#, E, F (bias around D-D#)
    
    # Instrument-specific tonic ranges
    "sarod": [10, 11, 0, 1, 2],               # A#, B, C, C#, D
    "sitar": [1, 2, 3],                       # C#, D, D#
    "bansuri": [2, 3, 4, 5],                  # D, D#, E, F
    "slide_guitar": [1, 2, 3],                # C#, D, D#
    
    # Default: all tonics allowed
    "autodetect": list(range(12)),
}


def get_tonic_candidates(
    source_type: str,
    vocalist_gender: str = None,
    instrument_type: str = "autodetect",
) -> List[int]:
    """
    Get list of valid tonic candidates based on source configuration.
    
    Args:
        source_type: "mixed", "instrumental", or "vocal"
        vocalist_gender: "male" or "female" (only for source_type="vocal")
        instrument_type: Instrument name (only for source_type="instrumental")
        
    Returns:
        List of pitch classes (0-11) to use as candidate tonics
    """
    if source_type == "vocal" and vocalist_gender:
        key = f"vocal_{vocalist_gender}"
        if key in TONIC_BIAS:
            return TONIC_BIAS[key].copy()
    
    if source_type == "instrumental" and instrument_type:
        if instrument_type in TONIC_BIAS:
            return TONIC_BIAS[instrument_type].copy()
    
    # Default: all tonics
    return list(range(12))


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Candidate:
    """A (raga, tonic) candidate for scoring."""
    
    tonic: int                    # 0-11 (pitch class of Sa)
    mask: Tuple[int, ...]         # 12-bit binary mask (absolute, not rotated)
    raga_names: List[str]         # Matching raga names
    intervals: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class ScoredCandidate:
    """Scoring result for a single candidate."""
    
    raga: str
    tonic: int
    tonic_name: str
    salience: int                 # Raw accompaniment count at tonic
    fit_score: float              # Final score (scaled)
    primary_score: float          # Sa + Pa/Ma bonus
    match_mass: float
    extra_mass: float
    observed_note_score: float    # Presence score
    loglike_norm: float
    raga_size: int
    match_diff: int               # |raga_size - detected_peaks|
    complexity_pen: float


# =============================================================================
# RAGA DATABASE
# =============================================================================

class RagaDatabase:
    """
    Load and query raga definitions from CSV.
    """
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
        # Build lookup tables
        self.interval_lookup: Dict[Tuple, List[dict]] = defaultdict(list)
        self.name_to_mask: Dict[str, Tuple[int, ...]] = {}
        self.all_ragas: List[dict] = []
        
        self._build_lookups()
    
    def _build_lookups(self):
        """Build interval and name lookup tables."""
        for _, row in self.df.iterrows():
            # Parse mask
            if "mask" in self.df.columns and pd.notna(row.get("mask")):
                raw = row["mask"]
                if isinstance(raw, str):
                    parts = [x.strip() for x in raw.split(",") if x.strip()]
                    try:
                        mask_abs = tuple(int(x) for x in parts)
                    except Exception:
                        mask_abs = None
                else:
                    mask_abs = None
            else:
                mask_abs = None
            
            if mask_abs is None:
                try:
                    # Try positional lookup using iloc (safest for 0-11 columns)
                    # We look for 12 columns that might hold the pitch class mask
                    # Assuming they are the first 12 or named 0-11
                    mask_abs = tuple(int(row.iloc[i]) for i in range(12))
                except Exception:
                    try:
                        mask_abs = tuple(int(row[str(i)]) for i in range(12))
                    except Exception:
                        continue
            
            # Get names
            names_raw = row.get("names") if "names" in self.df.columns else row.get("raga", f"raga_{_}")
            names = self._parse_names(names_raw)
            
            # Get pitch classes from mask
            mask_arr = np.array(mask_abs, dtype=int)
            pitch_classes = np.where(mask_arr == 1)[0]
            if len(pitch_classes) < 2:
                continue
            
            # Compute interval pattern
            pitch_classes_sorted = np.sort(pitch_classes)
            intervals = tuple(
                np.mod(np.diff(np.concatenate((pitch_classes_sorted, [pitch_classes_sorted[0] + 12]))), 12)
            )
            
            if sum(intervals) != 12:
                continue
            
            canonical = self._canonical_intervals(intervals)
            
            raga_entry = {
                "names": names,
                "mask": mask_abs,
                "pitch_classes": tuple(pitch_classes_sorted),
                "intervals": intervals,
                "size": len(pitch_classes),
            }
            
            self.interval_lookup[canonical].append(raga_entry)
            self.all_ragas.append(raga_entry)
            
            for name in names:
                self.name_to_mask[name.lower()] = mask_abs
    
    def _parse_names(self, names_raw) -> List[str]:
        if pd.isna(names_raw):
            return ["Unknown"]
        names_str = str(names_raw).strip()
        if names_str.startswith("[") and names_str.endswith("]"):
            names_str = names_str[1:-1].replace('"', "").replace("'", "")
        names = [n.strip() for n in names_str.split(",") if n.strip()]
        return names if names else ["Unknown"]
    
    def _canonical_intervals(self, intervals: Tuple[int, ...]) -> Tuple[int, ...]:
        if not intervals:
            return tuple()
        return min(tuple(np.roll(intervals, i)) for i in range(len(intervals)))
    
    def get_interval_lookup(self) -> Dict[Tuple, List[dict]]:
        return self.interval_lookup


# =============================================================================
# CANDIDATE GENERATION & SCORING
# =============================================================================

def generate_candidates(
    pitch_classes: Set[int],
    raga_db: RagaDatabase,
) -> List[Candidate]:
    """
    Generate (raga, tonic) candidates from detected pitch classes.
    """
    if not pitch_classes:
        return []
    
    candidates = []
    pitch_classes_arr = np.array(sorted(pitch_classes))
    k = len(pitch_classes_arr)
    
    intervals = np.mod(
        np.diff(np.concatenate((pitch_classes_arr, [pitch_classes_arr[0] + 12]))), 12
    )
    
    interval_lookup = raga_db.get_interval_lookup()
    
    for j in range(k):
        tonic = int(pitch_classes_arr[j])
        rotated_intervals = tuple(np.roll(intervals, -j))
        canonical = raga_db._canonical_intervals(rotated_intervals)
        
        if canonical in interval_lookup:
            raga_group = interval_lookup[canonical]
            matching_ragas = [r for r in raga_group if r["intervals"] == rotated_intervals]
            
            if matching_ragas:
                current_note = 0
                relative_notes = {0}
                for step in rotated_intervals[:-1]:
                    current_note += step
                    relative_notes.add(current_note)
                
                mask_rel = tuple(1 if i in relative_notes else 0 for i in range(12))
                all_names = []
                for r in matching_ragas:
                    all_names.extend(r["names"])
                
                candidates.append(Candidate(
                    tonic=tonic,
                    mask=mask_rel,
                    raga_names=all_names,
                    intervals=rotated_intervals,
                ))
    
    return candidates


def score_candidates_full(
    pitch_data_vocals: PitchData,
    pitch_data_accomp: Optional[PitchData],
    raga_db: RagaDatabase,
    detected_peak_count: int,
    instrument_mode: str = "autodetect",
    params: ScoringParams = DEFAULT_SCORING_PARAMS,
    tonic_candidates: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Full scoring pipeline matching the notebook exactly.
    
    This computes cent-level histograms, applies accompaniment salience filtering,
    and scores all ragas against all allowed tonics.
    
    Args:
        pitch_data_vocals: Pitch data from melody/vocals
        pitch_data_accomp: Pitch data from accompaniment (can be None)
        raga_db: Raga database
        detected_peak_count: Number of detected peaks
        instrument_mode: Source type for fallback tonic selection
        params: Scoring parameters
        tonic_candidates: Optional list of allowed tonics (from get_tonic_candidates)
    """
    EPS = params.EPS
    
    # === Build cent-level histogram for melody ===
    midi_vals_mel = pitch_data_vocals.midi_vals
    cent_vals_mel = (midi_vals_mel % 12) * 100.0
    
    num_bins = int(1200 / 1.0)  # 1-cent bins
    bin_edges = np.linspace(0.0, 1200.0, num_bins + 1)
    cent_hist, _ = np.histogram(cent_vals_mel, bins=bin_edges, range=(0.0, 1200.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # === Mass within window for each note ===
    def mass_within_window_for_note(note_idx):
        center = (note_idx * 100.0) % 1200.0
        diff = np.abs(bin_centers - center)
        diff = np.minimum(diff, 1200.0 - diff)  # circular
        return float(np.sum(cent_hist[diff <= params.WINDOW_CENTS]))
    
    note_masses_raw = np.array([mass_within_window_for_note(i) for i in range(12)], dtype=float)
    H_pc_arr = note_masses_raw.copy()
    
    # Normalized probability distribution
    p_pc = (H_pc_arr + EPS) / (np.sum(H_pc_arr) + 12 * EPS)
    
    # === Accompaniment salience (only if accompaniment available) ===
    has_accompaniment = pitch_data_accomp is not None and len(pitch_data_accomp.midi_vals) > 0
    
    if has_accompaniment:
        midi_vals_acc = pitch_data_accomp.midi_vals
        pitch_classes_acc = np.mod(np.round(midi_vals_acc), 12).astype(int)
        H_acc, _ = np.histogram(pitch_classes_acc, bins=12, range=(0, 12))
        salience_all_tonics = {t: int(H_acc[t]) for t in range(12)}
        max_acc = float(H_acc.max()) if H_acc.size and H_acc.max() > 0 else 1.0
        
        # Tonic filtering based on mean salience
        mean_salience = float(np.mean(list(salience_all_tonics.values())))
        surviving_tonics = [t for t, a in salience_all_tonics.items() if a > mean_salience]
        
        print(f"[SCORING] Mean accomp salience = {mean_salience:.1f}; {len(surviving_tonics)} surviving tonics")
    else:
        # No accompaniment - use melody histogram peaks as tonic hints
        salience_all_tonics = {t: 0 for t in range(12)}
        max_acc = 1.0
        mean_salience = 0.0
        surviving_tonics = []
        print("[SCORING] No accompaniment - using tonic bias or all tonics")
    
    # === Determine allowed tonics ===
    # Priority: explicit tonic_candidates > tonic bias > accompaniment filtering > all
    if tonic_candidates is not None and len(tonic_candidates) > 0:
        allowed_tonics = tonic_candidates.copy()
        print(f"[SCORING] Using tonic candidates from bias: {allowed_tonics}")
    elif has_accompaniment and len(surviving_tonics) > 0:
        allowed_tonics = surviving_tonics.copy()
    else:
        allowed_tonics = list(range(12))
    
    print(f"[SCORING] Using tonics: {allowed_tonics}")
    
    # === Score all ragas ===
    final_results = []
    
    for raga_entry in raga_db.all_ragas:
        mask_abs = np.array(raga_entry["mask"], dtype=int)
        names = raga_entry["names"]
        
        if mask_abs.sum() < 2:
            continue
        
        for tonic in allowed_tonics:
            # Tonic salience check for autodetect (only when we have accompaniment)
            tonic_sal = float(salience_all_tonics.get(tonic, 0))
            if has_accompaniment and tonic_candidates is None and instrument_mode == "autodetect" and tonic_sal <= mean_salience:
                continue
            
            # Rotate observed distribution
            p_rot = np.roll(p_pc, -tonic)
            
            raga_note_indices = np.where(mask_abs == 1)[0].tolist()
            raga_size = len(raga_note_indices)
            
            # Size penalty
            size_diff = abs(raga_size - detected_peak_count)
            size_penalty = params.MATCH_SIZE_GAMMA * (size_diff / 4.0)
            
            if raga_size < 2:
                continue
            
            # Match mass (fraction on allowed notes)
            match_mass = float(np.sum(p_rot[raga_note_indices]))
            extra_mass = float(1.0 - match_mass)
            
            # Presence (peak-normalized)
            peak = float(np.max(p_rot) + EPS)
            pres = (p_rot[raga_note_indices] / peak)
            
            if params.USE_PRESENCE_MEAN:
                observed_note_score = float(np.mean(pres)) if raga_size > 0 else 0.0
            else:
                observed_note_score = float(np.sum(pres) / (np.sqrt(raga_size) + EPS))
            
            # Log-likelihood
            sum_logp = float(np.sum(np.log(p_rot[raga_note_indices] + EPS)))
            baseline = -np.log(12.0)
            avg_logp = sum_logp / (raga_size + EPS)
            loglike_norm = 1.0 + (avg_logp / (-baseline + EPS))
            loglike_norm = max(0.0, min(1.0, loglike_norm))
            
            # Complexity penalty
            complexity_pen = max(0.0, (raga_size - 5) / 12.0)
            
            # Primary score (Sa + Pa/Ma bonus)
            if params.USE_NORM_PRIMARY:
                prim = float(p_rot[0])
                bonus_options = [0.0]
                if mask_abs[5] == 1:
                    bonus_options.append(float(p_rot[5]))
                if mask_abs[6] == 1:
                    bonus_options.append(float(p_rot[6]))
                if mask_abs[7] == 1:
                    bonus_options.append(float(p_rot[7]))
                primary_score = prim + max(bonus_options)
            else:
                H_rot_counts = np.roll(H_pc_arr, -tonic)
                sa_bonus = float(H_rot_counts[0])
                bonus_options = [0.0]
                if mask_abs[5] == 1:
                    bonus_options.append(float(H_rot_counts[5]))
                if mask_abs[6] == 1:
                    bonus_options.append(float(H_rot_counts[6]))
                if mask_abs[7] == 1:
                    bonus_options.append(float(H_rot_counts[7]))
                primary_score = sa_bonus + max(bonus_options)
            
            # Combined fit score
            fit_norm = (
                params.ALPHA_MATCH * match_mass +
                params.BETA_PRESENCE * observed_note_score +
                params.GAMMA_LOGLIKE * loglike_norm
            ) - (
                params.DELTA_EXTRA * extra_mass +
                params.COMPLEX_PENALTY * complexity_pen +
                size_penalty
            )
            
            # Tonic salience boost
            tonic_sal_norm = tonic_sal / (max_acc + EPS)
            fit_norm += params.TONIC_SALIENCE_WEIGHT * tonic_sal_norm
            
            fit_norm = max(-1.0, min(1.0, fit_norm))
            fit_score = float(fit_norm * params.SCALE)
            
            final_results.append({
                "raga": ", ".join(names),
                "tonic": int(tonic),
                "tonic_name": _tonic_to_name(tonic),
                "salience": int(tonic_sal),
                "fit_score": fit_score,
                "primary_score": float(primary_score),
                "match_mass": match_mass,
                "extra_mass": extra_mass,
                "observed_note_score": observed_note_score,
                "loglike_norm": loglike_norm,
                "raga_size": raga_size,
                "match_diff": size_diff,
                "complexity_pen": complexity_pen,
            })
    
    df = pd.DataFrame(final_results)
    
    if len(df) > 0:
        # Sort by fit_score -> primary_score -> salience
        # This matches the notebook behavior (fusion=False mode)
        df = df.sort_values(
            by=["fit_score", "primary_score", "salience"],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        df["rank"] = df.index + 1
    
    return df


# =============================================================================
# RAGA CORRECTION & FILTERING
# =============================================================================

def get_raga_notes(raga_db: RagaDatabase, raga_name: str, tonic: Union[int, str]) -> List[int]:
    """
    Get the valid notes for a specific raga relative to the given tonic.
    
    Args:
        raga_db: RagaDatabase instance
        raga_name: Name of the raga (can be comma-separated list, first match used)
        tonic: Tonic pitch class (0-11) or note name
        
    Returns:
        List of pitch classes (0-11) valid in this raga
    """
    candidate_names = [n.strip() for n in raga_name.split(',')]
    mask_abs = None
    matched_name = None
    
    for name in candidate_names:
        # 1. Exact match
        if raga_db.name_to_mask and name.lower() in raga_db.name_to_mask:
            mask_abs = raga_db.name_to_mask[name.lower()]
            matched_name = name
            break
            
        # 2. Fuzzy match
        for r in raga_db.all_ragas:
            for db_name in r["names"]:
                # Check if input name is part of DB name, or DB name is part of input name
                if name.lower() in db_name.lower() or db_name.lower() in name.lower():
                    mask_abs = r["mask"]
                    matched_name = db_name
                    break
            if mask_abs: break
        if mask_abs: break
    
    if mask_abs is None:
         print(f"[WARN] Raga '{raga_name}' (or sub-names) not found in DB. Allowing all notes.")
         return list(range(12))
    
    print(f"  [INFO] Found raga match: '{matched_name}'")
    
    tonic_pc = _parse_tonic(tonic) if isinstance(tonic, str) else int(tonic)
    
    mask_arr = np.array(mask_abs)
    relative_indices = np.where(mask_arr == 1)[0]
    
    # Transpose to actual tonic
    valid_pcs = [(pc + tonic_pc) % 12 for pc in relative_indices]
    
    return sorted(valid_pcs)


def snap_to_raga_notes(
    midi_notes: np.ndarray, 
    valid_pcs: List[int], 
    max_distance: float = 1.0, 
    discard_far: bool = False
) -> Tuple[np.ndarray, List[dict]]:
    """
    Snap MIDI notes to the nearest valid raga notes.
    """
    corrected_midi = []
    correction_info = []
    
    for original_midi in midi_notes:
        if np.isnan(original_midi):
            continue
            
        original_pc = int(round(original_midi)) % 12
        octave = int(round(original_midi)) // 12
        
        # Find closest valid PC
        distances = []
        for valid_pc in valid_pcs:
            dist1 = abs(original_pc - valid_pc)
            dist2 = 12 - dist1
            min_dist = min(dist1, dist2)
            distances.append((min_dist, valid_pc))
        
        min_distance, closest_pc = min(distances, key=lambda x: x[0])
        
        # Discard if too far
        if discard_far and min_distance > max_distance:
            correction_info.append({
                'original_midi': original_midi,
                'action': 'discarded'
            })
            continue
            
        # Correct logic
        if min_distance <= max_distance:
            # Reconstruct MIDI note in correct octave
            # We want closest MIDI note to original_midi that has pitch class closest_pc
            # Easy way: round original to nearest (octave*12 + closest_pc) equivalents
            
            # Candidates: (octave-1), octave, (octave+1)
            candidates = [
                (octave - 1) * 12 + closest_pc,
                octave * 12 + closest_pc,
                (octave + 1) * 12 + closest_pc
            ]
            
            corrected_midi_note = min(candidates, key=lambda x: abs(x - original_midi))
            
            action = 'corrected' if min_distance > 0 else 'unchanged'
            corrected_midi.append(corrected_midi_note)
            correction_info.append({
                'original_midi': original_midi,
                'corrected_midi': corrected_midi_note,
                'distance': min_distance,
                'action': action,
                'closest_pc': closest_pc
            })
        else:
            # Keep original if not discarding but too far to snap (fallback)
            corrected_midi.append(original_midi)
            correction_info.append({
                'original_midi': original_midi,
                'corrected_midi': original_midi,
                'distance': min_distance,
                'action': 'unchanged_far'
            })
            
    return np.array(corrected_midi), correction_info


def apply_raga_correction_to_notes(
    note_sequence: List['Note'], # Forward ref
    raga_db: RagaDatabase, 
    raga_name: str, 
    tonic: Union[int, str], 
    max_distance: float = 1.0, 
    keep_impure: bool = False,
) -> Tuple[List['Note'], dict, List[dict]]:
    """
    Apply raga-based filtering/correction to a sequence of Notes.
    """
    valid_pcs = get_raga_notes(raga_db, raga_name, tonic)
    
    corrected_sequence = []
    all_corrections = []
    
    # Stats
    stats = {
        'total': len(note_sequence),
        'unchanged': 0,
        'corrected': 0,
        'discarded': 0,
        'valid_pcs': valid_pcs
    }
    
    for note in note_sequence:
        midi_val = note.pitch_midi
        
        # Use snap logic on single note
        _, info_list = snap_to_raga_notes([midi_val], valid_pcs, max_distance, discard_far=not keep_impure)
        if not info_list:
            # Implicitly discarded if logic returned nothing
             stats['discarded'] += 1
             continue
             
        info = info_list[0]
        all_corrections.append(info)
        
        action = info['action']
        
        if action == 'discarded':
            stats['discarded'] += 1
            continue
            
        elif action == 'unchanged' or action == 'unchanged_far':
             stats['unchanged'] += 1
             corrected_sequence.append(note)
             
        elif action == 'corrected':
             if keep_impure:
                 # If keeping impure, we might just tag it or keep original? 
                 # User said: "make a flag to not delete the notes outside the raga? default behaviour should be to delete"
                 # Implicitly, if we KEEP them, we probably shouldn't correct them forcefully if they are far, 
                 # but 'corrected' implies they were close enough (within max_distance).
                 # If they are 'corrected', they are snapped.
                 # If they were far (> max_distance), 'snap_to' would mark them 'unchanged_far' if discard_far=False.
                 
                 # Logic for 'corrected' (within 1 semitone):
                 # We essentially quantize them to the raga.
                 # Let's create a NEW note with corrected pitch.
                 new_pitch = info['corrected_midi']
                 stats['corrected'] += 1
                 
                 # Create copy of note with new pitch

                 # Just instantiate new Note
                 new_note = Note(
                     start=note.start,
                     end=note.end,
                     pitch_midi=new_pitch,
                     pitch_hz=librosa.midi_to_hz(new_pitch),
                     confidence=note.confidence
                 )
                 corrected_sequence.append(new_note)
             else:
                 # Default behavior: Delete notes outside raga?
                 # Wait, user said "delete notes outside the raga".
                 # Notes "outside" are usually those FAR from raga notes.
                 # Notes close (within 1 st) are usually intonation errors designated for CORRECTION.
                 # So:
                 #   Close -> Correct (Snap)
                 #   Far -> Discard (Delete)
                 
                 # Case: Corrected (Close)
                 new_pitch = info['corrected_midi']
                 stats['corrected'] += 1
                 new_note = Note(
                     start=note.start,
                     end=note.end,
                     pitch_midi=new_pitch,
                     pitch_hz=librosa.midi_to_hz(new_pitch),
                     confidence=note.confidence
                 )
                 corrected_sequence.append(new_note)
                 
    stats['remaining'] = len(corrected_sequence)
    return corrected_sequence, stats, all_corrections


def _tonic_to_name(tonic: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return names[tonic % 12]


def _parse_tonic(tonic_str: str) -> int:
    """Parse tonic string (e.g. 'C#', 'Db', '1') to integer 0-11."""
    if not tonic_str:
        raise ValueError("Empty tonic string")
        
    # Handle integer input
    if isinstance(tonic_str, int):
        return tonic_str % 12
        
    s = str(tonic_str).strip().upper()
    
    # Handle numeric string
    if s.isdigit():
        return int(s) % 12
    
    # Handle note names
    note_map = {
        "C": 0, "B#": 0,
        "C#": 1, "DB": 1,
        "D": 2,
        "D#": 3, "EB": 3,
        "E": 4, "FB": 4,
        "F": 5, "E#": 5,
        "F#": 6, "GB": 6,
        "G": 7,
        "G#": 8, "AB": 8,
        "A": 9,
        "A#": 10, "BB": 10,
        "B": 11, "CB": 11,
        "SA": 0, # Assume Sa is relative 0 if passed, but usually we want absolute
    }
    
    if s in note_map:
        return note_map[s]
        
    raise ValueError(f"Invalid tonic: {tonic_str}")


class RagaScorer:
    """
    Score and rank candidates using either:
    - Full notebook algorithm (default)
    - Trained ML model (if provided)
    """
    
    def __init__(
        self,
        raga_db: RagaDatabase,
        model_path: Optional[str] = None,
        use_ml: bool = False,
        params: ScoringParams = DEFAULT_SCORING_PARAMS,
    ):
        self.raga_db = raga_db
        self.params = params
        self.use_ml = use_ml
        self.model = None
        self.scaler = None
        
        if use_ml and model_path and os.path.isfile(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        try:
            import joblib
            self.model = joblib.load(model_path)
            scaler_path = model_path.replace("_model.pkl", "_scaler.pkl")
            if os.path.isfile(scaler_path):
                self.scaler = joblib.load(scaler_path)
            print(f"[SCORER] Loaded ML model: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"[SCORER] Failed to load model: {e}")
            self.model = None
            self.use_ml = False
    
    def score(
        self,
        pitch_data_vocals: PitchData,
        pitch_data_accomp: PitchData,
        detected_peak_count: int,
        instrument_mode: str = "autodetect",
        tonic_candidates: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Score all raga candidates.
        
        Returns DataFrame sorted by score.
        """
        return score_candidates_full(
            pitch_data_vocals,
            pitch_data_accomp,
            self.raga_db,
            detected_peak_count,
            instrument_mode,
            self.params,
            tonic_candidates,
        )
