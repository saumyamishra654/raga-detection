"""
Analysis module: histogram construction, peak detection, and GMM fitting.

Provides:
- compute_cent_histograms: Build dual-resolution pitch class histograms
- detect_peaks: Cross-validated peak detection with pitch class mapping
- fit_gmm_to_peaks: Gaussian Mixture Model analysis for microtonal movements
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture

from .config import PipelineConfig
from .audio import PitchData


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HistogramData:
    """Container for dual-resolution cent histograms."""
    
    # Raw histograms
    high_res: np.ndarray          # 100-bin histogram
    low_res: np.ndarray           # 33-bin histogram
    
    # Smoothed histograms
    smoothed_high: np.ndarray
    smoothed_low: np.ndarray
    
    # Bin centers (cents)
    bin_centers_high: np.ndarray
    bin_centers_low: np.ndarray
    
    # Normalized versions (for plotting)
    high_res_norm: np.ndarray = field(default_factory=lambda: np.array([]))
    smoothed_high_norm: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        """Compute normalized versions if not provided."""
        if len(self.high_res_norm) == 0:
            max_val = self.high_res.max() if self.high_res.max() > 0 else 1
            self.high_res_norm = self.high_res / max_val
        if len(self.smoothed_high_norm) == 0:
            max_val = self.smoothed_high.max() if self.smoothed_high.max() > 0 else 1
            self.smoothed_high_norm = self.smoothed_high / max_val


@dataclass
class PeakData:
    """Container for peak detection results."""
    
    # High-resolution peaks
    high_res_indices: np.ndarray
    high_res_cents: np.ndarray
    
    # Low-resolution peaks
    low_res_indices: np.ndarray
    low_res_cents: np.ndarray
    
    # Cross-validated peaks
    validated_indices: np.ndarray
    validated_cents: np.ndarray
    
    # Pitch classes (0-11)
    pitch_classes: Set[int] = field(default_factory=set)
    
    # Peak details for summary
    peak_details: List[dict] = field(default_factory=list)


@dataclass
class GMMResult:
    """Container for GMM fitting results per peak."""
    
    peak_idx: int
    peak_cent: float
    nearest_note: int              # 0-11
    
    # GMM parameters
    means: np.ndarray              # Cent positions
    sigmas: np.ndarray             # Standard deviations
    weights: np.ndarray            # Component weights
    
    # Derived
    primary_mean: float = 0.0      # Mean of dominant component
    primary_sigma: float = 0.0     # Sigma of dominant component
    deviation_from_note: float = 0.0  # Cents from ideal note center


# =============================================================================
# HISTOGRAM CONSTRUCTION
# =============================================================================

def compute_cent_histograms(
    pitch_data: PitchData,
    bins_high: int = 100,
    bins_low: int = 33,
    sigma: float = 0.8,
    use_confidence_weights: bool = False,
) -> HistogramData:
    """
    Compute dual-resolution cent histograms with Gaussian smoothing.
    
    Args:
        pitch_data: PitchData containing midi_vals and optionally confidence
        bins_high: Number of bins for high-resolution histogram (default 100)
        bins_low: Number of bins for low-resolution histogram (default 33)
        sigma: Gaussian smoothing kernel width
        use_confidence_weights: If True, weight histogram by confidence values
        
    Returns:
        HistogramData with raw and smoothed histograms
    """
    # Convert MIDI to cents (0-1200 range, one octave)
    cent_vals = (pitch_data.midi_vals % 12) * 100.0
    
    # Prepare weights (confidence * frame_duration if available)
    weights = None
    if use_confidence_weights and pitch_data.confidence is not None and len(pitch_data.confidence) == len(cent_vals):
        weights = pitch_data.confidence
        # Optionally multiply by frame duration for time-weighted histogram
        if pitch_data.timestamps is not None and len(pitch_data.timestamps) > 1:
            frame_duration = np.median(np.diff(pitch_data.timestamps[:min(50, len(pitch_data.timestamps))]))
            weights = weights * frame_duration
    
    # High-resolution histogram (12 per bin with 100 bins)
    high_res, bin_edges_high = np.histogram(
        cent_vals, bins=bins_high, range=(0.0, 1200.0), weights=weights
    )
    bin_centers_high = (bin_edges_high[:-1] + bin_edges_high[1:]) / 2.0
    
    # Low-resolution histogram (~36 per bin with 33 bins)
    low_res, bin_edges_low = np.histogram(
        cent_vals, bins=bins_low, range=(0.0, 1200.0), weights=weights
    )
    bin_centers_low = (bin_edges_low[:-1] + bin_edges_low[1:]) / 2.0
    
    # Gaussian smoothing with circular wrap (octave is circular)
    smoothed_high = gaussian_filter1d(
        high_res.astype(float), sigma=sigma, mode="wrap"
    )
    smoothed_low = gaussian_filter1d(
        low_res.astype(float), sigma=sigma, mode="wrap"
    )
    
    return HistogramData(
        high_res=high_res.astype(float),
        low_res=low_res.astype(float),
        smoothed_high=smoothed_high,
        smoothed_low=smoothed_low,
        bin_centers_high=bin_centers_high,
        bin_centers_low=bin_centers_low,
    )


def compute_cent_histograms_from_config(
    pitch_data: PitchData, config: PipelineConfig
) -> HistogramData:
    """Compute histograms using configuration parameters."""
    return compute_cent_histograms(
        pitch_data,
        bins_high=config.histogram_bins_high,
        bins_low=config.histogram_bins_low,
        sigma=config.smoothing_sigma,
        use_confidence_weights=getattr(config, 'use_confidence_weights', True),
    )


# =============================================================================
# PEAK DETECTION
# =============================================================================

def detect_peaks(
    histogram: HistogramData,
    tolerance_cents: float = 35.0,
    peak_tolerance_cents: float = 45.0,
    prominence_high_factor: float = 0.03,
    prominence_low_factor: float = 0.01,
    debug: bool = False,
) -> PeakData:
    """
    Detect and cross-validate peaks, map to pitch classes.
    
    Args:
        histogram: HistogramData from compute_cent_histograms
        tolerance_cents: Window for mapping peaks to semitone centers
        peak_tolerance_cents: Window for cross-validation between resolutions
        prominence_high_factor: Prominence threshold factor for high-res
        prominence_low_factor: Prominence threshold factor for low-res
        debug: If True, print diagnostic information
        
    Returns:
        PeakData with validated peaks and pitch classes
    """
    # High-resolution peak detection
    # Notebook: prom_high = max(1.0, 0.03 * float(smoothed_H_100.max()))
    prom_high = max(1.0, prominence_high_factor * float(histogram.smoothed_high.max()))
    high_peaks, high_props = find_peaks(
        histogram.smoothed_high, prominence=prom_high, distance=2
    )
    
    # Add circular edge checks (bins 0 and max-1)
    high_peaks = _add_edge_peaks(histogram.smoothed_high, high_peaks, prom_high)
    
    high_cents = histogram.bin_centers_high[high_peaks]
    
    if debug:
        print(f"\n=== PEAK DETECTION DEBUG ===")
        print(f"High-res histogram max: {histogram.smoothed_high.max():.2f}")
        print(f"Prominence threshold (high-res): {prom_high:.2f} ({prominence_high_factor*100:.1f}% of max)")
        print(f"High-res peaks found: {len(high_peaks)}")
        print(f"High-res peak positions (cents): {high_cents}")
    
    # Low-resolution peak detection (on RAW histogram, per notebook)
    # Notebook: prom_low = max(0, 0.01 * float(H_mel_25.max()))
    prom_low = max(0.0, prominence_low_factor * float(histogram.low_res.max()))
    low_peaks, low_props = find_peaks(
        histogram.low_res, prominence=prom_low, distance=1
    )
    
    # Add circular edge checks
    low_peaks = _add_edge_peaks(histogram.low_res, low_peaks, prom_low)
    
    low_cents = histogram.bin_centers_low[low_peaks]
    
    if debug:
        print(f"\nLow-res histogram max: {histogram.low_res.max():.2f}")
        print(f"Prominence threshold (low-res): {prom_low:.2f} ({prominence_low_factor*100:.1f}% of max)")
        print(f"Low-res peaks found: {len(low_peaks)}")
        print(f"Low-res peak positions (cents): {low_cents}")
    
    # Cross-validate: high-res peaks must align with low-res within tolerance
    validated_indices = []
    validated_cents_list = []
    filtered_peaks = []  # Track filtered peaks for debug
    
    # Notebook logic:
    # is_validated = any(abs(sp_cent - rp_cent) <= peak_tolerance_cents for rp_cent in raw_peaks_cents)
    # The existing pipeline logic was functionally equivalent but used min() check
    
    for idx, cent in zip(high_peaks, high_cents):
        if len(low_cents) == 0:
            # With no low-res peaks available, accept every high-res peak as a fallback.
            validated_indices.append(idx)
            validated_cents_list.append(cent)
        else:
            # Circular distance to nearest low-res peak
            dists = np.abs((cent - low_cents + 600) % 1200 - 600)
            min_dist = np.min(dists)
            
            # We use circular distance so boundary cases (e.g., 10¢ vs 1190¢) wrap correctly.
            
            if min_dist <= peak_tolerance_cents:
                validated_indices.append(idx)
                validated_cents_list.append(cent)
            else:
                filtered_peaks.append((cent, min_dist))
    
    if debug and filtered_peaks:
        print(f"\n⚠ FILTERED PEAKS (failed cross-validation):")
        for cent, dist in filtered_peaks:
            pc = int(round(cent / 100)) % 12
            print(f"  - {cent:.1f}¢ (PC={pc}) → nearest low-res peak: {dist:.1f}¢ away (max allowed: {peak_tolerance_cents}¢)")
    
    validated_indices = np.array(validated_indices, dtype=int)
    validated_cents = np.array(validated_cents_list)
    
    # Map to pitch classes and remove duplicates (keep best match per note)
    note_centers = np.arange(0, 1200, 100)  # 0, 100, 200, ..., 1100
    
    # Temporary storage for grouping by pitch class
    peaks_by_note = {} # note_idx -> list of (distance, detail_dict, original_idx, cent)
    
    for idx, cent in zip(validated_indices, validated_cents):
        # Circular distance to nearest semitone center
        diffs = np.abs((cent - note_centers + 600) % 1200 - 600)
        nearest_note = int(np.argmin(diffs)) # 0-11
        note_dist = float(diffs[nearest_note])
        
        detail = {
            "bin_idx": int(idx),
            "cent_position": float(cent),
            "mapped_pc": nearest_note % 12,
            "distance_to_note": note_dist,
            "within_tolerance": note_dist <= tolerance_cents,
        }
        
        # Only peaks within tolerance contribute to merged pitch-classes; others are flagged as microtonal noise.
        
        if nearest_note not in peaks_by_note:
            peaks_by_note[nearest_note] = []
        peaks_by_note[nearest_note].append(detail)

    # Filter duplicates
    final_validated_indices = []
    final_validated_cents = []
    pitch_classes = set()
    peak_details = []
    
    for note_idx in sorted(peaks_by_note.keys()):
        candidates = peaks_by_note[note_idx]
        
        # Sort candidates by distance to note center (prefer closer to pure note)
        # Alternatively, could sort by peak prominence (magnitude in histogram)
        # But distance is safer for "identity" of the note.
        candidates.sort(key=lambda x: x["distance_to_note"])
        
        # Select best one
        best = candidates[0]
        
        # Add to final lists
        final_validated_indices.append(best["bin_idx"])
        final_validated_cents.append(best["cent_position"])
        peak_details.append(best)
        
        pitch_classes.add(best["mapped_pc"])
            
    # Overwrite return arrays with filtered ones
    validated_indices = np.array(final_validated_indices, dtype=int) if final_validated_indices else np.array([], dtype=int)
    validated_cents = np.array(final_validated_cents) if final_validated_cents else np.array([])
    
    return PeakData(
        high_res_indices=high_peaks,
        high_res_cents=high_cents,
        low_res_indices=low_peaks,
        low_res_cents=low_cents,
        validated_indices=validated_indices,
        validated_cents=validated_cents,
        pitch_classes=pitch_classes,
        peak_details=peak_details,
    )


def _add_edge_peaks(
    histogram: np.ndarray, peaks: np.ndarray, prominence: float
) -> np.ndarray:
    """Check for peaks at circular boundaries (bins 0 and N-1)."""
    n = len(histogram)
    peaks_list = list(peaks)
    
    # Check bin 0 (wraps to bin N-1)
    if 0 not in peaks:
        left = histogram[-1]
        center = histogram[0]
        right = histogram[1] if n > 1 else histogram[0]
        if center > left and center > right and center >= prominence:
            peaks_list.append(0)
    
    # Check bin N-1 (wraps to bin 0)
    if n - 1 not in peaks:
        left = histogram[-2] if n > 1 else histogram[-1]
        center = histogram[-1]
        right = histogram[0]
        if center > left and center > right and center >= prominence:
            peaks_list.append(n - 1)
    
    return np.array(sorted(peaks_list), dtype=int)


def detect_peaks_from_config(
    histogram: HistogramData, config: PipelineConfig, debug: bool = False
) -> PeakData:
    """Detect peaks using configuration parameters."""
    return detect_peaks(
        histogram,
        tolerance_cents=config.tolerance_cents,
        peak_tolerance_cents=config.peak_tolerance_cents,
        prominence_high_factor=config.prominence_high_factor,
        prominence_low_factor=config.prominence_low_factor,
        debug=debug,
    )


# =============================================================================
# GMM ANALYSIS
# =============================================================================

def fit_gmm_to_peaks(
    histogram: HistogramData,
    peak_indices: np.ndarray,
    window_cents: float = 150.0,
    n_components: int = 1,
    tonic: Optional[int] = None,
) -> List[GMMResult]:
    """
    Fit Gaussian Mixture Models around each detected peak.
    
    Analyzes microtonal variations by fitting GMMs to the histogram
    distribution within a window around each peak.
    
    If tonic is provided, results (cents) are relative to Tonic (Sa = 0).
    
    Args:
        histogram: HistogramData with smoothed histogram
        peak_indices: Indices of detected peaks in high-res histogram
        window_cents: Width of window around each peak (cents)
        n_components: Number of GMM components per peak
        tonic: Optional tonic pitch class (0-11). If set, rotates analysis.
        
    Returns:
        List of GMMResult with fitted parameters
    """
    results = []
    note_centers = np.arange(0, 1200, 100)
    
    tonic_offset = (tonic * 100) if tonic is not None else 0.0
    
    for peak_idx in peak_indices:
        # Original absolute peak position
        abs_peak_cent = histogram.bin_centers_high[peak_idx]
        
        # Shift to relative if tonic present
        if tonic is not None:
             # Relative to tonic, modulo 1200
             peak_cent = (abs_peak_cent - tonic_offset) % 1200
        else:
             peak_cent = abs_peak_cent
        
        # Extract window around peak (using ABSOLUTE positions to find bins)
        window_half = window_cents / 2
        
        # Find bins within window distance of the ABSOLUTE peak
        window_mask = np.zeros(len(histogram.bin_centers_high), dtype=bool)
        unwrapped_bins = []
        
        for i, bc in enumerate(histogram.bin_centers_high):
            # Distance in absolute cents (circular)
            dist = abs((bc - abs_peak_cent + 600) % 1200 - 600)
            if dist <= window_half:
                window_mask[i] = True
                
                # PREPARE SAMPLE VALUE
                # If tonic: convert bc to relative
                val = (bc - tonic_offset) % 1200 if tonic is not None else bc
                
                # UNWRAP val relative to peak_cent (critical for GMM)
                # This makes the distribution contiguous around peak_cent
                diff = (val - peak_cent + 600) % 1200 - 600
                unwrapped_val = peak_cent + diff
                unwrapped_bins.append(unwrapped_val)
        
        window_counts = histogram.smoothed_high[window_mask]
        
        # Filter if not enough data
        if len(unwrapped_bins) < n_components * 2:
            continue
            
        # Create weighted samples for GMM
        samples = []
        for val, count in zip(unwrapped_bins, window_counts):
            if count > 0:
                samples.extend([val] * int(count))
        
        if len(samples) < n_components * 2:
            continue
        
        samples = np.array(samples).reshape(-1, 1)
        
        # Fit GMM
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                random_state=42,
                covariance_type="full",
            )
            gmm.fit(samples)
        except Exception as e:
            print(f"GMM detection failed for peak {peak_idx}: {e}")
            continue
        
        means = np.asarray(gmm.means_).flatten()
        sigmas = np.sqrt(np.asarray(gmm.covariances_).flatten())
        weights = np.asarray(gmm.weights_)
        
        # Find dominant component
        dominant_idx = np.argmax(weights)
        primary_mean = float(means[dominant_idx])
        primary_sigma = float(sigmas[dominant_idx])
        
        # Find nearest note (Sargam or Absolute)
        # Note centers are 0, 100... 1100
        # If tonic used, 0=Sa.
        diffs = np.abs((peak_cent - note_centers + 600) % 1200 - 600)
        nearest_note_idx = int(np.argmin(diffs)) 
        # nearest_note is 0-11. If tonic, 0=Sa.
        nearest_note = nearest_note_idx
        
        # Deviation from note center (relative to nearest note)
        expected_cent = nearest_note * 100
        # Check deviation of PRIMARY MEAN from expected cent
        # The primary_mean is 'unwrapped' around peak_cent.
        # Ensure correct modulo arithmetic for deviation
        deviation = (primary_mean - expected_cent + 600) % 1200 - 600
        
        results.append(GMMResult(
            peak_idx=int(peak_idx),
            peak_cent=float(peak_cent), # Relative if tonic, else absolute
            nearest_note=nearest_note,
            means=means, # Relative if tonic
            sigmas=sigmas,
            weights=weights,
            primary_mean=primary_mean,
            primary_sigma=primary_sigma,
            deviation_from_note=float(deviation),
        ))
    
    return results


def compute_gmm_bias_cents(
    gmm_results: List[GMMResult],
    min_peaks: int = 3,
) -> Optional[float]:
    """
    Compute median deviation (cents) across GMM peaks.
    """
    if len(gmm_results) < min_peaks:
        return None
    deviations = [g.deviation_from_note for g in gmm_results if np.isfinite(g.deviation_from_note)]
    if len(deviations) < min_peaks:
        return None
    return float(np.median(deviations))


def fit_gmm_from_config(
    histogram: HistogramData,
    peak_data: PeakData,
    config: PipelineConfig,
) -> List[GMMResult]:
    """Fit GMM using configuration parameters."""
    return fit_gmm_to_peaks(
        histogram,
        peak_data.validated_indices,
        window_cents=config.gmm_window_cents,
        n_components=config.gmm_components,
    )
