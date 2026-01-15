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
) -> HistogramData:
    """
    Compute dual-resolution cent histograms with Gaussian smoothing.
    
    Args:
        pitch_data: PitchData containing midi_vals
        bins_high: Number of bins for high-resolution histogram (default 100)
        bins_low: Number of bins for low-resolution histogram (default 33)
        sigma: Gaussian smoothing kernel width
        
    Returns:
        HistogramData with raw and smoothed histograms
    """
    # Convert MIDI to cents (0-1200 range, one octave)
    cent_vals = (pitch_data.midi_vals % 12) * 100.0
    
    # High-resolution histogram (12¢ per bin with 100 bins)
    high_res, bin_edges_high = np.histogram(
        cent_vals, bins=bins_high, range=(0.0, 1200.0)
    )
    bin_centers_high = (bin_edges_high[:-1] + bin_edges_high[1:]) / 2.0
    
    # Low-resolution histogram (~36¢ per bin with 33 bins)
    low_res, bin_edges_low = np.histogram(
        cent_vals, bins=bins_low, range=(0.0, 1200.0)
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
) -> PeakData:
    """
    Detect and cross-validate peaks, map to pitch classes.
    
    Args:
        histogram: HistogramData from compute_cent_histograms
        tolerance_cents: Window for mapping peaks to semitone centers
        peak_tolerance_cents: Window for cross-validation between resolutions
        prominence_high_factor: Prominence threshold factor for high-res
        prominence_low_factor: Prominence threshold factor for low-res
        
    Returns:
        PeakData with validated peaks and pitch classes
    """
    # High-resolution peak detection
    prom_high = max(1.0, prominence_high_factor * histogram.smoothed_high.max())
    high_peaks, high_props = find_peaks(
        histogram.smoothed_high, prominence=prom_high, distance=2
    )
    
    # Add circular edge checks (bins 0 and max-1)
    high_peaks = _add_edge_peaks(histogram.smoothed_high, high_peaks, prom_high)
    
    high_cents = histogram.bin_centers_high[high_peaks]
    
    # Low-resolution peak detection (on RAW histogram, per notebook)
    prom_low = max(0.0, prominence_low_factor * histogram.low_res.max())
    low_peaks, low_props = find_peaks(
        histogram.low_res, prominence=prom_low, distance=1
    )
    
    # Add circular edge checks
    low_peaks = _add_edge_peaks(histogram.low_res, low_peaks, prom_low)
    
    low_cents = histogram.bin_centers_low[low_peaks]
    
    # Cross-validate: high-res peaks must align with low-res within tolerance
    validated_indices = []
    validated_cents_list = []
    
    for idx, cent in zip(high_peaks, high_cents):
        if len(low_cents) == 0:
            # No low-res peaks - accept all high-res peaks
            validated_indices.append(idx)
            validated_cents_list.append(cent)
        else:
            # Circular distance to nearest low-res peak
            dists = np.abs((cent - low_cents + 600) % 1200 - 600)
            min_dist = np.min(dists)
            if min_dist <= peak_tolerance_cents:
                validated_indices.append(idx)
                validated_cents_list.append(cent)
    
    validated_indices = np.array(validated_indices, dtype=int)
    validated_cents = np.array(validated_cents_list)
    
    # Map to pitch classes
    note_centers = np.arange(0, 1200, 100)  # 0, 100, 200, ..., 1100
    pitch_classes = set()
    peak_details = []
    
    for idx, cent in zip(validated_indices, validated_cents):
        # Circular distance to nearest semitone center
        diffs = np.abs((cent - note_centers + 600) % 1200 - 600)
        nearest_note = int(np.argmin(diffs))
        note_dist = float(diffs[nearest_note])
        
        if note_dist <= tolerance_cents:
            pitch_classes.add(nearest_note % 12)
        
        peak_details.append({
            "bin_idx": int(idx),
            "cent_position": float(cent),
            "mapped_pc": nearest_note % 12,
            "distance_to_note": note_dist,
            "within_tolerance": note_dist <= tolerance_cents,
        })
    
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
    histogram: HistogramData, config: PipelineConfig
) -> PeakData:
    """Detect peaks using configuration parameters."""
    return detect_peaks(
        histogram,
        tolerance_cents=config.tolerance_cents,
        peak_tolerance_cents=config.peak_tolerance_cents,
        prominence_high_factor=config.prominence_high_factor,
        prominence_low_factor=config.prominence_low_factor,
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
        
        means = gmm.means_.flatten()
        sigmas = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_
        
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
