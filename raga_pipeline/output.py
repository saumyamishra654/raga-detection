"""
Output module: visualization and HTML report generation.

Provides:
- Static plot generation (histograms, GMM, note segments)
- Interactive Plotly pitch contour with audio sync
- HTML report with embedded audio player and synchronized cursor

The HTML report includes JavaScript that synchronizes:
- Audio playhead position with Plotly cursor line
- Clicking on plot seeks audio to that timestamp
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Union, Tuple, TypedDict
import os
import json
import uuid
import io
import base64
from html import escape
from urllib.parse import quote
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from .config import PipelineConfig
from .audio import PitchData
from . import transcription
from .analysis import HistogramData, PeakData, GMMResult
from .raga import _parse_tonic
from .sequence import Note, Phrase, OFFSET_TO_SARGAM


# =============================================================================
# RESULTS CONTAINER
# =============================================================================

def _new_gmm_results() -> List[GMMResult]:
    return []


def _new_notes() -> List[Note]:
    return []


def _new_phrases() -> List[Phrase]:
    return []


def _new_phrase_clusters() -> Dict[int, List[Phrase]]:
    return {}


def _new_plot_paths() -> Dict[str, str]:
    return {}

@dataclass
class AnalysisResults:
    """Container for all analysis results."""
    
    config: PipelineConfig
    
    # Pitch data
    pitch_data_vocals: Optional[PitchData] = None
    pitch_data_stem: Optional[PitchData] = None
    pitch_data_accomp: Optional[PitchData] = None
    pitch_data_composite: Optional[PitchData] = None # New field for composite/full-mix pitch
    
    # Histogram analysis
    histogram_vocals: Optional[HistogramData] = None
    histogram_accomp: Optional[HistogramData] = None
    peaks_vocals: Optional[PeakData] = None
    
    # Raga matching
    candidates: Optional[pd.DataFrame] = None
    detected_tonic: Optional[int] = None
    detected_raga: Optional[str] = None
    
    # GMM analysis
    gmm_results: List[GMMResult] = field(default_factory=_new_gmm_results)
    gmm_bias_cents: Optional[float] = None
    
    # Sequence analysis
    notes: List[Note] = field(default_factory=_new_notes)
    phrases: List[Phrase] = field(default_factory=_new_phrases)
    phrase_clusters: Dict[int, List[Phrase]] = field(default_factory=_new_phrase_clusters)
    transition_matrix: Optional[np.ndarray] = None
    
    # Plot paths
    plot_paths: Dict[str, str] = field(default_factory=_new_plot_paths)


class TrackSpec(TypedDict):
    key: str
    label: str
    audio_id: str
    pitch_data: Optional[PitchData]


# =============================================================================
# STATIC PLOTS
# =============================================================================

def plot_histograms(
    histogram: HistogramData,
    peaks: PeakData,
    output_path: str,
    title: str = "Pitch Class Histogram",
    bias_cents: Optional[float] = None,
) -> str:
    """
    Plot dual-resolution histograms with peak annotations.
    
    Args:
        histogram: HistogramData from analysis
        peaks: PeakData with detected peaks
        output_path: Path to save PNG
        title: Plot title
        
    Returns:
        Path to saved figure
    """
    import matplotlib.ticker as mtick
    
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # === High-resolution (100-bin) with Gaussian smoothing ===
    ax1 = axes[0]
    
    # Normalize for probability mass display
    raw_norm = histogram.high_res_norm / max(histogram.high_res_norm.sum(), 1e-10)
    smoothed_norm = histogram.smoothed_high_norm / max(histogram.smoothed_high_norm.sum(), 1e-10)
    
    # Plot raw (faded) and smoothed (bold) histograms
    bin_centers_high = histogram.bin_centers_high
    raw_plot = raw_norm
    smoothed_plot = smoothed_norm
    if bias_cents is not None:
        rotated = (bin_centers_high - bias_cents) % 1200
        order = np.argsort(rotated)
        bin_centers_high = rotated[order]
        raw_plot = raw_plot[order]
        smoothed_plot = smoothed_plot[order]
    ax1.plot(bin_centers_high, raw_plot, color='lightsteelblue', alpha=0.4, label='Raw (100-bin)')
    ax1.plot(bin_centers_high, smoothed_plot, color='navy', linewidth=2, label='Smoothed (Gaussian)')
    
    # Mark validated peaks with X markers
    for idx in peaks.validated_indices:
        cent = histogram.bin_centers_high[idx]
        if bias_cents is not None:
            cent = (cent - bias_cents) % 1200
        height = smoothed_norm[idx]
        ax1.scatter([cent], [height], color='red', s=110, marker='x', zorder=6)
    
    # Add +/- 35 cent note zones (green shading)
    tolerance_cents = 35
    note_centers = np.arange(0, 1200, 100)
    for i, center in enumerate(note_centers):
        low = (center - tolerance_cents) % 1200
        high = (center + tolerance_cents) % 1200
        if low < high:
            ax1.axvspan(low, high, color='green', alpha=0.12, label='+/-35c Zone' if i == 0 else "")
        else:
            ax1.axvspan(0, high, color='green', alpha=0.12)
            ax1.axvspan(low, 1200, color='green', alpha=0.12)
    
    # Western note labels on X-axis
    ax1.set_xticks(np.arange(0, 1200, 100))
    ax1.set_xticklabels(NOTE_NAMES, fontsize=11)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    
    ax1.set_xlim(0, 1200)
    ax1.set_xlabel('Pitch class (Western notes)', fontsize=12)
    ax1.set_ylabel('Probability mass (%)', fontsize=12)
    ax1.set_title(f'{title} - High Resolution (100-bin, 12c)', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # === Low-resolution (33-bin) ===
    ax2 = axes[1]
    low_norm = histogram.smoothed_low / max(histogram.smoothed_low.max(), 1)
    bin_centers_low = histogram.bin_centers_low
    low_plot = low_norm
    if bias_cents is not None:
        rotated_low = (bin_centers_low - bias_cents) % 1200
        order_low = np.argsort(rotated_low)
        bin_centers_low = rotated_low[order_low]
        low_plot = low_plot[order_low]
    ax2.bar(bin_centers_low, low_plot,
            width=36, alpha=0.7, color='darkorange', label='Stable Peaks (33-bin)')
    
    for idx in peaks.low_res_indices:
        cent = histogram.bin_centers_low[idx]
        if bias_cents is not None:
            cent = (cent - bias_cents) % 1200
        ax2.axvline(cent, color='red', linestyle='--', alpha=0.5)
    
    ax2.set_xticks(np.arange(0, 1200, 100))
    ax2.set_xticklabels(NOTE_NAMES, fontsize=11)
    ax2.set_xlim(0, 1200)
    ax2.set_xlabel('Pitch class (Western notes)', fontsize=12)
    ax2.set_ylabel('Normalized Count')
    ax2.set_title(f'{title} - Low Resolution (33-bin, 36c)', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_absolute_note_histogram(
    note_midi_values: List[float],
    output_path: str,
    title: str = "Stationary Note Histogram (Octave-Wrapped Western Notes)",
    weights: Optional[List[float]] = None,
    ylabel: str = "Stationary event count",
) -> pd.DataFrame:
    """Plot octave-wrapped (12-bin) western-note histogram and return 12-column values."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi_rounded = np.asarray(np.round(note_midi_values), dtype=int)
    weight_values = None if weights is None else np.asarray(weights, dtype=float)

    if midi_rounded.size == 0 or (weight_values is not None and weight_values.size == 0):
        counts = np.zeros(12, dtype=float)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(12)
        ax.bar(x, counts, color='teal', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(note_names)
        ax.set_xlabel('Pitch Class (Octave-wrapped)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return pd.DataFrame([dict(zip(note_names, counts.tolist()))])

    pitch_classes = np.mod(midi_rounded, 12)
    if weight_values is not None and weight_values.shape[0] == pitch_classes.shape[0]:
        counts = np.bincount(pitch_classes, weights=weight_values, minlength=12).astype(float)
    else:
        counts = np.bincount(pitch_classes, minlength=12).astype(float)

    counts_df = pd.DataFrame([dict(zip(note_names, counts.tolist()))])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(12)
    ax.bar(x, counts, color='teal', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(note_names)
    ax.set_xlabel('Pitch Class (Octave-wrapped)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return counts_df


def plot_gmm_overlay(
    histogram: HistogramData,
    gmm_results: List[GMMResult],
    output_path: str,
    tonic: Optional[int] = None,
    bias_cents: Optional[float] = None,
) -> str:
    """
    Plot histogram with GMM fits overlaid.
    
    Args:
        histogram: HistogramData
        gmm_results: List of GMMResult from fit_gmm_to_peaks
        output_path: Path to save PNG
        
    Returns:
        Path to saved figure
    """
    from scipy import stats
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot histogram (optionally tonic-rotated)
    bin_centers = histogram.bin_centers_high
    counts = histogram.smoothed_high_norm
    if tonic is not None:
        offset = (tonic % 12) * 100.0
        bin_centers = (bin_centers - offset) % 1200
        order = np.argsort(bin_centers)
        bin_centers = bin_centers[order]
        counts = counts[order]
    if bias_cents is not None:
        bin_centers = (bin_centers - bias_cents) % 1200
        order = np.argsort(bin_centers)
        bin_centers = bin_centers[order]
        counts = counts[order]
    ax.bar(bin_centers, counts, width=12, alpha=0.6, color='steelblue', label='Histogram')
    
    # Overlay GMM fits
    x = np.linspace(0, 1200, 1200)
    x_plot = x
    if bias_cents is not None:
        x_plot = (x - bias_cents) % 1200
        order_x = np.argsort(x_plot)
        x_plot = x_plot[order_x]
    
    for gmm in gmm_results:
        color = plt.get_cmap("Set2")(gmm.nearest_note / 12)
        
        # Plot Gaussian components scaled as a mixture to the local peak height
        components = []
        for mean, sigma, weight in zip(gmm.means, gmm.sigmas, gmm.weights):
            components.append(weight * stats.norm.pdf(x, mean, sigma))
        mixture = np.sum(components, axis=0) if components else np.zeros_like(x)
        scale = histogram.smoothed_high_norm[gmm.peak_idx] / max(mixture.max(), 1e-10)
        for comp in components:
            if bias_cents is not None:
                ax.plot(x_plot, (comp * scale)[order_x], color=color, linewidth=1.5, alpha=0.8)
            else:
                ax.plot(x, comp * scale, color=color, linewidth=1.5, alpha=0.8)
        
        # Annotate with Western notation
        western_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        western = western_notes[gmm.nearest_note]
        peak_x = gmm.peak_cent
        if bias_cents is not None:
            peak_x = (peak_x - bias_cents) % 1200
        ax.annotate(
            f'{western}\nμ={gmm.primary_mean:.0f}¢\nσ={gmm.primary_sigma:.1f}¢',
            (peak_x, histogram.smoothed_high_norm[gmm.peak_idx]),
            textcoords='offset points', xytext=(0, 20),
            ha='center', fontsize=8,
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5)
        )
    
    ax.set_xlim(0, 1200)
    ax.set_xlabel('Cents')
    ax.set_ylabel('Normalized Count')
    ax.set_title('Histogram with GMM Fits (Microtonal Analysis)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_note_segments(
    pitch_data: PitchData,
    notes: List[Note],
    output_path: str,
    tonic: Optional[int] = None,
) -> str:
    """
    Plot detected note segments over pitch contour.
    
    Args:
        pitch_data: PitchData with pitch contour
        notes: List of detected notes
        output_path: Path to save PNG
        tonic: Tonic for sargam labels (optional)
        
    Returns:
        Path to saved figure
    """
    import librosa
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot pitch contour
    voiced_mask = pitch_data.voiced_mask
    if np.any(voiced_mask):
        times = pitch_data.timestamps[voiced_mask]
        midi = librosa.hz_to_midi(pitch_data.pitch_hz[voiced_mask])
        ax.scatter(times, midi, c='lightblue', s=1, alpha=0.5, label='Pitch contour')
    
    # Plot note segments
    colors = plt.get_cmap("Set3")(np.linspace(0, 1, 12))
    
    for note in notes:
        pc = int(round(note.pitch_midi)) % 12
        color = colors[pc]
        
        # Draw segment
        ax.hlines(note.pitch_midi, note.start, note.end, 
                  colors=color, linewidth=3, alpha=0.8)
        
        # Add label
        if tonic is not None:
            from .sequence import midi_to_sargam
            label = midi_to_sargam(note.pitch_midi, tonic)
        else:
            label = f'{note.pitch_midi:.0f}'
        
        ax.annotate(label, (note.start, note.pitch_midi),
                   textcoords='offset points', xytext=(2, 3),
                   fontsize=7, alpha=0.8)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MIDI Note')
    ax.set_title(f'Detected Notes ({len(notes)} notes)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


# =============================================================================
# INTERACTIVE PLOTLY VISUALIZATION
# =============================================================================

def create_pitch_contour_plotly(
    pitch_data: PitchData,
    tonic: int,
    raga_notes: Optional[Set[int]] = None,
    audio_duration: float = 0.0,
    show_rms_overlay: bool = False,
) -> str:
    """
    Generate Plotly figure HTML for pitch contour with sargam lines.
    
    Args:
        pitch_data: PitchData with pitch contour
        tonic: Tonic pitch class (0-11)
        raga_notes: Valid pitch classes for raga (optional)
        audio_duration: Total audio duration for x-axis
        
    Returns:
        JSON string for Plotly figure data
    """
    import librosa
    
    voiced_mask = pitch_data.voiced_mask
    
    if not np.any(voiced_mask):
        return json.dumps({"data": [], "layout": {}})
    
    voiced_times = pitch_data.timestamps[voiced_mask].tolist()
    voiced_midi = librosa.hz_to_midi(pitch_data.pitch_hz[voiced_mask]).tolist()
    
    # Determine pitch range
    if isinstance(tonic, str):
         # If tonic passed as string name? shouldn't be, but safer
         # We need int 0-11
         pass # Assume it's an int for now or handled upstream but let's cast
    
    tonic = int(tonic) if tonic is not None else 0
    tonic_midi_base = tonic + 48  # Octave 4
    median_midi = float(np.median(voiced_midi))
    median_octave = int((median_midi - tonic_midi_base) // 12)
    center_sa = tonic_midi_base + median_octave * 12
    min_midi = center_sa - 12
    max_midi = center_sa + 24
    
    total_duration = max(audio_duration, voiced_times[-1] if voiced_times else 1.0)
    
    # Build traces
    traces = []
    
    # Main pitch contour
    # Extract energy for voiced frames
    voiced_energy = []
    if len(pitch_data.energy) > 0:
        # Check alignment
        if len(pitch_data.energy) == len(pitch_data.timestamps):
            e_vals = pitch_data.energy[voiced_mask]
            # Sanitize: Replace nan/inf with 0.0
            e_vals = np.nan_to_num(e_vals, nan=0.0, posinf=1.0, neginf=0.0)
            voiced_energy = e_vals.tolist()
        else:
             # Fallback if length mismatch
             voiced_energy = [1.0] * len(voiced_midi)
    else:
        voiced_energy = [1.0] * len(voiced_midi)

    # Sanity check length
    if len(voiced_energy) != len(voiced_midi):
         voiced_energy = [1.0] * len(voiced_midi)

    # Main pitch contour
    traces.append({
        "x": voiced_times,
        "y": voiced_midi,
        "mode": "lines",
        "line": {"color": "royalblue", "width": 1.5},
        "name": "Pitch contour",
        "type": "scattergl",
        "customdata": voiced_energy,
        "hovertemplate": "Time: %{x:.2f}s<br>Pitch: %{y:.1f}<br>Energy: %{customdata:.2f}<extra></extra>",
    })

    # RMS energy overlay on a secondary y-axis (scaled into the MIDI range)
    if show_rms_overlay and len(pitch_data.energy) > 0:
        all_times = pitch_data.timestamps.tolist()
        raw_energy = np.nan_to_num(pitch_data.energy, nan=0.0, posinf=1.0, neginf=0.0)
        # Scale energy [0,1] into the lower portion of the MIDI range so it
        # doesn't obscure the pitch contour (bottom 30 % of visible range).
        midi_span = max_midi - min_midi
        scaled_energy = (raw_energy * 0.30 * midi_span + min_midi).tolist()
        traces.append({
            "x": all_times,
            "y": scaled_energy,
            "mode": "lines",
            "line": {"color": "rgba(0, 206, 209, 0.35)", "width": 1},
            "fill": "tozeroy",
            "fillcolor": "rgba(0, 206, 209, 0.08)",
            "name": "RMS Energy",
            "type": "scattergl",
            "yaxis": "y",
            "hovertemplate": "Time: %{x:.2f}s<br>Energy: %{customdata:.3f}<extra></extra>",
            "customdata": raw_energy.tolist(),
        })

    # Build shapes (sargam lines)
    shapes = []
    annotations = []
    
    OCTAVE_COLORS = {-2: "#8B4513", -1: "#FF6B6B", 0: "#2ECC71", 1: "#3498DB", 2: "#9B59B6"}
    
    for midi_val in range(int(min_midi), int(max_midi) + 1):
        pc = midi_val % 12
        offset = (pc - tonic) % 12
        
        if raga_notes is None or pc in raga_notes:
            label = OFFSET_TO_SARGAM.get(offset, "")
            octave_diff = int((midi_val - tonic_midi_base) // 12)
            
            if octave_diff < 0:
                label += "'" * abs(octave_diff)
            elif octave_diff > 0:
                label += "·" * octave_diff
            
            line_color = OCTAVE_COLORS.get(octave_diff, "gray")
            opacity = 0.3 if (raga_notes is not None and pc in raga_notes) else 0.12
            
            shapes.append({
                "type": "line",
                "x0": 0,
                "x1": total_duration,
                "y0": midi_val,
                "y1": midi_val,
                "line": {"color": line_color, "width": 0.7, "dash": "dash"},
                "opacity": opacity,
            })
            
            annotations.append({
                "x": total_duration,
                "y": midi_val,
                "text": label,
                "xanchor": "left",
                "yanchor": "middle",
                "font": {"size": 10, "color": line_color},
                "showarrow": False,
            })
    
    # Add cursor line (will be updated by JavaScript)
    shapes.append({
        "type": "line",
        "x0": 0,
        "x1": 0,
        "y0": 0,
        "y1": 1,
        "xref": "x",
        "yref": "paper",
        "line": {"color": "crimson", "width": 2},
    })
    
    layout = {
        "width": 900,
        "height": 420,
        "margin": {"l": 60, "r": 120, "t": 40, "b": 60},
        "xaxis": {"title": "Time (s)", "range": [0, total_duration]},
        "yaxis": {"title": "Pitch (MIDI)", "range": [min_midi - 2, max_midi + 2]},
        "showlegend": show_rms_overlay,
        "legend": {"x": 0, "y": 1, "bgcolor": "rgba(13,17,23,0.7)", "font": {"color": "#c9d1d9", "size": 11}},
        "shapes": shapes,
        "annotations": annotations,
    }
    
    return json.dumps({"data": traces, "layout": layout})


def plot_energy_distribution(
    energy_vals: np.ndarray,
    output_path: str,
    title: str = "Energy Distribution (RMS)",
    threshold: Optional[float] = None
):
    """
    Plot histogram of energy values.
    
    Args:
        energy_vals: Array of normalized energy values (0-1)
        output_path: Path to save plot
        title: Plot title
        threshold: Optional threshold to mark with a vertical line
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    sns.histplot(energy_vals, bins=50, kde=True, color='skyblue', edgecolor='black')
    
    if threshold is not None and threshold > 0:
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
        plt.legend()
        
    plt.title(title)
    plt.xlabel("Energy (Normalized RMS)")
    plt.ylabel("Frame Count")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def plot_energy_over_time(
    timestamps: np.ndarray,
    energy_vals: np.ndarray,
    output_path: str,
    title: str = "Energy Over Time (RMS)",
):
    """
    Plot energy values over time.
    """
    if len(timestamps) == 0 or len(energy_vals) == 0:
        return

    length = min(len(timestamps), len(energy_vals))
    times = np.asarray(timestamps[:length])
    energy = np.asarray(energy_vals[:length])
    energy = np.nan_to_num(energy, nan=0.0, posinf=1.0, neginf=0.0)

    plt.figure(figsize=(12, 4))
    plt.plot(times, energy, color='teal', linewidth=1.0, alpha=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (Normalized RMS)")
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_pitch_with_sargam_lines(
    time_axis: np.ndarray,
    pitch_values: np.ndarray,
    tonic: Union[int, str],
    raga_name: str,
    output_path: str,
    figsize: Tuple[int, int] = (15, 6),
    phrase_ranges: Optional[List[Tuple[float, float]]] = None,
):
    """
    Plot pitch contour with sargam lines.
    Slightly adapted from user snippet to save to file.
    """
    plt.figure(figsize=figsize)

    # Phrase regions: translucent yellow blocks with clear boundaries/labels.
    if phrase_ranges:
        y_top_for_labels = float(np.nanmax(pitch_values)) + 0.4
        for idx, (start_t, end_t) in enumerate(phrase_ranges):
            if not np.isfinite(start_t) or not np.isfinite(end_t) or end_t <= start_t:
                continue
            plt.axvspan(
                float(start_t),
                float(end_t),
                facecolor="#ffeb3b",
                edgecolor="none",
                linewidth=0.0,
                alpha=0.28,
                zorder=0,
            )
            plt.axvline(float(start_t), color="#000000", linestyle="-", linewidth=1.0, alpha=0.8, zorder=1)
            plt.axvline(float(end_t), color="#000000", linestyle="-", linewidth=1.0, alpha=0.8, zorder=1)
            plt.text(
                (float(start_t) + float(end_t)) / 2.0,
                y_top_for_labels,
                f"P{idx + 1}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#8a6d00",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#fff3a3", edgecolor="#eab308", alpha=0.9),
                zorder=3,
            )

    # Plot pitch
    plt.plot(time_axis, pitch_values, label='Pitch', color='blue', alpha=0.6, linewidth=1, zorder=2)
    
    # Clean tonic
    tonic_val = _parse_tonic(tonic) if isinstance(tonic, str) else int(tonic)
    
    # Sargam lines
    start_midi = int(np.nanmin(pitch_values)) - 1
    end_midi = int(np.nanmax(pitch_values)) + 1
    
    for midi_note in range(start_midi, end_midi + 1):
        offset = (midi_note - tonic_val) % 12
        octave = (midi_note - tonic_val) // 12 # Relative octave
        
        # Color coding for octaves
        if octave == 0:
            color = 'green' # Middle
            linestyle = '-'
            alpha = 0.5
        elif octave > 0:
            color = 'orange' # High
            linestyle = '--'
            alpha = 0.4
        else:
            color = 'purple' # Low
            linestyle = ':'
            alpha = 0.4
            
        sargam_label = OFFSET_TO_SARGAM.get(offset, "")
        if sargam_label:
            # Full label with octave indicator
            full_label = sargam_label
            if octave > 0: full_label += "·" * octave
            elif octave < 0: full_label += "'" * abs(octave)
            
            plt.axhline(y=midi_note, color=color, linestyle=linestyle, alpha=alpha, linewidth=0.8)
            plt.text(time_axis[-1], midi_note, f" {full_label}", va='center', fontsize=8, color=color)
            
    plt.xlabel("Time (s)")
    plt.ylabel("MIDI Pitch")
    plt.title(f"Pitch Contour with Sargam Lines (Tonic: {_tonic_name(tonic_val)}, Raga: {raga_name})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_transition_heatmap_v2(
    matrix: np.ndarray, 
    labels: List[str], 
    output_path: str
):
    """
    Plot enhanced transition matrix heatmap.
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize for coloring when raw counts arrive; otherwise display the supplied probabilities.
    
    plt.imshow(matrix, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='Transition Probability')
    
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    
    # Annotate
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            if val > 0.01: # Threshold for clutter
                text_color = "white" if val > 0.5 else "black"
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)
                
    plt.title("Note Transition Probability Matrix")
    plt.xlabel("To Note")
    plt.ylabel("From Note")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    
# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_detection_report(
    results: AnalysisResults,
    output_path: str,
) -> str:
    """
    Generate summary HTML report for Phase 1 (Detection).
    
    Includes:
    - Audio metadata
    - Histogram plots
    - Peak detection results
    - Raga candidate ranking
    
    Args:
        results: AnalysisResults container
        output_path: Path to save HTML file
        
    Returns:
        Path to generated report
    """
    sections = []
    
    # Audio Players Section (all 3 tracks)
    sections.append(_generate_audio_players_section(results))
    
    # Metadata section
    sections.append(_generate_metadata_section(results))
    
    # Histogram section (vocals)
    if 'histogram_vocals' in results.plot_paths:
        sections.append(f'''
        <section id="histogram-analysis">
            <h2>Vocal Pitch Distribution</h2>
            <img src="{os.path.basename(results.plot_paths['histogram_vocals'])}" 
                 alt="Vocal pitch histogram" style="max-width: 100%;">
        </section>
        ''')

        # Stationary-note histogram section (duration-weighted octave-wrapped western notes)
    if 'stationary_note_histogram' in results.plot_paths:
        sections.append(f'''
        <section id="stationary-note-histogram">
                            <h2>Stationary Note Distribution (Duration-Weighted, Octave-Wrapped)</h2>
            <img src="{os.path.basename(results.plot_paths['stationary_note_histogram'])}"
                                 alt="Duration-weighted stationary octave-wrapped note histogram" style="max-width: 100%;">
        </section>
        ''')
    
    # Histogram section (accompaniment)
    if 'histogram_accomp' in results.plot_paths:
        sections.append(f'''
        <section id="histogram-accomp">
            <h2>Accompaniment Pitch Distribution</h2>
            <img src="{os.path.basename(results.plot_paths['histogram_accomp'])}" 
                 alt="Accompaniment pitch histogram" style="max-width: 100%;">
        </section>
        ''')
    
    # Peak detection section
    if results.peaks_vocals is not None:
        sections.append(_generate_peak_section(results.peaks_vocals))
    
    # Raga ranking section
    if results.candidates is not None and len(results.candidates) > 0:
        sections.append(_generate_ranking_section(results.candidates))
        
    # Assemble HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raga Detection Summary - {results.config.filename}</title>
    <style>
        {_get_css_styles()}
    </style>
</head>
<body>
    <header>
        <h1>Raga Detection Summary</h1>
        <p class="subtitle">{results.config.filename}</p>
    </header>
    <main>
        {"".join(sections)}
    </main>
    <footer>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><em>Run 'analyze' phase with selected tonic/raga for full details.</em></p>
    </footer>
    <script>
        {_get_audio_sync_js()}
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path

def generate_html_report(
    results: AnalysisResults,
    output_path: str,
) -> str:
    """
    Generate comprehensive HTML report with embedded visualizations.
    
    The report includes:
    - Interactive audio player with synchronized pitch contour
    - Histogram and peak detection plots
    - Raga candidate ranking
    - Musical transcription
    - Transition matrix heatmap
    
    Args:
        results: AnalysisResults container
        output_path: Path to save HTML file
        
    Returns:
        Path to generated report
    """
    
    # Generate unique IDs for audio sync
    vocals_id = f"vocals-{uuid.uuid4().hex[:8]}"
    accomp_id = f"accomp-{uuid.uuid4().hex[:8]}"
    
    analysis_uses_composite = (
        getattr(results.config, "melody_source", "separated") == "composite"
        and results.pitch_data_composite is not None
    )
    analysis_audio_path = _require_audio_path(results.config) if analysis_uses_composite else results.config.vocals_path
    analysis_track_label = "Composite Analysis" if analysis_uses_composite else "Vocals Analysis"

    # Get audio duration
    try:
        import librosa
        audio_duration = librosa.get_duration(path=analysis_audio_path)
    except Exception:
        audio_duration = 0.0
    
    # Build HTML sections
    sections = []
    
    # Metadata section
    sections.append(_generate_metadata_section(results))
    
    # Vocals player section
    if results.pitch_data_vocals is not None:
        tonic = results.detected_tonic or 0
        raga_notes = None
        
        pitch_json = create_pitch_contour_plotly(
            results.pitch_data_vocals,
            tonic,
            raga_notes,
            audio_duration,
            show_rms_overlay=getattr(results.config, 'show_rms_overlay', True),
        )
        
        sections.append(_generate_audio_player_section(
            analysis_audio_path,
            pitch_json,
            vocals_id,
            analysis_track_label,
            tonic,
            transcription_json=_serialize_notes(results.notes, tonic) if results.notes else "[]"
        ))
    
    # Accompaniment player section
    if results.pitch_data_accomp is not None:
        tonic = results.detected_tonic or 0
        
        pitch_json = create_pitch_contour_plotly(
            results.pitch_data_accomp,
            tonic,
            None,
            audio_duration,
        )
        
        sections.append(_generate_audio_player_section(
            results.config.accompaniment_path,
            pitch_json,
            accomp_id,
            "Accompaniment Analysis",
            tonic,
        ))
    
    # Histogram section
    if 'histogram_vocals' in results.plot_paths:
        sections.append(f'''
        <section id="histogram-analysis">
            <h2>Pitch Distribution</h2>
            <img src="{os.path.basename(results.plot_paths['histogram_vocals'])}" 
                 alt="Pitch histogram" style="max-width: 100%;">
        </section>
        ''')
    
    # Peak detection section
    if results.peaks_vocals is not None:
        sections.append(_generate_peak_section(results.peaks_vocals))
    
    # Raga ranking section
    if results.candidates is not None and len(results.candidates) > 0:
        sections.append(_generate_ranking_section(results.candidates))
    
    # Transcription section
    if results.notes:
        tonic = results.detected_tonic or 0
        sections.append(_generate_transcription_section(results.notes, results.phrases, tonic))
    
    # Transition matrix section
    if results.transition_matrix is not None:
        sections.append(_generate_transition_section(results.transition_matrix))
    
    # GMM section
    if 'gmm_overlay' in results.plot_paths:
        sections.append(f'''
        <section id="gmm-analysis">
            <h2>Microtonal Analysis (GMM)</h2>
            <img src="{os.path.basename(results.plot_paths['gmm_overlay'])}" 
                 alt="GMM analysis" style="max-width: 100%;">
        </section>
        ''')
    
    # Assemble full HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raga Detection Report - {results.config.filename}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        {_get_css_styles()}
    </style>
</head>
<body>
    <header>
        <h1>Raga Detection Report</h1>
        <p class="subtitle">{results.config.filename}</p>
    </header>
    <main>
        {"".join(sections)}
    </main>
    <footer>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </footer>
    <script>
        {_get_audio_sync_js()}
    </script>
</body>
</html>
'''
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path


def _generate_metadata_section(results: AnalysisResults) -> str:
    """Generate metadata section HTML."""
    config = results.config
    
    return f'''
    <section id="metadata">
        <h2>Analysis Metadata</h2>
        <table class="metadata-table">
            <tr><td>Audio File</td><td>{config.filename}</td></tr>
            <tr><td>Separator</td><td>{config.separator_engine} ({config.demucs_model})</td></tr>
            <tr><td>Vocal Confidence</td><td>{config.vocal_confidence}</td></tr>
            <tr><td>Histogram Bins</td><td>{config.histogram_bins_high} (high-res), {config.histogram_bins_low} (low-res)</td></tr>
            <tr><td>Detected Tonic</td><td>{_tonic_name(results.detected_tonic) if results.detected_tonic else 'N/A'}</td></tr>
            <tr><td>Top Raga</td><td>{results.detected_raga or 'N/A'}</td></tr>
        </table>
    </section>
    '''


def _generate_audio_player_section(
    audio_path: str,
    pitch_json: str,
    player_id: str,
    title: str,
    tonic: int,
    transcription_json: str = "[]",
) -> str:
    """Generate audio player with synchronized pitch contour."""
    
    audio_filename = os.path.basename(audio_path)
    source_tags = _build_audio_source_tags(audio_filename)
    
    return f'''
    <section id="{player_id}-section" class="audio-player-section">
        <h2>{title}</h2>
        <div class="audio-player-container">
            <audio id="{player_id}-audio" controls>
                {source_tags}
            </audio>
            
            <div class="controls-container" style="margin-bottom: 10px; padding: 10px; background: #161b22; border-radius: 4px; display: flex; gap: 10px; align-items: center;">
                <label for="{player_id}-energy-slider" style="font-size: 12px; color: #8b949e;">Energy Filter:</label>
                <input type="range" id="{player_id}-energy-slider" min="0" max="100" value="0" style="flex-grow: 1;">
                <span id="{player_id}-energy-val" style="font-family: monospace; font-size: 12px; min-width: 40px; text-align: right;">0%</span>
            </div>

            <div id="{player_id}-plot" class="pitch-plot"></div>
            
            <!-- Karaoke Overlay -->
            <div id="{player_id}-karaoke" class="karaoke-container">
                <!-- Spans will be injected here by JS -->
            </div>

            <div class="time-display">
                <span id="{player_id}-time">0.00s</span> | 
                <span id="{player_id}-sargam">--</span>
            </div>
        </div>
        <script>
            (function() {{
                var plotData = {pitch_json};
                var transcriptionData = {transcription_json};
                var plotDiv = document.getElementById('{player_id}-plot');
                Plotly.newPlot(plotDiv, plotData.data, plotData.layout);
                
                // Setup audio sync
                setupAudioSync('{player_id}', {tonic}, transcriptionData);
            }})();
        </script>
    </section>
    '''


def _generate_peak_section(peaks: PeakData) -> str:
    """Generate peak detection summary."""
    
    # Western note names map
    PC_TO_WESTERN = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 
                     6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
    
    rows = []
    for detail in peaks.peak_details:
        status = "✓" if detail['within_tolerance'] else "✗"
        pc = detail['mapped_pc']
        western = PC_TO_WESTERN.get(pc, '?')
        rows.append(f'''
            <tr>
                <td>{detail['cent_position']:.1f}¢</td>
                <td>{western} ({pc})</td>
                <td>{detail['distance_to_note']:.1f}¢</td>
                <td>{status}</td>
            </tr>
        ''')
    
    pc_list = ', '.join(PC_TO_WESTERN.get(pc, '?') for pc in sorted(peaks.pitch_classes))
    
    return f'''
    <section id="peak-detection">
        <h2>Peak Detection</h2>
        <p><strong>Detected Pitch Classes:</strong> {pc_list}</p>
        <table class="data-table">
            <thead>
                <tr><th>Cent Position</th><th>Mapped Note</th><th>Distance</th><th>Valid</th></tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </section>
    '''


def _generate_ranking_section(candidates: pd.DataFrame) -> str:
    """Generate raga ranking table with all scoring parameters."""
    
    # Generate rows for top 40
    rows = []
    for idx, (_, row) in enumerate(candidates.head(40).iterrows(), start=1):
        rank = int(row["rank"]) if "rank" in row else idx
        hidden_class = ' class="hidden-row"' if rank > 20 else ''
        rows.append(f'''
            <tr{hidden_class}>
                <td>{rank}</td>
                <td class="raga-name">{row['raga']}</td>
                <td>{row.get('tonic_name', row['tonic'])}</td>
                <td class="score">{row['fit_score']:.2f}</td>
                <td>{row['match_mass']:.3f}</td>
                <td>{row.get('extra_mass', 0):.3f}</td>
                <td>{row.get('observed_note_score', 0):.3f}</td>
                <td>{row.get('loglike_norm', 0):.3f}</td>
                <td>{row.get('primary_score', 0):.3f}</td>
                <td>{row.get('salience', 0)}</td>
                <td>{row.get('raga_size', 0)}</td>
                <td>{row.get('match_diff', 0)}</td>
                <td>{row.get('complexity_pen', 0):.3f}</td>
            </tr>
        ''')
    
    return f'''
    <section id="raga-ranking">
        <h2>Raga Candidates</h2>
        <p>Showing top 40 candidates.</p>
        <div class="table-scroll-container">
            <table class="data-table ranking-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Raga</th>
                        <th>Tonic</th>
                        <th>Score</th>
                        <th>Match</th>
                        <th>Extra</th>
                        <th>Presence</th>
                        <th>LogLike</th>
                        <th>Primary</th>
                        <th>Salience</th>
                        <th>Size</th>
                        <th>Δ Size</th>
                        <th>Complex</th>
                    </tr>
                </thead>
                <tbody id="ranking-tbody">
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        <button id="show-more-btn" onclick="toggleMoreRows()">Show More (21-40)</button>
    </section>
    '''


def _generate_audio_players_section(results: AnalysisResults) -> str:
    """Generate audio players section with all 3 tracks."""
    config = results.config
    
    # Get relative paths from stem_dir
    vocals_rel = os.path.basename(config.vocals_path)
    accomp_rel = os.path.basename(config.accompaniment_path)
    original_audio_path = _require_audio_path(config)
    original_rel = os.path.relpath(original_audio_path, config.stem_dir)
    original_local = os.path.basename(original_audio_path)
    original_sources = _build_audio_source_tags(original_local, original_rel)
    vocals_sources = _build_audio_source_tags(vocals_rel, vocals_rel)
    accomp_sources = _build_audio_source_tags(accomp_rel, accomp_rel)
    original_id = "detection-original-player"
    vocals_id = "detection-vocals-player"
    accomp_id = "detection-accomp-player"
    
    return f'''
    <section id="audio-tracks">
        <h2>Audio Tracks</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
            <div>
                <p><strong>Original</strong></p>
                <audio id="{original_id}" controls style="width:100%">{original_sources}</audio>
            </div>
            <div>
                <p><strong>Vocals</strong></p>
                <audio id="{vocals_id}" controls style="width:100%">{vocals_sources}</audio>
            </div>
            <div>
                <p><strong>Accompaniment</strong></p>
                <audio id="{accomp_id}" controls style="width:100%">{accomp_sources}</audio>
            </div>
        </div>
        <script>
            (function() {{
                var audioIds = {json.dumps([original_id, vocals_id, accomp_id])};
                audioIds.forEach(function(activeId) {{
                    var activeEl = document.getElementById(activeId);
                    if (!activeEl) return;
                    activeEl.addEventListener("play", function() {{
                        audioIds.forEach(function(otherId) {{
                            if (otherId === activeId) return;
                            var otherEl = document.getElementById(otherId);
                            if (otherEl && !otherEl.paused) {{
                                otherEl.pause();
                            }}
                        }});
                    }});
                }});
            }})();
        </script>
    </section>
    '''


def _build_audio_source_tags(primary_path: str, secondary_path: Optional[str] = None) -> str:
    """Build one or two <source> tags so browsers can fall back across locations."""
    paths: List[str] = [primary_path]
    if secondary_path and secondary_path != primary_path:
        paths.append(secondary_path)

    source_tags: List[str] = []
    for path in paths:
        # Encode URL path safely so spaces and special characters in filenames
        # (for example "My Song.mp3") load correctly in browser audio sources.
        encoded_path = quote(path.replace("\\", "/"), safe="/")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".wav":
            mime_type = "audio/wav"
        elif ext == ".mp3":
            mime_type = "audio/mpeg"
        elif ext == ".flac":
            mime_type = "audio/flac"
        elif ext == ".m4a":
            mime_type = "audio/mp4"
        elif ext == ".mp4":
            mime_type = "audio/mp4"
        elif ext == ".aac":
            mime_type = "audio/aac"
        else:
            mime_type = "audio/*"
        source_tags.append(f'<source src="{encoded_path}" type="{mime_type}">')

    return "".join(source_tags)


def _require_audio_path(config: PipelineConfig) -> str:
    """Return a concrete audio path for report-generation paths that require local audio."""
    if not config.audio_path:
        raise ValueError("Audio path is required for report generation but is missing.")
    return config.audio_path


def _strip_octave_markers(label: str) -> str:
    """Remove octave glyphs from a sargam label."""
    return label.replace("'", "").replace("·", "").strip()


def _estimate_median_sa_octave(notes: List[Note], tonic: int) -> int:
    """
    Estimate recording-specific Sa octave anchor from note data.

    Preference order:
    1) Median octave of notes whose pitch class matches Sa.
    2) Median octave of all notes.
    3) Fallback to octave index 5 (near middle register for tonic pitch classes).
    """
    if not notes:
        return 5

    tonic_pc = int(round(tonic)) % 12
    sa_octaves: List[int] = []
    all_octaves: List[int] = []

    for note in notes:
        midi_rounded = int(round(note.pitch_midi))
        octave_index = int(round((midi_rounded - tonic_pc) / 12.0))
        all_octaves.append(octave_index)
        if midi_rounded % 12 == tonic_pc:
            sa_octaves.append(octave_index)

    source = sa_octaves if sa_octaves else all_octaves
    if not source:
        return 5
    return int(round(float(np.median(source))))


def _format_note_sargam_for_report(
    note: Note,
    tonic: int,
    median_sa_octave: int,
    octave_marker_threshold: int = 3,
) -> str:
    """
    Format note sargam for report display.

    Octave markers are suppressed near the singer's median Sa and only shown for
    large deviations (>= `octave_marker_threshold` octaves from that anchor).
    """
    base = _strip_octave_markers(note.sargam or "")
    if not base:
        base = OFFSET_TO_SARGAM.get(int(round(note.pitch_midi - tonic)) % 12, "?")

    tonic_pc = int(round(tonic)) % 12
    midi_rounded = int(round(note.pitch_midi))
    octave_index = int(round((midi_rounded - tonic_pc) / 12.0))
    relative_octave = octave_index - median_sa_octave

    # User-facing report policy: suppress octave markers unless the note is
    # extremely low relative to the recording's median Sa anchor.
    if relative_octave <= -octave_marker_threshold:
        return base + ("'" * abs(relative_octave))
    return base


def _generate_transcription_section(
    notes: List[Note],
    phrases: List[Phrase],
    tonic: int,
    audio_element_ids: Optional[List[str]] = None,
) -> str:
    """Generate phrase-by-phrase musical transcription section."""
    section_id = f"transcription-{uuid.uuid4().hex[:8]}"
    visible_limit = 10
    median_sa_octave = _estimate_median_sa_octave(notes, tonic)
    phrase_rows: List[str] = []
    for i, phrase in enumerate(phrases):
        note_spans: List[str] = []
        for note_idx, n in enumerate(phrase.notes):
            label = _format_note_sargam_for_report(
                n,
                tonic=tonic,
                median_sa_octave=median_sa_octave,
            )
            note_spans.append(
                (
                    f'<span class="transcription-note" data-start="{n.start:.3f}" '
                    f'data-end="{n.end:.3f}" data-phrase-index="{i}" data-note-index="{note_idx}">'
                    f"{escape(label)}</span>"
                )
            )
        phrase_text = " ".join(note_spans).strip() if note_spans else "<span class=\"transcription-note transcription-note-empty\">--</span>"
        row_style = "display: none;" if i >= visible_limit else ""
        phrase_rows.append(
            f'''
            <div class="transcription-phrase-row {'transcription-phrase-hidden' if i >= visible_limit else ''}" style="padding: 10px 12px; border: 1px solid #30363d; border-radius: 8px; background: #0d1117; {row_style}">
                <div style="font-size: 12px; color: #8b949e; margin-bottom: 4px;">Phrase {i+1} · {phrase.start:.2f}s to {phrase.end:.2f}s</div>
                <div class="transcription-phrase-notes" style="font-size: 16px; line-height: 1.8; color: #c9d1d9; word-break: break-word;">{phrase_text}</div>
            </div>
            '''
        )

    phrases_html = "".join(phrase_rows) if phrase_rows else "<p>No phrase transcription available.</p>"
    hidden_count = max(0, len(phrases) - visible_limit)
    show_more_html = ""
    if hidden_count > 0:
        show_more_html = f'''
        <button type="button" id="{section_id}-toggle" style="margin-top: 10px; padding: 8px 14px; border-radius: 6px; border: 1px solid #30363d; background: #21262d; color: #c9d1d9; cursor: pointer;">
            Show more ({hidden_count} more phrases)
        </button>
        '''

    sync_script_html = ""
    if phrases and audio_element_ids:
        sync_script_html = f"""
        <script>
        (function() {{
            var root = document.getElementById("{section_id}");
            if (!root) return;
            var audioIds = {json.dumps(audio_element_ids)};
            if (!audioIds || !audioIds.length) return;

            var noteEls = Array.from(root.querySelectorAll(".transcription-note[data-start][data-end]"));
            if (!noteEls.length) return;

            var notes = noteEls
                .map(function(el) {{
                    return {{
                        el: el,
                        start: parseFloat(el.dataset.start || "0"),
                        end: parseFloat(el.dataset.end || "0")
                    }};
                }})
                .filter(function(n) {{
                    return Number.isFinite(n.start) && Number.isFinite(n.end) && n.end >= n.start;
                }})
                .sort(function(a, b) {{
                    return (a.start - b.start) || (a.end - b.end);
                }});

            if (!notes.length) return;

            var activeAudio = null;
            var lastActiveIdx = -1;

            function findActiveNoteIndex(t) {{
                var lo = 0;
                var hi = notes.length - 1;
                var best = -1;
                while (lo <= hi) {{
                    var mid = (lo + hi) >> 1;
                    if (notes[mid].start <= t) {{
                        best = mid;
                        lo = mid + 1;
                    }} else {{
                        hi = mid - 1;
                    }}
                }}
                if (best < 0) return -1;
                var tolerance = 0.04;
                if (t <= notes[best].end + tolerance) return best;
                return -1;
            }}

            function setActiveByIndex(idx) {{
                if (idx === lastActiveIdx) return;
                if (lastActiveIdx >= 0 && lastActiveIdx < notes.length) {{
                    notes[lastActiveIdx].el.classList.remove("active");
                }}
                if (idx >= 0 && idx < notes.length) {{
                    notes[idx].el.classList.add("active");
                }}
                lastActiveIdx = idx;
            }}

            function updateAtTime(t) {{
                var idx = findActiveNoteIndex(t);
                setActiveByIndex(idx);
            }}

            function getPlayingAudio() {{
                for (var i = 0; i < audioIds.length; i += 1) {{
                    var el = document.getElementById(audioIds[i]);
                    if (el && !el.paused && !el.ended) {{
                        return el;
                    }}
                }}
                return null;
            }}

            function resolveActiveAudio(preferredEl) {{
                var playing = getPlayingAudio();
                if (playing) {{
                    activeAudio = playing;
                    return activeAudio;
                }}
                if (preferredEl && !preferredEl.ended) {{
                    activeAudio = preferredEl;
                    return activeAudio;
                }}
                if (activeAudio && !activeAudio.ended) {{
                    return activeAudio;
                }}
                for (var j = 0; j < audioIds.length; j += 1) {{
                    var fallbackEl = document.getElementById(audioIds[j]);
                    if (fallbackEl) {{
                        activeAudio = fallbackEl;
                        return activeAudio;
                    }}
                }}
                return null;
            }}

            audioIds.forEach(function(id) {{
                var el = document.getElementById(id);
                if (!el) return;
                el.addEventListener("play", function() {{
                    activeAudio = resolveActiveAudio(el);
                    updateAtTime((activeAudio ? activeAudio.currentTime : el.currentTime) || 0);
                }});
                el.addEventListener("timeupdate", function() {{
                    if (activeAudio === el || !activeAudio || activeAudio.paused) {{
                        activeAudio = resolveActiveAudio(el);
                        updateAtTime((activeAudio ? activeAudio.currentTime : el.currentTime) || 0);
                    }}
                }});
                el.addEventListener("seeked", function() {{
                    activeAudio = resolveActiveAudio(el);
                    updateAtTime((activeAudio ? activeAudio.currentTime : el.currentTime) || 0);
                }});
            }});

            function frameSync() {{
                var el = resolveActiveAudio();
                if (el) {{
                    updateAtTime(el.currentTime || 0);
                }}
                requestAnimationFrame(frameSync);
            }}
            frameSync();
        }})();
        </script>
        """

    return f'''
    <section id="{section_id}">
        <h2>Musical Transcription</h2>
        <p><strong>Total Notes:</strong> {len(notes)} | <strong>Phrases:</strong> {len(phrases)}</p>
        <style>
        #{section_id} .transcription-note {{
            display: inline-block;
            margin: 1px 2px;
            padding: 1px 6px;
            border-radius: 6px;
            border: 1px solid transparent;
            transition: background-color 120ms linear, color 120ms linear, border-color 120ms linear;
        }}
        #{section_id} .transcription-note.active {{
            background: #f2cc60;
            color: #0d1117;
            border-color: #f2cc60;
            box-shadow: 0 0 8px rgba(242, 204, 96, 0.45);
        }}
        #{section_id} .transcription-note-empty {{
            opacity: 0.7;
        }}
        </style>
        <div class="phrase-list" style="display: grid; gap: 8px;">
            <h3>Phrase-by-Phrase Transcription</h3>
            {phrases_html}
        </div>
        {show_more_html}
        <script>
        (function() {{
            var root = document.getElementById("{section_id}");
            if (!root) return;
            var btn = document.getElementById("{section_id}-toggle");
            if (!btn) return;
            var hiddenRows = root.querySelectorAll(".transcription-phrase-hidden");
            var expanded = false;
            btn.addEventListener("click", function() {{
                expanded = !expanded;
                hiddenRows.forEach(function(row) {{
                    row.style.display = expanded ? "block" : "none";
                }});
                btn.textContent = expanded ? "Show less" : "Show more ({hidden_count} more phrases)";
            }});
        }})();
        </script>
        {sync_script_html}
    </section>
    '''


def _generate_transition_section(matrix: np.ndarray) -> str:
    """Generate transition matrix heatmap section."""
    
    # Western note labels (absolute, not sargam)
    western_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    header = "<th></th>" + "".join(f"<th>{note}</th>" for note in western_labels)
    
    rows = []
    for i, row_data in enumerate(matrix):
        cells = f"<td><strong>{western_labels[i]}</strong></td>"
        for j, val in enumerate(row_data):
            intensity = int(val * 255)
            color = f"rgba(52, 152, 219, {val:.2f})"  # Blue color
            cells += f'<td style="background-color: {color}; color: {"white" if val > 0.5 else "black"}">{ val:.2f}</td>'
        rows.append(f"<tr>{cells}</tr>")
    
    return f'''
    <section id="transition-matrix">
        <h2>Note Transition Matrix</h2>
        <p>Probability of transitioning from row note to column note (absolute pitch classes).</p>
        <table class="transition-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </section>
    '''


def _tonic_name(tonic: Optional[int]) -> str:
    """Convert tonic to note name."""
    if tonic is None:
        return "Unknown"
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return names[tonic % 12]


def _get_css_styles() -> str:
    return '''
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap');
        
        * { box-sizing: border-box; }
        body {
            font-family: 'Space Grotesk', 'JetBrains Mono', 'Fira Code', monospace;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #0d1117;
            color: #c9d1d9;
        }
        header {
            text-align: center;
            padding: 30px 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e94560;
            border-radius: 12px;
            margin-bottom: 30px;
            border: 1px solid #30363d;
        }
        header h1 { margin: 0; color: #e94560; font-weight: 700; letter-spacing: -1px; }
        header .subtitle { opacity: 0.8; color: #8b949e; }
        section {
            background: #161b22;
            padding: 24px;
            margin-bottom: 20px;
            border-radius: 12px;
            border: 1px solid #30363d;
        }
        h2 {
            color: #58a6ff;
            border-bottom: 2px solid #e94560;
            padding-bottom: 10px;
            margin-top: 0;
            font-weight: 600;
        }
        .audio-player-container {
            padding: 15px;
            background: #21262d;
            border-radius: 8px;
        }
        audio { 
            width: 100%; 
            margin-bottom: 10px; 
            filter: invert(1) hue-rotate(180deg);
        }
        .pitch-plot { width: 100%; height: 420px; }
        .time-display {
            padding: 10px;
            background: #21262d;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            color: #e94560;
        }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { padding: 8px 12px; text-align: left; border: 1px solid #30363d; }
        th { background: #21262d; color: #e94560; font-weight: 600; }
        tr:nth-child(even) { background: #161b22; }
        tr:nth-child(odd) { background: #0d1117; }
        .metadata-table td:first-child { font-weight: bold; width: 200px; color: #58a6ff; }
        .transition-table th, .transition-table td { 
            text-align: center; 
            padding: 4px 8px; 
            font-size: 12px;
        }
        .sargam-notation pre {
            background: #21262d;
            color: #7ee787;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'JetBrains Mono', monospace;
        }
        /* Ranking table styles */
        .table-scroll-container {
            overflow-x: auto;
            margin: 15px 0;
        }
        .ranking-table {
            min-width: 1000px;
        }
        .ranking-table th, .ranking-table td {
            white-space: nowrap;
            padding: 6px 10px;
            font-size: 13px;
        }
        .ranking-table .raga-name {
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .ranking-table .score {
            font-weight: bold;
            color: #7ee787;
        }
        .hidden-row {
            display: none;
        }
        #show-more-btn {
            background: #e94560;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            margin-top: 10px;
            font-weight: 600;
            font-family: 'Space Grotesk', sans-serif;
            transition: all 0.2s;
        }
        #show-more-btn:hover {
            background: #f05a73;
            transform: translateY(-1px);
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #8b949e;
        }
        img {
            border-radius: 8px;
            border: 1px solid #30363d;
        }
        a { color: #58a6ff; }
        a:hover { color: #e94560; }

        /* Karaoke Overlay */
        .karaoke-container {
            margin: 10px 0;
            padding: 15px;
            background: #0d1117;
            border-radius: 8px;
            overflow-x: auto;
            white-space: nowrap;
            text-align: center;
            border: 1px solid #30363d;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            position: relative;
        }
        .karaoke-word {
            display: inline-block;
            padding: 5px 15px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 18px;
            color: #8b949e;
            opacity: 0.4;
            transition: all 0.2s ease;
            cursor: pointer;
            border-radius: 4px;
        }
        .karaoke-word:hover {
            background: #21262d;
            opacity: 0.8;
        }
        .karaoke-word.active {
            opacity: 1.0;
            color: #e94560;
            transform: scale(1.15);
            font-weight: bold;
            text-shadow: 0 0 10px rgba(233, 69, 96, 0.4);
            background: rgba(233, 69, 96, 0.1);
        }
    '''


def _get_audio_sync_js() -> str:
    """Get JavaScript for audio-plot synchronization."""
    return '''
        // Toggle show more/less rows in ranking table
        var showingMore = false;
        function toggleMoreRows() {
            var hiddenRows = document.querySelectorAll('.hidden-row');
            var btn = document.getElementById('show-more-btn');
            showingMore = !showingMore;
            hiddenRows.forEach(function(row) {
                row.style.display = showingMore ? 'table-row' : 'none';
            });
            btn.textContent = showingMore ? 'Show Less (1-20)' : 'Show More (21-40)';
        }
        
        function setupAudioSync(playerId, tonic, transcriptionData) {
            var audio = document.getElementById(playerId + '-audio');
            var plotDiv = document.getElementById(playerId + '-plot');
            var timeDisplay = document.getElementById(playerId + '-time');
            var sargamDisplay = document.getElementById(playerId + '-sargam');
            var slider = document.getElementById(playerId + '-energy-slider');
            var sliderVal = document.getElementById(playerId + '-energy-val');
            var karaokeDiv = document.getElementById(playerId + '-karaoke');
            
            if (!audio || !plotDiv) return;
            
            var SARGAM = {0:'Sa',1:'re',2:'Re',3:'ga',4:'Ga',5:'ma',6:'Ma',7:'Pa',8:'dha',9:'Dha',10:'ni',11:'Ni'};
            
            // Store original Y data to restore later
            if (plotDiv.data && plotDiv.data[0] && !plotDiv._originalY) {
                plotDiv._originalY = plotDiv.data[0].y.slice();
            }
            
            // Slider Logic
            if (slider && sliderVal) {
                slider.addEventListener('input', function() {
                    var thresh = parseInt(this.value) / 100.0;
                    sliderVal.textContent = this.value + '%';
                    
                    var trace = plotDiv.data[0];
                    if (!trace || !plotDiv._originalY) return;
                    
                    var energy = trace.customdata;
                    var originalY = plotDiv._originalY;
                    var newY = [];
                    
                    // If no energy data, do nothing
                    if (!energy || energy.length !== originalY.length) return;
                    
                    for (var i = 0; i < originalY.length; i++) {
                        if (energy[i] >= thresh) {
                            newY.push(originalY[i]);
                        } else {
                            newY.push(null); // Gap
                        }
                    }
                    
                    Plotly.restyle(plotDiv, {'y': [newY]}, [0]);
                });
            }
            
            // Get cursor shape index (last shape)
            var shapes = plotDiv.layout.shapes || [];
            var cursorIdx = shapes.length - 1;
            
            // Karaoke Setup
            var karaokeSpans = [];
            if (karaokeDiv && transcriptionData && transcriptionData.length > 0) {
                // Clear existing
                karaokeDiv.innerHTML = '';
                
                // Add spacer to start
                var spacerStart = document.createElement('div');
                spacerStart.style.minWidth = "50%";
                karaokeDiv.appendChild(spacerStart);

                transcriptionData.forEach(function(note, idx) {
                    var span = document.createElement('span');
                    span.className = 'karaoke-word';
                    span.textContent = note.sargam;
                    span.dataset.start = note.start;
                    span.dataset.end = note.end;
                    span.dataset.idx = idx;
                    
                    // Click to seek
                    span.onclick = function() {
                        audio.currentTime = note.start;
                    };
                    
                    karaokeDiv.appendChild(span);
                    karaokeSpans.push(span);
                });
                
                // Add spacer to end
                var spacerEnd = document.createElement('div');
                spacerEnd.style.minWidth = "50%";
                karaokeDiv.appendChild(spacerEnd);
            }
            
            // Audio timeupdate -> cursor position
            audio.addEventListener('timeupdate', function() {
                var t = audio.currentTime;
                
                // Update cursor line
                Plotly.relayout(plotDiv, {
                    ['shapes[' + cursorIdx + '].x0']: t,
                    ['shapes[' + cursorIdx + '].x1']: t
                });
                
                // Update time display
                timeDisplay.textContent = t.toFixed(2) + 's';
                
                // Update Karaoke
                if (karaokeSpans.length > 0) {
                    var activeSpan = null;
                    
                    // Find active note (simple linear search is fine for < 1000 notes usually)
                    // Start from the last active index to limit redundant scans.
                    for (var i = 0; i < karaokeSpans.length; i++) {
                        var span = karaokeSpans[i];
                        var start = parseFloat(span.dataset.start);
                        var end = parseFloat(span.dataset.end);
                        
                        if (t >= start && t <= end) {
                            activeSpan = span;
                            if (!span.classList.contains('active')) {
                                span.classList.add('active');
                                // Scroll into view logic
                                // Center the active element
                                var containerCenter = karaokeDiv.offsetWidth / 2;
                                var spanCenter = span.offsetLeft + (span.offsetWidth / 2);
                                // We are using spacers, so offsetLeft is relative to the container scroll content
                                // scrollLeft = spanCenter - containerCenter
                                
                                karaokeDiv.scrollTo({
                                    left: spanCenter - containerCenter,
                                    behavior: 'smooth'
                                });
                            }
                        } else {
                            if (span.classList.contains('active')) {
                                span.classList.remove('active');
                            }
                        }
                    }
                }

                // Estimate current MIDI from plot data (Legacy/Fallback)
                try {
                    var trace = plotDiv.data[0];
                    if (trace && trace.x && trace.y) {
                        var idx = trace.x.findIndex(function(x) { return x >= t; });
                        if (idx > 0) {
                            var midi = trace.y[idx];
                            // Handle nulls (filtered points)
                            if (midi === null || midi === undefined) {
                                sargamDisplay.textContent = '--';
                            } else {
                                var offset = Math.round(midi - tonic) % 12;
                                if (offset < 0) offset += 12;
                                sargamDisplay.textContent = SARGAM[offset] || '--';
                            }
                        }
                    }
                } catch(e) {}
            });
            
            // Click on plot -> seek audio
            plotDiv.on('plotly_click', function(data) {
                if (data.points && data.points[0]) {
                    audio.currentTime = data.points[0].x;
                }
            });
        }
    '''



# =============================================================================
# SCROLLABLE PITCH PLOT (User Requested)
# =============================================================================

def plot_pitch_wide_to_base64_with_legend(
    pitch_data: PitchData, 
    tonic_ref: Union[int, str], 
    raga_name: str = "Unknown",
    raga_notes: Optional[Set[int]] = None,
    transcription_smoothing_ms: float = 70.0,
    transcription_min_duration: float = 0.04,
    transcription_derivative_threshold: float = 2.0,
    figsize_width: int = 80, 
    figsize_height: int = 7, 
    dpi: int = 100, 
    join_gap_threshold: float = 1.0,
    show_rms_overlay: bool = True,
    overlay_energy: Optional[np.ndarray] = None,
    overlay_timestamps: Optional[np.ndarray] = None,
    overlay_label: str = "RMS Energy",
    phrase_ranges: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[str, str, int, float, float]:
    """
    Generate wide pitch contour and overlay sargam lines, returning base64 images.
    Returns: (legend_b64, plot_b64, pixel_width, x_axis_start, x_axis_end)
    """
    import io
    import base64
    import librosa
    from matplotlib.ticker import MultipleLocator
    
    # Resolve tonic
    tonic_midi_base: float
    tonic_midi: int
    tonic_label: str
    if isinstance(tonic_ref, (int, np.integer)):
        tonic_midi = int(tonic_ref) % 12
        tonic_midi_base = float(tonic_midi + 60) # Default to middle C octave if int
        tonic_label = _tonic_name(tonic_ref)
    else:
        try:
            tonic_midi_base = float(librosa.note_to_midi(str(tonic_ref) + '4'))
            tonic_midi = int(tonic_midi_base) % 12
            tonic_label = str(tonic_ref)
        except Exception:
            tonic_midi_base = 60.0
            tonic_midi = 0
            tonic_label = "C"

    # Colors
    OCTAVE_COLORS = {
        -2: '#8B4513',  # Brown
        -1: '#FF6B6B',  # Red
        0:  '#2ECC71',  # Green
        1:  '#3498DB',  # Blue
        2:  '#9B59B6',  # Purple
    }

    # Prepare data
    pitch_hz = pitch_data.pitch_hz
    timestamps = pitch_data.timestamps
    voiced_mask = pitch_data.voiced_mask
    frame_period = pitch_data.frame_period
    
    if not np.any(voiced_mask):
         # Return empty placeholders
         return "", "", 100, 0.0, 1.0

    voiced_midi = librosa.hz_to_midi(pitch_hz[voiced_mask])
    voiced_midi_rounded = np.round(voiced_midi).astype(int)
    
    # Calculate range dynamically based on data min/max
    data_min = np.min(voiced_midi)
    data_max = np.max(voiced_midi)
    
    min_m = np.floor(data_min) - 2
    max_m = np.ceil(data_max) + 2
    
    rng = np.arange(int(np.floor(min_m)), int(np.ceil(max_m)) + 1)
    
    # --- Main Plot ---
    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height), dpi=dpi)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.1) # Min margins

    # Phrase regions in yellow so phrase boundaries are visible on the scroll plot.
    if phrase_ranges:
        label_y = float(max_m) - 0.35
        for idx, (start_t, end_t) in enumerate(phrase_ranges):
            if not np.isfinite(start_t) or not np.isfinite(end_t) or end_t <= start_t:
                continue
            ax.axvspan(
                float(start_t),
                float(end_t),
                facecolor="#ffeb3b",
                edgecolor="none",
                linewidth=0.0,
                alpha=0.16,
                zorder=0,
            )
            ax.axvline(float(start_t), color="#000000", linestyle="-", linewidth=0.7, alpha=0.65, zorder=1)
            ax.axvline(float(end_t), color="#000000", linestyle="-", linewidth=0.7, alpha=0.65, zorder=1)
            ax.text(
                (float(start_t) + float(end_t)) / 2.0,
                label_y,
                f"P{idx + 1}",
                ha="center",
                va="top",
                fontsize=7,
                color="#000000",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="#fff3a3", edgecolor="#000000", alpha=0.9),
                zorder=2,
            )
    
    voiced_times = timestamps[voiced_mask]
    
    # Plot contour segments
    time_diffs = np.diff(voiced_times)
    gap_trigger = frame_period * 4
    gap_indices = np.where(time_diffs > gap_trigger)[0]
    
    segments = []
    start_idx = 0
    for gap_idx in gap_indices:
        end_idx = gap_idx + 1
        segments.append((start_idx, end_idx))
        start_idx = end_idx
    segments.append((start_idx, len(voiced_times)))
    
    # Join small gaps
    joined_segments = []
    if segments:
        curr = segments[0]
        for next_seg in segments[1:]:
            # if gap < threshold, merge
            t_end_prev = voiced_times[curr[1]-1]
            t_start_next = voiced_times[next_seg[0]]
            if t_start_next - t_end_prev <= join_gap_threshold:
                curr = (curr[0], next_seg[1])
            else:
                joined_segments.append(curr)
                curr = next_seg
        joined_segments.append(curr)
        
    for seg in joined_segments:
        s, e = seg
        ax.plot(voiced_times[s:e], voiced_midi[s:e], color='tab:blue', linewidth=1.5, alpha=0.9)

    # --- Transcription Overlay ---
    # Detect events
    transcription_events = transcription.detect_stationary_events(
        pitch_hz=pitch_hz,
        timestamps=timestamps,
        voicing_mask=voiced_mask, # Note: voiced_mask is boolean array same length as pitch_hz
        tonic=tonic_midi_base,
        snap_mode='chromatic', # Default to chromatic for now
        smoothing_sigma_ms=transcription_smoothing_ms,
        derivative_threshold=transcription_derivative_threshold,
        min_event_duration=transcription_min_duration
    )
    
    # Plot events
    for i, event in enumerate(transcription_events):
        # Draw horizontal line for the note
        if i == 0:
            ax.hlines(
                y=event.snapped_midi,
                xmin=event.start,
                xmax=event.end,
                colors='orange',
                linewidth=3,
                alpha=0.8,
                label='Transcribed'
            )
        else:
            ax.hlines(
                y=event.snapped_midi,
                xmin=event.start,
                xmax=event.end,
                colors='orange',
                linewidth=3,
                alpha=0.8
            )

    # --- Inflection Points Overlay ---
    # Detect turning points (derivative = 0)
    inflection_times, inflection_pitches = transcription.detect_pitch_inflection_points(
        pitch_hz=pitch_hz,
        timestamps=timestamps,
        voicing_mask=voiced_mask,
        smoothing_sigma_ms=transcription_smoothing_ms, # Use same smoothing as transcription
    )
    
    # Filter out inflection points that are already covered by stationary events (orange bars)
    if len(inflection_times) > 0 and transcription_events:
        keep_mask = np.ones(len(inflection_times), dtype=bool)
        for event in transcription_events:
            # Mark points inside this event's duration as False using strict boundaries.
            overlap = (inflection_times >= event.start) & (inflection_times <= event.end)
            keep_mask[overlap] = False
            
        inflection_times = inflection_times[keep_mask]
        inflection_pitches = inflection_pitches[keep_mask]
    
    if len(inflection_times) > 0:
         # Plot vertical lines for inflection points
         ax.vlines(
             x=inflection_times,
             ymin=min_m,
             ymax=max_m,
             colors='red',
             linestyles=':',
             linewidth=0.8,
             alpha=0.4,
             label='Inflection'
         )
         # Optionally mark points
         ax.scatter(inflection_times, inflection_pitches, c='red', s=10, alpha=0.5, zorder=3)

    # --- RMS Energy Overlay ---
    # Scaled energy is drawn as a filled area in the bottom 25 % of the plot.
    e_vals_src = overlay_energy if overlay_energy is not None else pitch_data.energy
    e_times_src = overlay_timestamps if overlay_timestamps is not None else timestamps
    if show_rms_overlay and e_vals_src is not None and len(e_vals_src) > 0:
        e_vals_arr = np.asarray(e_vals_src, dtype=float)
        e_times_arr = np.asarray(e_times_src, dtype=float)
        e_len = min(len(e_vals_arr), len(e_times_arr))
        if e_len > 0:
            e_vals = np.nan_to_num(e_vals_arr[:e_len], nan=0.0, posinf=1.0, neginf=0.0)
            e_times = e_times_arr[:e_len]
            midi_span = (max_m + 1) - (min_m - 1)
            scaled_e = e_vals * 0.25 * midi_span + (min_m - 1)
            ax.fill_between(
                e_times, min_m - 1, scaled_e,
                color='darkcyan', alpha=0.15, zorder=0, label=overlay_label
            )
            ax.plot(e_times, scaled_e, color='darkcyan', linewidth=0.6, alpha=0.4, zorder=0)

    ax.set_ylim(min_m - 1, max_m + 1)
    x_axis_start = 0.0
    x_axis_end = max(float(timestamps[-1]), 1.0)
    ax.set_xlim(x_axis_start, x_axis_end)
    ax.set_xlabel('Time (s)')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(2)) # Tick every 2 seconds
    ax.set_title(f"Pitch Contour Analysis ({raga_name} / {tonic_label})")

    # Horizontal Lines
    # Recording-relative Sa baseline for display labels:
    # avoid dense octave markers for normal ranges.
    sa_mask = (voiced_midi_rounded % 12) == tonic_midi
    if np.any(sa_mask):
        median_sa_octave = int(round(float(np.median(np.round((voiced_midi[sa_mask] - tonic_midi_base) / 12.0)))))
    else:
        median_sa_octave = int(round(float(np.median(np.round((voiced_midi - tonic_midi_base) / 12.0)))))

    for midi_val in rng:
        pc = int(midi_val) % 12
        offset = (pc - tonic_midi) % 12
        
        if raga_notes is None or pc in raga_notes:
            octave_diff = int((midi_val - tonic_midi_base) // 12)
            line_color = OCTAVE_COLORS.get(octave_diff, 'gray')
            line_alpha = 0.4 if (raga_notes is not None and pc in raga_notes) else 0.15
            linestyle = '--'
            ax.axhline(y=midi_val, color=line_color, alpha=line_alpha, linewidth=0.8, linestyle=linestyle)

    # Save Main
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, transparent=False)
    plt.close(fig)
    plot_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    pixel_width = int(figsize_width * dpi)

    # --- Legend Plot (Sticky Side) ---
    legend_fig, legend_ax = plt.subplots(figsize=(2, figsize_height), dpi=dpi)
    legend_fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(min_m - 1, max_m + 1)
    legend_ax.axis('off')
    
    for midi_val in rng:
        pc = int(midi_val) % 12
        offset = (pc - tonic_midi) % 12
        
        if raga_notes is None or pc in raga_notes:
            label = OFFSET_TO_SARGAM.get(offset, '')
            octave_diff = int((midi_val - tonic_midi_base) // 12)
            
            # Only show octave markers when far from the recording's median Sa.
            relative_octave = octave_diff - median_sa_octave
            if relative_octave <= -3:
                label += "'" * abs(relative_octave)
            
            text_color = OCTAVE_COLORS.get(octave_diff, 'gray')
            # Stronger text
            legend_ax.axhline(y=midi_val, color=text_color, alpha=0.4, linewidth=0.8)
            legend_ax.text(0.9, midi_val, label, fontsize=10, color=text_color, fontweight='bold',
                           ha='right', va='center', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))

    lbuf = io.BytesIO()
    legend_fig.savefig(lbuf, format='png', dpi=dpi, transparent=True)
    plt.close(legend_fig)
    legend_b64 = base64.b64encode(lbuf.getvalue()).decode('ascii')

    return legend_b64, plot_b64, pixel_width, x_axis_start, x_axis_end

def create_scrollable_pitch_plot_html(
    pitch_data: PitchData,
    tonic: int,
    raga_name: str,
    audio_element_ids: List[str],
    raga_notes: Optional[Set[int]] = None,
    transcription_smoothing_ms: float = 70.0,
    transcription_min_duration: float = 0.04,
    transcription_derivative_threshold: float = 2.0,
    show_rms_overlay: bool = True,
    overlay_energy: Optional[np.ndarray] = None,
    overlay_timestamps: Optional[np.ndarray] = None,
    overlay_label: str = "RMS Energy",
    phrase_ranges: Optional[List[Tuple[float, float]]] = None,
) -> str:
    """
    Create HTML component for scrollable pitch plot with audio sync.
    """
    legend_b64, plot_b64, px_width, x_axis_start, x_axis_end = plot_pitch_wide_to_base64_with_legend(
        pitch_data, tonic, raga_name, raga_notes, 
        transcription_smoothing_ms=transcription_smoothing_ms,
        transcription_min_duration=transcription_min_duration,
        transcription_derivative_threshold=transcription_derivative_threshold,
        show_rms_overlay=show_rms_overlay,
        overlay_energy=overlay_energy,
        overlay_timestamps=overlay_timestamps,
        overlay_label=overlay_label,
        phrase_ranges=phrase_ranges,
    )
    
    if not plot_b64:
        return "<p>No pitch data for validation plot.</p>"
        
    unique_id = f"sp_{uuid.uuid4().hex[:6]}"
    duration = x_axis_end - x_axis_start
    
    html = f"""
    <div class="scroll-plot-wrapper" id="{unique_id}-wrapper" style="border: 1px solid #30363d; border-radius: 8px; overflow: hidden; margin: 20px 0; background: #0d1117;">
        <div style="display: flex; height: 700px; position: relative;">
            <!-- Legend (Sticky) -->
            <div style="flex: 0 0 200px; background: #161b22; border-right: 1px solid #30363d; z-index: 10; display: flex; align-items: center; justify-content: center;">
                <img src="data:image/png;base64,{legend_b64}" style="height: 100%; object-fit: contain; width: 100%;">
            </div>
            
            <!-- Scrollable Content -->
            <div id="{unique_id}-container" style="flex: 1; overflow-x: auto; position: relative; background: #0d1117;">
                <div style="width: {px_width}px; height: 100%; position: relative;">
                    <img src="data:image/png;base64,{plot_b64}" style="width: 100%; height: 100%; display: block;">
                    <!-- Cursor Line -->
                    <div id="{unique_id}-cursor" style="position: absolute; left: 0; top: 0; bottom: 0; width: 2px; background: #e94560; box-shadow: 0 0 5px #e94560; pointer-events: none; z-index: 5;"></div>
                </div>
            </div>
        </div>
        <div style="padding: 10px; background: #161b22; border-top: 1px solid #30363d; display: flex; justify-content: space-between; color: #8b949e; font-size: 12px;">
             <span>Total Duration: {duration:.2f}s</span>
             <span id="{unique_id}-status">Synced to Audio</span>
        </div>
    </div>
    
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        const trackIds = {json.dumps(audio_element_ids)};
        const container = document.getElementById("{unique_id}-container");
        const cursor = document.getElementById("{unique_id}-cursor");
        const xStart = {x_axis_start};
        const xEnd = {x_axis_end};
        const totalDuration = xEnd - xStart;
        const pixelWidth = {px_width};
        
        if (container && cursor) {{
            console.log("Initializing sync for {unique_id}");
            
            // Plot margins (matches python subplots_adjust)
            const marginL = 0.005;
            const marginR = 0.995;
            const plotStartPx = marginL * pixelWidth;
            const plotEndPx = marginR * pixelWidth;
            let seekSnapUntil = 0;
            let activeAudio = null;

            function clamp(v, lo, hi) {{
                return Math.max(lo, Math.min(hi, v));
            }}

            function timeToX(t) {{
                const safeT = clamp(t, xStart, xEnd);
                const pct = totalDuration > 0 ? ((safeT - xStart) / totalDuration) : 0;
                return plotStartPx + (pct * (plotEndPx - plotStartPx));
            }}

            function xToTime(x) {{
                const safeX = clamp(x, plotStartPx, plotEndPx);
                const pct = (safeX - plotStartPx) / Math.max(plotEndPx - plotStartPx, 1e-6);
                return xStart + (pct * totalDuration);
            }}

            function seekAllTracks(targetTime) {{
                trackIds.forEach(function(id) {{
                    const el = document.getElementById(id);
                    if (!el) return;
                    try {{
                        el.currentTime = targetTime;
                    }} catch (_err) {{
                        // Ignore tracks that reject seek before metadata is ready.
                    }}
                }});
            }}

            function getPlayingAudio() {{
                for (const id of trackIds) {{
                    const el = document.getElementById(id);
                    if (el && !el.paused && !el.ended) {{
                        return el;
                    }}
                }}
                return null;
            }}

            function resolveActiveAudio(preferredEl) {{
                const playing = getPlayingAudio();
                if (playing) {{
                    activeAudio = playing;
                    return activeAudio;
                }}
                if (preferredEl && !preferredEl.ended) {{
                    activeAudio = preferredEl;
                    return activeAudio;
                }}
                if (activeAudio && !activeAudio.ended) {{
                    return activeAudio;
                }}
                for (const id of trackIds) {{
                    const el = document.getElementById(id);
                    if (el) {{
                        activeAudio = el;
                        return activeAudio;
                    }}
                }}
                return null;
            }}

            function renderAtTime(t, follow) {{
                const x = timeToX(t);
                cursor.style.left = x + 'px';

                if (!follow) {{
                    return;
                }}

                const viewWidth = container.clientWidth;
                const maxScroll = Math.max(0, container.scrollWidth - viewWidth);
                const targetScroll = clamp(x - (viewWidth / 2), 0, maxScroll);

                if ((performance.now() < seekSnapUntil) || (Math.abs(container.scrollLeft - targetScroll) > 100)) {{
                    container.scrollLeft = targetScroll;
                }} else {{
                    container.scrollLeft = container.scrollLeft * 0.85 + targetScroll * 0.15;
                }}
            }}
            
            // Sync Loop
            function updateSync() {{
                const src = resolveActiveAudio();
                if (src) {{
                    const t = src.currentTime || 0;
                    const follow = (!src.paused && !src.ended) || (performance.now() < seekSnapUntil);
                    renderAtTime(t, follow);
                }}
                requestAnimationFrame(updateSync);
            }}
            updateSync();
            
            // Allow click to seek
            container.addEventListener('click', function(e) {{
                const rect = container.getBoundingClientRect();
                const clickX = e.clientX - rect.left + container.scrollLeft;
                const seekT = xToTime(clickX);

                if (isFinite(seekT) && seekT >= xStart && seekT <= xEnd) {{
                    console.log("Seeking to:", seekT);
                    const sourceBeforeSeek = getPlayingAudio() || activeAudio;
                    // Seek ALL tracks to keep them in sync if swapped
                    seekAllTracks(seekT);
                    activeAudio = resolveActiveAudio(sourceBeforeSeek || undefined);
                    const t = (activeAudio ? activeAudio.currentTime : seekT) || seekT;
                    renderAtTime(t, true);
                    seekSnapUntil = performance.now() + 300;
                }}
            }});

            // Keep cursor/timeline aligned immediately after native seek events too.
            trackIds.forEach(function(id) {{
                const el = document.getElementById(id);
                if (!el) return;
                el.addEventListener('play', function() {{
                    activeAudio = resolveActiveAudio(el);
                    renderAtTime((activeAudio ? activeAudio.currentTime : el.currentTime) || 0, true);
                }});
                el.addEventListener('seeked', function() {{
                    activeAudio = resolveActiveAudio(el);
                    const t = (activeAudio ? activeAudio.currentTime : el.currentTime) || 0;
                    renderAtTime(t, true);
                    seekSnapUntil = performance.now() + 300;
                }});
                el.addEventListener('loadedmetadata', function() {{
                    if (!activeAudio) {{
                        activeAudio = resolveActiveAudio(el);
                        renderAtTime((activeAudio ? activeAudio.currentTime : el.currentTime) || 0, false);
                    }}
                }});
                el.addEventListener('pause', function() {{
                    if (activeAudio === el) {{
                        activeAudio = resolveActiveAudio();
                    }}
                }});
                el.addEventListener('ended', function() {{
                    if (activeAudio === el) {{
                        activeAudio = resolveActiveAudio();
                    }}
                }});
            }});
        }}
    }});
    </script>
    """
    return html


def save_notes_to_csv(notes: List['Note'], output_path: str):
    """
    Save list of Note objects to CSV.
    """
    import csv
    if not notes:
        return

    fieldnames = [
        'start', 'end', 'duration', 
        'pitch_midi', 'pitch_hz', 'confidence',
        'pitch_class', 'sargam', 'energy'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for note in notes:
            writer.writerow({
                'start': f"{note.start:.3f}",
                'end': f"{note.end:.3f}",
                'duration': f"{note.duration:.3f}",
                'pitch_midi': f"{note.pitch_midi:.2f}",
                'pitch_hz': f"{note.pitch_hz:.1f}",
                'confidence': f"{note.confidence:.2f}",
                'pitch_class': note.pitch_class,
                'sargam': note.sargam,
                'energy': f"{getattr(note, 'energy', 0.0):.4f}"
            })


def _generate_karaoke_section(
    phrases: List['Phrase'],
    tonic: int,
    audio_element_ids: List[str],
) -> str:
    """
    Generate a phrase-level karaoke section that scrolls vertically like
    lyric UIs.
    """
    if not phrases:
        return ""

    uid = f"karaoke_{uuid.uuid4().hex[:6]}"
    flat_notes: List[Note] = [note for phrase in phrases for note in phrase.notes]
    median_sa_octave = _estimate_median_sa_octave(flat_notes, tonic)

    # Build phrase data.
    phrase_data = []
    note_timeline = []
    for idx, phrase in enumerate(phrases):
        note_items = []
        for n in phrase.notes:
            label = _format_note_sargam_for_report(
                n,
                tonic=tonic,
                median_sa_octave=median_sa_octave,
            )
            if label:
                note_item = {
                    "label": label,
                    "start": round(n.start, 3),
                    "end": round(n.end, 3),
                    "phrase_idx": idx,
                    "note_idx": len(note_timeline),
                }
                note_items.append(note_item)
                note_timeline.append(note_item)

        if not note_items:
            continue
        phrase_text = " ".join(item["label"] for item in note_items).strip()
        phrase_data.append({
            "idx": idx + 1,
            "start": round(phrase.start, 3),
            "end": round(phrase.end, 3),
            "text": phrase_text,
            "notes": note_items,
        })

    if not phrase_data or not note_timeline:
        return ""

    phrases_json = json.dumps(phrase_data)
    notes_json = json.dumps(note_timeline)

    return f'''
    <section id="{uid}-section">
        <h2>Phrase Karaoke</h2>

        <style>
        #{uid}-lyrics .karaoke-phrase-note {{
            display: inline-block;
            margin: 1px 2px;
            padding: 2px 6px;
            border-radius: 6px;
            border: 1px solid transparent;
            transition: all 0.16s ease;
        }}
        #{uid}-lyrics .karaoke-phrase-note.sung {{
            background: rgba(242, 204, 96, 0.16);
            border-color: rgba(242, 204, 96, 0.45);
            color: #f0f6fc;
        }}
        #{uid}-lyrics .karaoke-phrase-note.current {{
            background: #f2cc60;
            border-color: #f2cc60;
            color: #111827;
            box-shadow: 0 0 8px rgba(242, 204, 96, 0.35);
        }}
        </style>

        <div id="{uid}-lyrics" style="
            overflow-y: auto;
            padding: 14px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            height: 320px;
            position: relative;
        ">
        </div>

        <div style="
            display: flex;
            justify-content: space-between;
            padding: 6px 4px 0;
            font-size: 11px;
            color: #484f58;
        ">
            <span id="{uid}-time">0.00 s</span>
            <span id="{uid}-phrase">--</span>
        </div>

        <script>
        (function() {{
            var phrases = {phrases_json};
            var noteTimeline = {notes_json};
            var listEl = document.getElementById("{uid}-lyrics");
            var timeEl = document.getElementById("{uid}-time");
            var phraseEl = document.getElementById("{uid}-phrase");
            var trackIds = {json.dumps(audio_element_ids)};
            if (!listEl || !phrases.length || !noteTimeline.length) return;

            var rows = [];
            var phraseNoteSpans = [];
            var BASE_STYLE = "width:100%; padding:10px 12px; border-radius:10px; " +
                "border:1px solid #30363d; background:#161b22; color:#8b949e; " +
                "opacity:0.45; transform:scale(0.98); transition:all 0.2s ease; " +
                "cursor:pointer; line-height:1.5;";
            var ACTIVE_STYLE = "border-color:#e94560; color:#ffffff; opacity:1; " +
                "transform:scale(1.01); box-shadow:0 0 0 1px rgba(233,69,96,0.25) inset;";

            phrases.forEach(function(p) {{
                var row = document.createElement("div");
                row.dataset.start = p.start;
                row.dataset.end = p.end;
                row.dataset.idx = p.idx;
                row.style.cssText = BASE_STYLE;

                var meta = document.createElement("div");
                meta.style.cssText = "font-size:11px;color:#8b949e;margin-bottom:6px;";
                meta.textContent = "Phrase " + p.idx + " · " + p.start.toFixed(2) + "s - " + p.end.toFixed(2) + "s";

                var phraseLine = document.createElement("div");
                phraseLine.style.cssText = "font-size:17px;word-break:break-word;";
                p.notes.forEach(function(n, notePos) {{
                    var noteSpan = document.createElement("span");
                    noteSpan.className = "karaoke-phrase-note";
                    noteSpan.textContent = n.label;
                    noteSpan.dataset.noteIdx = String(n.note_idx);
                    noteSpan.addEventListener("click", function(evt) {{
                        evt.stopPropagation();
                        trackIds.forEach(function(id) {{
                            var el = document.getElementById(id);
                            if (!el) return;
                            try {{
                                el.currentTime = n.start;
                            }} catch (_err) {{
                            }}
                        }});
                        queueSync(getPlayingAudio() || activeAudio || null, true);
                    }});
                    phraseLine.appendChild(noteSpan);
                    phraseNoteSpans[n.note_idx] = noteSpan;
                    if (notePos < p.notes.length - 1) {{
                        phraseLine.appendChild(document.createTextNode(" "));
                    }}
                }});

                row.appendChild(meta);
                row.appendChild(phraseLine);

                row.addEventListener("click", function() {{
                    trackIds.forEach(function(id) {{
                        var el = document.getElementById(id);
                        if (!el) return;
                        try {{
                            el.currentTime = p.start;
                        }} catch (_err) {{
                        }}
                    }});
                    queueSync(getPlayingAudio() || activeAudio || null, true);
                }});

                listEl.appendChild(row);
                rows.push(row);
            }});

            // Binary search for active phrase at time t.
            function findActivePhrase(t) {{
                var lo = 0, hi = phrases.length - 1;
                while (lo <= hi) {{
                    var mid = (lo + hi) >>> 1;
                    if (t < phrases[mid].start) hi = mid - 1;
                    else if (t > phrases[mid].end) lo = mid + 1;
                    else return mid;
                }}
                return -1;
            }}

            // Binary search for active token at time t.
            function findActiveToken(t) {{
                var lo = 0, hi = noteTimeline.length - 1;
                while (lo <= hi) {{
                    var mid = (lo + hi) >>> 1;
                    if (t < noteTimeline[mid].start) hi = mid - 1;
                    else if (t > noteTimeline[mid].end) lo = mid + 1;
                    else return mid;
                }}
                return -1;
            }}

            function findLastCompletedToken(t) {{
                var lo = 0, hi = noteTimeline.length - 1, best = -1;
                while (lo <= hi) {{
                    var mid = (lo + hi) >>> 1;
                    if (noteTimeline[mid].end <= t) {{
                        best = mid;
                        lo = mid + 1;
                    }} else {{
                        hi = mid - 1;
                    }}
                }}
                return best;
            }}

            function setPhraseInactive(phraseIdx) {{
                if (phraseIdx < 0 || phraseIdx >= rows.length) return;
                rows[phraseIdx].style.cssText = BASE_STYLE;
            }}

            function setPhraseActive(phraseIdx) {{
                if (phraseIdx < 0 || phraseIdx >= rows.length) return;
                rows[phraseIdx].style.cssText = BASE_STYLE + ACTIVE_STYLE;
            }}

            function setTokenSung(tokenIdx, isSung) {{
                var phraseNote = phraseNoteSpans[tokenIdx];
                if (phraseNote) phraseNote.classList.toggle("sung", isSung);
            }}

            function setTokenCurrent(tokenIdx, isCurrent) {{
                var phraseNote = phraseNoteSpans[tokenIdx];
                if (phraseNote) phraseNote.classList.toggle("current", isCurrent);
            }}

            var activeAudio = null;
            var lastActive = -1;
            var lastCurrentToken = -1;
            var lastCompletedToken = -1;
            var sungAppliedIdx = -1;
            var sungRafScheduled = false;
            var syncRafScheduled = false;
            var pendingPreferredAudio = null;
            var pendingFromSeek = false;

            function getPlayingAudio() {{
                for (var i = 0; i < trackIds.length; i++) {{
                    var el = document.getElementById(trackIds[i]);
                    if (el && !el.paused && !el.ended) return el;
                }}
                return null;
            }}

            function resolveActiveAudio(preferredEl) {{
                var playing = getPlayingAudio();
                if (playing) {{
                    activeAudio = playing;
                    return activeAudio;
                }}
                if (preferredEl && !preferredEl.ended) {{
                    activeAudio = preferredEl;
                    return activeAudio;
                }}
                if (activeAudio && !activeAudio.ended) {{
                    return activeAudio;
                }}
                for (var i = 0; i < trackIds.length; i++) {{
                    var fallback = document.getElementById(trackIds[i]);
                    if (fallback) {{
                        activeAudio = fallback;
                        return activeAudio;
                    }}
                }}
                return null;
            }}

            function queueSync(preferredEl, fromSeek) {{
                if (preferredEl) {{
                    pendingPreferredAudio = preferredEl;
                }}
                if (fromSeek) {{
                    pendingFromSeek = true;
                }}
                if (syncRafScheduled) return;
                syncRafScheduled = true;
                requestAnimationFrame(function() {{
                    syncRafScheduled = false;
                    var preferred = pendingPreferredAudio;
                    var seekFlag = pendingFromSeek;
                    pendingPreferredAudio = null;
                    pendingFromSeek = false;
                    syncAtCurrentTime(preferred, seekFlag);
                }});
            }}

            function processSungUpdates() {{
                sungRafScheduled = false;
                if (sungAppliedIdx === lastCompletedToken) return;

                var chunkSize = 48;
                if (sungAppliedIdx < lastCompletedToken) {{
                    var hi = Math.min(lastCompletedToken, sungAppliedIdx + chunkSize);
                    for (var up = sungAppliedIdx + 1; up <= hi; up++) {{
                        setTokenSung(up, true);
                    }}
                    sungAppliedIdx = hi;
                }} else {{
                    var loExclusive = Math.max(lastCompletedToken, sungAppliedIdx - chunkSize);
                    for (var down = sungAppliedIdx; down > loExclusive; down--) {{
                        setTokenSung(down, false);
                    }}
                    sungAppliedIdx = loExclusive;
                }}

                if (sungAppliedIdx !== lastCompletedToken) {{
                    sungRafScheduled = true;
                    requestAnimationFrame(processSungUpdates);
                }}
            }}

            function scheduleSungUpdates() {{
                if (sungRafScheduled) return;
                sungRafScheduled = true;
                requestAnimationFrame(processSungUpdates);
            }}

            function syncAtCurrentTime(preferredEl, fromSeek) {{
                var audio = resolveActiveAudio(preferredEl || null);
                if (!audio) return;

                var t = audio.currentTime;
                if (timeEl) timeEl.textContent = t.toFixed(2) + " s";

                var completedIdx = findLastCompletedToken(t);
                if (completedIdx !== lastCompletedToken) {{
                    lastCompletedToken = completedIdx;
                    scheduleSungUpdates();
                }}

                var tokenIdx = findActiveToken(t);
                if (tokenIdx !== lastCurrentToken && lastCurrentToken >= 0) {{
                    setTokenCurrent(lastCurrentToken, false);
                }}
                if (tokenIdx >= 0 && tokenIdx !== lastCurrentToken) {{
                    setTokenCurrent(tokenIdx, true);
                }}
                if (tokenIdx < 0 && lastCurrentToken >= 0) {{
                    setTokenCurrent(lastCurrentToken, false);
                }}
                lastCurrentToken = tokenIdx;

                var prevActive = lastActive;
                var idx = findActivePhrase(t);

                if (idx !== prevActive && prevActive >= 0) {{
                    setPhraseInactive(prevActive);
                }}
                if (idx < 0) {{
                    if (phraseEl) phraseEl.textContent = "--";
                    lastActive = -1;
                    return;
                }}

                var phraseChanged = idx !== prevActive;
                if (phraseChanged) {{
                    setPhraseActive(idx);
                    if (phraseEl) {{
                        phraseEl.textContent = "Phrase " + phrases[idx].idx;
                    }}
                }}

                if (phraseChanged || prevActive === -1) {{
                    var row = rows[idx];
                    var targetTop = row.offsetTop - (listEl.clientHeight / 2) + (row.offsetHeight / 2);
                    if (fromSeek) {{
                        listEl.scrollTop = Math.max(0, targetTop);
                    }} else {{
                        listEl.scrollTo({{ top: Math.max(0, targetTop), behavior: "smooth" }});
                    }}
                }}

                lastActive = idx;
            }}

            // Prime once on load.
            queueSync(null, false);

            // Bind to all audio timing events for low-lag updates after seeks.
            trackIds.forEach(function(id) {{
                var el = document.getElementById(id);
                if (!el) return;
                ["timeupdate", "seeked", "play", "loadedmetadata"].forEach(function(evt) {{
                    el.addEventListener(evt, function() {{
                        queueSync(el, evt === "seeked");
                    }});
                }});
                el.addEventListener("pause", function() {{
                    if (activeAudio === el) {{
                        activeAudio = resolveActiveAudio();
                    }}
                    queueSync(activeAudio || null, false);
                }});
                el.addEventListener("ended", function() {{
                    if (activeAudio === el) {{
                        activeAudio = resolveActiveAudio();
                    }}
                    queueSync(activeAudio || null, false);
                }});
            }});

            function frameSync() {{
                var playing = getPlayingAudio();
                if (playing) {{
                    queueSync(playing, false);
                }}
                requestAnimationFrame(frameSync);
            }}
            frameSync();
        }})();
        </script>
    </section>
    '''


@dataclass
class AnalysisStats:
    correction_summary: dict
    pattern_analysis: dict  # Replaced minimal motif list with full dict
    raga_name: str
    tonic: str
    transition_matrix_path: str
    pitch_plot_path: str


def generate_analysis_report(
    results: 'AnalysisResults',
    stats: AnalysisStats,
    output_dir: str
) -> str:
    """
    Generate detailed analysis report (Phase 2).
    """
    report_path = os.path.join(output_dir, "analysis_report.html")
    
    # Relative paths for audio assets
    original_audio_path = _require_audio_path(results.config)
    vocals_rel = os.path.relpath(results.config.vocals_path, output_dir)
    accomp_rel = os.path.relpath(results.config.accompaniment_path, output_dir)
    original_rel = os.path.relpath(original_audio_path, output_dir)
    vocals_local = os.path.basename(results.config.vocals_path)
    accomp_local = os.path.basename(results.config.accompaniment_path)
    original_local = os.path.basename(original_audio_path)

    original_sources = _build_audio_source_tags(original_local, original_rel)
    vocals_sources = _build_audio_source_tags(vocals_local, vocals_rel)
    accomp_sources = _build_audio_source_tags(accomp_local, accomp_rel)
    
    # Audio IDs for sync
    vocals_id = "vocals-player"
    accomp_id = "accomp-player"
    original_id = "original-player"
    audio_element_ids = [original_id, vocals_id, accomp_id]
    
    analysis_source_key = "original" if results.config.melody_source == "composite" else "vocals"
    analysis_source_label = (
        "Original Mix (Composite Pitch/Energy)"
        if analysis_source_key == "original"
        else "Vocals Stem (Separated Melody Pitch/Energy)"
    )

    vocals_pitch_data = results.pitch_data_stem
    if vocals_pitch_data is None and results.config.melody_source == "separated":
        vocals_pitch_data = results.pitch_data_vocals

    original_pitch_data = results.pitch_data_composite
    if original_pitch_data is None and results.config.melody_source == "composite":
        original_pitch_data = results.pitch_data_vocals

    track_specs: List[TrackSpec] = [
        {
            "key": "original",
            "label": "Original Mix",
            "audio_id": original_id,
            "pitch_data": original_pitch_data,
        },
        {
            "key": "vocals",
            "label": "Vocals Stem",
            "audio_id": vocals_id,
            "pitch_data": vocals_pitch_data,
        },
        {
            "key": "accompaniment",
            "label": "Accompaniment",
            "audio_id": accomp_id,
            "pitch_data": results.pitch_data_accomp,
        },
    ]
    available_tracks = [spec for spec in track_specs if spec["pitch_data"] is not None]
    default_track_key = analysis_source_key if any(
        spec["key"] == analysis_source_key for spec in available_tracks
    ) else (available_tracks[0]["key"] if available_tracks else None)
    default_energy_key = default_track_key

    track_buttons_html = ""
    energy_buttons_html = ""
    track_panels_html = ""
    track_switch_js = ""
    playback_controls_html = """
    <div id="playback-rate-controls" style="margin: 0 0 16px 0;">
        <p style="font-size: 0.86em; color: #8b949e; margin: 0 0 6px 0;">Playback Speed (for transcription verification):</p>
        <div style="display:flex; flex-wrap:wrap; gap:8px;">
            <button type="button" data-playback-rate="1" style="padding:8px 12px; border-radius:6px; border:1px solid #30363d; background:#21262d; color:#c9d1d9; cursor:pointer;">1x</button>
            <button type="button" data-playback-rate="0.5" style="padding:8px 12px; border-radius:6px; border:1px solid #30363d; background:#21262d; color:#c9d1d9; cursor:pointer;">0.5x</button>
            <button type="button" data-playback-rate="0.25" style="padding:8px 12px; border-radius:6px; border:1px solid #30363d; background:#21262d; color:#c9d1d9; cursor:pointer;">0.25x</button>
        </div>
    </div>
    """
    phrase_ranges = [
        (float(phrase.start), float(phrase.end))
        for phrase in results.phrases
        if phrase.end > phrase.start
    ]

    if available_tracks:
        track_button_parts = []
        energy_button_parts = []
        panel_parts = []
        available_track_keys = [spec["key"] for spec in available_tracks]

        # Always show the 3 energy-source buttons (disable unavailable ones).
        for spec in track_specs:
            key = spec["key"]
            label = spec["label"]
            is_available = key in available_track_keys
            disabled_attr = "" if is_available else "disabled"
            disabled_style = "" if is_available else "opacity:0.45; cursor:not-allowed;"
            energy_button_parts.append(
                f'<button type="button" id="energy-btn-{key}" data-energy-key="{key}" {disabled_attr} '
                f'style="padding:8px 12px; border-radius:6px; border:1px solid #30363d; '
                f'background:#21262d; color:#c9d1d9; cursor:pointer; {disabled_style}">{label}</button>'
            )

        # Pre-generate standalone energy histogram per overlay source.
        energy_visual_html_by_key: Dict[str, str] = {}
        for e_spec in available_tracks:
            e_key = e_spec["key"]
            e_label = e_spec["label"]
            e_pitch = e_spec["pitch_data"]
            if e_pitch is None:
                continue

            energy_histogram_html = "<p>No energy data available.</p>"
            try:
                e_vals = e_pitch.energy
                if len(e_vals) > 0:
                    energy_hist_path = os.path.join(output_dir, f"energy_histogram_{e_key}.png")
                    plot_energy_distribution(
                        e_vals,
                        energy_hist_path,
                        title=f"{e_label} Energy Distribution (Threshold: {results.config.energy_threshold})",
                        threshold=results.config.energy_threshold
                    )
                    e_hist_rel = os.path.relpath(energy_hist_path, output_dir)
                    energy_histogram_html = f'''
                    <div class="stat-box" style="text-align: center; background: #0d1117; margin-top: 20px;">
                        <h3>Energy Analysis ({e_label})</h3>
                        <img src="{e_hist_rel}" style="max-width: 100%; border-radius: 6px;">
                    </div>
                    '''
            except Exception as e:
                energy_histogram_html = f"<p>Error generating energy histogram for {e_label}: {e}</p>"

            energy_visual_html_by_key[e_key] = energy_histogram_html

        for spec in available_tracks:
            key = spec["key"]
            label = spec["label"]
            audio_id = spec["audio_id"]
            pitch_data = spec["pitch_data"]
            if pitch_data is None:
                continue
            panel_visible = (key == default_track_key)

            track_button_parts.append(
                f'<button type="button" id="track-btn-{key}" data-track-key="{key}" '
                f'style="padding:8px 12px; border-radius:6px; border:1px solid #30363d; '
                f'background:#21262d; color:#c9d1d9; cursor:pointer;">{label}</button>'
            )

            overlay_parts = []
            for e_spec in available_tracks:
                e_key = e_spec["key"]
                e_label = e_spec["label"]
                e_pitch = e_spec["pitch_data"]
                if e_pitch is None:
                    continue
                overlay_visible = (key == default_track_key and e_key == default_energy_key)

                try:
                    scroll_plot_html = create_scrollable_pitch_plot_html(
                        pitch_data,
                        tonic=results.detected_tonic or 0,
                        raga_name=stats.raga_name,
                        audio_element_ids=audio_element_ids,
                        transcription_smoothing_ms=results.config.transcription_smoothing_ms,
                        transcription_min_duration=results.config.transcription_min_duration,
                        transcription_derivative_threshold=results.config.transcription_derivative_threshold,
                        show_rms_overlay=getattr(results.config, 'show_rms_overlay', True),
                        overlay_energy=e_pitch.energy,
                        overlay_timestamps=e_pitch.timestamps,
                        overlay_label=f"{e_label} RMS",
                        phrase_ranges=phrase_ranges,
                    )
                except Exception as e:
                    scroll_plot_html = (
                        f"<p>Error generating scrollable plot for pitch={label}, overlay={e_label}: {e}</p>"
                    )

                overlay_parts.append(f'''
                <div class="energy-overlay-panel" id="overlay-panel-{key}-{e_key}" data-track-key="{key}" data-energy-key="{e_key}" style="display: {'block' if overlay_visible else 'none'};">
                    <h3>Pitch: {label} | Amplitude Overlay: {e_label}</h3>
                    <p style="font-size: 0.9em; color: #8b949e;">Pitch stays on <strong>{label}</strong>. Energy overlay is from <strong>{e_label}</strong>. Phrase spans are highlighted in yellow.</p>
                    {scroll_plot_html}
                </div>
                ''')

            panel_parts.append(f'''
            <div class="track-analysis-panel" id="track-panel-{key}" data-track-key="{key}" style="display: {'block' if panel_visible else 'none'};">
                {"".join(overlay_parts)}
            </div>
            ''')

        if len(track_button_parts) > 1:
            track_buttons_html = f'''
            <p style="font-size: 0.86em; color: #8b949e; margin: 4px 0 6px 0;">Pitch Track (what contour is shown):</p>
            <div id="track-analysis-selector" style="display:flex; flex-wrap:wrap; gap:8px; margin: 10px 0 16px 0;">
                {"".join(track_button_parts)}
            </div>
            '''
        if energy_button_parts:
            energy_buttons_html = f'''
            <p style="font-size: 0.86em; color: #8b949e; margin: 0 0 6px 0;">Amplitude Overlay Source (independent):</p>
            <div id="energy-overlay-selector" style="display:flex; flex-wrap:wrap; gap:8px; margin: 0 0 16px 0;">
                {"".join(energy_button_parts)}
            </div>
            '''
        track_panels_html = "".join(panel_parts)

        track_to_audio = {spec["key"]: spec["audio_id"] for spec in available_tracks}
        track_switch_js = f"""
        <script>
        document.addEventListener("DOMContentLoaded", function() {{
            var availableTrackKeys = {json.dumps(available_track_keys)};
            var availableEnergyKeys = {json.dumps(available_track_keys)};
            var trackToAudio = {json.dumps(track_to_audio)};
            var audioIds = {json.dumps(audio_element_ids)};
            var defaultTrack = {json.dumps(default_track_key)};
            var defaultEnergy = {json.dumps(default_energy_key)};
            var currentTrack = defaultTrack;
            var currentEnergy = defaultEnergy;
            var playbackRates = [1, 0.5, 0.25];

            function setActiveTrackButton(trackKey) {{
                availableTrackKeys.forEach(function(key) {{
                    var btn = document.getElementById("track-btn-" + key);
                    if (!btn) return;
                    if (key === trackKey) {{
                        btn.style.background = "#e94560";
                        btn.style.color = "#ffffff";
                        btn.style.borderColor = "#e94560";
                    }} else {{
                        btn.style.background = "#21262d";
                        btn.style.color = "#c9d1d9";
                        btn.style.borderColor = "#30363d";
                    }}
                }});
            }}

            function setActiveEnergyButton(energyKey) {{
                availableEnergyKeys.forEach(function(key) {{
                    var btn = document.getElementById("energy-btn-" + key);
                    if (!btn || btn.disabled) return;
                    if (key === energyKey) {{
                        btn.style.background = "#238636";
                        btn.style.color = "#ffffff";
                        btn.style.borderColor = "#238636";
                    }} else {{
                        btn.style.background = "#21262d";
                        btn.style.color = "#c9d1d9";
                        btn.style.borderColor = "#30363d";
                    }}
                }});
            }}

            function showOverlay(trackKey, energyKey) {{
                var resolvedEnergy = availableEnergyKeys.indexOf(energyKey) >= 0 ? energyKey : availableEnergyKeys[0];
                availableEnergyKeys.forEach(function(eKey) {{
                    var panel = document.getElementById("overlay-panel-" + trackKey + "-" + eKey);
                    if (panel) {{
                        panel.style.display = (eKey === resolvedEnergy) ? "block" : "none";
                    }}
                }});
                currentEnergy = resolvedEnergy;
                setActiveEnergyButton(currentEnergy);
            }}

            function showTrackPanel(trackKey) {{
                availableTrackKeys.forEach(function(key) {{
                    var panel = document.getElementById("track-panel-" + key);
                    if (panel) {{
                        panel.style.display = (key === trackKey) ? "block" : "none";
                    }}
                }});
                currentTrack = trackKey;
                setActiveTrackButton(trackKey);
                showOverlay(trackKey, currentEnergy);
            }}

            function setActivePlaybackRateButton(rate) {{
                playbackRates.forEach(function(candidate) {{
                    var selector = '#playback-rate-controls button[data-playback-rate="' + candidate + '"]';
                    var btn = document.querySelector(selector);
                    if (!btn) return;
                    if (Math.abs(candidate - rate) < 1e-9) {{
                        btn.style.background = "#e3b341";
                        btn.style.color = "#0d1117";
                        btn.style.borderColor = "#e3b341";
                    }} else {{
                        btn.style.background = "#21262d";
                        btn.style.color = "#c9d1d9";
                        btn.style.borderColor = "#30363d";
                    }}
                }});
            }}

            function applyPlaybackRate(rate) {{
                audioIds.forEach(function(audioId) {{
                    var audioEl = document.getElementById(audioId);
                    if (audioEl) {{
                        audioEl.playbackRate = rate;
                    }}
                }});
                setActivePlaybackRateButton(rate);
            }}

            // Keep audio playback exclusive: only one track can play at a time.
            audioIds.forEach(function(activeId) {{
                var activeEl = document.getElementById(activeId);
                if (!activeEl) return;
                activeEl.addEventListener("play", function() {{
                    audioIds.forEach(function(otherId) {{
                        if (otherId === activeId) return;
                        var otherEl = document.getElementById(otherId);
                        if (otherEl && !otherEl.paused) {{
                            otherEl.pause();
                        }}
                    }});
                }});
            }});

            availableTrackKeys.forEach(function(key) {{
                var btn = document.getElementById("track-btn-" + key);
                if (btn) {{
                    btn.addEventListener("click", function() {{
                        showTrackPanel(key);
                    }});
                }}
            }});

            availableEnergyKeys.forEach(function(key) {{
                var btn = document.getElementById("energy-btn-" + key);
                if (btn && !btn.disabled) {{
                    btn.addEventListener("click", function() {{
                        showOverlay(currentTrack, key);
                    }});
                }}
            }});

            var rateButtons = document.querySelectorAll("#playback-rate-controls button[data-playback-rate]");
            rateButtons.forEach(function(btn) {{
                btn.addEventListener("click", function() {{
                    var rawRate = btn.getAttribute("data-playback-rate");
                    if (!rawRate) return;
                    var parsedRate = parseFloat(rawRate);
                    if (!isFinite(parsedRate)) return;
                    applyPlaybackRate(parsedRate);
                }});
            }});

            applyPlaybackRate(1.0);

            if (defaultTrack) {{
                showTrackPanel(defaultTrack);
            }}
        }});
        </script>
        """
    else:
        track_panels_html = "<p>No pitch data available for interactive analysis.</p>"

    # Transcription html
    transcription_html = _generate_transcription_section(
        results.notes,
        results.phrases,
        results.detected_tonic or 0,
        audio_element_ids=None,
    )

    # Phrase karaoke section (top interactive view)
    karaoke_html = ""
    if results.phrases:
        karaoke_html = _generate_karaoke_section(
            results.phrases,
            results.detected_tonic or 0,
            audio_element_ids=audio_element_ids,
        )

    # Pattern Analysis HTML (Motifs + Aaroh/Avroh)
    pattern_html = ""
    if stats.pattern_analysis:
        p = stats.pattern_analysis
        
        # Motifs
        motif_items = []
        if 'common_motifs' in p:
            for m in p['common_motifs']:
                motif_items.append(f"<li><strong>{m['pattern']}</strong>: {m['count']} occurrences</li>")
        
        # Aaroh RUns
        aaroh_items = []
        if 'aaroh_stats' in p:
            for run, count in p['aaroh_stats'].items():
                aaroh_items.append(f"<li><strong>{run}</strong>: {count}</li>")

        # Avroh Runs
        avroh_items = []
        if 'avroh_stats' in p:
            for run, count in p['avroh_stats'].items():
                avroh_items.append(f"<li><strong>{run}</strong>: {count}</li>")

        # Aaroh/Avroh checker
        checker_html = ""
        checker = p.get("aaroh_avroh_checker")
        reference = p.get("aaroh_avroh_reference", {})
        if checker:
            note_rows = []
            for row in checker.get("note_evidence", []):
                if not row.get("sufficient_evidence"):
                    continue
                note_rows.append(
                    "<tr>"
                    f"<td>{row.get('note', '')}</td>"
                    f"<td>{row.get('asc_edges', 0)}</td>"
                    f"<td>{row.get('desc_edges', 0)}</td>"
                    f"<td>{row.get('asc_ratio', 0.0):.2f}</td>"
                    f"<td>{row.get('desc_ratio', 0.0):.2f}</td>"
                    f"<td>{row.get('expected_asc', 0.0):.1f}</td>"
                    f"<td>{row.get('expected_desc', 0.0):.1f}</td>"
                    "</tr>"
                )

            ref_name = reference.get("matched_name", stats.raga_name)
            ref_aaroh = reference.get("aaroh_raw", "N/A")
            ref_avroh = reference.get("avroh_raw", "N/A")

            checker_html = f"""
            <div class="stat-box">
                <strong>Aaroh/Avroh Checker</strong>
                <p><strong>Reference:</strong> {ref_name}</p>
                <p><strong>Aaroh:</strong> {ref_aaroh}</p>
                <p><strong>Avroh:</strong> {ref_avroh}</p>
                <p><strong>Score:</strong> {checker.get('score', 0.0):.3f}
                   ({checker.get('matched_checks', 0)}/{checker.get('total_checks', 0)} checks)</p>
                <p><strong>Missing Aaroh:</strong> {", ".join(checker.get('missing_aaroh', [])) or "None"}</p>
                <p><strong>Missing Avroh:</strong> {", ".join(checker.get('missing_avroh', [])) or "None"}</p>
                <p><strong>Unexpected Aaroh:</strong> {", ".join(checker.get('unexpected_aaroh', [])) or "None"}</p>
                <p><strong>Unexpected Avroh:</strong> {", ".join(checker.get('unexpected_avroh', [])) or "None"}</p>
                <div style="overflow-x:auto;">
                    <table style="width:100%; border-collapse: collapse; margin-top: 8px;">
                        <thead>
                            <tr>
                                <th style="text-align:left; border-bottom:1px solid #30363d;">Note</th>
                                <th style="text-align:left; border-bottom:1px solid #30363d;">Asc Edges</th>
                                <th style="text-align:left; border-bottom:1px solid #30363d;">Desc Edges</th>
                                <th style="text-align:left; border-bottom:1px solid #30363d;">Asc Ratio</th>
                                <th style="text-align:left; border-bottom:1px solid #30363d;">Desc Ratio</th>
                                <th style="text-align:left; border-bottom:1px solid #30363d;">Exp Asc</th>
                                <th style="text-align:left; border-bottom:1px solid #30363d;">Exp Desc</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(note_rows) or "<tr><td colspan='7'>No notes with sufficient directional evidence.</td></tr>"}
                        </tbody>
                    </table>
                </div>
            </div>
            """
                
        pattern_html = f"""
        <section id="patterns">
            <h2>Pattern Analysis</h2>
            <div class="stats-grid">
                 <div class="stat-box"><strong>Common Motifs</strong><ul>{"".join(motif_items) or "<li>None</li>"}</ul></div>
                 <div class="stat-box"><strong>Aaroh (Ascending) Runs</strong><ul>{"".join(aaroh_items) or "<li>None</li>"}</ul></div>
                 <div class="stat-box"><strong>Avroh (Descending) Runs</strong><ul>{"".join(avroh_items) or "<li>None</li>"}</ul></div>
                 {checker_html}
            </div>
        </section>
        """
        
    # Correction Stats HTML
    correction_html = ""
    if stats.correction_summary:
        s = stats.correction_summary
        correction_html = f"""
        <section id="correction">
            <h2>Raga Correction Summary</h2>
            <div class="stats-grid">
                <div class="stat-box"><strong>Total Notes:</strong> {s.get('total', 0)}</div>
                <div class="stat-box"><strong>Unchanged:</strong> {s.get('unchanged', 0)}</div>
                <div class="stat-box"><strong>Corrected (Snapped):</strong> {s.get('corrected', 0)}</div>
                <div class="stat-box"><strong>Discarded:</strong> {s.get('discarded', 0)}</div>
                <div class="stat-box"><strong>Valid Notes Remaining:</strong> {s.get('remaining', 0)}</div>
            </div>
        </section>
        """
    # GMM Analysis HTML
    gmm_html = ""
    if 'gmm_overlay' in results.plot_paths:
        gmm_rel = os.path.relpath(results.plot_paths['gmm_overlay'], output_dir)
        gmm_html = f"""
        <section id="gmm-analysis">
            <h2>Microtonal Analysis (GMM)</h2>
            <div class="viz-container">
                <img src="{gmm_rel}" alt="GMM Analysis" style="width:100%; max-width:800px;">
                <p style="font-size: 0.9em; color: #8b949e; text-align: center;">Gaussian Mixture Model fit to histogram peaks for microtonal precision.</p>
            </div>
        </section>
        """

    # Images HTML
    images_html = ""
    tm_rel = os.path.relpath(stats.transition_matrix_path, output_dir)
    # Pitch contour static image (optional, since we have scrollable one now)
    # But user might still want the standard one.
    pp_rel = os.path.relpath(stats.pitch_plot_path, output_dir)
    images_html = f"""
    <section id="visualizations">
        <h2>Detailed Visualizations</h2>
        <div class="viz-container">
            <h3>Standard Pitch Contour</h3>
            <img src="{pp_rel}" alt="Pitch Contour" style="width:100%; max-width:1000px;">
        </div>
        <div class="viz-container">
            <h3>Transition Matrix</h3>
            <img src="{tm_rel}" alt="Transition Matrix" style="width:100%; max-width:600px;">
        </div>
    </section>
    """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Raga Analysis Report: {stats.raga_name}</title>
        
        <style>
            {_get_css_styles()}
            /* Custom styles for scroll plot */
            .scroll-plot-wrapper {{
                transition: box-shadow 0.3s ease;
            }}
            .scroll-plot-wrapper:hover {{
                box-shadow: 0 0 15px rgba(233, 69, 96, 0.2);
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>Raga Analysis Report</h1>
            <p class="subtitle">{stats.raga_name} (Tonic: {stats.tonic})</p>
        </header>

        <main>
            <!-- Audio Tracks Section with IDs -->
            <section id="audio-tracks">
                <h2>Audio Tracks & Interactive Analysis</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-bottom: 20px;">
                    <div>
                        <p><strong>Original</strong></p>
                        <audio id="{original_id}" controls style="width:100%">{original_sources}</audio>
                    </div>
                    <div>
                        <p><strong>Vocals Stem</strong></p>
                        <audio id="{vocals_id}" controls style="width:100%">{vocals_sources}</audio>
                    </div>
                    <div>
                        <p><strong>Accompaniment</strong></p>
                        <audio id="{accomp_id}" controls style="width:100%">{accomp_sources}</audio>
                    </div>
                </div>
                {playback_controls_html}
                
                <p style="font-size: 0.9em; color: #8b949e;">
                    Analysis source for note detection: <strong>{analysis_source_label}</strong>.
                    Pitch track and amplitude overlay are controlled by the selectors below. Audio playback only moves the playhead/cursor.
                </p>
                {track_buttons_html}
                {energy_buttons_html}
                {track_panels_html}
                {karaoke_html}
            </section>

            {correction_html}

            {transcription_html}

            {pattern_html}

            {gmm_html}

            {images_html}
        </main>

        <footer>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </footer>
        {track_switch_js}
    </body>
    </html>
    """
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(html_content)
        
    return report_path


def _serialize_notes(notes: List['Note'], tonic: int) -> str:
    """Serialize notes to JSON for JS consumption."""
    data = []
    # If tonic is passed, verify it's int
    if isinstance(tonic, str):
        # Fallback if somehow string passed
        tonic = 0 

    for note in notes:
        data.append({
            'start': round(note.start, 3),
            'end': round(note.end, 3),
            'sargam': note.sargam or "?"  # Ensure sargam is populated
        })
    return json.dumps(data)
