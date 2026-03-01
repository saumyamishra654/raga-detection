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
from pathlib import Path
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
    transcription_derivative_timestamps: Optional[np.ndarray] = None
    transcription_derivative_values: Optional[np.ndarray] = None
    transcription_derivative_voiced_mask: Optional[np.ndarray] = None
    
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


def plot_note_duration_histogram(
    notes: List[Note],
    output_path: str,
    title: str = "Note Duration Distribution",
) -> str:
    """
    Plot histogram of transcribed note durations (seconds).

    Args:
        notes: Transcribed notes.
        output_path: Path to save PNG.
        title: Plot title.

    Returns:
        Path to saved figure.
    """
    durations = np.asarray(
        [float(note.duration) for note in notes if float(note.duration) > 0.0],
        dtype=float,
    )

    plt.figure(figsize=(10, 5))

    if durations.size == 0:
        plt.text(
            0.5, 0.5,
            "No valid note durations available",
            ha="center",
            va="center",
            fontsize=12,
            color="#8b949e",
            transform=plt.gca().transAxes,
        )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Count")
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()
        return output_path

    bins = int(np.clip(np.sqrt(durations.size) * 2, 12, 60))
    plt.hist(durations, bins=bins, color="#4ea1ff", edgecolor="#1f3b5c", alpha=0.85)

    mean_dur = float(np.mean(durations))
    median_dur = float(np.median(durations))
    plt.axvline(mean_dur, color="#ff7b72", linestyle="--", linewidth=1.2, label=f"Mean: {mean_dur:.3f}s")
    plt.axvline(median_dur, color="#f2cc60", linestyle="-.", linewidth=1.2, label=f"Median: {median_dur:.3f}s")

    plt.xlabel("Duration (seconds)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return output_path


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
    bias_cents: Optional[float] = None,
):
    """
    Plot pitch contour with sargam lines.
    Slightly adapted from user snippet to save to file.
    """
    plt.figure(figsize=figsize)

    bias_semitones = (float(bias_cents) / 100.0) if bias_cents is not None else 0.0
    pitch_plot_values = np.asarray(pitch_values, dtype=float) - bias_semitones

    # Phrase regions: translucent yellow blocks with clear boundaries/labels.
    if phrase_ranges:
        y_top_for_labels = float(np.nanmax(pitch_plot_values)) + 0.4
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
    plt.plot(time_axis, pitch_plot_values, label='Pitch', color='blue', alpha=0.6, linewidth=1, zorder=2)
    
    # Clean tonic
    tonic_val = _parse_tonic(tonic) if isinstance(tonic, str) else int(tonic)
    
    # Sargam lines
    start_midi = int(np.nanmin(pitch_plot_values)) - 1
    end_midi = int(np.nanmax(pitch_plot_values)) + 1
    
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
    """Generate detection audio players section for available tracks only."""
    config = results.config

    original_audio_path = _require_audio_path(config)
    original_rel = os.path.relpath(original_audio_path, config.stem_dir)
    original_local = os.path.basename(original_audio_path)

    tracks: List[Tuple[str, str, str]] = [
        ("Original", "detection-original-player", _build_audio_source_tags(original_local, original_rel))
    ]

    vocals_path = config.vocals_path
    if os.path.isfile(vocals_path):
        vocals_rel = os.path.basename(vocals_path)
        tracks.append(
            ("Vocals", "detection-vocals-player", _build_audio_source_tags(vocals_rel, vocals_rel))
        )

    accompaniment_path = config.accompaniment_path
    if os.path.isfile(accompaniment_path):
        accomp_rel = os.path.basename(accompaniment_path)
        tracks.append(
            ("Accompaniment", "detection-accomp-player", _build_audio_source_tags(accomp_rel, accomp_rel))
        )

    track_cards = "".join(
        f'''
            <div>
                <p><strong>{escape(label)}</strong></p>
                <audio id="{audio_id}" controls style="width:100%">{sources}</audio>
            </div>
        '''
        for label, audio_id, sources in tracks
    )
    audio_ids = [audio_id for _, audio_id, _ in tracks]

    return f'''
    <section id="audio-tracks">
        <h2>Audio Tracks</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
            {track_cards}
        </div>
        <script>
            (function() {{
                var audioIds = {json.dumps(audio_ids)};
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


def _build_transcription_editor_payload(
    notes: List[Note],
    phrases: List[Phrase],
    tonic: int,
) -> Dict[str, Any]:
    serialized_notes: List[Dict[str, Any]] = []
    serialized_phrases: List[Dict[str, Any]] = []

    source_phrases = phrases
    if not source_phrases and notes:
        source_phrases = [Phrase(notes=list(notes))]

    note_counter = 1
    phrase_counter = 1
    for phrase in source_phrases:
        phrase_id = f"p{phrase_counter:04d}"
        phrase_counter += 1
        phrase_note_ids: List[str] = []
        phrase_notes = sorted(list(phrase.notes), key=lambda n: (n.start, n.end, n.pitch_midi))
        for note in phrase_notes:
            note_id = f"n{note_counter:05d}"
            note_counter += 1
            label = _strip_octave_markers(note.sargam or "")
            if not label:
                tonic_pc = int(round(tonic)) % 12
                label = OFFSET_TO_SARGAM.get(int(round(note.pitch_midi - tonic_pc)) % 12, "")
            pitch_class = int(round(note.pitch_midi)) % 12
            serialized_notes.append(
                {
                    "id": note_id,
                    "start": float(note.start),
                    "end": float(note.end),
                    "pitch_midi": float(note.pitch_midi),
                    "pitch_hz": float(note.pitch_hz),
                    "raw_pitch_midi": float(getattr(note, "raw_pitch_midi", note.pitch_midi)),
                    "snapped_pitch_midi": float(getattr(note, "snapped_pitch_midi", note.pitch_midi)),
                    "corrected_pitch_midi": float(getattr(note, "corrected_pitch_midi", note.pitch_midi)),
                    "rendered_pitch_midi": float(getattr(note, "pitch_midi", note.pitch_midi)),
                    "confidence": float(note.confidence),
                    "energy": float(getattr(note, "energy", 0.0)),
                    "sargam": label,
                    "pitch_class": pitch_class,
                }
            )
            phrase_note_ids.append(note_id)

        if phrase_note_ids:
            starts = [item["start"] for item in serialized_notes if item["id"] in phrase_note_ids]
            ends = [item["end"] for item in serialized_notes if item["id"] in phrase_note_ids]
            serialized_phrases.append(
                {
                    "id": phrase_id,
                    "start": min(starts) if starts else float(phrase.start),
                    "end": max(ends) if ends else float(phrase.end),
                    "note_ids": phrase_note_ids,
                }
            )

    tonic_pc = int(round(tonic)) % 12
    base_sa_midi = tonic_pc + 60
    sargam_options: List[Dict[str, Any]] = []
    for offset in range(12):
        sargam_options.append(
            {
                "offset": offset,
                "label": OFFSET_TO_SARGAM.get(offset, f"?{offset}"),
                "midi": int(base_sa_midi + offset),
            }
        )

    return {
        "tonic": tonic_pc,
        "notes": serialized_notes,
        "phrases": serialized_phrases,
        "sargam_options": sargam_options,
    }


def _generate_transcription_editor_section(
    notes: List[Note],
    phrases: List[Phrase],
    tonic: int,
) -> str:
    section_id = f"transcription-editor-{uuid.uuid4().hex[:8]}"
    payload_json = json.dumps(_build_transcription_editor_payload(notes, phrases, tonic))
    template = """
    <section id="__SECTION_ID__" style="margin-top: 24px;">
        <h2>Transcription Editor (Experimental)</h2>
        <p style="font-size: 0.9em; color: #8b949e; margin-top: -2px;">
            Use plot point/range selection as context, then adjust notes/phrases. Saves are versioned and produce edited reports.
        </p>
        <style>
        #__SECTION_ID__ .editor-shell {
            border: 1px solid #30363d;
            border-radius: 10px;
            background: #0d1117;
            overflow: hidden;
        }
        #__SECTION_ID__ .editor-toolbar {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 10px;
            border-bottom: 1px solid #30363d;
            background: #11161d;
        }
        #__SECTION_ID__ .editor-toolbar button,
        #__SECTION_ID__ .editor-toolbar select,
        #__SECTION_ID__ .editor-toolbar input {
            border: 1px solid #30363d;
            border-radius: 6px;
            background: #21262d;
            color: #c9d1d9;
            padding: 6px 10px;
            font-size: 12px;
        }
        #__SECTION_ID__ .editor-toolbar button {
            cursor: pointer;
        }
        #__SECTION_ID__ .editor-toolbar button:disabled {
            opacity: 0.5;
            cursor: default;
        }
        #__SECTION_ID__ .editor-status {
            margin-left: auto;
            color: #8b949e;
            font-size: 12px;
            align-self: center;
        }
        #__SECTION_ID__ .editor-default {
            color: #8b949e;
            font-size: 12px;
            align-self: center;
            border: 1px solid #30363d;
            border-radius: 999px;
            padding: 4px 10px;
            background: #0d1117;
        }
        #__SECTION_ID__ .editor-grid {
            display: grid;
            gap: 12px;
            grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr);
            padding: 12px;
        }
        #__SECTION_ID__ .editor-card {
            border: 1px solid #30363d;
            border-radius: 8px;
            background: #11161d;
            padding: 10px;
        }
        #__SECTION_ID__ .editor-card h3 {
            margin: 0 0 8px 0;
            font-size: 14px;
            color: #f0f6fc;
        }
        #__SECTION_ID__ .editor-field-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 8px 0;
        }
        #__SECTION_ID__ .editor-field-row label {
            font-size: 12px;
            color: #8b949e;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        #__SECTION_ID__ .editor-field-row input,
        #__SECTION_ID__ .editor-field-row select {
            border: 1px solid #30363d;
            border-radius: 6px;
            background: #0d1117;
            color: #c9d1d9;
            padding: 5px 8px;
            font-size: 12px;
            min-width: 70px;
        }
        #__SECTION_ID__ .editor-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }
        #__SECTION_ID__ .editor-actions button {
            border: 1px solid #30363d;
            border-radius: 6px;
            background: #21262d;
            color: #c9d1d9;
            padding: 6px 10px;
            font-size: 12px;
            cursor: pointer;
        }
        #__SECTION_ID__ .editor-phrase-list {
            max-height: 420px;
            overflow-y: auto;
            display: grid;
            gap: 8px;
        }
        #__SECTION_ID__ .editor-phrase-row {
            border: 1px solid #30363d;
            border-radius: 8px;
            background: #0d1117;
            padding: 8px;
        }
        #__SECTION_ID__ .editor-phrase-row.active {
            border-color: #58a6ff;
            box-shadow: 0 0 0 1px rgba(88, 166, 255, 0.25) inset;
        }
        #__SECTION_ID__ .editor-phrase-meta {
            display: flex;
            gap: 8px;
            align-items: center;
            font-size: 11px;
            color: #8b949e;
            margin-bottom: 6px;
        }
        #__SECTION_ID__ .editor-note-chip-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        #__SECTION_ID__ .editor-note-chip {
            border: 1px solid #30363d;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 12px;
            background: #161b22;
            color: #c9d1d9;
            cursor: pointer;
        }
        #__SECTION_ID__ .editor-note-chip.active {
            background: #f2cc60;
            color: #111827;
            border-color: #f2cc60;
        }
        @media (max-width: 1100px) {
            #__SECTION_ID__ .editor-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        <div class="editor-shell">
            <div class="editor-toolbar">
                <button type="button" id="__SECTION_ID__-undo">Undo</button>
                <button type="button" id="__SECTION_ID__-redo">Redo</button>
                <button type="button" id="__SECTION_ID__-save" style="background:#238636; border-color:#238636; color:#ffffff;">Save</button>
                <button type="button" id="__SECTION_ID__-save-new">Save as New Version</button>
                <button type="button" id="__SECTION_ID__-reset">Reset</button>
                <select id="__SECTION_ID__-versions" title="Saved versions"></select>
                <button type="button" id="__SECTION_ID__-load-version">Load Version</button>
                <button type="button" id="__SECTION_ID__-delete-version">Delete Version</button>
                <button type="button" id="__SECTION_ID__-regenerate-version">Regenerate Report</button>
                <button type="button" id="__SECTION_ID__-set-default-selected">Set Selected Default</button>
                <button type="button" id="__SECTION_ID__-set-default-original">Set Original Default</button>
                <span class="editor-default" id="__SECTION_ID__-default-label">Default: Original</span>
                <span class="editor-status" id="__SECTION_ID__-status">Editor ready.</span>
            </div>
            <div class="editor-grid">
                <div class="editor-card">
                    <h3>Selection + Actions</h3>
                    <div id="__SECTION_ID__-selection" style="font-size:12px; color:#8b949e; margin-bottom: 8px;">
                        Plot selection: none
                    </div>
                    <div id="__SECTION_ID__-stats" style="font-size:12px; color:#8b949e;">
                        Notes: 0 | Phrases: 0
                    </div>

                    <div class="editor-field-row" style="margin-top: 10px;">
                        <label>Sargam
                            <select id="__SECTION_ID__-add-sargam"></select>
                        </label>
                        <label>Octave
                            <select id="__SECTION_ID__-add-octave"></select>
                        </label>
                        <label>MIDI
                            <input id="__SECTION_ID__-add-midi" type="text" value="-" readonly>
                        </label>
                    </div>
                    <div class="editor-actions">
                        <button type="button" id="__SECTION_ID__-add-note-range">Add Note From Range</button>
                        <button type="button" id="__SECTION_ID__-delete-range">Delete Notes In Range</button>
                        <button type="button" id="__SECTION_ID__-delete-note">Delete Selected Note</button>
                    </div>

                    <div class="editor-field-row" style="margin-top: 12px;">
                        <label>Phrase start
                            <input id="__SECTION_ID__-phrase-start" type="number" step="0.001">
                        </label>
                        <label>Phrase end
                            <input id="__SECTION_ID__-phrase-end" type="number" step="0.001">
                        </label>
                        <button type="button" id="__SECTION_ID__-apply-phrase-bounds">Apply Phrase Bounds (Snapped)</button>
                    </div>
                    <div class="editor-actions">
                        <button type="button" id="__SECTION_ID__-merge-phrases">Merge Checked Phrases</button>
                        <button type="button" id="__SECTION_ID__-split-phrase">Split Phrase At Selected Note</button>
                    </div>

                    <div class="editor-field-row" style="margin-top: 12px;">
                        <label>Note start
                            <input id="__SECTION_ID__-note-start" type="number" step="0.001">
                        </label>
                        <label>Note end
                            <input id="__SECTION_ID__-note-end" type="number" step="0.001">
                        </label>
                        <button type="button" id="__SECTION_ID__-apply-note-bounds">Resize Selected Note</button>
                    </div>
                </div>
                <div class="editor-card">
                    <h3>Phrase Notes</h3>
                    <div class="editor-phrase-list" id="__SECTION_ID__-phrase-list"></div>
                </div>
            </div>
        </div>
        <script>
        (function() {
            var root = document.getElementById("__SECTION_ID__");
            if (!root) return;
            var initialPayload = __EDITOR_PAYLOAD__;
            var selectionEl = document.getElementById("__SECTION_ID__-selection");
            var statsEl = document.getElementById("__SECTION_ID__-stats");
            var phraseListEl = document.getElementById("__SECTION_ID__-phrase-list");
            var statusEl = document.getElementById("__SECTION_ID__-status");
            var undoBtn = document.getElementById("__SECTION_ID__-undo");
            var redoBtn = document.getElementById("__SECTION_ID__-redo");
            var saveBtn = document.getElementById("__SECTION_ID__-save");
            var saveNewBtn = document.getElementById("__SECTION_ID__-save-new");
            var resetBtn = document.getElementById("__SECTION_ID__-reset");
            var versionsSelect = document.getElementById("__SECTION_ID__-versions");
            var loadVersionBtn = document.getElementById("__SECTION_ID__-load-version");
            var deleteVersionBtn = document.getElementById("__SECTION_ID__-delete-version");
            var regenerateBtn = document.getElementById("__SECTION_ID__-regenerate-version");
            var setDefaultSelectedBtn = document.getElementById("__SECTION_ID__-set-default-selected");
            var setDefaultOriginalBtn = document.getElementById("__SECTION_ID__-set-default-original");
            var defaultLabelEl = document.getElementById("__SECTION_ID__-default-label");
            var addSargamSelect = document.getElementById("__SECTION_ID__-add-sargam");
            var addOctaveSelect = document.getElementById("__SECTION_ID__-add-octave");
            var addMidiInput = document.getElementById("__SECTION_ID__-add-midi");
            var addNoteRangeBtn = document.getElementById("__SECTION_ID__-add-note-range");
            var deleteRangeBtn = document.getElementById("__SECTION_ID__-delete-range");
            var deleteNoteBtn = document.getElementById("__SECTION_ID__-delete-note");
            var phraseStartInput = document.getElementById("__SECTION_ID__-phrase-start");
            var phraseEndInput = document.getElementById("__SECTION_ID__-phrase-end");
            var applyPhraseBoundsBtn = document.getElementById("__SECTION_ID__-apply-phrase-bounds");
            var mergePhrasesBtn = document.getElementById("__SECTION_ID__-merge-phrases");
            var splitPhraseBtn = document.getElementById("__SECTION_ID__-split-phrase");
            var noteStartInput = document.getElementById("__SECTION_ID__-note-start");
            var noteEndInput = document.getElementById("__SECTION_ID__-note-end");
            var applyNoteBoundsBtn = document.getElementById("__SECTION_ID__-apply-note-bounds");

            function deepClone(v) {
                return JSON.parse(JSON.stringify(v));
            }

            function asNumber(v, fallback) {
                var parsed = Number(v);
                if (!isFinite(parsed)) return fallback;
                return parsed;
            }

            function formatSec(v) {
                var n = Number(v);
                if (!isFinite(n)) return "n/a";
                return n.toFixed(3) + "s";
            }

            function escapeHtml(raw) {
                return String(raw || "")
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#39;");
            }

            var noteSeq = 0;
            var phraseSeq = 0;
            function initCountersFromState(snapshot) {
                noteSeq = 0;
                phraseSeq = 0;
                (snapshot.notes || []).forEach(function(note) {
                    var match = String(note.id || "").match(/(\\d+)$/);
                    if (match) noteSeq = Math.max(noteSeq, Number(match[1]));
                });
                (snapshot.phrases || []).forEach(function(phrase) {
                    var match = String(phrase.id || "").match(/(\\d+)$/);
                    if (match) phraseSeq = Math.max(phraseSeq, Number(match[1]));
                });
            }
            function nextNoteId() {
                noteSeq += 1;
                return "n" + String(noteSeq).padStart(5, "0");
            }
            function nextPhraseId() {
                phraseSeq += 1;
                return "p" + String(phraseSeq).padStart(4, "0");
            }

            function normalizeState(rawState) {
                var draft = deepClone(rawState || {});
                var notesRaw = Array.isArray(draft.notes) ? draft.notes : [];
                var phrasesRaw = Array.isArray(draft.phrases) ? draft.phrases : [];

                var notes = [];
                var noteById = {};
                notesRaw.forEach(function(note, idx) {
                    var noteId = String((note && note.id) || "").trim() || ("n" + String(idx + 1).padStart(5, "0"));
                    if (noteById[noteId]) return;
                    var start = asNumber(note.start, 0);
                    var end = asNumber(note.end, start + 0.001);
                    if (end < start) {
                        var tmp = start;
                        start = end;
                        end = tmp;
                    }
                    if (end <= start) end = start + 0.001;
                    var pitchMidi = asNumber(note.pitch_midi, 60);
                    var pitchHz = asNumber(note.pitch_hz, 440 * Math.pow(2, (pitchMidi - 69) / 12));
                    if (!isFinite(pitchHz) || pitchHz <= 0) {
                        pitchHz = 440 * Math.pow(2, (pitchMidi - 69) / 12);
                    }
                    var confidence = asNumber(note.confidence, 0.8);
                    var energy = asNumber(note.energy, 0.0);
                    var pitchClass = isFinite(Number(note.pitch_class))
                        ? ((Math.round(Number(note.pitch_class)) % 12) + 12) % 12
                        : ((Math.round(pitchMidi) % 12) + 12) % 12;
                    var normalized = {
                        id: noteId,
                        start: start,
                        end: end,
                        pitch_midi: pitchMidi,
                        pitch_hz: pitchHz,
                        raw_pitch_midi: asNumber(note.raw_pitch_midi, pitchMidi),
                        snapped_pitch_midi: asNumber(note.snapped_pitch_midi, pitchMidi),
                        corrected_pitch_midi: asNumber(note.corrected_pitch_midi, pitchMidi),
                        rendered_pitch_midi: asNumber(note.rendered_pitch_midi, pitchMidi),
                        confidence: confidence,
                        energy: energy,
                        sargam: String((note && note.sargam) || "").replace(/[\\u00B7'`’]+/g, "").trim(),
                        pitch_class: pitchClass
                    };
                    notes.push(normalized);
                    noteById[noteId] = normalized;
                });

                notes.sort(function(a, b) {
                    if (a.start !== b.start) return a.start - b.start;
                    if (a.end !== b.end) return a.end - b.end;
                    return a.id.localeCompare(b.id);
                });

                var assigned = {};
                var phrases = [];
                phrasesRaw.forEach(function(phrase, idx) {
                    var phraseId = String((phrase && phrase.id) || "").trim() || ("p" + String(idx + 1).padStart(4, "0"));
                    var seenLocal = {};
                    var noteIds = [];
                    (Array.isArray(phrase.note_ids) ? phrase.note_ids : []).forEach(function(rawId) {
                        var noteId = String(rawId || "").trim();
                        if (!noteById[noteId]) return;
                        if (assigned[noteId]) return;
                        if (seenLocal[noteId]) return;
                        seenLocal[noteId] = true;
                        assigned[noteId] = true;
                        noteIds.push(noteId);
                    });
                    if (!noteIds.length) return;
                    noteIds.sort(function(a, b) {
                        return noteById[a].start - noteById[b].start;
                    });
                    var start = noteById[noteIds[0]].start;
                    var end = noteById[noteIds[0]].end;
                    noteIds.forEach(function(noteId) {
                        start = Math.min(start, noteById[noteId].start);
                        end = Math.max(end, noteById[noteId].end);
                    });
                    phrases.push({
                        id: phraseId,
                        start: start,
                        end: end,
                        note_ids: noteIds
                    });
                });

                notes.forEach(function(note) {
                    if (assigned[note.id]) return;
                    var phraseId = nextPhraseId();
                    phrases.push({
                        id: phraseId,
                        start: note.start,
                        end: note.end,
                        note_ids: [note.id]
                    });
                });

                phrases.sort(function(a, b) {
                    if (a.start !== b.start) return a.start - b.start;
                    if (a.end !== b.end) return a.end - b.end;
                    return a.id.localeCompare(b.id);
                });

                return {
                    tonic: asNumber(draft.tonic, 0),
                    notes: notes,
                    phrases: phrases,
                    sargam_options: Array.isArray(draft.sargam_options) ? draft.sargam_options : []
                };
            }

            var state = normalizeState(initialPayload);
            initCountersFromState(state);
            var undoStack = [];
            var redoStack = [];
            var selectedNoteId = state.notes.length ? state.notes[0].id : null;
            var selectedPhraseId = state.phrases.length ? state.phrases[0].id : null;
            var mergePhraseIds = {};
            var currentRange = null;
            var currentPoint = null;
            var savedSnapshot = JSON.stringify(state);
            var activeVersions = [];
            var currentVersionId = null;
            var defaultSelection = "original";
            var defaultReportUrl = null;

            function isDirty() {
                return JSON.stringify(state) !== savedSnapshot;
            }

            function setStatus(msg, isError) {
                statusEl.style.color = isError ? "#ff7b72" : "#8b949e";
                statusEl.textContent = msg;
            }

            function updateDefaultLabel() {
                if (!defaultLabelEl) return;
                if (defaultSelection && defaultSelection !== "original") {
                    defaultLabelEl.textContent = "Default: " + defaultSelection;
                } else {
                    defaultLabelEl.textContent = "Default: Original";
                }
            }

            function getNoteById(noteId, inState) {
                var snapshot = inState || state;
                for (var i = 0; i < snapshot.notes.length; i += 1) {
                    if (snapshot.notes[i].id === noteId) return snapshot.notes[i];
                }
                return null;
            }

            function getPhraseById(phraseId, inState) {
                var snapshot = inState || state;
                for (var i = 0; i < snapshot.phrases.length; i += 1) {
                    if (snapshot.phrases[i].id === phraseId) return snapshot.phrases[i];
                }
                return null;
            }

            function findPhraseIdForNote(noteId, inState) {
                var snapshot = inState || state;
                for (var i = 0; i < snapshot.phrases.length; i += 1) {
                    if (snapshot.phrases[i].note_ids.indexOf(noteId) >= 0) return snapshot.phrases[i].id;
                }
                return null;
            }

            function updatePickerOptions() {
                var options = Array.isArray(state.sargam_options) ? state.sargam_options : [];
                addSargamSelect.innerHTML = options.map(function(item, idx) {
                    var selected = idx === 0 ? " selected" : "";
                    return '<option value="' + escapeHtml(String(item.label || "")) + '" data-offset="' + Number(item.offset || 0) + '"' + selected + ">" +
                        escapeHtml(String(item.label || "")) + "</option>";
                }).join("");
                var octaveParts = [];
                for (var octave = 2; octave <= 7; octave += 1) {
                    var selected = octave === 5 ? " selected" : "";
                    octaveParts.push('<option value="' + octave + '"' + selected + ">" + octave + "</option>");
                }
                addOctaveSelect.innerHTML = octaveParts.join("");
                updateMidiPreview();
            }

            function selectedPickerMidi() {
                var selectedOption = addSargamSelect.options[addSargamSelect.selectedIndex];
                if (!selectedOption) return null;
                var offset = asNumber(selectedOption.getAttribute("data-offset"), 0);
                var octave = asNumber(addOctaveSelect.value, 5);
                var tonicPc = ((Math.round(asNumber(state.tonic, 0)) % 12) + 12) % 12;
                var midi = (octave * 12) + tonicPc + offset;
                return midi;
            }

            function updateMidiPreview() {
                var midi = selectedPickerMidi();
                addMidiInput.value = isFinite(midi) ? Number(midi).toFixed(2) : "-";
            }

            function collectBoundaries() {
                var boundaries = [];
                state.notes.forEach(function(note) {
                    boundaries.push(note.start);
                    boundaries.push(note.end);
                });
                boundaries.sort(function(a, b) { return a - b; });
                return boundaries;
            }

            function nearestBoundary(value) {
                var boundaries = collectBoundaries();
                if (!boundaries.length) return value;
                var best = boundaries[0];
                var bestDist = Math.abs(best - value);
                for (var i = 1; i < boundaries.length; i += 1) {
                    var dist = Math.abs(boundaries[i] - value);
                    if (dist < bestDist) {
                        best = boundaries[i];
                        bestDist = dist;
                    }
                }
                return best;
            }

            function selectNearestNoteAtTime(t) {
                if (!state.notes.length) return;
                var tolerance = 0.02;
                var best = null;
                var bestDist = Number.POSITIVE_INFINITY;
                state.notes.forEach(function(note) {
                    if ((note.start - tolerance) <= t && t <= (note.end + tolerance)) {
                        var span = note.end - note.start;
                        if (best === null || span < (best.end - best.start)) {
                            best = note;
                        }
                        return;
                    }
                    var dist = 0;
                    if (t < note.start) dist = note.start - t;
                    else if (t > note.end) dist = t - note.end;
                    if (dist < bestDist) {
                        bestDist = dist;
                        best = note;
                    }
                });
                if (best) {
                    selectedNoteId = best.id;
                    selectedPhraseId = findPhraseIdForNote(best.id, state);
                }
            }

            function commitMutation(label, mutator) {
                var before = deepClone(state);
                var next = deepClone(state);
                mutator(next);
                next = normalizeState(next);
                undoStack.push(before);
                if (undoStack.length > 200) undoStack.shift();
                redoStack = [];
                state = next;
                if (selectedNoteId && !getNoteById(selectedNoteId)) selectedNoteId = null;
                if (selectedPhraseId && !getPhraseById(selectedPhraseId)) selectedPhraseId = null;
                if (!selectedPhraseId && state.phrases.length) selectedPhraseId = state.phrases[0].id;
                if (!selectedNoteId && state.notes.length) selectedNoteId = state.notes[0].id;
                if (selectedNoteId) {
                    var phraseForNote = findPhraseIdForNote(selectedNoteId, state);
                    if (phraseForNote) selectedPhraseId = phraseForNote;
                }
                initCountersFromState(state);
                render();
                setStatus(label + (isDirty() ? " (unsaved)" : ""), false);
            }

            function loadState(payload, label, versionId) {
                state = normalizeState(payload);
                initCountersFromState(state);
                undoStack = [];
                redoStack = [];
                mergePhraseIds = {};
                selectedNoteId = state.notes.length ? state.notes[0].id : null;
                selectedPhraseId = state.phrases.length ? state.phrases[0].id : null;
                savedSnapshot = JSON.stringify(state);
                if (typeof versionId !== "undefined") {
                    currentVersionId = versionId;
                }
                render();
                setStatus(label, false);
            }

            function renderPhraseList() {
                if (!state.phrases.length) {
                    phraseListEl.innerHTML = "<div style='color:#8b949e; font-size:12px;'>No phrases available.</div>";
                    return;
                }
                var noteById = {};
                state.notes.forEach(function(note) { noteById[note.id] = note; });
                var html = state.phrases.map(function(phrase, idx) {
                    var activeClass = phrase.id === selectedPhraseId ? " active" : "";
                    var noteChips = phrase.note_ids.map(function(noteId) {
                        var note = noteById[noteId];
                        if (!note) return "";
                        var label = note.sargam || Number(note.pitch_midi).toFixed(2);
                        var isActive = note.id === selectedNoteId ? " active" : "";
                        return "<span class='editor-note-chip" + isActive + "' data-note-id='" + escapeHtml(note.id) + "'>" +
                            escapeHtml(label) + " (" + formatSec(note.end - note.start) + ")" +
                            "</span>";
                    }).join("");
                    var checked = mergePhraseIds[phrase.id] ? " checked" : "";
                    return (
                        "<div class='editor-phrase-row" + activeClass + "' data-phrase-id='" + escapeHtml(phrase.id) + "'>" +
                        "<div class='editor-phrase-meta'>" +
                        "<input type='checkbox' class='editor-merge-checkbox' data-phrase-id='" + escapeHtml(phrase.id) + "'" + checked + ">" +
                        "<button type='button' class='editor-select-phrase' data-phrase-id='" + escapeHtml(phrase.id) + "' " +
                        "style='border:1px solid #30363d; border-radius:6px; background:#21262d; color:#c9d1d9; cursor:pointer; font-size:11px;'>" +
                        "Phrase " + (idx + 1) + "</button>" +
                        "<span>" + formatSec(phrase.start) + " - " + formatSec(phrase.end) + "</span>" +
                        "<span>(" + phrase.note_ids.length + " notes)</span>" +
                        "</div>" +
                        "<div class='editor-note-chip-wrap'>" + noteChips + "</div>" +
                        "</div>"
                    );
                }).join("");
                phraseListEl.innerHTML = html;
            }

            function renderSelection() {
                var parts = [];
                if (currentPoint !== null && isFinite(currentPoint)) {
                    parts.push("Point " + formatSec(currentPoint));
                }
                if (currentRange) {
                    var lo = Math.min(currentRange.start, currentRange.end);
                    var hi = Math.max(currentRange.start, currentRange.end);
                    parts.push("Range " + formatSec(lo) + " - " + formatSec(hi));
                }
                if (selectedPhraseId) parts.push("Phrase " + escapeHtml(selectedPhraseId));
                if (selectedNoteId) parts.push("Note " + escapeHtml(selectedNoteId));
                selectionEl.textContent = parts.length ? ("Plot selection: " + parts.join(" | ")) : "Plot selection: none";
            }

            function renderFieldValues() {
                var phrase = selectedPhraseId ? getPhraseById(selectedPhraseId) : null;
                if (phrase) {
                    phraseStartInput.value = phrase.start.toFixed(3);
                    phraseEndInput.value = phrase.end.toFixed(3);
                } else {
                    phraseStartInput.value = "";
                    phraseEndInput.value = "";
                }
                var note = selectedNoteId ? getNoteById(selectedNoteId) : null;
                if (note) {
                    noteStartInput.value = note.start.toFixed(3);
                    noteEndInput.value = note.end.toFixed(3);
                } else {
                    noteStartInput.value = "";
                    noteEndInput.value = "";
                }
            }

            function renderButtons() {
                undoBtn.disabled = undoStack.length === 0;
                redoBtn.disabled = redoStack.length === 0;
                deleteNoteBtn.disabled = !selectedNoteId;
                splitPhraseBtn.disabled = !(selectedNoteId && selectedPhraseId);
                applyPhraseBoundsBtn.disabled = !selectedPhraseId;
                applyNoteBoundsBtn.disabled = !selectedNoteId;
                deleteRangeBtn.disabled = !currentRange;
                addNoteRangeBtn.disabled = !currentRange;
            }

            function renderStats() {
                statsEl.textContent = "Notes: " + state.notes.length + " | Phrases: " + state.phrases.length;
            }

            function render() {
                renderPhraseList();
                renderSelection();
                renderFieldValues();
                renderButtons();
                renderStats();
            }

            function parseReportContextFromUrl() {
                var parts = window.location.pathname.split("/").filter(Boolean);
                if (parts.length < 3 || parts[0] !== "local-report") {
                    return null;
                }
                return {
                    dirToken: parts[1],
                    reportName: decodeURIComponent(parts.slice(2).join("/"))
                };
            }

            var reportContext = parseReportContextFromUrl();

            function buildApiPath(suffix) {
                if (!reportContext) return null;
                return "/api/transcription-edits/" +
                    encodeURIComponent(reportContext.dirToken) +
                    "/" +
                    encodeURIComponent(reportContext.reportName) +
                    "/" +
                    suffix;
            }

            function updateVersionActionButtons() {
                var selected = versionsSelect.value;
                var selectedIsNew = (!selected || selected === "__new__");
                loadVersionBtn.disabled = selectedIsNew || !activeVersions.length;
                deleteVersionBtn.disabled = selectedIsNew || !activeVersions.length;
                regenerateBtn.disabled = selectedIsNew || !activeVersions.length;
                setDefaultSelectedBtn.disabled = selectedIsNew || !activeVersions.length || selected === defaultSelection;
                setDefaultOriginalBtn.disabled = defaultSelection === "original";
            }

            function setVersionOptions(versions, preferredVersionId) {
                activeVersions = Array.isArray(versions) ? versions.slice() : [];

                var optionParts = ["<option value='__new__'>Create new version...</option>"];
                activeVersions.forEach(function(version) {
                    var isDefaultVersion = defaultSelection && version.version_id === defaultSelection;
                    var defaultTag = isDefaultVersion ? " [default]" : "";
                    optionParts.push(
                        "<option value='" + escapeHtml(version.version_id) + "'>" +
                        escapeHtml(version.version_id + " (" + version.created_at + ")" + defaultTag) +
                        "</option>"
                    );
                });
                versionsSelect.innerHTML = optionParts.join("");
                versionsSelect.disabled = false;

                var targetVersionId = preferredVersionId || currentVersionId || null;
                if (targetVersionId) {
                    var hasTarget = activeVersions.some(function(version) {
                        return version.version_id === targetVersionId;
                    });
                    versionsSelect.value = hasTarget ? targetVersionId : "__new__";
                } else {
                    versionsSelect.value = "__new__";
                }
                updateDefaultLabel();
                updateVersionActionButtons();
            }

            async function refreshVersions(preferredVersionId) {
                if (!reportContext) {
                    setVersionOptions([], null);
                    return;
                }
                var endpoint = buildApiPath("versions");
                if (!endpoint) return;
                var response = await fetch(endpoint);
                if (!response.ok) {
                    throw new Error("Failed to fetch versions.");
                }
                var data = await response.json();
                defaultSelection = String((data && data.default_selection) || "original");
                defaultReportUrl = (data && data.default_report_url) ? String(data.default_report_url) : null;
                setVersionOptions(data.versions || [], preferredVersionId || data.latest_version_id || null);
                return data;
            }

            async function loadDefaultSelection() {
                if (!reportContext) {
                    setStatus("Save/load available only when opened via local app route.", false);
                    saveBtn.disabled = true;
                    saveNewBtn.disabled = true;
                    deleteVersionBtn.disabled = true;
                    loadVersionBtn.disabled = true;
                    regenerateBtn.disabled = true;
                    setDefaultSelectedBtn.disabled = true;
                    setDefaultOriginalBtn.disabled = true;
                    return;
                }

                if (defaultSelection && defaultSelection !== "original") {
                    var hasDefault = activeVersions.some(function(version) {
                        return version.version_id === defaultSelection;
                    });
                    if (hasDefault) {
                        await loadSpecificVersion(defaultSelection);
                        setStatus("Loaded default saved version " + defaultSelection + ".", false);
                        return;
                    }
                }

                loadState(initialPayload, "Loaded base auto-transcription.", null);
                updateVersionActionButtons();
            }

            async function loadSpecificVersion(versionId) {
                if (!reportContext || !versionId || versionId === "__new__") return;
                var endpoint = buildApiPath("version/" + encodeURIComponent(versionId));
                if (!endpoint) return;
                var response = await fetch(endpoint);
                if (!response.ok) {
                    throw new Error("Failed to load version " + versionId + ".");
                }
                var data = await response.json();
                if (!data || !data.has_version || !data.payload) {
                    throw new Error("Version payload missing.");
                }
                loadState(data.payload, "Loaded version " + versionId + ".", versionId);
                await refreshVersions(versionId);
            }

            async function setDefaultSelectionTo(selectionValue) {
                if (!reportContext) {
                    setStatus("Cannot set default outside local app report route.", true);
                    return;
                }
                var selection = String(selectionValue || "").trim();
                if (!selection) {
                    setStatus("Default selection is required.", true);
                    return;
                }
                var endpoint = buildApiPath("default");
                if (!endpoint) return;
                endpoint += "?default_selection=" + encodeURIComponent(selection);
                var response = await fetch(endpoint, { method: "POST" });
                var data = await response.json();
                if (!response.ok) {
                    throw new Error((data && data.detail) ? data.detail : "Failed to update default selection.");
                }
                defaultSelection = String((data && data.default_selection) || "original");
                defaultReportUrl = (data && data.default_report_url) ? String(data.default_report_url) : null;
                setVersionOptions(data.versions || [], currentVersionId || null);
                return data;
            }

            function payloadForSave() {
                return {
                    notes: state.notes.map(function(note) {
                        return {
                            id: note.id,
                            start: note.start,
                            end: note.end,
                            pitch_midi: note.pitch_midi,
                            pitch_hz: note.pitch_hz,
                            raw_pitch_midi: note.raw_pitch_midi,
                            snapped_pitch_midi: note.snapped_pitch_midi,
                            corrected_pitch_midi: note.corrected_pitch_midi,
                            rendered_pitch_midi: note.rendered_pitch_midi,
                            confidence: note.confidence,
                            energy: note.energy,
                            sargam: note.sargam,
                            pitch_class: note.pitch_class
                        };
                    }),
                    phrases: state.phrases.map(function(phrase) {
                        return {
                            id: phrase.id,
                            start: phrase.start,
                            end: phrase.end,
                            note_ids: phrase.note_ids.slice()
                        };
                    })
                };
            }

            function buildSaveEndpoint(createNewVersion) {
                var endpoint = buildApiPath("save");
                if (!endpoint) return null;
                var selectedVersion = versionsSelect.value;
                var params = new URLSearchParams();
                if (createNewVersion || selectedVersion === "__new__") {
                    params.set("create_new_version", "true");
                } else {
                    var targetVersion = selectedVersion || currentVersionId || "";
                    if (targetVersion && targetVersion !== "__new__") {
                        params.set("target_version_id", targetVersion);
                    }
                }
                var query = params.toString();
                if (!query) return endpoint;
                return endpoint + "?" + query;
            }

            async function saveCurrentVersion(createNewVersion) {
                if (!reportContext) {
                    setStatus("Cannot save outside local app report route.", true);
                    return;
                }
                var endpoint = buildSaveEndpoint(!!createNewVersion);
                if (!endpoint) return;
                saveBtn.disabled = true;
                saveNewBtn.disabled = true;
                setStatus(createNewVersion ? "Saving as new version..." : "Saving selected version...", false);
                try {
                    var response = await fetch(endpoint, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payloadForSave())
                    });
                    var data = await response.json();
                    if (!response.ok) {
                        throw new Error((data && data.detail) ? data.detail : "Save failed.");
                    }
                    if (data && data.payload) {
                        var savedVersionId = data.version ? data.version.version_id : null;
                        loadState(
                            data.payload,
                            "",
                            savedVersionId
                        );
                    }
                    var activeVersionId = data.version ? data.version.version_id : null;
                    defaultSelection = String((data && data.default_selection) || "original");
                    defaultReportUrl = (data && data.default_report_url) ? String(data.default_report_url) : null;
                    setVersionOptions(data.versions || [], activeVersionId);
                    var modeText = (data && data.save_mode === "updated") ? "Updated" : "Created";
                    if (data.version && data.version.report_url) {
                        setStatus(
                            modeText + " " + data.version.version_id + ". Edited report: " + data.version.report_url,
                            false
                        );
                    } else {
                        setStatus(modeText + " transcription edit version.", false);
                    }
                } catch (err) {
                    setStatus(String(err && err.message ? err.message : err), true);
                } finally {
                    saveBtn.disabled = false;
                    saveNewBtn.disabled = false;
                }
            }

            async function deleteSelectedVersion() {
                if (!reportContext) {
                    setStatus("Cannot delete outside local app report route.", true);
                    return;
                }
                var selectedVersion = versionsSelect.value;
                if (!selectedVersion || selectedVersion === "__new__") {
                    setStatus("Choose a saved version to delete.", true);
                    return;
                }
                var okay = window.confirm("Delete version " + selectedVersion + "? This cannot be undone.");
                if (!okay) return;

                var endpoint = buildApiPath("version/" + encodeURIComponent(selectedVersion));
                if (!endpoint) return;
                deleteVersionBtn.disabled = true;
                setStatus("Deleting version " + selectedVersion + "...", false);
                try {
                    var response = await fetch(endpoint, { method: "DELETE" });
                    var data = await response.json();
                    if (!response.ok) {
                        throw new Error((data && data.detail) ? data.detail : "Delete failed.");
                    }
                    if (currentVersionId === selectedVersion) {
                        currentVersionId = null;
                    }
                    defaultSelection = String((data && data.default_selection) || "original");
                    defaultReportUrl = (data && data.default_report_url) ? String(data.default_report_url) : null;
                    var latestAfterDelete = data.latest_version_id || null;
                    setVersionOptions(data.versions || [], latestAfterDelete);
                    if (latestAfterDelete) {
                        await loadSpecificVersion(latestAfterDelete);
                    } else {
                        loadState(initialPayload, "Deleted " + selectedVersion + ". No saved versions remain.", null);
                        setVersionOptions([], null);
                    }
                    if (latestAfterDelete) {
                        setStatus("Deleted " + selectedVersion + ".", false);
                    }
                } catch (err) {
                    setStatus(String(err && err.message ? err.message : err), true);
                } finally {
                    updateVersionActionButtons();
                }
            }

            async function regenerateSelectedVersion() {
                if (!reportContext) {
                    setStatus("Cannot regenerate outside local app report route.", true);
                    return;
                }
                var selectedVersion = versionsSelect.value;
                if (!selectedVersion || selectedVersion === "__new__") {
                    setStatus("Choose a saved version to regenerate.", true);
                    return;
                }
                var endpoint = buildApiPath(
                    "version/" + encodeURIComponent(selectedVersion) + "/regenerate"
                );
                if (!endpoint) return;
                regenerateBtn.disabled = true;
                setStatus("Regenerating report for " + selectedVersion + "...", false);
                try {
                    if (isDirty()) {
                        setStatus("Unsaved changes detected. Saving before regenerate...", false);
                        var saveEndpoint = buildSaveEndpoint(false);
                        if (!saveEndpoint) {
                            throw new Error("Unable to resolve save endpoint before regenerate.");
                        }
                        var saveResponse = await fetch(saveEndpoint, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(payloadForSave())
                        });
                        var saveData = await saveResponse.json();
                        if (!saveResponse.ok) {
                            throw new Error((saveData && saveData.detail) ? saveData.detail : "Save before regenerate failed.");
                        }
                        if (saveData && saveData.payload) {
                            var savedVersionId = saveData.version ? saveData.version.version_id : selectedVersion;
                            loadState(saveData.payload, "", savedVersionId);
                            selectedVersion = savedVersionId || selectedVersion;
                        }
                        defaultSelection = String((saveData && saveData.default_selection) || "original");
                        defaultReportUrl = (saveData && saveData.default_report_url) ? String(saveData.default_report_url) : null;
                        setVersionOptions(saveData.versions || [], selectedVersion);
                        endpoint = buildApiPath(
                            "version/" + encodeURIComponent(selectedVersion) + "/regenerate"
                        );
                        if (!endpoint) {
                            throw new Error("Unable to resolve regenerate endpoint after save.");
                        }
                    }
                    var response = await fetch(endpoint, { method: "POST" });
                    var data = await response.json();
                    if (!response.ok) {
                        throw new Error((data && data.detail) ? data.detail : "Regenerate failed.");
                    }
                    if (data && data.payload) {
                        loadState(data.payload, "", selectedVersion);
                    }
                    await refreshVersions(selectedVersion);
                    if (data && data.version && data.version.report_url) {
                        setStatus(
                            "Regenerated " + selectedVersion + ". Edited report: " + data.version.report_url,
                            false
                        );
                        try {
                            document.dispatchEvent(
                                new CustomEvent("raga-transcription-report-regenerated", {
                                    detail: {
                                        report_url: String(data.version.report_url),
                                        version_id: String(selectedVersion),
                                    },
                                })
                            );
                        } catch (_err) {
                            // Ignore event dispatch failures.
                        }
                    } else {
                        setStatus("Regenerated " + selectedVersion + ".", false);
                    }
                } catch (err) {
                    setStatus(String(err && err.message ? err.message : err), true);
                } finally {
                    updateVersionActionButtons();
                }
            }

            async function setSelectedVersionAsDefault() {
                var selectedVersion = versionsSelect.value;
                if (!selectedVersion || selectedVersion === "__new__") {
                    setStatus("Choose a saved version to set as default.", true);
                    return;
                }
                setDefaultSelectedBtn.disabled = true;
                setStatus("Setting default transcription to " + selectedVersion + "...", false);
                try {
                    await setDefaultSelectionTo(selectedVersion);
                    setStatus("Default transcription set to " + selectedVersion + ".", false);
                } catch (err) {
                    setStatus(String(err && err.message ? err.message : err), true);
                } finally {
                    updateVersionActionButtons();
                }
            }

            async function setOriginalAsDefault() {
                setDefaultOriginalBtn.disabled = true;
                setStatus("Setting default transcription to original...", false);
                try {
                    await setDefaultSelectionTo("original");
                    setStatus("Default transcription set to original.", false);
                } catch (err) {
                    setStatus(String(err && err.message ? err.message : err), true);
                } finally {
                    updateVersionActionButtons();
                }
            }

            function handleUndo() {
                if (!undoStack.length) return;
                var current = deepClone(state);
                var previous = undoStack.pop();
                redoStack.push(current);
                state = normalizeState(previous);
                initCountersFromState(state);
                render();
                setStatus("Undo applied" + (isDirty() ? " (unsaved)" : ""), false);
            }

            function handleRedo() {
                if (!redoStack.length) return;
                var current = deepClone(state);
                var next = redoStack.pop();
                undoStack.push(current);
                state = normalizeState(next);
                initCountersFromState(state);
                render();
                setStatus("Redo applied" + (isDirty() ? " (unsaved)" : ""), false);
            }

            function handleDeleteSelectedNote() {
                if (!selectedNoteId) return;
                var noteId = selectedNoteId;
                commitMutation("Deleted selected note", function(next) {
                    next.notes = next.notes.filter(function(note) { return note.id !== noteId; });
                    next.phrases.forEach(function(phrase) {
                        phrase.note_ids = phrase.note_ids.filter(function(id) { return id !== noteId; });
                    });
                    next.phrases = next.phrases.filter(function(phrase) { return phrase.note_ids.length > 0; });
                });
            }

            function handleDeleteRange() {
                if (!currentRange) return;
                var lo = Math.min(currentRange.start, currentRange.end);
                var hi = Math.max(currentRange.start, currentRange.end);
                commitMutation("Deleted notes in selected range", function(next) {
                    var removeIds = {};
                    next.notes.forEach(function(note) {
                        if (note.end >= lo && note.start <= hi) {
                            removeIds[note.id] = true;
                        }
                    });
                    next.notes = next.notes.filter(function(note) { return !removeIds[note.id]; });
                    next.phrases.forEach(function(phrase) {
                        phrase.note_ids = phrase.note_ids.filter(function(noteId) { return !removeIds[noteId]; });
                    });
                    next.phrases = next.phrases.filter(function(phrase) { return phrase.note_ids.length > 0; });
                });
            }

            function handleMergePhrases() {
                var ids = Object.keys(mergePhraseIds).filter(function(id) { return !!mergePhraseIds[id]; });
                if (ids.length < 2) {
                    setStatus("Check at least two phrases to merge.", true);
                    return;
                }
                commitMutation("Merged selected phrases", function(next) {
                    var noteById = {};
                    next.notes.forEach(function(note) {
                        noteById[note.id] = note;
                    });

                    var selected = [];
                    next.phrases.forEach(function(phrase, idx) {
                        if (ids.indexOf(phrase.id) >= 0) {
                            selected.push({ phrase: phrase, index: idx });
                        }
                    });
                    if (selected.length < 2) return;

                    selected.sort(function(a, b) {
                        if (a.phrase.start !== b.phrase.start) return a.phrase.start - b.phrase.start;
                        if (a.phrase.end !== b.phrase.end) return a.phrase.end - b.phrase.end;
                        return a.index - b.index;
                    });

                    var mergedNoteIds = [];
                    var seen = {};
                    selected.forEach(function(item) {
                        item.phrase.note_ids.forEach(function(noteId) {
                            if (seen[noteId]) return;
                            if (!noteById[noteId]) return;
                            seen[noteId] = true;
                            mergedNoteIds.push(noteId);
                        });
                    });
                    if (!mergedNoteIds.length) return;

                    mergedNoteIds.sort(function(a, b) {
                        var noteA = noteById[a];
                        var noteB = noteById[b];
                        if (noteA.start !== noteB.start) return noteA.start - noteB.start;
                        if (noteA.end !== noteB.end) return noteA.end - noteB.end;
                        return a.localeCompare(b);
                    });

                    var mergedStart = noteById[mergedNoteIds[0]].start;
                    var mergedEnd = noteById[mergedNoteIds[0]].end;
                    mergedNoteIds.forEach(function(noteId) {
                        mergedStart = Math.min(mergedStart, noteById[noteId].start);
                        mergedEnd = Math.max(mergedEnd, noteById[noteId].end);
                    });

                    var keepId = selected[0].phrase.id;
                    var insertIndex = selected[0].index;
                    for (var i = 1; i < selected.length; i += 1) {
                        insertIndex = Math.min(insertIndex, selected[i].index);
                    }
                    next.phrases = next.phrases.filter(function(phrase) { return ids.indexOf(phrase.id) < 0; });
                    if (insertIndex < 0) insertIndex = 0;
                    if (insertIndex > next.phrases.length) insertIndex = next.phrases.length;
                    next.phrases.splice(insertIndex, 0, {
                        id: keepId,
                        start: mergedStart,
                        end: mergedEnd,
                        note_ids: mergedNoteIds
                    });
                    selectedPhraseId = keepId;
                    mergePhraseIds = {};
                });
            }

            function handleSplitPhrase() {
                if (!selectedPhraseId || !selectedNoteId) {
                    setStatus("Select a phrase and a split note first.", true);
                    return;
                }
                var phrase = getPhraseById(selectedPhraseId);
                if (!phrase) return;
                var splitIdx = phrase.note_ids.indexOf(selectedNoteId);
                if (splitIdx <= 0 || splitIdx >= phrase.note_ids.length - 1) {
                    setStatus("Split note must be inside phrase (not first/last).", true);
                    return;
                }
                var phraseId = selectedPhraseId;
                var noteId = selectedNoteId;
                commitMutation("Split phrase at selected note", function(next) {
                    var target = null;
                    for (var i = 0; i < next.phrases.length; i += 1) {
                        if (next.phrases[i].id === phraseId) {
                            target = next.phrases[i];
                            break;
                        }
                    }
                    if (!target) return;
                    var idx = target.note_ids.indexOf(noteId);
                    if (idx <= 0 || idx >= target.note_ids.length - 1) return;
                    var rightIds = target.note_ids.slice(idx + 1);
                    target.note_ids = target.note_ids.slice(0, idx + 1);
                    next.phrases.push({
                        id: nextPhraseId(),
                        start: 0,
                        end: 0,
                        note_ids: rightIds
                    });
                });
            }

            function handleApplyPhraseBounds() {
                if (!selectedPhraseId) return;
                var requestedStart = asNumber(phraseStartInput.value, NaN);
                var requestedEnd = asNumber(phraseEndInput.value, NaN);
                if (!isFinite(requestedStart) || !isFinite(requestedEnd)) {
                    setStatus("Phrase start/end must be valid numbers.", true);
                    return;
                }
                var snappedStart = nearestBoundary(requestedStart);
                var snappedEnd = nearestBoundary(requestedEnd);
                if (snappedEnd < snappedStart) {
                    var tmp = snappedStart;
                    snappedStart = snappedEnd;
                    snappedEnd = tmp;
                }
                if (snappedEnd <= snappedStart) {
                    setStatus("Phrase bounds collapse to zero after snapping.", true);
                    return;
                }
                var phraseId = selectedPhraseId;
                commitMutation("Updated phrase bounds (snapped to note boundaries)", function(next) {
                    var selectedIds = next.notes
                        .filter(function(note) { return note.end >= snappedStart && note.start <= snappedEnd; })
                        .map(function(note) { return note.id; });
                    if (!selectedIds.length) return;
                    next.phrases.forEach(function(phrase) {
                        if (phrase.id !== phraseId) {
                            phrase.note_ids = phrase.note_ids.filter(function(noteId) {
                                return selectedIds.indexOf(noteId) < 0;
                            });
                        }
                    });
                    var target = null;
                    next.phrases.forEach(function(phrase) {
                        if (phrase.id === phraseId) target = phrase;
                    });
                    if (!target) {
                        target = {
                            id: phraseId,
                            start: 0,
                            end: 0,
                            note_ids: []
                        };
                        next.phrases.push(target);
                    }
                    target.note_ids = selectedIds.slice();
                    next.phrases = next.phrases.filter(function(phrase) { return phrase.note_ids.length > 0; });
                });
            }

            function handleApplyNoteBounds() {
                if (!selectedNoteId) return;
                var requestedStart = asNumber(noteStartInput.value, NaN);
                var requestedEnd = asNumber(noteEndInput.value, NaN);
                if (!isFinite(requestedStart) || !isFinite(requestedEnd)) {
                    setStatus("Note start/end must be valid numbers.", true);
                    return;
                }
                if (requestedEnd < requestedStart) {
                    var tmp = requestedStart;
                    requestedStart = requestedEnd;
                    requestedEnd = tmp;
                }
                if (requestedEnd <= requestedStart) requestedEnd = requestedStart + 0.001;
                var noteId = selectedNoteId;
                commitMutation("Resized selected note", function(next) {
                    next.notes.forEach(function(note) {
                        if (note.id !== noteId) return;
                        note.start = requestedStart;
                        note.end = requestedEnd;
                    });
                });
            }

            function handleAddNoteFromRange() {
                if (!currentRange) {
                    setStatus("Select a range on the plot before adding a note.", true);
                    return;
                }
                var lo = Math.min(currentRange.start, currentRange.end);
                var hi = Math.max(currentRange.start, currentRange.end);
                if (hi <= lo) hi = lo + 0.04;
                var midi = selectedPickerMidi();
                if (!isFinite(midi)) {
                    setStatus("Could not resolve MIDI for selected picker values.", true);
                    return;
                }
                var selectedOption = addSargamSelect.options[addSargamSelect.selectedIndex];
                var label = selectedOption ? String(selectedOption.value || selectedOption.textContent || "").trim() : "";
                commitMutation("Added note from selected range", function(next) {
                    var noteId = nextNoteId();
                    var note = {
                        id: noteId,
                        start: lo,
                        end: hi,
                        pitch_midi: midi,
                        pitch_hz: 440 * Math.pow(2, (midi - 69) / 12),
                        raw_pitch_midi: midi,
                        snapped_pitch_midi: midi,
                        corrected_pitch_midi: midi,
                        rendered_pitch_midi: midi,
                        confidence: 0.95,
                        energy: 0.0,
                        sargam: label,
                        pitch_class: ((Math.round(midi) % 12) + 12) % 12
                    };
                    next.notes.push(note);

                    var targetPhrase = null;
                    if (selectedPhraseId) {
                        next.phrases.forEach(function(phrase) {
                            if (phrase.id === selectedPhraseId) targetPhrase = phrase;
                        });
                    }
                    if (!targetPhrase) {
                        next.phrases.forEach(function(phrase) {
                            if (targetPhrase) return;
                            if (phrase.end >= lo && phrase.start <= hi) targetPhrase = phrase;
                        });
                    }
                    if (!targetPhrase) {
                        targetPhrase = {
                            id: nextPhraseId(),
                            start: lo,
                            end: hi,
                            note_ids: []
                        };
                        next.phrases.push(targetPhrase);
                    }
                    targetPhrase.note_ids.push(noteId);
                    selectedNoteId = noteId;
                    selectedPhraseId = targetPhrase.id;
                });
            }

            phraseListEl.addEventListener("click", function(evt) {
                var noteTarget = evt.target.closest("[data-note-id]");
                if (noteTarget) {
                    selectedNoteId = noteTarget.getAttribute("data-note-id");
                    selectedPhraseId = findPhraseIdForNote(selectedNoteId, state);
                    render();
                    return;
                }
                var phraseTarget = evt.target.closest("[data-phrase-id]");
                if (phraseTarget && evt.target.classList.contains("editor-select-phrase")) {
                    selectedPhraseId = phraseTarget.getAttribute("data-phrase-id");
                    var phrase = getPhraseById(selectedPhraseId);
                    if (phrase && phrase.note_ids.length) {
                        selectedNoteId = phrase.note_ids[0];
                    }
                    render();
                }
            });

            phraseListEl.addEventListener("change", function(evt) {
                if (!evt.target.classList.contains("editor-merge-checkbox")) return;
                var phraseId = evt.target.getAttribute("data-phrase-id");
                mergePhraseIds[phraseId] = !!evt.target.checked;
            });

            document.addEventListener("raga-transcription-selection", function(evt) {
                if (!evt || !evt.detail) return;
                var detail = evt.detail;
                if (detail.mode === "none") {
                    currentRange = null;
                    currentPoint = null;
                    render();
                    return;
                }
                if (detail.mode === "point") {
                    currentPoint = asNumber(detail.time, null);
                    currentRange = null;
                    if (isFinite(currentPoint)) {
                        selectNearestNoteAtTime(currentPoint);
                    }
                    render();
                    return;
                }
                if (detail.mode === "range") {
                    var start = asNumber(detail.start, NaN);
                    var end = asNumber(detail.end, NaN);
                    if (isFinite(start) && isFinite(end)) {
                        currentRange = { start: start, end: end };
                        currentPoint = null;
                        selectNearestNoteAtTime((start + end) * 0.5);
                        render();
                    }
                }
            });

            undoBtn.addEventListener("click", handleUndo);
            redoBtn.addEventListener("click", handleRedo);
            saveBtn.addEventListener("click", function() { saveCurrentVersion(false); });
            saveNewBtn.addEventListener("click", function() { saveCurrentVersion(true); });
            resetBtn.addEventListener("click", function() {
                loadState(initialPayload, "Reset editor to base auto-transcription.", null);
                setVersionOptions(activeVersions, currentVersionId);
            });
            loadVersionBtn.addEventListener("click", function() {
                var selectedVersion = versionsSelect.value;
                if (!selectedVersion || selectedVersion === "__new__") return;
                loadSpecificVersion(selectedVersion).catch(function(err) {
                    setStatus(String(err && err.message ? err.message : err), true);
                });
            });
            deleteVersionBtn.addEventListener("click", function() {
                deleteSelectedVersion().catch(function(err) {
                    setStatus(String(err && err.message ? err.message : err), true);
                });
            });
            regenerateBtn.addEventListener("click", function() {
                regenerateSelectedVersion().catch(function(err) {
                    setStatus(String(err && err.message ? err.message : err), true);
                });
            });
            setDefaultSelectedBtn.addEventListener("click", function() {
                setSelectedVersionAsDefault().catch(function(err) {
                    setStatus(String(err && err.message ? err.message : err), true);
                });
            });
            setDefaultOriginalBtn.addEventListener("click", function() {
                setOriginalAsDefault().catch(function(err) {
                    setStatus(String(err && err.message ? err.message : err), true);
                });
            });
            versionsSelect.addEventListener("change", function() {
                updateVersionActionButtons();
            });
            addSargamSelect.addEventListener("change", updateMidiPreview);
            addOctaveSelect.addEventListener("change", updateMidiPreview);
            addNoteRangeBtn.addEventListener("click", handleAddNoteFromRange);
            deleteRangeBtn.addEventListener("click", handleDeleteRange);
            deleteNoteBtn.addEventListener("click", handleDeleteSelectedNote);
            mergePhrasesBtn.addEventListener("click", handleMergePhrases);
            splitPhraseBtn.addEventListener("click", handleSplitPhrase);
            applyPhraseBoundsBtn.addEventListener("click", handleApplyPhraseBounds);
            applyNoteBoundsBtn.addEventListener("click", handleApplyNoteBounds);

            updatePickerOptions();
            render();

            Promise.resolve()
                .then(function() {
                    return refreshVersions();
                })
                .then(function() {
                    return loadDefaultSelection();
                })
                .then(function() {
                    return refreshVersions(currentVersionId);
                })
                .catch(function(err) {
                    setStatus(String(err && err.message ? err.message : err), true);
                });
        })();
        </script>
    </section>
    """
    return template.replace("__SECTION_ID__", section_id).replace("__EDITOR_PAYLOAD__", payload_json)


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
    transcription_energy_threshold: float = 0.0,
    figsize_width: int = 80, 
    figsize_height: int = 7, 
    dpi: int = 100, 
    join_gap_threshold: float = 0.3,
    show_rms_overlay: bool = True,
    overlay_energy: Optional[np.ndarray] = None,
    overlay_timestamps: Optional[np.ndarray] = None,
    overlay_label: str = "RMS Energy",
    phrase_ranges: Optional[List[Tuple[float, float]]] = None,
    transcription_notes: Optional[List[Note]] = None,
    bias_cents: Optional[float] = None,
) -> Tuple[str, str, int, float, float, float, float]:
    """
    Generate wide pitch contour and overlay sargam lines, returning base64 images.
    Returns:
        (
            legend_b64,
            plot_b64,
            pixel_width,
            x_axis_start,
            x_axis_end,
            y_axis_min,
            y_axis_max,
        )
    """
    import io
    import base64
    import math
    import re
    from matplotlib.ticker import MultipleLocator

    DEFAULT_SECONDS_AT_BASE_WIDTH = 200.0
    MIN_PLOT_WIDTH_PX = 1200
    TARGET_MAJOR_TICKS = 220

    def _hz_to_midi(values: np.ndarray) -> np.ndarray:
        safe = np.asarray(values, dtype=float)
        safe = np.maximum(safe, 1e-6)
        return 69.0 + 12.0 * np.log2(safe / 440.0)

    def _note_name_to_midi(note_text: str) -> Optional[float]:
        pattern = re.compile(r"^\s*([A-Ga-g])([#b]?)(-?\d+)?\s*$")
        match = pattern.match(note_text)
        if not match:
            return None
        letter = match.group(1).upper()
        accidental = match.group(2)
        octave_text = match.group(3)
        octave = int(octave_text) if octave_text is not None else 4
        base_pc = {
            "C": 0,
            "D": 2,
            "E": 4,
            "F": 5,
            "G": 7,
            "A": 9,
            "B": 11,
        }.get(letter)
        if base_pc is None:
            return None
        if accidental == "#":
            base_pc += 1
        elif accidental == "b":
            base_pc -= 1
        pc = base_pc % 12
        return float((octave + 1) * 12 + pc)
    
    # Resolve tonic
    tonic_midi_base: float
    tonic_midi: int
    tonic_label: str
    if isinstance(tonic_ref, (int, np.integer)):
        tonic_midi = int(tonic_ref) % 12
        tonic_midi_base = float(tonic_midi + 60) # Default to middle C octave if int
        tonic_label = _tonic_name(tonic_ref)
    else:
        resolved_tonic = _note_name_to_midi(str(tonic_ref))
        if resolved_tonic is None:
            resolved_tonic = _note_name_to_midi(f"{tonic_ref}4")
        if resolved_tonic is not None:
            tonic_midi_base = float(resolved_tonic)
            tonic_midi = int(round(tonic_midi_base)) % 12
            tonic_label = str(tonic_ref)
        else:
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
         return "", "", 100, 0.0, 1.0, 0.0, 1.0

    bias_semitones = (float(bias_cents) / 100.0) if bias_cents is not None else 0.0
    voiced_midi = _hz_to_midi(pitch_hz[voiced_mask]) - bias_semitones
    voiced_midi_rounded = np.round(voiced_midi).astype(int)

    x_axis_start = 0.0
    x_axis_end = max(float(timestamps[-1]), 1.0)
    duration_seconds = max(x_axis_end - x_axis_start, 1.0)

    # Keep horizontal scale strictly proportional to recording length.
    base_width_px = max(int(figsize_width * dpi), MIN_PLOT_WIDTH_PX)
    pixels_per_second = base_width_px / DEFAULT_SECONDS_AT_BASE_WIDTH
    pixel_width = max(int(round(duration_seconds * pixels_per_second)), MIN_PLOT_WIDTH_PX)
    dynamic_figsize_width = pixel_width / float(dpi)

    def _pick_major_tick_step(total_seconds: float) -> float:
        candidates = [
            2, 5, 10, 15, 20, 30,
            60, 120, 180, 300, 600,
            900, 1200, 1800,
        ]
        for step in candidates:
            if (total_seconds / step) <= TARGET_MAJOR_TICKS:
                return float(step)
        coarse = max(total_seconds / TARGET_MAJOR_TICKS, 1800.0)
        return float(math.ceil(coarse / 1800.0) * 1800.0)
    
    # Calculate range dynamically based on data min/max
    data_min = np.min(voiced_midi)
    data_max = np.max(voiced_midi)
    
    min_m = np.floor(data_min) - 2
    max_m = np.ceil(data_max) + 2
    
    rng = np.arange(int(np.floor(min_m)), int(np.ceil(max_m)) + 1)
    
    # --- Main Plot ---
    fig, ax = plt.subplots(figsize=(dynamic_figsize_width, figsize_height), dpi=dpi)
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

    normalized_phrase_ranges: List[Tuple[float, float]] = []
    if phrase_ranges:
        for start_t, end_t in phrase_ranges:
            if not np.isfinite(start_t) or not np.isfinite(end_t):
                continue
            lo = float(min(start_t, end_t))
            hi = float(max(start_t, end_t))
            if hi <= lo:
                continue
            normalized_phrase_ranges.append((lo, hi))

    def _shared_phrase_window(t_a: float, t_b: float) -> Optional[Tuple[float, float]]:
        if not normalized_phrase_ranges:
            return None
        eps = 1e-9
        for p_start, p_end in normalized_phrase_ranges:
            if (p_start - eps) <= t_a <= (p_end + eps) and (p_start - eps) <= t_b <= (p_end + eps):
                return (p_start, p_end)
        return None
    
    # Join small gaps
    joined_segments = []
    if segments:
        curr = segments[0]
        for next_seg in segments[1:]:
            # if gap < threshold, merge
            t_end_prev = voiced_times[curr[1]-1]
            t_start_next = voiced_times[next_seg[0]]
            gap_is_small = (t_start_next - t_end_prev) <= join_gap_threshold
            same_phrase = _shared_phrase_window(float(t_end_prev), float(t_start_next))
            can_merge = gap_is_small and (same_phrase is not None or not normalized_phrase_ranges)
            if can_merge:
                curr = (curr[0], next_seg[1])
            else:
                joined_segments.append(curr)
                curr = next_seg
        joined_segments.append(curr)
        
    for seg in joined_segments:
        s, e = seg
        ax.plot(voiced_times[s:e], voiced_midi[s:e], color='tab:blue', linewidth=1.5, alpha=0.9)

    # --- Transcription Overlay ---
    # If a note timeline is provided (for example edited transcription),
    # render directly from notes so the plot reflects saved edits.
    if transcription_notes:
        rendered_any = False
        rendered_stationary = False
        rendered_inflection = False
        problematic_points: List[Dict[str, float]] = []
        for note in transcription_notes:
            start_t = float(getattr(note, "start", 0.0))
            end_t = float(getattr(note, "end", 0.0))
            pitch_midi = float(getattr(note, "pitch_midi", np.nan))
            if not np.isfinite(start_t) or not np.isfinite(end_t) or not np.isfinite(pitch_midi):
                continue
            if end_t <= start_t:
                continue

            confidence = float(getattr(note, "confidence", 1.0))
            is_inflection = confidence < 0.95
            raw_pitch = float(getattr(note, "raw_pitch_midi", np.nan))
            snapped_pitch = float(getattr(note, "snapped_pitch_midi", pitch_midi))
            corrected_pitch = float(getattr(note, "corrected_pitch_midi", pitch_midi))
            color = 'red' if is_inflection else 'orange'
            line_width = 2.0 if is_inflection else 3.0
            alpha = 0.75 if is_inflection else 0.82
            label = None
            if is_inflection and not rendered_inflection:
                label = "Inflection"
                rendered_inflection = True
            elif (not is_inflection) and not rendered_stationary:
                label = "Transcribed"
                rendered_stationary = True

            ax.hlines(
                y=pitch_midi,
                xmin=start_t,
                xmax=end_t,
                colors=color,
                linewidth=line_width,
                alpha=alpha,
                label=label,
                zorder=3,
            )
            rendered_any = True

            if is_inflection:
                rendered_mismatch = not np.isclose(
                    pitch_midi,
                    corrected_pitch,
                    atol=1e-9,
                )
                detached_from_raw = np.isfinite(raw_pitch) and abs(pitch_midi - raw_pitch) > 1.0
                if rendered_mismatch or detached_from_raw:
                    problematic_points.append(
                        {
                            "time": (start_t + end_t) * 0.5,
                            "raw_pitch_midi": raw_pitch,
                            "snapped_pitch_midi": snapped_pitch,
                            "corrected_pitch_midi": corrected_pitch,
                            "rendered_pitch_midi": pitch_midi,
                        }
                    )

        if rendered_inflection:
            inflection_points_x = []
            inflection_points_y = []
            for note in transcription_notes:
                confidence = float(getattr(note, "confidence", 1.0))
                if confidence >= 0.95:
                    continue
                start_t = float(getattr(note, "start", 0.0))
                end_t = float(getattr(note, "end", 0.0))
                pitch_midi = float(getattr(note, "pitch_midi", np.nan))
                if not np.isfinite(start_t) or not np.isfinite(end_t) or not np.isfinite(pitch_midi):
                    continue
                inflection_points_x.append((start_t + end_t) * 0.5)
                inflection_points_y.append(pitch_midi)
            if inflection_points_x:
                ax.scatter(
                    inflection_points_x,
                    inflection_points_y,
                    c='red',
                    s=10,
                    alpha=0.5,
                    zorder=4,
                )

        # Fallback for corrupted payloads: preserve legacy detection overlay.
        if not rendered_any:
            transcription_notes = None
        elif problematic_points:
            print(f"[PLOT_DIAG] Found {len(problematic_points)} problematic inflection points")
            for diag in problematic_points:
                print(
                    "[PLOT_DIAG] "
                    f"t={diag['time']:.3f}s "
                    f"raw={diag['raw_pitch_midi']:.4f} "
                    f"snapped={diag['snapped_pitch_midi']:.4f} "
                    f"corrected={diag['corrected_pitch_midi']:.4f} "
                    f"rendered={diag['rendered_pitch_midi']:.4f}"
                )

    if not transcription_notes:
        try:
            transcription_events = transcription.detect_stationary_events(
                pitch_hz=pitch_hz,
                timestamps=timestamps,
                voicing_mask=voiced_mask, # Note: voiced_mask is boolean array same length as pitch_hz
                tonic=tonic_midi_base,
                energy=pitch_data.energy,
                energy_threshold=transcription_energy_threshold,
                snap_mode='chromatic', # Default to chromatic for now
                smoothing_sigma_ms=transcription_smoothing_ms,
                derivative_threshold=transcription_derivative_threshold,
                min_event_duration=transcription_min_duration
                ,bias_cents=(bias_cents or 0.0)
            )
        except Exception:
            transcription_events = []

        for i, event in enumerate(transcription_events):
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

        try:
            inflection_times, inflection_pitches = transcription.detect_pitch_inflection_points(
                pitch_hz=pitch_hz,
                timestamps=timestamps,
                voicing_mask=voiced_mask,
                smoothing_sigma_ms=transcription_smoothing_ms, # Use same smoothing as transcription
            )
        except Exception:
            inflection_times = np.array([])
            inflection_pitches = np.array([])

        if len(inflection_times) > 0 and transcription_events:
            keep_mask = np.ones(len(inflection_times), dtype=bool)
            for event in transcription_events:
                overlap = (inflection_times >= event.start) & (inflection_times <= event.end)
                keep_mask[overlap] = False

            inflection_times = inflection_times[keep_mask]
            inflection_pitches = inflection_pitches[keep_mask]

        if len(inflection_times) > 0:
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
    ax.set_xlim(x_axis_start, x_axis_end)
    ax.set_xlabel('Time (s)')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(_pick_major_tick_step(duration_seconds)))
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

    y_axis_min = float(min_m - 1)
    y_axis_max = float(max_m + 1)
    return legend_b64, plot_b64, pixel_width, x_axis_start, x_axis_end, y_axis_min, y_axis_max

def create_scrollable_pitch_plot_html(
    pitch_data: PitchData,
    tonic: int,
    raga_name: str,
    audio_element_ids: List[str],
    raga_notes: Optional[Set[int]] = None,
    transcription_smoothing_ms: float = 70.0,
    transcription_min_duration: float = 0.04,
    transcription_derivative_threshold: float = 2.0,
    transcription_energy_threshold: float = 0.0,
    show_rms_overlay: bool = True,
    overlay_energy: Optional[np.ndarray] = None,
    overlay_timestamps: Optional[np.ndarray] = None,
    overlay_label: str = "RMS Energy",
    phrase_ranges: Optional[List[Tuple[float, float]]] = None,
    transcription_notes: Optional[List[Note]] = None,
    bias_cents: Optional[float] = None,
    hover_pitch_derivative_timestamps: Optional[np.ndarray] = None,
    hover_pitch_derivative_values: Optional[np.ndarray] = None,
    hover_pitch_derivative_voiced_mask: Optional[np.ndarray] = None,
) -> str:
    """
    Create HTML component for scrollable pitch plot with audio sync.
    """
    (
        legend_b64,
        plot_b64,
        px_width,
        x_axis_start,
        x_axis_end,
        y_axis_min,
        y_axis_max,
    ) = plot_pitch_wide_to_base64_with_legend(
        pitch_data, tonic, raga_name, raga_notes, 
        transcription_smoothing_ms=transcription_smoothing_ms,
        transcription_min_duration=transcription_min_duration,
        transcription_derivative_threshold=transcription_derivative_threshold,
        transcription_energy_threshold=transcription_energy_threshold,
        show_rms_overlay=show_rms_overlay,
        overlay_energy=overlay_energy,
        overlay_timestamps=overlay_timestamps,
        overlay_label=overlay_label,
        phrase_ranges=phrase_ranges,
        transcription_notes=transcription_notes,
        bias_cents=bias_cents,
    )
    
    if not plot_b64:
        return "<p>No pitch data for validation plot.</p>"

    unique_id = f"sp_{uuid.uuid4().hex[:6]}"
    duration = x_axis_end - x_axis_start

    def _safe_json_float(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    overlay_energy_payload: List[Optional[float]] = []
    if overlay_energy is not None:
        overlay_energy_payload = [_safe_json_float(v) for v in np.asarray(overlay_energy)]

    overlay_timestamps_payload: List[Optional[float]] = []
    if overlay_timestamps is not None:
        overlay_timestamps_payload = [_safe_json_float(v) for v in np.asarray(overlay_timestamps)]

    hover_pitch_timestamps_payload: List[float] = []
    hover_pitch_midi_payload: List[float] = []
    try:
        ts_arr = np.asarray(pitch_data.timestamps, dtype=float)
        hz_arr = np.asarray(pitch_data.pitch_hz, dtype=float)
        voiced_arr = np.asarray(pitch_data.voiced_mask, dtype=bool)
        valid = np.isfinite(ts_arr) & np.isfinite(hz_arr) & (hz_arr > 0)
        if voiced_arr.shape == valid.shape:
            valid = valid & voiced_arr
        if np.any(valid):
            ts_valid = ts_arr[valid]
            midi_valid = 69.0 + 12.0 * np.log2(np.maximum(hz_arr[valid], 1e-6) / 440.0)
            if bias_cents is not None:
                midi_valid = midi_valid - (float(bias_cents) / 100.0)
            for ts_val, midi_val in zip(ts_valid.tolist(), midi_valid.tolist()):
                safe_ts = _safe_json_float(ts_val)
                safe_midi = _safe_json_float(midi_val)
                if safe_ts is None or safe_midi is None:
                    continue
                hover_pitch_timestamps_payload.append(safe_ts)
                hover_pitch_midi_payload.append(safe_midi)
    except Exception:
        # Hover pitch sampling is optional; fallback keeps inspector functional.
        hover_pitch_timestamps_payload = []
        hover_pitch_midi_payload = []

    hover_derivative_timestamps_payload: List[float] = []
    hover_derivative_values_payload: List[float] = []
    try:
        if (
            hover_pitch_derivative_timestamps is not None
            and hover_pitch_derivative_values is not None
        ):
            d_ts = np.asarray(hover_pitch_derivative_timestamps, dtype=float)
            d_values = np.asarray(hover_pitch_derivative_values, dtype=float)
            valid = np.isfinite(d_ts) & np.isfinite(d_values)
            if hover_pitch_derivative_voiced_mask is not None:
                d_voiced = np.asarray(hover_pitch_derivative_voiced_mask, dtype=bool)
                if d_voiced.shape == valid.shape:
                    valid = valid & d_voiced
            if np.any(valid):
                for ts_val, d_val in zip(d_ts[valid].tolist(), d_values[valid].tolist()):
                    safe_ts = _safe_json_float(ts_val)
                    safe_d = _safe_json_float(d_val)
                    if safe_ts is None or safe_d is None:
                        continue
                    hover_derivative_timestamps_payload.append(safe_ts)
                    hover_derivative_values_payload.append(safe_d)
    except Exception:
        hover_derivative_timestamps_payload = []
        hover_derivative_values_payload = []

    try:
        tonic_pitch_class = (
            _parse_tonic(tonic)
            if isinstance(tonic, str)
            else int(round(float(tonic))) % 12
        )
    except Exception:
        tonic_pitch_class = 0

    note_payload: List[Dict[str, Any]] = []
    for note_idx, note in enumerate(transcription_notes or []):
        start = _safe_json_float(note.start)
        end = _safe_json_float(note.end)
        if start is None or end is None:
            continue
        note_id_raw = str(getattr(note, "id", "") or "").strip()
        note_id = note_id_raw if note_id_raw else f"n{note_idx + 1:05d}"
        note_payload.append(
            {
                "id": note_id,
                "start": start,
                "end": end,
                "sargam": str(getattr(note, "sargam", "") or ""),
                "pitch_midi": _safe_json_float(getattr(note, "pitch_midi", None)),
                "raw_pitch_midi": _safe_json_float(getattr(note, "raw_pitch_midi", None)),
                "snapped_pitch_midi": _safe_json_float(getattr(note, "snapped_pitch_midi", getattr(note, "pitch_midi", None))),
                "corrected_pitch_midi": _safe_json_float(getattr(note, "corrected_pitch_midi", getattr(note, "pitch_midi", None))),
                "rendered_pitch_midi": _safe_json_float(getattr(note, "pitch_midi", None)),
                "energy": _safe_json_float(getattr(note, "energy", None)),
                "confidence": _safe_json_float(getattr(note, "confidence", None)),
            }
        )
    
    phrase_payload: List[Dict[str, Any]] = []
    for phrase_idx, phrase_range in enumerate(phrase_ranges or []):
        if not isinstance(phrase_range, (list, tuple)) or len(phrase_range) < 2:
            continue
        start = _safe_json_float(phrase_range[0])
        end = _safe_json_float(phrase_range[1])
        if start is None or end is None:
            continue
        lo = min(start, end)
        hi = max(start, end)
        if hi <= lo:
            continue
        phrase_payload.append(
            {
                "id": f"p{phrase_idx + 1:04d}",
                "start": lo,
                "end": hi,
            }
        )

    html = f"""
    <div class="scroll-plot-wrapper" id="{unique_id}-wrapper" style="border: 1px solid #30363d; border-radius: 8px; overflow: hidden; margin: 20px 0; background: #0d1117;">
        <div style="display: flex; height: 700px; position: relative;">
            <!-- Legend (Sticky) -->
            <div style="flex: 0 0 200px; background: #161b22; border-right: 1px solid #30363d; z-index: 10; display: flex; align-items: center; justify-content: center;">
                <img src="data:image/png;base64,{legend_b64}" style="height: 100%; object-fit: contain; width: 100%;">
            </div>
            
            <!-- Scrollable Content -->
            <div id="{unique_id}-container" style="flex: 1; overflow-x: auto; position: relative; background: #0d1117;">
                <div id="{unique_id}-plot-layer" style="width: {px_width}px; height: 100%; position: relative; user-select: none;">
                    <img id="{unique_id}-plot-image" src="data:image/png;base64,{plot_b64}" draggable="false" style="width: 100%; height: 100%; display: block; user-select: none; -webkit-user-drag: none;">
                    <!-- Inspector markers -->
                    <div id="{unique_id}-range-band" style="position: absolute; left: 0; top: 0; bottom: 0; width: 0; display: none; background: rgba(56,139,253,0.18); border: 1px solid rgba(56,139,253,0.7); pointer-events: none; z-index: 4;"></div>
                    <div id="{unique_id}-point-marker" style="position: absolute; left: 0; top: 0; bottom: 0; width: 2px; background: #e3b341; display: none; pointer-events: none; z-index: 6;"></div>
                    <div id="{unique_id}-hover-v-guide" style="position: absolute; left: 0; top: 0; width: 0; height: 0; display: none; border-left: 1px dotted rgba(88,62,14,0.98); pointer-events: none; z-index: 6;"></div>
                    <div id="{unique_id}-hover-h-guide" style="position: absolute; left: 0; top: 0; width: 0; height: 0; display: none; border-top: 1px dotted rgba(88,62,14,0.98); pointer-events: none; z-index: 6;"></div>
                    <div id="{unique_id}-hover-tooltip" style="position: absolute; left: 0; top: 0; display: none; max-width: 320px; padding: 6px 8px; border-radius: 6px; border: 1px solid #30363d; background: rgba(17, 22, 29, 0.95); color: #c9d1d9; font-size: 11px; line-height: 1.4; pointer-events: none; z-index: 7; box-shadow: 0 2px 8px rgba(0,0,0,0.45);"></div>
                    <!-- Cursor Line -->
                    <div id="{unique_id}-cursor" style="position: absolute; left: 0; top: 0; bottom: 0; width: 2px; background: #e94560; box-shadow: 0 0 5px #e94560; pointer-events: none; z-index: 5;"></div>
                </div>
            </div>
        </div>
        <div style="padding: 10px; background: #161b22; border-top: 1px solid #30363d; display: flex; justify-content: space-between; color: #8b949e; font-size: 12px;">
             <span>Total Duration: {duration:.2f}s</span>
             <span id="{unique_id}-status">Synced to Audio</span>
        </div>
        <div id="{unique_id}-inspector" style="padding: 10px; background: #11161d; border-top: 1px solid #30363d; color: #c9d1d9; font-size: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap;">
                <span><strong>Selection:</strong> <span id="{unique_id}-selection-type">None</span></span>
                <button type="button" id="{unique_id}-clear-selection" style="padding: 4px 10px; border-radius: 6px; border: 1px solid #30363d; background: #21262d; color: #c9d1d9; cursor: pointer;">Clear selection</button>
            </div>
            <div id="{unique_id}-selection-time" style="margin-top: 6px; color: #8b949e;">Click for point query or drag for range query.</div>
            <div id="{unique_id}-selection-energy" style="margin-top: 4px; color: #8b949e;">Energy ({escape(overlay_label)}): unavailable</div>
            <div id="{unique_id}-selection-notes" style="margin-top: 8px; color: #c9d1d9;">No selection.</div>
        </div>
    </div>
    
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        const trackIds = {json.dumps(audio_element_ids)};
        const overlayEnergyValues = {json.dumps(overlay_energy_payload)};
        const overlayEnergyTimestamps = {json.dumps(overlay_timestamps_payload)};
        const hoverPitchTimestamps = {json.dumps(hover_pitch_timestamps_payload)};
        const hoverPitchMidi = {json.dumps(hover_pitch_midi_payload)};
        const hoverDerivativeTimestamps = {json.dumps(hover_derivative_timestamps_payload)};
        const hoverDerivativeValues = {json.dumps(hover_derivative_values_payload)};
        const transcriptionNotesRaw = {json.dumps(note_payload)};
        const phraseRangesRaw = {json.dumps(phrase_payload)};
        const overlayLabel = {json.dumps(overlay_label)};
        const tonicPitchClass = {tonic_pitch_class};
        const clearSelectionEventName = "raga-scroll-selection-clear";
        const container = document.getElementById("{unique_id}-container");
        const plotLayer = document.getElementById("{unique_id}-plot-layer");
        const plotImage = document.getElementById("{unique_id}-plot-image");
        const cursor = document.getElementById("{unique_id}-cursor");
        const pointMarker = document.getElementById("{unique_id}-point-marker");
        const hoverVGuide = document.getElementById("{unique_id}-hover-v-guide");
        const hoverHGuide = document.getElementById("{unique_id}-hover-h-guide");
        const hoverTooltip = document.getElementById("{unique_id}-hover-tooltip");
        const rangeBand = document.getElementById("{unique_id}-range-band");
        const selectionTypeEl = document.getElementById("{unique_id}-selection-type");
        const selectionTimeEl = document.getElementById("{unique_id}-selection-time");
        const selectionEnergyEl = document.getElementById("{unique_id}-selection-energy");
        const selectionNotesEl = document.getElementById("{unique_id}-selection-notes");
        const clearSelectionBtn = document.getElementById("{unique_id}-clear-selection");
        const xStart = {x_axis_start};
        const xEnd = {x_axis_end};
        const yAxisMin = {y_axis_min};
        const yAxisMax = {y_axis_max};
        const totalDuration = xEnd - xStart;
        const pixelWidth = {px_width};

        if (
            container &&
            cursor &&
            pointMarker &&
            hoverVGuide &&
            hoverHGuide &&
            hoverTooltip &&
            rangeBand &&
            selectionTypeEl &&
            selectionTimeEl &&
            selectionEnergyEl &&
            selectionNotesEl &&
            clearSelectionBtn
        ) {{
            console.log("Initializing sync + inspector for {unique_id}");
            
            // Plot margins (matches python subplots_adjust)
            const marginL = 0.005;
            const marginR = 0.995;
            const marginT = 0.05;
            const marginB = 0.10;
            const plotStartPx = marginL * pixelWidth;
            const plotEndPx = marginR * pixelWidth;
            const dragThresholdPx = 8;
            const pointToleranceSec = 0.02;
            const inspectorRowsVisible = 5;
            const inspectorRowHeightPx = 24;
            let seekSnapUntil = 0;
            let activeAudio = null;
            let isPointerDown = false;
            let pointerStartX = 0;
            let pointerCurrentX = 0;
            let lastPointerEventMeta = null;

            function isSelectionDebugEnabled() {{
                try {{
                    if (typeof window !== "undefined" && window.__RAGA_SELECTION_DEBUG__) {{
                        return true;
                    }}
                }} catch (_err) {{
                    // Ignore probe failures.
                }}
                try {{
                    if (
                        typeof window !== "undefined" &&
                        window.localStorage &&
                        window.localStorage.getItem("ragaSelectionDebug") === "1"
                    ) {{
                        return true;
                    }}
                }} catch (_err) {{
                    // Ignore private/incognito storage failures.
                }}
                return false;
            }}

            function selectionTrace(stage, payload) {{
                const entry = Object.assign(
                    {{
                        stage: stage,
                        sourcePlotId: "{unique_id}",
                        atMs: Date.now(),
                    }},
                    payload || {{}}
                );
                try {{
                    const key = "__RAGA_SELECTION_TRACE__";
                    const trace = Array.isArray(window[key]) ? window[key] : [];
                    trace.push(entry);
                    if (trace.length > 300) {{
                        trace.shift();
                    }}
                    window[key] = trace;
                }} catch (_err) {{
                    // Ignore cross-realm/window storage issues.
                }}
                if (isSelectionDebugEnabled()) {{
                    try {{
                        console.log("[RAGA_SELECTION][iframe][" + stage + "]", entry);
                    }} catch (_err) {{
                        // Ignore console failures.
                    }}
                }}
            }}

            const energyFrames = [];
            const frameCount = Math.min(overlayEnergyValues.length, overlayEnergyTimestamps.length);
            for (let i = 0; i < frameCount; i += 1) {{
                const ts = Number(overlayEnergyTimestamps[i]);
                const e = Number(overlayEnergyValues[i]);
                if (isFinite(ts) && isFinite(e)) {{
                    energyFrames.push({{ t: ts, e: e }});
                }}
            }}

            const pitchFrames = [];
            const pitchFrameCount = Math.min(hoverPitchTimestamps.length, hoverPitchMidi.length);
            for (let i = 0; i < pitchFrameCount; i += 1) {{
                const ts = Number(hoverPitchTimestamps[i]);
                const midi = Number(hoverPitchMidi[i]);
                if (isFinite(ts) && isFinite(midi)) {{
                    pitchFrames.push({{ t: ts, midi: midi }});
                }}
            }}

            const derivativeFrames = [];
            const derivativeFrameCount = Math.min(hoverDerivativeTimestamps.length, hoverDerivativeValues.length);
            for (let i = 0; i < derivativeFrameCount; i += 1) {{
                const ts = Number(hoverDerivativeTimestamps[i]);
                const d = Number(hoverDerivativeValues[i]);
                if (isFinite(ts) && isFinite(d)) {{
                    derivativeFrames.push({{ t: ts, d: d }});
                }}
            }}

            const transcriptionNotes = transcriptionNotesRaw
                .map(function(note) {{
                    const start = Number(note.start);
                    const end = Number(note.end);
                    return {{
                        id: String(note.id || ""),
                        start: start,
                        end: end,
                        sargam: (note.sargam || "").toString(),
                        pitch_midi: Number(note.pitch_midi),
                        raw_pitch_midi: Number(note.raw_pitch_midi),
                        snapped_pitch_midi: Number(note.snapped_pitch_midi),
                        corrected_pitch_midi: Number(note.corrected_pitch_midi),
                        rendered_pitch_midi: Number(note.rendered_pitch_midi),
                        energy: Number(note.energy),
                        confidence: Number(note.confidence),
                    }};
                }})
                .filter(function(note) {{
                    return isFinite(note.start) && isFinite(note.end) && (note.end >= note.start);
                }})
                .sort(function(a, b) {{
                    if (a.start !== b.start) return a.start - b.start;
                    return a.end - b.end;
                }});

            const phraseRanges = phraseRangesRaw
                .map(function(phrase) {{
                    const start = Number(phrase.start);
                    const end = Number(phrase.end);
                    return {{
                        id: String(phrase.id || ""),
                        start: start,
                        end: end,
                    }};
                }})
                .filter(function(phrase) {{
                    return isFinite(phrase.start) && isFinite(phrase.end) && (phrase.end > phrase.start);
                }})
                .sort(function(a, b) {{
                    if (a.start !== b.start) return a.start - b.start;
                    return a.end - b.end;
                }});

            function clamp(v, lo, hi) {{
                return Math.max(lo, Math.min(hi, v));
            }}

            function formatSeconds(v) {{
                return isFinite(v) ? v.toFixed(3) : "n/a";
            }}

            function formatMaybe(v, digits) {{
                if (!isFinite(v)) return "n/a";
                return v.toFixed(digits);
            }}

            function formatPitchTrace(note) {{
                return (
                    "raw " + formatMaybe(note.raw_pitch_midi, 3) +
                    " | snapped " + formatMaybe(note.snapped_pitch_midi, 3) +
                    " | corrected " + formatMaybe(note.corrected_pitch_midi, 3) +
                    " | rendered " + formatMaybe(note.rendered_pitch_midi, 3)
                );
            }}

            function centsDistance(notePitch, correctedPitch) {{
                if (!isFinite(notePitch) || !isFinite(correctedPitch)) {{
                    return NaN;
                }}
                return Math.abs((notePitch - correctedPitch) * 100.0);
            }}

            function formatCentsDistance(value) {{
                if (!isFinite(value)) {{
                    return "n/a";
                }}
                return value.toFixed(1) + "c";
            }}

            function formatPitchDistance(note) {{
                return (
                    "raw " + formatCentsDistance(centsDistance(note.raw_pitch_midi, note.corrected_pitch_midi)) +
                    " | snapped " + formatCentsDistance(centsDistance(note.snapped_pitch_midi, note.corrected_pitch_midi)) +
                    " | rendered " + formatCentsDistance(centsDistance(note.rendered_pitch_midi, note.corrected_pitch_midi))
                );
            }}

            function nearestPitchSample(t) {{
                if (!pitchFrames.length) {{
                    return null;
                }}

                let lo = 0;
                let hi = pitchFrames.length - 1;
                while (lo < hi) {{
                    const mid = Math.floor((lo + hi) / 2);
                    if (pitchFrames[mid].t < t) {{
                        lo = mid + 1;
                    }} else {{
                        hi = mid;
                    }}
                }}

                let idx = lo;
                if (
                    idx > 0 &&
                    Math.abs(pitchFrames[idx - 1].t - t) <= Math.abs(pitchFrames[idx].t - t)
                ) {{
                    idx = idx - 1;
                }}
                return pitchFrames[idx];
            }}

            function nearestDerivativeSample(t) {{
                if (!derivativeFrames.length) {{
                    return null;
                }}

                let lo = 0;
                let hi = derivativeFrames.length - 1;
                while (lo < hi) {{
                    const mid = Math.floor((lo + hi) / 2);
                    if (derivativeFrames[mid].t < t) {{
                        lo = mid + 1;
                    }} else {{
                        hi = mid;
                    }}
                }}

                let idx = lo;
                if (
                    idx > 0 &&
                    Math.abs(derivativeFrames[idx - 1].t - t) <= Math.abs(derivativeFrames[idx].t - t)
                ) {{
                    idx = idx - 1;
                }}
                return derivativeFrames[idx];
            }}

            function hoverPitchDerivativeLineHtml(derivativeSample) {{
                if (!derivativeSample || !isFinite(derivativeSample.d)) {{
                    return "<div style='margin-top:4px; color:#8b949e;'><strong>Pitch derivative @ t:</strong> unavailable</div>";
                }}

                const semitonesPerSec = derivativeSample.d;
                const centsPerSec = semitonesPerSec * 100.0;
                const semitoneText = (semitonesPerSec >= 0 ? "+" : "") + semitonesPerSec.toFixed(3);
                const centsText = (centsPerSec >= 0 ? "+" : "") + centsPerSec.toFixed(1);
                return (
                    "<div style='margin-top:4px;'>" +
                    "<strong>Pitch derivative @ t:</strong> " +
                    "<span>" + escapeHtml(semitoneText) + " st/s (" + escapeHtml(centsText) + " c/s)</span>" +
                    "</div>"
                );
            }}

            function midiToWesternLabel(midi) {{
                if (!isFinite(midi)) {{
                    return "";
                }}
                const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
                const rounded = Math.round(midi);
                const pc = ((rounded % 12) + 12) % 12;
                const octave = Math.floor(rounded / 12) - 1;
                return names[pc] + String(octave);
            }}

            function midiToSargamLabel(midi) {{
                if (!isFinite(midi)) {{
                    return "";
                }}
                const labels = ["Sa", "re", "Re", "ga", "Ga", "ma", "Ma", "Pa", "dha", "Dha", "ni", "Ni"];
                const rounded = Math.round(midi);
                const offset = ((rounded - tonicPitchClass) % 12 + 12) % 12;
                return labels[offset] || "";
            }}

            function hoverPitchLineHtml(sample) {{
                if (!sample || !isFinite(sample.midi)) {{
                    return "<div style='margin-top:4px; color:#8b949e;'><strong>Y-axis pitch @ t:</strong> unavailable</div>";
                }}
                const sargam = midiToSargamLabel(sample.midi);
                const western = midiToWesternLabel(sample.midi);
                const label = sargam
                    ? (sargam + " · " + western)
                    : western;
                return (
                    "<div style='margin-top:4px;'>" +
                    "<strong>Y-axis pitch @ t:</strong> " +
                    "<span>" + escapeHtml(label) + " (midi " + sample.midi.toFixed(2) + ")</span>" +
                    "</div>"
                );
            }}

            function escapeHtml(raw) {{
                return String(raw)
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#39;");
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

            function midiToY(midi) {{
                if (!isFinite(midi) || !isFinite(yAxisMin) || !isFinite(yAxisMax) || yAxisMax <= yAxisMin) {{
                    return NaN;
                }}
                const plotHeight = Math.max(plotLayer.clientHeight || container.clientHeight || 1, 1);
                const plotTopPx = marginT * plotHeight;
                const plotBottomPx = (1 - marginB) * plotHeight;
                const frac = clamp((midi - yAxisMin) / (yAxisMax - yAxisMin), 0, 1);
                return plotBottomPx - (frac * Math.max(plotBottomPx - plotTopPx, 1));
            }}

            function xFromEvent(evt) {{
                const containerRect = container.getBoundingClientRect();
                const layerRect = plotLayer.getBoundingClientRect();
                const rawXFromContainer = evt.clientX - containerRect.left + container.scrollLeft;
                const rawXFromLayer = evt.clientX - layerRect.left;
                const rawX = rawXFromContainer;
                const clampedX = clamp(rawX, plotStartPx, plotEndPx);
                lastPointerEventMeta = {{
                    eventType: String((evt && evt.type) || ""),
                    clientX: Number(evt && evt.clientX),
                    clientY: Number(evt && evt.clientY),
                    containerLeft: containerRect.left,
                    containerScrollLeft: container.scrollLeft,
                    layerLeft: layerRect.left,
                    rawXFromContainer: rawXFromContainer,
                    rawXFromLayer: rawXFromLayer,
                    xChosen: rawX,
                    xClamped: clampedX,
                }};
                return clampedX;
            }}

            function yFromEvent(evt) {{
                const layerRect = plotLayer.getBoundingClientRect();
                const rawY = evt.clientY - layerRect.top;
                return clamp(rawY, 0, Math.max(plotLayer.clientHeight, 0));
            }}

            function hideHoverGuides() {{
                hoverVGuide.style.display = "none";
                hoverVGuide.style.height = "0px";
                hoverHGuide.style.display = "none";
                hoverHGuide.style.width = "0px";
            }}

            function showHoverGuides(x, midi) {{
                const y = midiToY(midi);
                if (!isFinite(y)) {{
                    hideHoverGuides();
                    return;
                }}
                const safeX = clamp(x, plotStartPx, plotEndPx);
                const plotHeight = Math.max(plotLayer.clientHeight || container.clientHeight || 1, 1);
                const plotBottomPx = (1 - marginB) * plotHeight;
                const safeY = clamp(y, marginT * plotHeight, plotBottomPx);

                const vHeight = Math.max(0, plotBottomPx - safeY);
                hoverVGuide.style.left = safeX + "px";
                hoverVGuide.style.top = safeY + "px";
                hoverVGuide.style.height = vHeight + "px";
                hoverVGuide.style.display = "block";

                const hWidth = Math.max(0, safeX - plotStartPx);
                hoverHGuide.style.left = plotStartPx + "px";
                hoverHGuide.style.top = safeY + "px";
                hoverHGuide.style.width = hWidth + "px";
                hoverHGuide.style.display = "block";
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

            function setDefaultInspector() {{
                selectionTypeEl.textContent = "None";
                selectionTimeEl.textContent = "Click for point query or drag for range query.";
                selectionEnergyEl.textContent = "Energy (" + overlayLabel + "): unavailable";
                selectionNotesEl.textContent = "No selection.";
            }}

            function emitSelectionUpdate(detail) {{
                try {{
                    const eventDetail = Object.assign(
                        {{
                            sourcePlotId: "{unique_id}",
                            overlayLabel: overlayLabel,
                        }},
                        detail || {{}}
                    );
                    selectionTrace("iframe_emit", {{
                        mode: eventDetail.mode || null,
                        time: isFinite(Number(eventDetail.time)) ? Number(eventDetail.time) : null,
                        start: isFinite(Number(eventDetail.start)) ? Number(eventDetail.start) : null,
                        end: isFinite(Number(eventDetail.end)) ? Number(eventDetail.end) : null,
                        noteCount: Array.isArray(eventDetail.notes) ? eventDetail.notes.length : 0,
                        phraseCount: Array.isArray(eventDetail.phrases) ? eventDetail.phrases.length : 0,
                        pointer: lastPointerEventMeta,
                    }});
                    document.dispatchEvent(
                        new CustomEvent("raga-transcription-selection", {{
                            detail: eventDetail
                        }})
                    );
                }} catch (_err) {{
                    // Ignore event bridge failures.
                }}
            }}

            function clearSelection() {{
                pointMarker.style.display = "none";
                rangeBand.style.display = "none";
                hideHoverTooltip();
                setDefaultInspector();
                emitSelectionUpdate({{ mode: "none" }});
            }}

            function nearestEnergySample(t) {{
                if (!energyFrames.length) {{
                    return null;
                }}

                let lo = 0;
                let hi = energyFrames.length - 1;
                while (lo < hi) {{
                    const mid = Math.floor((lo + hi) / 2);
                    if (energyFrames[mid].t < t) {{
                        lo = mid + 1;
                    }} else {{
                        hi = mid;
                    }}
                }}

                let idx = lo;
                if (
                    idx > 0 &&
                    Math.abs(energyFrames[idx - 1].t - t) <= Math.abs(energyFrames[idx].t - t)
                ) {{
                    idx = idx - 1;
                }}
                return energyFrames[idx];
            }}

            function computeRangeEnergy(t0, t1) {{
                if (!energyFrames.length) {{
                    return null;
                }}
                const lo = Math.min(t0, t1);
                const hi = Math.max(t0, t1);
                let count = 0;
                let sum = 0;
                let minVal = Number.POSITIVE_INFINITY;
                let maxVal = Number.NEGATIVE_INFINITY;

                energyFrames.forEach(function(frame) {{
                    if (frame.t < lo || frame.t > hi) {{
                        return;
                    }}
                    count += 1;
                    sum += frame.e;
                    if (frame.e < minVal) minVal = frame.e;
                    if (frame.e > maxVal) maxVal = frame.e;
                }});

                if (!count) {{
                    return null;
                }}
                return {{
                    min: minVal,
                    mean: sum / count,
                    max: maxVal,
                    count: count,
                }};
            }}

            function noteDistanceToTime(note, t) {{
                if (t < note.start) return note.start - t;
                if (t > note.end) return t - note.end;
                return 0;
            }}

            function normalizeSargamLabel(raw) {{
                return String(raw || "").replace(/[·'`’]+/g, "").trim();
            }}

            function formatNoteSymbol(note) {{
                const label = normalizeSargamLabel(note.sargam);
                if (label) {{
                    return label;
                }}
                if (isFinite(note.pitch_midi)) {{
                    return note.pitch_midi.toFixed(2);
                }}
                return "?";
            }}

            function noteKindLabel(note) {{
                if (isFinite(note.confidence) && note.confidence < 0.95) {{
                    return "Inflection";
                }}
                return "Stationary";
            }}

            function noteBreakdownTableHtml(notes) {{
                const rows = notes.map(function(note) {{
                    const duration = Math.max(0, note.end - note.start);
                    const noteLabel = formatNoteSymbol(note) + " (" + noteKindLabel(note) + ")";
                    return (
                        "<tr>" +
                        "<td style='padding:4px 6px; border-top:1px solid #30363d;'>" + escapeHtml(noteLabel) + "</td>" +
                        "<td style='padding:4px 6px; border-top:1px solid #30363d;'>" + formatSeconds(duration) + "s</td>" +
                        "<td style='padding:4px 6px; border-top:1px solid #30363d;'>midi " + formatMaybe(note.pitch_midi, 2) + "</td>" +
                        "<td style='padding:4px 6px; border-top:1px solid #30363d; color:#8b949e;'>" + escapeHtml(formatPitchDistance(note)) + "</td>" +
                        "<td style='padding:4px 6px; border-top:1px solid #30363d; color:#8b949e;'>" + escapeHtml(formatPitchTrace(note)) + "</td>" +
                        "</tr>"
                    );
                }}).join("");
                const maxHeight = (inspectorRowsVisible * inspectorRowHeightPx) + 30;
                return (
                    "<div style='margin-top:6px; max-height:" + maxHeight + "px; overflow-y:auto; border:1px solid #30363d; border-radius:6px;'>" +
                    "<table style='width:100%; border-collapse:collapse; font-size:12px;'>" +
                    "<thead>" +
                    "<tr>" +
                    "<th style='text-align:left; padding:5px 6px; background:#161b22;'>Note</th>" +
                    "<th style='text-align:left; padding:5px 6px; background:#161b22;'>Duration</th>" +
                    "<th style='text-align:left; padding:5px 6px; background:#161b22;'>MIDI</th>" +
                    "<th style='text-align:left; padding:5px 6px; background:#161b22;'>Dist to Corrected</th>" +
                    "<th style='text-align:left; padding:5px 6px; background:#161b22;'>Pitch Trace</th>" +
                    "</tr>" +
                    "</thead>" +
                    "<tbody>" + rows + "</tbody>" +
                    "</table>" +
                    "</div>"
                );
            }}

            function renderPointSelection(t, energySample, notes, usedNearest) {{
                selectionTypeEl.textContent = "Point";
                selectionTimeEl.textContent = "Time: t = " + formatSeconds(t) + "s";
                if (energySample) {{
                    selectionEnergyEl.textContent =
                        "Energy (" +
                        overlayLabel +
                        "): " +
                        energySample.e.toFixed(4) +
                        " @ " +
                        formatSeconds(energySample.t) +
                        "s";
                }} else {{
                    selectionEnergyEl.textContent = "Energy (" + overlayLabel + "): unavailable";
                }}

                if (!notes.length) {{
                    selectionNotesEl.innerHTML = "<strong>Transcription:</strong> No transcription notes available.";
                    return;
                }}

                const header = usedNearest ? "Nearest transcription note" : "Active transcription note(s)";
                const summary = notes.map(function(note) {{ return formatNoteSymbol(note); }}).join(" ");
                selectionNotesEl.innerHTML =
                    "<strong>" + escapeHtml(header) + ":</strong> " +
                    escapeHtml(summary) +
                    noteBreakdownTableHtml(notes);
            }}

            function renderRangeSelection(t0, t1, energyStats, notes) {{
                const lo = Math.min(t0, t1);
                const hi = Math.max(t0, t1);
                selectionTypeEl.textContent = "Range";
                selectionTimeEl.textContent =
                    "Range: " + formatSeconds(lo) + "s - " + formatSeconds(hi) +
                    "s (duration " + formatSeconds(hi - lo) + "s)";

                if (energyStats) {{
                    selectionEnergyEl.textContent =
                        "Energy (" + overlayLabel + "): min " +
                        energyStats.min.toFixed(4) +
                        ", mean " +
                        energyStats.mean.toFixed(4) +
                        ", max " +
                        energyStats.max.toFixed(4) +
                        " (" + energyStats.count + " frames)";
                }} else {{
                    selectionEnergyEl.textContent =
                        "Energy (" + overlayLabel + "): unavailable in selected range";
                }}

                if (!notes.length) {{
                    selectionNotesEl.innerHTML = "<strong>Transcription:</strong> No overlapping transcription notes.";
                    return;
                }}

                const summary = notes.map(function(note) {{ return formatNoteSymbol(note); }}).join(" ");
                selectionNotesEl.innerHTML =
                    "<strong>Transcription summary:</strong> " +
                    escapeHtml(summary) +
                    noteBreakdownTableHtml(notes);
            }}

            function resolveNotesAtTime(t) {{
                let overlapping = transcriptionNotes.filter(function(note) {{
                    return (note.start - pointToleranceSec) <= t && t <= (note.end + pointToleranceSec);
                }});
                let usedNearest = false;

                if (!overlapping.length && transcriptionNotes.length) {{
                    let nearest = transcriptionNotes[0];
                    let bestDistance = noteDistanceToTime(nearest, t);
                    for (let i = 1; i < transcriptionNotes.length; i += 1) {{
                        const candidate = transcriptionNotes[i];
                        const distance = noteDistanceToTime(candidate, t);
                        if (distance < bestDistance) {{
                            nearest = candidate;
                            bestDistance = distance;
                        }}
                    }}
                    overlapping = [nearest];
                    usedNearest = true;
                }}

                return {{
                    notes: overlapping,
                    usedNearest: usedNearest,
                }};
            }}

            function resolvePhrasesAtTime(t) {{
                return phraseRanges.filter(function(phrase) {{
                    return phrase.start <= t && t <= phrase.end;
                }});
            }}

            function resolvePhrasesInRange(lo, hi) {{
                return phraseRanges.filter(function(phrase) {{
                    return phrase.end >= lo && phrase.start <= hi;
                }});
            }}

            function hideHoverTooltip() {{
                hideHoverGuides();
                hoverTooltip.style.display = "none";
                hoverTooltip.innerHTML = "";
            }}

            function renderHoverTooltip(plotX, pointerY) {{
                const x = clamp(plotX, plotStartPx, plotEndPx);
                const t = xToTime(x);
                const pitchSample = nearestPitchSample(t);
                const derivativeSample = nearestDerivativeSample(t);
                const resolved = resolveNotesAtTime(t);
                const notes = resolved.notes;

                if (!pitchSample && !notes.length) {{
                    hideHoverTooltip();
                    return;
                }}

                if (pitchSample && isFinite(pitchSample.t) && isFinite(pitchSample.midi)) {{
                    showHoverGuides(timeToX(pitchSample.t), pitchSample.midi);
                }} else {{
                    hideHoverGuides();
                }}

                const nearestTag = resolved.usedNearest ? " (nearest)" : "";
                const pitchLine = hoverPitchLineHtml(pitchSample);
                const derivativeLine = hoverPitchDerivativeLineHtml(derivativeSample);
                let noteSection = "<div style='margin-top:4px; color:#8b949e;'><strong>Nearest transcribed note:</strong> unavailable</div>";
                if (notes.length) {{
                    const noteHeader = resolved.usedNearest
                        ? "Nearest transcribed note"
                        : "Active transcribed note(s)";
                    const maxTooltipItems = 2;
                    const rows = notes.slice(0, maxTooltipItems).map(function(note) {{
                        const symbol = formatNoteSymbol(note);
                        const kind = noteKindLabel(note);
                        const region = formatSeconds(note.start) + "s - " + formatSeconds(note.end) + "s";
                        const cents = formatPitchDistance(note);
                        return (
                            "<div style='margin-top:4px;'>" +
                            "<strong>" + escapeHtml(symbol) + "</strong> " +
                            "<span style='color:#8b949e;'>" + escapeHtml(kind) + "</span><br>" +
                            "<span style='color:#8b949e;'>region " + escapeHtml(region) + "</span><br>" +
                            "<span style='color:#8b949e;'>dist to corrected: " + escapeHtml(cents) + "</span>" +
                            "</div>"
                        );
                    }}).join("");
                    const remaining = notes.length - maxTooltipItems;
                    noteSection =
                        "<div style='margin-top:4px;'><strong>" + escapeHtml(noteHeader) + ":</strong></div>" +
                        rows +
                        (remaining > 0
                            ? "<div style='margin-top:4px; color:#8b949e;'>+" + remaining + " more note(s)</div>"
                            : "");
                }}

                hoverTooltip.innerHTML =
                    "<div><strong>t=" + escapeHtml(formatSeconds(t)) + "s" + escapeHtml(nearestTag) + "</strong></div>" +
                    pitchLine +
                    derivativeLine +
                    noteSection;

                const maxLeft = Math.max(0, pixelWidth - 340);
                const maxTop = Math.max(0, container.clientHeight - 120);
                hoverTooltip.style.left = clamp(x + 14, 0, maxLeft) + "px";
                hoverTooltip.style.top = clamp(pointerY + 14, 0, maxTop) + "px";
                hoverTooltip.style.display = "block";
            }}

            function runPointQuery(plotX, shouldSeek) {{
                const x = clamp(plotX, plotStartPx, plotEndPx);
                const t = xToTime(x);
                const energySample = nearestEnergySample(t);
                const resolved = resolveNotesAtTime(t);
                const overlapping = resolved.notes;
                const usedNearest = resolved.usedNearest;
                const phrases = resolvePhrasesAtTime(t);
                const pointDetail = {{
                    mode: "point",
                    time: t,
                    start: t,
                    end: t,
                    notes: overlapping,
                    phrases: phrases,
                    usedNearest: usedNearest,
                    energy: energySample ? energySample.e : null,
                    energySampleTime: energySample ? energySample.t : null,
                    debug: {{
                        inputPlotX: plotX,
                        clampedPlotX: x,
                        plotStartPx: plotStartPx,
                        plotEndPx: plotEndPx,
                        xStart: xStart,
                        xEnd: xEnd,
                        pointer: lastPointerEventMeta,
                    }},
                }};

                pointMarker.style.display = "block";
                pointMarker.style.left = x + "px";
                rangeBand.style.display = "none";
                renderPointSelection(t, energySample, overlapping, usedNearest);
                emitSelectionUpdate(pointDetail);

                if (shouldSeek && isFinite(t)) {{
                    const sourceBeforeSeek = getPlayingAudio() || activeAudio;
                    seekAllTracks(t);
                    activeAudio = resolveActiveAudio(sourceBeforeSeek || undefined);
                    const syncedT = (activeAudio ? activeAudio.currentTime : t) || t;
                    renderAtTime(syncedT, true);
                    seekSnapUntil = performance.now() + 300;
                }}
            }}

            function runRangeQuery(x0, x1) {{
                const left = clamp(Math.min(x0, x1), plotStartPx, plotEndPx);
                const right = clamp(Math.max(x0, x1), plotStartPx, plotEndPx);
                const width = Math.max(1, right - left);
                const t0 = xToTime(left);
                const t1 = xToTime(right);
                const lo = Math.min(t0, t1);
                const hi = Math.max(t0, t1);
                const notes = transcriptionNotes.filter(function(note) {{
                    return note.end >= lo && note.start <= hi;
                }});
                const phrases = resolvePhrasesInRange(lo, hi);
                const energyStats = computeRangeEnergy(lo, hi);

                rangeBand.style.display = "block";
                rangeBand.style.left = left + "px";
                rangeBand.style.width = width + "px";
                pointMarker.style.display = "none";
                renderRangeSelection(lo, hi, energyStats, notes);
                emitSelectionUpdate({{
                    mode: "range",
                    start: lo,
                    end: hi,
                    notes: notes,
                    phrases: phrases,
                    energyMin: energyStats ? energyStats.min : null,
                    energyMean: energyStats ? energyStats.mean : null,
                    energyMax: energyStats ? energyStats.max : null,
                    debug: {{
                        inputStartX: x0,
                        inputEndX: x1,
                        clampedStartX: left,
                        clampedEndX: right,
                        plotStartPx: plotStartPx,
                        plotEndPx: plotEndPx,
                        xStart: xStart,
                        xEnd: xEnd,
                        pointer: lastPointerEventMeta,
                    }},
                }});
            }}

            function updateDragPreview(startX, currentX) {{
                const left = Math.min(startX, currentX);
                const width = Math.max(1, Math.abs(currentX - startX));
                rangeBand.style.display = "block";
                rangeBand.style.left = left + "px";
                rangeBand.style.width = width + "px";
                pointMarker.style.display = "none";
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

            setDefaultInspector();

            [container, plotLayer, plotImage].forEach(function(target) {{
                if (!target) return;
                target.addEventListener("dragstart", function(e) {{
                    e.preventDefault();
                }});
            }});

            container.addEventListener("mousedown", function(e) {{
                if (e.button !== 0) {{
                    return;
                }}
                e.preventDefault();
                hideHoverTooltip();
                isPointerDown = true;
                pointerStartX = xFromEvent(e);
                pointerCurrentX = pointerStartX;
            }});

            container.addEventListener("mousemove", function(e) {{
                if (isPointerDown) {{
                    return;
                }}
                renderHoverTooltip(xFromEvent(e), yFromEvent(e));
            }});

            container.addEventListener("mouseleave", function() {{
                if (!isPointerDown) {{
                    hideHoverTooltip();
                }}
            }});

            document.addEventListener("mousemove", function(e) {{
                if (!isPointerDown) {{
                    return;
                }}
                pointerCurrentX = xFromEvent(e);
                if (Math.abs(pointerCurrentX - pointerStartX) >= dragThresholdPx) {{
                    updateDragPreview(pointerStartX, pointerCurrentX);
                }}
            }});

            document.addEventListener("mouseup", function(e) {{
                if (!isPointerDown) {{
                    return;
                }}
                pointerCurrentX = xFromEvent(e);
                const movedPx = Math.abs(pointerCurrentX - pointerStartX);
                isPointerDown = false;
                hideHoverTooltip();

                if (movedPx >= dragThresholdPx) {{
                    runRangeQuery(pointerStartX, pointerCurrentX);
                }} else {{
                    runPointQuery(pointerCurrentX, true);
                }}
            }});

            clearSelectionBtn.addEventListener("click", function() {{
                clearSelection();
            }});

            document.addEventListener("keydown", function(e) {{
                if (e.key === "Escape") {{
                    clearSelection();
                }}
            }});

            document.addEventListener(clearSelectionEventName, function() {{
                clearSelection();
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
            color: #79c0ff;
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


def _relpath_or_raw(path_value: Any, root_dir: str) -> Any:
    if path_value is None:
        return None
    path_text = str(path_value)
    if not path_text:
        return path_text
    try:
        return os.path.relpath(path_text, root_dir)
    except Exception:
        return path_text


def _abspath_or_raw(path_value: Any) -> str:
    if path_value is None:
        return ""
    path_text = str(path_value).strip()
    if not path_text:
        return ""
    try:
        return str(Path(path_text).expanduser().resolve())
    except Exception:
        return path_text


def _to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if isinstance(value, np.generic):
        return _to_json_compatible(value.item())

    if isinstance(value, np.ndarray):
        return [_to_json_compatible(item) for item in value.tolist()]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(item) for item in value]

    return str(value)


def _build_analysis_report_metadata(
    results: 'AnalysisResults',
    stats: AnalysisStats,
    output_dir: str,
    report_path: str,
) -> Dict[str, Any]:
    stem_dir = os.path.abspath(output_dir)
    config = results.config
    tonic_for_editor = int(results.detected_tonic) if results.detected_tonic is not None else 0
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "report_filename": os.path.basename(report_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "audio_path": _abspath_or_raw(getattr(config, "audio_path", "")),
            "vocals_path": _abspath_or_raw(getattr(config, "vocals_path", "")),
            "accompaniment_path": _abspath_or_raw(getattr(config, "accompaniment_path", "")),
            "melody_source": str(getattr(config, "melody_source", "separated") or "separated"),
            "transcription_smoothing_ms": float(getattr(config, "transcription_smoothing_ms", 70.0)),
            "transcription_min_duration": float(getattr(config, "transcription_min_duration", 0.04)),
            "transcription_derivative_threshold": float(getattr(config, "transcription_derivative_threshold", 2.0)),
            "energy_threshold": float(getattr(config, "energy_threshold", 0.0)),
            "show_rms_overlay": bool(getattr(config, "show_rms_overlay", True)),
        },
        "detected": {
            "tonic": int(results.detected_tonic) if results.detected_tonic is not None else None,
            "raga": str(results.detected_raga) if results.detected_raga else None,
        },
        "stats": {
            "correction_summary": _to_json_compatible(stats.correction_summary),
            "pattern_analysis": _to_json_compatible(stats.pattern_analysis),
            "raga_name": stats.raga_name,
            "tonic": stats.tonic,
            "transition_matrix_path": _relpath_or_raw(stats.transition_matrix_path, stem_dir),
            "pitch_plot_path": _relpath_or_raw(stats.pitch_plot_path, stem_dir),
        },
        "plot_paths": {
            key: _relpath_or_raw(path, stem_dir)
            for key, path in (results.plot_paths or {}).items()
        },
        "pitch_csv_paths": {
            "original": ["composite_pitch_data.csv"],
            "vocals": ["melody_pitch_data.csv", "vocals_pitch_data.csv"],
            "accompaniment": ["accompaniment_pitch_data.csv"],
        },
        # Base payload for local-app transcription editing workspace.
        "transcription_edit_payload": _build_transcription_editor_payload(
            results.notes,
            results.phrases,
            tonic_for_editor,
        ),
        "transcription_derivative_profile": {
            "timestamps": _to_json_compatible(results.transcription_derivative_timestamps),
            "values": _to_json_compatible(results.transcription_derivative_values),
            "voiced_mask": _to_json_compatible(results.transcription_derivative_voiced_mask),
        },
    }
    return _to_json_compatible(metadata)


def _write_analysis_report_metadata(
    results: 'AnalysisResults',
    stats: AnalysisStats,
    output_dir: str,
    report_path: str,
) -> None:
    metadata = _build_analysis_report_metadata(results, stats, output_dir, report_path)
    metadata_path = str(Path(report_path).with_suffix(".meta.json"))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)


def generate_analysis_report(
    results: 'AnalysisResults',
    stats: AnalysisStats,
    output_dir: str,
    report_filename: str = "analysis_report.html",
) -> str:
    """
    Generate detailed analysis report (Phase 2).
    """
    report_path = os.path.join(output_dir, report_filename)
    
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

                derivative_timestamps_for_plot: Optional[np.ndarray] = None
                derivative_values_for_plot: Optional[np.ndarray] = None
                derivative_voiced_for_plot: Optional[np.ndarray] = None
                if (
                    results.transcription_derivative_timestamps is not None
                    and results.transcription_derivative_values is not None
                    and results.pitch_data_vocals is not None
                    and pitch_data is results.pitch_data_vocals
                ):
                    derivative_timestamps_for_plot = results.transcription_derivative_timestamps
                    derivative_values_for_plot = results.transcription_derivative_values
                    derivative_voiced_for_plot = results.transcription_derivative_voiced_mask

                try:
                    scroll_plot_html = create_scrollable_pitch_plot_html(
                        pitch_data,
                        tonic=results.detected_tonic or 0,
                        raga_name=stats.raga_name,
                        audio_element_ids=audio_element_ids,
                        transcription_smoothing_ms=results.config.transcription_smoothing_ms,
                        transcription_min_duration=results.config.transcription_min_duration,
                        transcription_derivative_threshold=results.config.transcription_derivative_threshold,
                        transcription_energy_threshold=results.config.energy_threshold,
                        show_rms_overlay=getattr(results.config, 'show_rms_overlay', True),
                        overlay_energy=e_pitch.energy,
                        overlay_timestamps=e_pitch.timestamps,
                        overlay_label=f"{e_label} RMS",
                        phrase_ranges=phrase_ranges,
                        transcription_notes=results.notes,
                        bias_cents=results.gmm_bias_cents,
                        hover_pitch_derivative_timestamps=derivative_timestamps_for_plot,
                        hover_pitch_derivative_values=derivative_values_for_plot,
                        hover_pitch_derivative_voiced_mask=derivative_voiced_for_plot,
                    )
                except Exception as e:
                    scroll_plot_html = (
                        f"<p>Error generating scrollable plot for pitch={label}, overlay={e_label}: {e}</p>"
                    )

                overlay_parts.append(f'''
                <div class="energy-overlay-panel" id="overlay-panel-{key}-{e_key}" data-track-key="{key}" data-energy-key="{e_key}" style="display: {'block' if overlay_visible else 'none'};">
                    <h3>Pitch: {label} | Amplitude Overlay: {e_label}</h3>
                    <p style="font-size: 0.9em; color: #8b949e;">Pitch track uses <strong>{label}</strong>; amplitude overlay uses <strong>{e_label}</strong>. Stationary transcription overlays are orange, inflections are red, and phrase spans are highlighted in yellow.</p>
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
                document.dispatchEvent(
                    new CustomEvent("raga-scroll-selection-clear", {{
                        detail: {{ trackKey: trackKey, energyKey: currentEnergy }}
                    }})
                );
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
    note_duration_html = ""
    if 'note_duration_histogram' in results.plot_paths:
        note_duration_rel = os.path.relpath(results.plot_paths['note_duration_histogram'], output_dir)
        note_duration_html = f"""
        <div class="viz-container">
            <h3>Note Duration Histogram</h3>
            <img src="{note_duration_rel}" alt="Note Duration Histogram" style="width:100%; max-width:900px;">
        </div>
        """
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
        {note_duration_html}
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
    try:
        _write_analysis_report_metadata(results, stats, output_dir, report_path)
    except Exception as exc:
        print(f"[WARN] Failed to write analysis report metadata: {exc}")
        
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
