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
from typing import List, Dict, Set, Optional, Any, Union, Tuple
import os
import json
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from .config import PipelineConfig
from .audio import PitchData
from .analysis import HistogramData, PeakData, GMMResult
from .sequence import Note, Phrase, OFFSET_TO_SARGAM


# =============================================================================
# RESULTS CONTAINER
# =============================================================================

@dataclass
class AnalysisResults:
    """Container for all analysis results."""
    
    config: PipelineConfig
    
    # Pitch data
    pitch_data_vocals: Optional[PitchData] = None
    pitch_data_accomp: Optional[PitchData] = None
    
    # Histogram analysis
    histogram_vocals: Optional[HistogramData] = None
    histogram_accomp: Optional[HistogramData] = None
    peaks_vocals: Optional[PeakData] = None
    
    # Raga matching
    candidates: Optional[pd.DataFrame] = None
    detected_tonic: Optional[int] = None
    detected_raga: Optional[str] = None
    
    # GMM analysis
    gmm_results: List[GMMResult] = field(default_factory=list)
    
    # Sequence analysis
    notes: List[Note] = field(default_factory=list)
    phrases: List[Phrase] = field(default_factory=list)
    phrase_clusters: Dict[int, List[Phrase]] = field(default_factory=dict)
    transition_matrix: Optional[np.ndarray] = None
    
    # Plot paths
    plot_paths: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# STATIC PLOTS
# =============================================================================

def plot_histograms(
    histogram: HistogramData,
    peaks: PeakData,
    output_path: str,
    title: str = "Pitch Class Histogram",
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
    ax1.plot(histogram.bin_centers_high, raw_norm, color='lightsteelblue', alpha=0.4, label='Raw (100-bin)')
    ax1.plot(histogram.bin_centers_high, smoothed_norm, color='navy', linewidth=2, label='Smoothed (Gaussian)')
    
    # Mark validated peaks with X markers
    for idx in peaks.validated_indices:
        cent = histogram.bin_centers_high[idx]
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
    ax2.bar(histogram.bin_centers_low, low_norm,
            width=36, alpha=0.7, color='darkorange', label='Stable Peaks (33-bin)')
    
    for idx in peaks.low_res_indices:
        cent = histogram.bin_centers_low[idx]
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


def plot_gmm_overlay(
    histogram: HistogramData,
    gmm_results: List[GMMResult],
    output_path: str,
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
    
    # Plot histogram
    ax.bar(histogram.bin_centers_high, histogram.smoothed_high_norm,
           width=12, alpha=0.6, color='steelblue', label='Histogram')
    
    # Overlay GMM fits
    x = np.linspace(0, 1200, 1200)
    
    for gmm in gmm_results:
        color = plt.cm.Set2(gmm.nearest_note / 12)
        
        # Plot Gaussian for each component
        for mean, sigma, weight in zip(gmm.means, gmm.sigmas, gmm.weights):
            y = weight * stats.norm.pdf(x, mean, sigma)
            y = y * histogram.smoothed_high_norm[gmm.peak_idx] / max(y.max(), 1e-10)
            ax.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
        
        # Annotate with Western notation
        western_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        western = western_notes[gmm.nearest_note]
        ax.annotate(
            f'{western}\nμ={gmm.primary_mean:.0f}¢\nσ={gmm.primary_sigma:.1f}¢',
            (gmm.peak_cent, histogram.smoothed_high_norm[gmm.peak_idx]),
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
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
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
    traces.append({
        "x": voiced_times,
        "y": voiced_midi,
        "mode": "lines",
        "line": {"color": "royalblue", "width": 1.5},
        "name": "Pitch contour",
        "type": "scattergl",
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
        "showlegend": False,
        "shapes": shapes,
        "annotations": annotations,
    }
    
    return json.dumps({"data": traces, "layout": layout})


def plot_pitch_with_sargam_lines(
    time_axis: np.ndarray,
    pitch_values: np.ndarray,
    tonic: Union[int, str],
    raga_name: str,
    output_path: str,
    figsize: Tuple[int, int] = (15, 6)
):
    """
    Plot pitch contour with sargam lines.
    Slightly adapted from user snippet to save to file.
    """
    plt.figure(figsize=figsize)
    
    # Plot pitch
    plt.plot(time_axis, pitch_values, label='Pitch', color='blue', alpha=0.6, linewidth=1)
    
    # Clean tonic
    tonic_val = _parse_tonic(tonic) if isinstance(tonic, str) else int(tonic)
    
    # Sargam lines
    patches_list = []
    
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
    plt.title(f"Pitch Contour with Sargam Lines (Tonic: {_tonic_name(tonic)}, Raga: {raga_name})")
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
    
    # Normalize for coloring if raw counts passed? Assume probability matrix passed usually, but check
    # If integer matrix, maybe normalize for color? Actually commonly passed counts / row_sums probability matrix.
    # Just display what's passed.
    
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
    
    # Get audio duration
    try:
        import librosa
        audio_duration = librosa.get_duration(path=results.config.vocals_path)
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
        )
        
        sections.append(_generate_audio_player_section(
            results.config.vocals_path,
            pitch_json,
            vocals_id,
            "Vocals Analysis",
            tonic,
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
) -> str:
    """Generate audio player with synchronized pitch contour."""
    
    audio_filename = os.path.basename(audio_path)
    
    return f'''
    <section id="{player_id}-section" class="audio-player-section">
        <h2>{title}</h2>
        <div class="audio-player-container">
            <audio id="{player_id}-audio" controls>
                <source src="{audio_filename}" type="audio/wav">
            </audio>
            <div id="{player_id}-plot" class="pitch-plot"></div>
            <div class="time-display">
                <span id="{player_id}-time">0.00s</span> | 
                <span id="{player_id}-sargam">--</span>
            </div>
        </div>
        <script>
            (function() {{
                var plotData = {pitch_json};
                var plotDiv = document.getElementById('{player_id}-plot');
                Plotly.newPlot(plotDiv, plotData.data, plotData.layout);
                
                // Setup audio sync
                setupAudioSync('{player_id}', {tonic});
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
    for i, row in candidates.head(40).iterrows():
        rank = row.get('rank', i+1)
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
        <p>Showing top 40 candidates. Scroll horizontally to see all parameters.</p>
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
    original_rel = os.path.relpath(config.audio_path, config.stem_dir)
    
    return f'''
    <section id="audio-tracks">
        <h2>Audio Tracks</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
            <div>
                <p><strong>Original</strong></p>
                <audio controls src="{original_rel}" style="width:100%"></audio>
            </div>
            <div>
                <p><strong>Vocals</strong></p>
                <audio controls src="{vocals_rel}" style="width:100%"></audio>
            </div>
            <div>
                <p><strong>Accompaniment</strong></p>
                <audio controls src="{accomp_rel}" style="width:100%"></audio>
            </div>
        </div>
    </section>
    '''


def _generate_transcription_section(notes: List[Note], phrases: List[Phrase], tonic: int) -> str:
    """Generate musical transcription section."""
    from .sequence import transcribe_notes
    
    full_transcription = transcribe_notes(notes, tonic)
    
    phrase_html = ""
    for i, phrase in enumerate(phrases[:10]):
        sargam = " ".join(OFFSET_TO_SARGAM.get(int(round(n.pitch_midi - tonic)) % 12, '?') for n in phrase.notes)
        phrase_html += f'<p><strong>Phrase {i+1}</strong> ({phrase.start:.2f}s - {phrase.end:.2f}s): {sargam}</p>'
    
    return f'''
    <section id="transcription">
        <h2>Musical Transcription</h2>
        <p><strong>Total Notes:</strong> {len(notes)} | <strong>Phrases:</strong> {len(phrases)}</p>
        <div class="sargam-notation">
            <h3>Full Transcription</h3>
            <pre>{full_transcription}</pre>
        </div>
        <div class="phrase-list">
            <h3>Detected Phrases</h3>
            {phrase_html}
        </div>
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
    """Get CSS styles for the report - Dark mode with funky font."""
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
        
        function setupAudioSync(playerId, tonic) {
            var audio = document.getElementById(playerId + '-audio');
            var plotDiv = document.getElementById(playerId + '-plot');
            var timeDisplay = document.getElementById(playerId + '-time');
            var sargamDisplay = document.getElementById(playerId + '-sargam');
            
            if (!audio || !plotDiv) return;
            
            var SARGAM = {0:'Sa',1:'re',2:'Re',3:'ga',4:'Ga',5:'ma',6:'Ma',7:'Pa',8:'dha',9:'Dha',10:'ni',11:'Ni'};
            
            // Get cursor shape index (last shape)
            var shapes = plotDiv.layout.shapes || [];
            var cursorIdx = shapes.length - 1;
            
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
                
                // Estimate current MIDI from plot data
                try {
                    var trace = plotDiv.data[0];
                    if (trace && trace.x && trace.y) {
                        var idx = trace.x.findIndex(function(x) { return x >= t; });
                        if (idx > 0) {
                            var midi = trace.y[idx];
                            var offset = Math.round(midi - tonic) % 12;
                            if (offset < 0) offset += 12;
                            sargamDisplay.textContent = SARGAM[offset] || '--';
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
    figsize_width: int = 80, 
    figsize_height: int = 7, 
    dpi: int = 100, 
    join_gap_threshold: float = 1.0
) -> Tuple[str, str, int]:
    """
    Generate wide pitch contour and overlay sargam lines, returning base64 images.
    Returns: (legend_b64, plot_b64, pixel_width)
    """
    import io
    import base64
    import librosa
    from matplotlib.ticker import MultipleLocator
    
    # Resolve tonic
    if isinstance(tonic_ref, (int, np.integer)):
        tonic_midi = tonic_ref % 12
        tonic_midi_base = tonic_midi + 60 # Default to middle C octave if int
        tonic_label = _tonic_name(tonic_ref)
    else:
        try:
            tonic_midi_base = librosa.note_to_midi(str(tonic_ref) + '4')
            tonic_midi = int(tonic_midi_base) % 12
            tonic_label = str(tonic_ref)
        except Exception:
            tonic_midi_base = 60
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
         return "", "", 100

    voiced_midi = librosa.hz_to_midi(pitch_hz[voiced_mask])
    
    # Calculate range dynamically based on data min/max
    data_min = np.min(voiced_midi)
    data_max = np.max(voiced_midi)
    
    min_m = np.floor(data_min) - 2
    max_m = np.ceil(data_max) + 2
    
    rng = np.arange(int(np.floor(min_m)), int(np.ceil(max_m)) + 1)
    
    # --- Main Plot ---
    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height), dpi=dpi)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.1) # Min margins
    
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

    ax.set_ylim(min_m - 1, max_m + 1)
    ax.set_xlim(0, max(timestamps[-1], 1.0))
    ax.set_xlabel('Time (s)')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(2)) # Tick every 2 seconds
    ax.set_title(f"Pitch Contour Analysis ({raga_name} / {tonic_label})")

    # Horizontal Lines
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
            
            # Markers
            if octave_diff < 0: label += "'" * abs(octave_diff)
            elif octave_diff > 0: label += "·" * octave_diff
            
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

    return legend_b64, plot_b64, pixel_width


    return html

def create_scrollable_pitch_plot_html(
    pitch_data: PitchData,
    tonic: int,
    raga_name: str,
    audio_element_id: str,
    raga_notes: Optional[Set[int]] = None
) -> str:
    """
    Create HTML component for scrollable pitch plot with audio sync.
    """
    legend_b64, plot_b64, px_width = plot_pitch_wide_to_base64_with_legend(
        pitch_data, tonic, raga_name, raga_notes
    )
    
    if not plot_b64:
        return "<p>No pitch data for validation plot.</p>"
        
    unique_id = f"sp_{uuid.uuid4().hex[:6]}"
    duration = pitch_data.timestamps[-1] if len(pitch_data.timestamps) > 0 else 1.0
    
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
        const audio = document.getElementById("{audio_element_id}");
        const container = document.getElementById("{unique_id}-container");
        const cursor = document.getElementById("{unique_id}-cursor");
        const status = document.getElementById("{unique_id}-status");
        
        const totalDuration = {duration};
        const pixelWidth = {px_width};
        
        if (!audio) {{
            console.error("Audio element not found: {audio_element_id}");
            if (status) status.textContent = "Error: Audio player not found";
            return;
        }}
        
        if (audio && container && cursor) {{
            console.log("Initializing sync for {unique_id}");
            
            // Plot margins (matches python subplots_adjust)
            const marginL = 0.005;
            const marginR = 0.995;
            const plotContentWidthPct = marginR - marginL;
            
            // Sync Loop
            function updateSync() {{
                if (!audio.paused) {{
                    const t = audio.currentTime;
                    // Safety: clamp t
                    const t_safe = Math.max(0, Math.min(t, totalDuration));
                    
                    // Map time to x position accounting for margins
                    // t=0 -> marginL
                    // t=dur -> marginR
                    const pct = t_safe / totalDuration;
                    const x_pct = marginL + (pct * plotContentWidthPct);
                    const x = x_pct * pixelWidth;
                    
                    // Move cursor
                    cursor.style.left = x + 'px';
                    
                    // Auto-scroll request (only if playing)
                    const viewWidth = container.clientWidth;
                    const targetScroll = x - (viewWidth / 2);
                    
                    // Smooth vs instant scroll
                    if (Math.abs(container.scrollLeft - targetScroll) > 100) {{
                        // Jump if far behind
                        container.scrollLeft = targetScroll;
                    }} else {{
                         // Smooth follow
                         container.scrollLeft = container.scrollLeft * 0.95 + targetScroll * 0.05;
                    }}
                }}
                requestAnimationFrame(updateSync);
            }}
            updateSync();
            
            // Allow click to seek
            container.addEventListener('click', function(e) {{
                const rect = container.getBoundingClientRect();
                const clickX = e.clientX - rect.left + container.scrollLeft;
                
                // Inverse mapping: x -> t
                const x_pct = clickX / pixelWidth;
                const pct = (x_pct - marginL) / plotContentWidthPct;
                const seekT = pct * totalDuration;
                
                if (isFinite(seekT) && seekT >= 0 && seekT <= totalDuration) {{
                    console.log("Seeking to:", seekT);
                    audio.currentTime = seekT;
                    // Update cursor immediately
                    cursor.style.left = clickX + 'px';
                }}
            }});
            
            // Debug info
            audio.addEventListener('play', () => console.log("Audio started"));
            audio.addEventListener('pause', () => console.log("Audio paused"));
        }}
    }});
    </script>
    """
    return html

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
    vocals_rel = os.path.relpath(results.config.vocals_path, output_dir)
    accomp_rel = os.path.relpath(results.config.accompaniment_path, output_dir)
    original_rel = os.path.relpath(results.config.audio_path, output_dir)
    
    # Audio IDs for sync
    vocals_id = "vocals-player"
    accomp_id = "accomp-player"
    
    # Scrollable Plot HTML
    scroll_plot_html = ""
    if results.pitch_data_vocals:
        # Generate the scrollable plot synced to vocals
        try:
            scroll_plot_html = create_scrollable_pitch_plot_html(
                results.pitch_data_vocals,
                tonic=results.detected_tonic or 0,
                raga_name=stats.raga_name,
                audio_element_id=vocals_id
            )
        except Exception as e:
            scroll_plot_html = f"<p>Error generating scrollable plot: {e}</p>"

    # Transcription html
    transcription_html = _generate_transcription_section(results.notes, results.phrases, results.detected_tonic)
    
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
                
        pattern_html = f"""
        <section id="patterns">
            <h2>Pattern Analysis</h2>
            <div class="stats-grid">
                 <div class="stat-box"><strong>Common Motifs</strong><ul>{"".join(motif_items) or "<li>None</li>"}</ul></div>
                 <div class="stat-box"><strong>Aaroh (Ascending) Runs</strong><ul>{"".join(aaroh_items) or "<li>None</li>"}</ul></div>
                 <div class="stat-box"><strong>Avroh (Descending) Runs</strong><ul>{"".join(avroh_items) or "<li>None</li>"}</ul></div>
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
                        <audio controls src="{original_rel}" style="width:100%"></audio>
                    </div>
                    <div>
                        <p><strong>Vocals (Analysis Source)</strong></p>
                        <audio id="{vocals_id}" controls src="{vocals_rel}" style="width:100%"></audio>
                    </div>
                    <div>
                        <p><strong>Accompaniment</strong></p>
                        <audio id="{accomp_id}" controls src="{accomp_rel}" style="width:100%"></audio>
                    </div>
                </div>
                
                <!-- Scrollable Plot -->
                <h3>Interactive Pitch Analysis (Synced to Vocals)</h3>
                <p style="font-size: 0.9em; color: #8b949e;">Scroll horizontally or play audio to follow the analysis. Color-coded by octave.</p>
                {scroll_plot_html}
            </section>

            {correction_html}

            {transcription_html}

            {pattern_html}

            {images_html}
        </main>

        <footer>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </footer>
    </body>
    </html>
    """
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(html_content)
        
    return report_path
