"""
Audio processing module: stem separation and pitch extraction.

Provides:
- separate_stems: Split audio into vocals and accompaniment using Demucs/Spleeter
- extract_pitch: Extract f0 pitch data using SwiftF0 with caching
- PitchData: Dataclass containing pitch extraction results

Note: torch and demucs are imported lazily to avoid import errors if not installed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import os
import numpy as np
import pandas as pd
import librosa

from .config import PipelineConfig


@dataclass
class PitchData:
    """Container for pitch extraction results."""
    
    # Core pitch data
    timestamps: np.ndarray       # Time axis (seconds)
    pitch_hz: np.ndarray         # Detected pitch (Hz), 0 for unvoiced
    confidence: np.ndarray       # Detection confidence [0-1]
    voicing: np.ndarray          # Boolean voicing mask
    
    # Derived data
    valid_freqs: np.ndarray      # Voiced frequencies only (Hz)
    midi_vals: np.ndarray        # Voiced MIDI note values
    
    # Metadata
    frame_period: float = 0.01   # Frame period (seconds)
    audio_path: str = ""
    
    @property
    def voiced_mask(self) -> np.ndarray:
        """Boolean mask for voiced frames."""
        return (self.pitch_hz > 0) & self.voicing
    
    @property
    def voiced_times(self) -> np.ndarray:
        """Timestamps for voiced frames only."""
        return self.timestamps[self.voiced_mask]
    
    @property
    def cent_vals(self) -> np.ndarray:
        """Cent values (0-1200) for voiced frames."""
        return (self.midi_vals % 12) * 100.0
    
    def apply_confidence_threshold(self, threshold: float) -> "PitchData":
        """Return new PitchData with updated voicing based on confidence threshold."""
        new_voicing = self.voicing & (self.confidence >= threshold)
        new_voiced_mask = (self.pitch_hz > 0) & new_voicing
        new_valid_freqs = self.pitch_hz[new_voiced_mask]
        new_midi_vals = librosa.hz_to_midi(new_valid_freqs) if len(new_valid_freqs) > 0 else np.array([])
        
        return PitchData(
            timestamps=self.timestamps,
            pitch_hz=self.pitch_hz,
            confidence=self.confidence,
            voicing=new_voicing,
            valid_freqs=new_valid_freqs,
            midi_vals=new_midi_vals,
            frame_period=self.frame_period,
            audio_path=self.audio_path,
        )


# =============================================================================
# STEM SEPARATION
# =============================================================================

def separate_stems(
    audio_path: str,
    output_dir: str,
    engine: str = "demucs",
    model: str = "htdemucs",
    device: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Separate audio into vocals and accompaniment stems.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Base output directory
        engine: 'demucs' or 'spleeter'
        model: Demucs model name (ignored for spleeter)
        device: 'cuda', 'cpu', or None (auto-detect)
        
    Returns:
        Tuple of (vocals_path, accompaniment_path)
        
    Note:
        Results are cached - if stems already exist, separation is skipped.
    """
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_dir, model, filename)
    
    vocals_path = os.path.join(stem_dir, "vocals.wav")
    accompaniment_path = os.path.join(stem_dir, "accompaniment.wav")
    
    # Check cache
    if os.path.isfile(vocals_path) and os.path.isfile(accompaniment_path):
        print(f"[CACHE] Stems already exist in '{stem_dir}'")
        return vocals_path, accompaniment_path
    
    os.makedirs(stem_dir, exist_ok=True)
    
    if engine.lower() == "spleeter":
        return _separate_spleeter(audio_path, stem_dir, vocals_path, accompaniment_path)
    else:
        return _separate_demucs(audio_path, stem_dir, vocals_path, accompaniment_path, model, device)


def _separate_demucs(
    audio_path: str,
    stem_dir: str,
    vocals_path: str,
    accompaniment_path: str,
    model_name: str = "htdemucs",
    device: Optional[str] = None,
) -> Tuple[str, str]:
    """Separate using Demucs."""
    # Lazy imports to avoid errors if not installed
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    import scipy.io.wavfile as wavfile
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[DEMUCS] Loading model: {model_name}...")
    model = get_model(model_name)
    model.to(device)
    
    print(f"[DEMUCS] Loading audio: {os.path.basename(audio_path)}...")
    audio_file = AudioFile(audio_path)
    wav = audio_file.read(
        streams=0,
        samplerate=model.samplerate,
        channels=model.audio_channels,
    )
    
    print(f"[DEMUCS] Separating on {device}...")
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    
    with torch.no_grad():
        sources = apply_model(
            model,
            wav.unsqueeze(0).to(device),
            device=device,
            progress=True,
        )[0]
    
    sources = sources * ref.std() + ref.mean()
    
    # Extract vocals and accompaniment
    stem_names = model.sources
    vocals_idx = stem_names.index("vocals") if "vocals" in stem_names else 0
    vocals = sources[vocals_idx].cpu()
    
    # Sum all other stems for accompaniment
    accompaniment = sum(
        sources[i].cpu() for i in range(len(stem_names)) if i != vocals_idx
    )
    
    # Save stems using scipy (avoids torchcodec dependency)
    def save_wav(tensor, path, samplerate):
        # tensor shape: (channels, samples) -> transpose to (samples, channels)
        audio_np = tensor.numpy().T
        # Clip to [-1, 1] and convert to int16
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wavfile.write(path, samplerate, audio_int16)
    
    save_wav(vocals, vocals_path, model.samplerate)
    save_wav(accompaniment, accompaniment_path, model.samplerate)
    
    print(f"[DEMUCS] Saved vocals: {vocals_path}")
    print(f"[DEMUCS] Saved accompaniment: {accompaniment_path}")
    
    return vocals_path, accompaniment_path


def _separate_spleeter(
    audio_path: str,
    stem_dir: str,
    vocals_path: str,
    accompaniment_path: str,
) -> Tuple[str, str]:
    """Separate using Spleeter."""
    from spleeter.separator import Separator
    
    print("[SPLEETER] Initializing 2-stem separator...")
    separator = Separator("spleeter:2stems")
    
    print(f"[SPLEETER] Separating: {os.path.basename(audio_path)}...")
    separator.separate_to_file(audio_path, stem_dir, codec="wav")
    
    # Spleeter creates a subdirectory named after the input file
    filename_no_ext = os.path.splitext(os.path.basename(audio_path))[0]
    spleeter_out_dir = os.path.join(stem_dir, filename_no_ext)
    
    spleeter_vocals = os.path.join(spleeter_out_dir, "vocals.wav")
    spleeter_accomp = os.path.join(spleeter_out_dir, "accompaniment.wav")
    
    # Move files to the expected location (stem_dir root)
    if os.path.exists(spleeter_vocals) and spleeter_vocals != vocals_path:
        import shutil
        shutil.move(spleeter_vocals, vocals_path)
    if os.path.exists(spleeter_accomp) and spleeter_accomp != accompaniment_path:
        import shutil
        shutil.move(spleeter_accomp, accompaniment_path)
    
    # Clean up empty subdirectory
    if os.path.exists(spleeter_out_dir) and not os.listdir(spleeter_out_dir):
        os.rmdir(spleeter_out_dir)
    
    print(f"[SPLEETER] Saved vocals: {vocals_path}")
    print(f"[SPLEETER] Saved accompaniment: {accompaniment_path}")
    
    return vocals_path, accompaniment_path


def load_audio_direct(
    audio_path: str,
    output_dir: str,
    source_type: str = "instrumental",
) -> Tuple[str, Optional[str]]:
    """
    Load audio directly without stem separation.
    
    For instrumental or solo vocal recordings, skip stem separation 
    and use the original audio as the melody source.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Output directory for the processed file
        source_type: "instrumental" or "vocal"
        
    Returns:
        Tuple of (melody_path, accompaniment_path or None)
        accompaniment_path is always None since no separation is done.
    """
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_dir, "direct", filename)
    os.makedirs(stem_dir, exist_ok=True)
    
    # Name the output based on source type for clarity
    if source_type == "vocal":
        melody_filename = "vocals.wav"
    else:
        melody_filename = "melody.wav"
    
    melody_path = os.path.join(stem_dir, melody_filename)
    
    # Check if already exists
    if os.path.isfile(melody_path):
        print(f"[CACHE] Direct audio already exists: {melody_path}")
        return melody_path, None
    
    # Copy audio to output (convert to WAV if needed)
    import shutil
    import subprocess
    
    input_ext = os.path.splitext(audio_path)[1].lower()
    if input_ext == ".wav":
        # Just copy
        shutil.copy2(audio_path, melody_path)
        print(f"[DIRECT] Copied audio to: {melody_path}")
    else:
        # Use ffmpeg for conversion (more reliable than librosa for various formats)
        try:
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ar", "44100", "-ac", "2",
                melody_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")
            print(f"[DIRECT] Converted to WAV using ffmpeg: {melody_path}")
        except FileNotFoundError:
            # Fallback to soundfile if ffmpeg not available
            import soundfile as sf
            audio_data, sr = sf.read(audio_path)
            sf.write(melody_path, audio_data, sr)
            print(f"[DIRECT] Converted to WAV using soundfile: {melody_path}")
    
    # Return None for accompaniment since we're not separating
    return melody_path, None


# =============================================================================
# PITCH EXTRACTION
# =============================================================================

def extract_pitch(
    audio_path: str,
    output_dir: str,
    prefix: str,
    fmin: float,
    fmax: float,
    confidence_threshold: float = 0.9,
    force_recompute: bool = False,
) -> PitchData:
    """
    Extract pitch using SwiftF0 with caching.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory for cached results
        prefix: Prefix for output files (e.g., 'vocals', 'accompaniment')
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz)
        confidence_threshold: Minimum confidence for valid pitch
        force_recompute: Ignore cached results
        
    Returns:
        PitchData containing pitch extraction results
    """
    csv_path = os.path.join(output_dir, f"{prefix}_pitch_data.csv")
    
    # Try loading from cache
    if os.path.isfile(csv_path) and not force_recompute:
        print(f"[CACHE] Loading pitch data from: {csv_path}")
        pitch_data = load_pitch_from_csv(csv_path, audio_path)
        
        # Apply confidence threshold
        if confidence_threshold > 0:
            print(f"[INFO] Applying confidence threshold = {confidence_threshold}")
            pitch_data = pitch_data.apply_confidence_threshold(confidence_threshold)
        
        return pitch_data
    
    # Run SwiftF0
    print(f"[SWIFTF0] Extracting pitch from: {os.path.basename(audio_path)}")
    
    from swift_f0 import SwiftF0, export_to_csv
    
    detector = SwiftF0(fmin=fmin, fmax=fmax, confidence_threshold=confidence_threshold)
    result = detector.detect_from_file(audio_path)
    
    # Export to CSV
    os.makedirs(output_dir, exist_ok=True)
    export_to_csv(result, csv_path)
    print(f"[WRITE] Exported CSV: {csv_path}")
    
    # Convert to PitchData
    pitch_hz = np.asarray(result.pitch_hz)
    timestamps = np.asarray(result.timestamps) if hasattr(result, "timestamps") else np.arange(len(pitch_hz)) * 0.01
    confidence = np.asarray(result.confidence) if hasattr(result, "confidence") else np.ones_like(pitch_hz)
    voicing = np.asarray(result.voicing) if hasattr(result, "voicing") else (pitch_hz > 0)
    
    voiced_mask = (pitch_hz > 0) & voicing & (confidence >= confidence_threshold)
    valid_freqs = pitch_hz[voiced_mask]
    midi_vals = librosa.hz_to_midi(valid_freqs) if len(valid_freqs) > 0 else np.array([])
    
    frame_period = result.frame_period if hasattr(result, "frame_period") else 0.01
    
    return PitchData(
        timestamps=timestamps,
        pitch_hz=pitch_hz,
        confidence=confidence,
        voicing=voicing,
        valid_freqs=valid_freqs,
        midi_vals=midi_vals,
        frame_period=frame_period,
        audio_path=audio_path,
    )


def load_pitch_from_csv(csv_path: str, audio_path: str = "") -> PitchData:
    """
    Load cached pitch data from CSV.
    
    Args:
        csv_path: Path to CSV file
        audio_path: Original audio path (for metadata)
        
    Returns:
        PitchData instance
    """
    df = pd.read_csv(csv_path)
    
    # Handle column name variants
    ts_col = "time" if "time" in df.columns else "timestamp"
    pitch_col = "pitch_hz" if "pitch_hz" in df.columns else "f0"
    conf_col = "confidence" if "confidence" in df.columns else "conf"
    voiced_col = "voicing" if "voicing" in df.columns else ("voiced" if "voiced" in df.columns else None)
    
    timestamps = pd.to_numeric(df[ts_col], errors="coerce").to_numpy()
    timestamps = np.nan_to_num(timestamps, nan=0.0)
    
    pitch_hz = pd.to_numeric(df[pitch_col], errors="coerce").to_numpy()
    pitch_hz = np.nan_to_num(pitch_hz, nan=0.0)
    
    confidence = pd.to_numeric(df[conf_col], errors="coerce").to_numpy()
    confidence = np.nan_to_num(confidence, nan=0.0)
    
    if voiced_col and voiced_col in df.columns:
        voiced_raw = df[voiced_col].astype(str).fillna("").str.strip().str.lower()
        voicing = voiced_raw.isin(["1", "1.0", "true", "t", "yes", "y"]).to_numpy()
    else:
        voicing = pitch_hz > 0
    
    # Compute derived values
    voiced_mask = (pitch_hz > 0) & voicing
    valid_freqs = pitch_hz[voiced_mask]
    midi_vals = librosa.hz_to_midi(valid_freqs) if len(valid_freqs) > 0 else np.array([])
    
    # Estimate frame period
    if len(timestamps) > 1:
        frame_period = float(np.median(np.diff(timestamps)))
    else:
        frame_period = 0.01
    
    return PitchData(
        timestamps=timestamps,
        pitch_hz=pitch_hz,
        confidence=confidence,
        voicing=voicing,
        valid_freqs=valid_freqs,
        midi_vals=midi_vals,
        frame_period=frame_period,
        audio_path=audio_path,
    )


def extract_pitch_from_config(config: PipelineConfig, stem: str = "vocals") -> PitchData:
    """
    Convenience function to extract pitch using PipelineConfig.
    
    Args:
        config: Pipeline configuration
        stem: 'vocals' or 'accompaniment'
        
    Returns:
        PitchData instance
    """
    if stem == "vocals":
        audio_path = config.vocals_path
        confidence = config.vocal_confidence
    else:
        audio_path = config.accompaniment_path
        confidence = config.accomp_confidence
    
    fmin = librosa.note_to_hz(config.fmin_note)
    fmax = librosa.note_to_hz(config.fmax_note)
    
    return extract_pitch(
        audio_path=audio_path,
        output_dir=config.stem_dir,
        prefix=stem,
        fmin=fmin,
        fmax=fmax,
        confidence_threshold=confidence,
        force_recompute=config.force_recompute,
    )
