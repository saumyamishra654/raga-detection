"""
Audio processing module: stem separation and pitch extraction.

Provides:
- separate_stems: Split audio into vocals and accompaniment using Demucs/Spleeter
- extract_pitch: Extract f0 pitch data using SwiftF0 with caching
- PitchData: Dataclass containing pitch extraction results

Note: torch and demucs are imported lazily to avoid import errors if not installed.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, cast
import glob
import os
import re
import shutil
import subprocess
import numpy as np
import pandas as pd
import librosa

from .config import PipelineConfig


@dataclass
class PitchData:
    """Container for pitch extraction results."""
    # Core pitch data columns for analysis
    # Core pitch data
    timestamps: np.ndarray       # Time axis (seconds)
    pitch_hz: np.ndarray         # Detected pitch (Hz), 0 for unvoiced
    confidence: np.ndarray       # Detection confidence [0-1]
    voicing: np.ndarray          # Boolean voicing mask

    # Derived data
    valid_freqs: np.ndarray      # Voiced frequencies only (Hz)
    midi_vals: np.ndarray        # Voiced MIDI note values
    energy: np.ndarray = field(default_factory=lambda: np.array([]))  # Normalized RMS energy (0-1); field() ensures an empty array by default.

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
    def duration(self) -> float:
        """Total duration in seconds."""
        if len(self.timestamps) > 0:
            return self.timestamps[-1]
        return 0.0

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
            energy=self.energy,
            frame_period=self.frame_period,
            audio_path=self.audio_path,
        )


# =============================================================================
# EXTERNAL AUDIO INGEST
# =============================================================================

def _sanitize_filename_base(filename_base: str) -> str:
    """Sanitize filename base for cross-platform safe MP3 output."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", filename_base.strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise ValueError("Filename cannot be empty after sanitization.")
    return sanitized


def _parse_time_to_seconds(value: str) -> float:
    """Parse time text into seconds. Supports SS, MM:SS, HH:MM:SS."""
    raw = value.strip()
    if not raw:
        raise ValueError("Time value cannot be empty.")

    parts = raw.split(":")
    if len(parts) == 1:
        try:
            seconds = float(parts[0])
        except ValueError as exc:
            raise ValueError(f"Invalid time value: {value}") from exc
        if seconds < 0:
            raise ValueError(f"Time must be non-negative: {value}")
        return seconds

    if len(parts) > 3:
        raise ValueError(f"Invalid time format: {value}")

    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid time format: {value}") from exc

    if any(p < 0 for p in nums):
        raise ValueError(f"Time must be non-negative: {value}")

    if len(nums) == 2:
        minutes, seconds = nums
        return minutes * 60 + seconds

    hours, minutes, seconds = nums
    return hours * 3600 + minutes * 60 + seconds


def _get_audio_duration_seconds(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required for duration checks but was not found in PATH.")

    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration_txt = proc.stdout.strip()
    if not duration_txt:
        raise RuntimeError(f"Failed to read audio duration: {audio_path}")
    return float(duration_txt)


def _trim_audio_segment(audio_path: str, start_s: float, end_s: float) -> None:
    """Trim an audio file in-place to [start_s, end_s]."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for trimming but was not found in PATH.")

    temp_path = f"{audio_path}.trim.tmp.mp3"
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{start_s:.3f}",
                "-to",
                f"{end_s:.3f}",
                "-i",
                audio_path,
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "2",
                temp_path,
            ],
            check=True,
        )
        os.replace(temp_path, audio_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def download_youtube_audio(
    yt_url: str,
    audio_dir: str,
    filename_base: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> str:
    """
    Download audio from YouTube and save as MP3.

    Args:
        yt_url: YouTube URL
        audio_dir: Target directory where MP3 should be saved
        filename_base: Output filename base (without extension)
        start_time: Optional trim start time (SS, MM:SS, or HH:MM:SS)
        end_time: Optional trim end time (SS, MM:SS, or HH:MM:SS)

    Returns:
        Absolute path to saved MP3 file

    Raises:
        ValueError: If required inputs are missing
        FileExistsError: If target output already exists
        RuntimeError: If yt-dlp or ffmpeg is unavailable or download fails
    """
    if not yt_url:
        raise ValueError("YouTube URL is required.")
    if not audio_dir:
        raise ValueError("Audio directory is required.")
    if not filename_base:
        raise ValueError("Filename is required.")

    target_dir = os.path.abspath(audio_dir)
    os.makedirs(target_dir, exist_ok=True)

    safe_base = _sanitize_filename_base(filename_base)
    target_mp3_path = os.path.join(target_dir, f"{safe_base}.mp3")

    if os.path.exists(target_mp3_path):
        raise FileExistsError(
            f"Target audio already exists: {target_mp3_path}. "
            "Choose a different --filename or remove the existing file."
        )

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for YouTube audio extraction but was not found in PATH.")

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is required for --yt download support. Install it with: pip install yt-dlp"
        ) from exc

    output_template = os.path.join(target_dir, f"{safe_base}.%(ext)s")
    base_opts = {
        "noplaylist": True,
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    attempts = [
        {
            "name": "android/tv audio priority",
            "opts": {
                "format": "140/251/bestaudio[acodec!=none]/bestaudio/best",
                "extractor_args": {"youtube": {"player_client": ["android", "tv"]}},
            },
        },
        {
            "name": "android + ipv4 fallback",
            "opts": {
                "format": "140/251/bestaudio[acodec!=none]/bestaudio/best",
                "extractor_args": {"youtube": {"player_client": ["android"]}},
                "force_ipv4": True,
            },
        },
        {
            "name": "hls audio fallback",
            "opts": {
                "format": "bestaudio[protocol*=m3u8]/bestaudio/best",
                "extractor_args": {"youtube": {"player_client": ["tv"]}},
            },
        },
    ]

    errors: list[str] = []
    download_succeeded = False
    for attempt in attempts:
        for temp_file in glob.glob(os.path.join(target_dir, f"{safe_base}.*")):
            if temp_file != target_mp3_path and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

        merged_opts = {**base_opts, **attempt["opts"]}
        try:
            with yt_dlp.YoutubeDL(cast(Any, merged_opts)) as ydl:
                ydl.download([yt_url])
            if os.path.isfile(target_mp3_path):
                download_succeeded = True
                break
        except Exception as exc:
            errors.append(f"{attempt['name']}: {exc}")

    if not download_succeeded or not os.path.isfile(target_mp3_path):
        detail = errors[-1] if errors else "unknown download failure"
        raise RuntimeError(
            "Failed to download YouTube audio after multiple fallback attempts. "
            f"Last error: {detail}. "
            "Try updating yt-dlp (`pip install -U yt-dlp`) and retry. "
            "If the video is restricted, try another URL."
        )

    track_duration = _get_audio_duration_seconds(target_mp3_path)
    trim_start = _parse_time_to_seconds(start_time) if start_time else 0.0
    trim_end = _parse_time_to_seconds(end_time) if end_time else track_duration

    duration_epsilon = 1e-3
    if trim_start >= trim_end:
        raise ValueError(
            f"Invalid trim range: start time ({trim_start:.3f}s) must be less than "
            f"end time ({trim_end:.3f}s)."
        )
    if trim_start > track_duration + duration_epsilon:
        raise ValueError(
            f"Start time ({trim_start:.3f}s) exceeds track duration ({track_duration:.3f}s)."
        )
    if trim_end > track_duration + duration_epsilon:
        raise ValueError(
            f"End time ({trim_end:.3f}s) exceeds track duration ({track_duration:.3f}s)."
        )

    trim_start = max(0.0, min(trim_start, track_duration))
    trim_end = max(0.0, min(trim_end, track_duration))

    if trim_start > 0.0 or trim_end < (track_duration - duration_epsilon):
        _trim_audio_segment(target_mp3_path, trim_start, trim_end)

    return target_mp3_path


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
    # Use 'spleeter' or the Demucs model name as the subdirectory
    subdir = "spleeter" if engine.lower() == "spleeter" else model
    stem_dir = os.path.join(output_dir, subdir, filename)

    vocals_path = os.path.join(stem_dir, "vocals.mp3")
    accompaniment_path = os.path.join(stem_dir, "accompaniment.mp3")
    legacy_vocals_path = os.path.join(stem_dir, "vocals.wav")
    legacy_accomp_path = os.path.join(stem_dir, "accompaniment.wav")

    # Check cache
    if os.path.isfile(vocals_path) and os.path.isfile(accompaniment_path):
        print(f"[CACHE] Stems already exist in '{stem_dir}'")
        return vocals_path, accompaniment_path

    if os.path.isfile(legacy_vocals_path) and os.path.isfile(legacy_accomp_path):
        print(f"[CACHE] Found legacy WAV stems in '{stem_dir}', converting to MP3")
        _convert_wav_to_mp3(legacy_vocals_path, vocals_path)
        _convert_wav_to_mp3(legacy_accomp_path, accompaniment_path)
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
    audio_file = AudioFile(Path(audio_path))
    wav = audio_file.read(
        streams=slice(0, 1),
        samplerate=model.samplerate,
        channels=model.audio_channels,
    )
    # demucs.apply_model expects (channels, length); squeeze optional stream dim
    if wav.dim() == 3:
        wav = wav[0]

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

    # Save stems via temp WAV then encode to MP3 to reduce size.
    def save_wav(tensor, path, samplerate):
        # tensor shape: (channels, samples) -> transpose to (samples, channels)
        audio_np = tensor.numpy().T
        # Clip to [-1, 1] and convert to int16
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wavfile.write(path, samplerate, audio_int16)

    def save_mp3(tensor, path, samplerate):
        temp_wav = f"{path}.tmp.wav"
        save_wav(tensor, temp_wav, samplerate)
        try:
            _convert_wav_to_mp3(temp_wav, path)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    save_mp3(vocals, vocals_path, model.samplerate)
    save_mp3(accompaniment, accompaniment_path, model.samplerate)

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
    separator.separate_to_file(audio_path, stem_dir, codec=cast(Any, "mp3"))

    # Spleeter creates a subdirectory named after the input file
    filename_no_ext = os.path.splitext(os.path.basename(audio_path))[0]
    spleeter_out_dir = os.path.join(stem_dir, filename_no_ext)

    spleeter_vocals = os.path.join(spleeter_out_dir, "vocals.mp3")
    spleeter_accomp = os.path.join(spleeter_out_dir, "accompaniment.mp3")

    # Move files to the expected location (stem_dir root)
    if os.path.exists(spleeter_vocals) and spleeter_vocals != vocals_path:
        shutil.move(spleeter_vocals, vocals_path)
    if os.path.exists(spleeter_accomp) and spleeter_accomp != accompaniment_path:
        shutil.move(spleeter_accomp, accompaniment_path)

    # Clean up empty subdirectory
    if os.path.exists(spleeter_out_dir) and not os.listdir(spleeter_out_dir):
        os.rmdir(spleeter_out_dir)

    print(f"[SPLEETER] Saved vocals: {vocals_path}")
    print(f"[SPLEETER] Saved accompaniment: {accompaniment_path}")

    return vocals_path, accompaniment_path


def _convert_wav_to_mp3(wav_path: str, mp3_path: str) -> None:
    """Convert a WAV file to MP3 using ffmpeg and remove the WAV on success."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; cannot encode MP3 stems.")

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            wav_path,
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
            mp3_path,
        ],
        check=True,
    )

    if os.path.exists(wav_path):
        os.remove(wav_path)


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


def _fit_array_to_length(values: np.ndarray, target_len: int, fill_value: float = 0.0) -> np.ndarray:
    """Trim or pad a 1D array to a target length."""
    if len(values) > target_len:
        return values[:target_len]
    if len(values) < target_len:
        return np.pad(values, (0, target_len - len(values)), mode="constant", constant_values=fill_value)
    return values


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
    energy_metric: str = "rms",
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
        energy_metric: 'rms' (peak-normalised) or 'log_amp' (dBFS, percentile-normalised)

    Returns:
        PitchData containing pitch extraction results with energy
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

    # Calculate Energy
    y, sr = librosa.load(audio_path, sr=None)
    pitch_hz = np.asarray(result.pitch_hz)
    target_len = len(pitch_hz)
    timestamps = (
        np.asarray(result.timestamps)
        if hasattr(result, "timestamps")
        else np.arange(target_len, dtype=float) * 0.01
    )

    # Align RMS hop with the actual SwiftF0 timeline.
    inferred_frame_period = None
    if len(timestamps) > 1:
        dt = np.diff(timestamps.astype(float))
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) > 0:
            inferred_frame_period = float(np.median(dt))

    raw_frame_period = getattr(result, "frame_period", None)
    try:
        raw_frame_period = float(raw_frame_period) if raw_frame_period is not None else None
    except (TypeError, ValueError):
        raw_frame_period = None

    if raw_frame_period is not None and np.isfinite(raw_frame_period) and raw_frame_period > 0:
        frame_period = raw_frame_period
    elif inferred_frame_period is not None:
        frame_period = inferred_frame_period
    else:
        frame_period = 0.01

    if (
        inferred_frame_period is not None
        and abs(frame_period - inferred_frame_period) > 5e-4
    ):
        print(
            f"[AUDIO] frame_period mismatch (result={frame_period:.6f}s, "
            f"timestamps={inferred_frame_period:.6f}s). Using timestamp-derived value."
        )
        frame_period = inferred_frame_period

    hop_length = max(1, int(round(sr * frame_period)))

    if energy_metric == "log_amp":
        # ----- Log-amplitude (dBFS) with percentile normalisation -----
        print("[AUDIO] Calculating log-amplitude (dBFS) energy...")
        # Frame-level RMS as the amplitude proxy, then convert to dB
        rms = librosa.feature.rms(y=y, hop_length=hop_length, center=False)[0]
        rms = _fit_array_to_length(rms, target_len)

        eps = 1e-10  # avoid log(0)
        db = 20.0 * np.log10(rms + eps)  # dBFS (full-scale relative to 1.0)

        # Percentile-based normalisation: map [p5, p95] -> [0, 1], clamp
        p5 = np.percentile(db, 5)
        p95 = np.percentile(db, 95)
        if p95 - p5 < 1e-6:  # near-constant energy
            energy_norm = np.ones_like(db) * 0.5
        else:
            energy_norm = (db - p5) / (p95 - p5)
            energy_norm = np.clip(energy_norm, 0.0, 1.0)
        print(f"[AUDIO] dBFS range: [{db.min():.1f}, {db.max():.1f}] dB  "
              f"percentile window: [{p5:.1f}, {p95:.1f}] dB")
    else:
        # ----- RMS with peak normalisation (original behaviour) -----
        print("[AUDIO] Calculating RMS energy...")
        rms = librosa.feature.rms(y=y, hop_length=hop_length, center=False)[0]
        rms = _fit_array_to_length(rms, target_len)

        # Normalize energy (0-1)
        energy_max = rms.max() if rms.max() > 0 else 1.0
        energy_norm = rms / energy_max

    rms_len_after_align = len(rms)
    ts_len = len(timestamps)
    ts_start = float(timestamps[0]) if ts_len > 0 else 0.0
    ts_end = float(timestamps[-1]) if ts_len > 0 else 0.0
    print(
        "[AUDIO] Energy alignment: "
        f"sr={sr}, frame_period={frame_period:.6f}s, hop={hop_length}, "
        f"target_len={target_len}, rms_len={rms_len_after_align}, ts_len={ts_len}, "
        f"ts_range=[{ts_start:.3f}, {ts_end:.3f}], metric={energy_metric}, center=False, pad=zero"
    )

    # Append energy as a separate column because SwiftF0 result objects are immutable.

    # Export to CSV first
    os.makedirs(output_dir, exist_ok=True)
    export_to_csv(result, csv_path)

    # Append energy to CSV
    df = pd.read_csv(csv_path)
    # Ensure lengths match
    if len(df) != len(energy_norm):
        print(f"[WARN] Length mismatch: CSV={len(df)}, Energy={len(energy_norm)}")
        min_len = min(len(df), len(energy_norm))
        df = df.iloc[:min_len]
        energy_norm = energy_norm[:min_len]

    df["energy"] = energy_norm
    df.to_csv(csv_path, index=False)
    print(f"[WRITE] Exported CSV with energy: {csv_path}")

    # Convert to PitchData
    confidence = np.asarray(result.confidence) if hasattr(result, "confidence") else np.ones_like(pitch_hz)
    voicing = np.asarray(result.voicing) if hasattr(result, "voicing") else (pitch_hz > 0)

    # Ensure all arrays match min_len if truncation occurred
    if len(pitch_hz) != len(energy_norm):
        min_len = min(len(pitch_hz), len(energy_norm))
        pitch_hz = pitch_hz[:min_len]
        timestamps = timestamps[:min_len]
        confidence = confidence[:min_len]
        voicing = voicing[:min_len]
        energy_norm = energy_norm[:min_len]

    voiced_mask = (pitch_hz > 0) & voicing & (confidence >= confidence_threshold)
    valid_freqs = pitch_hz[voiced_mask]
    midi_vals = librosa.hz_to_midi(valid_freqs) if len(valid_freqs) > 0 else np.array([])

    return PitchData(
        timestamps=timestamps,
        pitch_hz=pitch_hz,
        confidence=confidence,
        voicing=voicing,
        valid_freqs=valid_freqs,
        midi_vals=midi_vals,
        energy=energy_norm,
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

    # Load energy if exists
    if 'energy' in df.columns:
        energy = pd.to_numeric(df['energy'], errors="coerce").to_numpy()
        energy = np.nan_to_num(energy, nan=0.0)
    else:
        energy = np.ones_like(pitch_hz)  # Default to max energy if missing

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
        energy=energy,
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

    fmin = float(librosa.note_to_hz(config.fmin_note))
    fmax = float(librosa.note_to_hz(config.fmax_note))

    return extract_pitch(
        audio_path=audio_path,
        output_dir=config.stem_dir,
        prefix=stem,
        fmin=fmin,
        fmax=fmax,
        confidence_threshold=confidence,
        force_recompute=config.force_recompute,
        energy_metric=getattr(config, 'energy_metric', 'rms'),
    )
