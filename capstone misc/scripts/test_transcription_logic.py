
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from raga_pipeline import transcription

def test_synthetic_vibrato():
    print("Testing Stationary Point Transcription on Synthetic Data...")
    
    # 1. Generate Synthetic Data
    # 3 second audio: 
    # 0.0-1.0s: Note C4 (MIDI 60) - Stable
    # 1.0-2.0s: Note E4 (MIDI 64) - With Vibrato (+/- 0.5 semitones, 5Hz)
    # 2.0-3.0s: Glide to G4 (MIDI 67)
    
    fs = 100 # 100Hz frame rate (10ms)
    duration = 3.0
    times = np.linspace(0, duration, int(duration * fs))
    
    pitch_midi = np.zeros_like(times)
    
    # C4
    pitch_midi[(times < 1.0)] = 60.0
    
    # E4 with Vibrato
    # Vibrato: 5Hz sine wave, amplitude 0.5 semitones
    mask_vibrato = (times >= 1.0) & (times < 2.0)
    vibrato = 0.5 * np.sin(2 * np.pi * 5 * (times[mask_vibrato] - 1.0))
    pitch_midi[mask_vibrato] = 64.0 + vibrato
    
    # Glide
    mask_glide = (times >= 2.0)
    pitch_midi[mask_glide] = np.linspace(64.0, 67.0, np.sum(mask_glide))
    
    # Voicing (all voiced)
    voicing = np.ones_like(times, dtype=bool)
    
    # Convert to Hz for API
    pitch_hz = 440.0 * (2.0 ** ((pitch_midi - 69.0) / 12.0))
    
    # 2. Run Transcription
    events = transcription.detect_stationary_events(
        pitch_hz=pitch_hz,
        timestamps=times,
        voicing_mask=voicing,
        tonic=60, # C4 is tonic (Sa)
        smoothing_sigma_ms=100, # Strong smoothing for vibrato
        derivative_threshold=1.5,
        min_event_duration=0.2,
        snap_mode='chromatic'
    )
    
    # 3. Analyze Results
    print(f"Found {len(events)} events:")
    for i, e in enumerate(events):
        print(f"Event {i+1}: {e.start:.2f}s - {e.end:.2f}s | MIDI: {e.pitch_midi:.2f} -> Snapped: {e.snapped_midi:.1f} ({e.sargam})")
    
    # Validation
    # Expect Event 1: ~0.0-1.0s, MIDI 60 (Sa)
    # Expect Event 2: ~1.0-2.0s, MIDI 64 (Ga) - Smoothing should handle vibrato
    # Expect NO Event 3 (Glide) - Derivative should be high OR if slope is low?
    # Glide 64->67 over 1s = 3 semitones/sec. Threshold is 1.5. So should be rejected.
    
    passed = True
    if len(events) != 2:
        print("FAIL: Expected 2 events (C4, E4)")
        passed = False
    else:
        # Check C4
        if abs(events[0].snapped_midi - 60) > 0.1:
            print("FAIL: Event 1 not C4")
            passed = False
            
        # Check E4 (smoothing check)
        if abs(events[1].snapped_midi - 64) > 0.5:
             print(f"FAIL: Event 2 not E4 (Got {events[1].snapped_midi:.2f}) - Vibrato smoothing failed?")
             passed = False
             
    if passed:
        print("SUCCESS: Synthetic test passed!")

if __name__ == "__main__":
    test_synthetic_vibrato()
