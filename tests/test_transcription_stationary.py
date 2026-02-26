import unittest

import numpy as np

try:
    from raga_pipeline import transcription

    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


@unittest.skipUnless(IMPORT_OK, "transcription module unavailable in current environment")
class TranscriptionStationaryTests(unittest.TestCase):
    def test_stationary_notes_use_snapped_pitch(self) -> None:
        # Build a stable microtonal contour so snapped and raw MIDI differ.
        midi_pitch = 69.33  # Between A4 and A#4
        hz_pitch = 440.0 * (2.0 ** ((midi_pitch - 69.0) / 12.0))

        timestamps = np.arange(0.0, 2.0, 0.01, dtype=float)
        pitch_hz = np.full_like(timestamps, hz_pitch)
        voicing = np.ones_like(timestamps, dtype=bool)
        energy = np.ones_like(timestamps, dtype=float)

        events = transcription.detect_stationary_events(
            pitch_hz=pitch_hz,
            timestamps=timestamps,
            voicing_mask=voicing,
            tonic=60,  # C4 as tonic anchor
            energy=energy,
            energy_threshold=0.0,
            smoothing_sigma_ms=0.0,
            derivative_threshold=10.0,
            min_event_duration=0.05,
        )
        self.assertTrue(events, "Expected at least one stationary event")
        evt = events[0]
        self.assertGreater(abs(evt.pitch_midi - evt.snapped_midi), 0.05)

        notes = transcription.transcribe_to_notes(
            pitch_hz=pitch_hz,
            timestamps=timestamps,
            voicing_mask=voicing,
            tonic=60,
            energy=energy,
            energy_threshold=0.0,
            smoothing_sigma_ms=0.0,
            derivative_threshold=10.0,
            min_event_duration=0.05,
            transcription_min_duration=0.05,
        )
        self.assertTrue(notes, "Expected at least one transcribed note")
        self.assertAlmostEqual(notes[0].pitch_midi, evt.snapped_midi, places=6)


if __name__ == "__main__":
    unittest.main()
