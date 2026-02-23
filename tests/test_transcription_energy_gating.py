import unittest

import numpy as np

try:
    from raga_pipeline import transcription

    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


def _midi_to_hz(midi_vals: np.ndarray) -> np.ndarray:
    return 440.0 * (2.0 ** ((midi_vals - 69.0) / 12.0))


@unittest.skipUnless(IMPORT_OK, "transcription module unavailable in current environment")
class TranscriptionEnergyGatingTests(unittest.TestCase):
    def test_inflection_energy_sampled_and_kept_when_above_threshold(self) -> None:
        timestamps = np.arange(0.0, 2.0, 0.01, dtype=float)
        midi = 60.0 + 0.6 * np.sin(2.0 * np.pi * 3.0 * timestamps)
        pitch_hz = _midi_to_hz(midi)
        voicing = np.ones_like(timestamps, dtype=bool)
        energy = np.full_like(timestamps, 0.2, dtype=float)

        notes = transcription.transcribe_to_notes(
            pitch_hz=pitch_hz,
            timestamps=timestamps,
            voicing_mask=voicing,
            tonic=60,
            energy=energy,
            energy_threshold=0.1,
            smoothing_sigma_ms=0.0,
            derivative_threshold=0.0,
            min_event_duration=0.1,
            transcription_min_duration=0.1,
        )
        self.assertTrue(notes, "Expected inflection notes above threshold")
        inflections = [n for n in notes if n.confidence < 0.99]
        self.assertTrue(inflections, "Expected at least one inflection note")
        for note in inflections:
            self.assertGreaterEqual(note.energy, 0.1)
            self.assertAlmostEqual(note.energy, 0.2, places=3)

    def test_inflection_notes_dropped_when_energy_below_threshold(self) -> None:
        timestamps = np.arange(0.0, 2.0, 0.01, dtype=float)
        midi = 60.0 + 0.6 * np.sin(2.0 * np.pi * 3.0 * timestamps)
        pitch_hz = _midi_to_hz(midi)
        voicing = np.ones_like(timestamps, dtype=bool)
        energy = np.full_like(timestamps, 0.05, dtype=float)

        notes = transcription.transcribe_to_notes(
            pitch_hz=pitch_hz,
            timestamps=timestamps,
            voicing_mask=voicing,
            tonic=60,
            energy=energy,
            energy_threshold=0.1,
            smoothing_sigma_ms=0.0,
            derivative_threshold=0.0,
            min_event_duration=0.1,
            transcription_min_duration=0.1,
        )
        self.assertEqual(len(notes), 0, "Inflection notes below threshold should be removed")

    def test_stationary_and_inflection_paths_share_threshold_logic(self) -> None:
        timestamps = np.arange(0.0, 2.0, 0.01, dtype=float)
        midi = np.empty_like(timestamps)
        split_idx = len(timestamps) // 2
        midi[:split_idx] = 60.0
        midi[split_idx:] = 62.0 + 0.6 * np.sin(2.0 * np.pi * 3.0 * timestamps[split_idx:])
        pitch_hz = _midi_to_hz(midi)
        voicing = np.ones_like(timestamps, dtype=bool)

        energy = np.empty_like(timestamps, dtype=float)
        energy[:split_idx] = 0.2
        energy[split_idx:] = 0.05

        notes = transcription.transcribe_to_notes(
            pitch_hz=pitch_hz,
            timestamps=timestamps,
            voicing_mask=voicing,
            tonic=60,
            energy=energy,
            energy_threshold=0.1,
            smoothing_sigma_ms=0.0,
            derivative_threshold=4.0,
            min_event_duration=0.05,
            transcription_min_duration=0.05,
        )

        self.assertTrue(notes, "Expected retained stationary notes above threshold")
        self.assertTrue(any(n.confidence >= 0.99 for n in notes), "Expected stationary notes")
        self.assertTrue(all(n.energy >= 0.1 for n in notes), "All notes should respect threshold")


if __name__ == "__main__":
    unittest.main()
