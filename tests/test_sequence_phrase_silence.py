import unittest
import numpy as np

from raga_pipeline.sequence import Note, Phrase, detect_phrases_by_silence


def _make_note(start: float, end: float, midi: float = 60.0) -> Note:
    return Note(
        start=start, end=end, pitch_midi=midi,
        pitch_hz=261.6, confidence=1.0, energy=0.5,
    )


class TestDetectPhrasesBySilence(unittest.TestCase):

    def _make_energy_track(self, duration: float, dt: float = 0.01):
        """Return (energy, timestamps) arrays for a given duration, all loud."""
        n = int(duration / dt) + 1
        timestamps = np.linspace(0, duration, n)
        energy = np.ones(n) * 0.5
        return energy, timestamps

    def test_no_silence_single_phrase(self):
        """All energy above threshold -> one phrase with all notes."""
        notes = [_make_note(0.0, 0.3), _make_note(0.35, 0.6), _make_note(0.65, 1.0)]
        energy, ts = self._make_energy_track(1.0)
        phrases = detect_phrases_by_silence(
            notes, energy, ts, silence_threshold=0.10, silence_min_duration=0.25,
        )
        self.assertEqual(len(phrases), 1)
        self.assertEqual(len(phrases[0].notes), 3)

    def test_clear_silence_splits_into_two(self):
        """A 0.5s silent gap in the middle -> two phrases."""
        notes = [_make_note(0.0, 0.4), _make_note(1.0, 1.4)]
        energy, ts = self._make_energy_track(1.5)
        # Insert silence from 0.5s to 0.95s
        silence_mask = (ts >= 0.45) & (ts <= 0.95)
        energy[silence_mask] = 0.01
        phrases = detect_phrases_by_silence(
            notes, energy, ts, silence_threshold=0.10, silence_min_duration=0.25,
        )
        self.assertEqual(len(phrases), 2)
        self.assertEqual(len(phrases[0].notes), 1)
        self.assertEqual(len(phrases[1].notes), 1)

    def test_short_silence_no_split(self):
        """Silence shorter than silence_min_duration -> no split."""
        notes = [_make_note(0.0, 0.4), _make_note(0.7, 1.0)]
        energy, ts = self._make_energy_track(1.0)
        # Insert only 0.15s of silence (below 0.25s min)
        silence_mask = (ts >= 0.45) & (ts <= 0.60)
        energy[silence_mask] = 0.01
        phrases = detect_phrases_by_silence(
            notes, energy, ts, silence_threshold=0.10, silence_min_duration=0.25,
        )
        self.assertEqual(len(phrases), 1)
        self.assertEqual(len(phrases[0].notes), 2)

    def test_min_phrase_duration_filter(self):
        """Phrases shorter than min_phrase_duration get merged into neighbor."""
        notes = [
            _make_note(0.0, 0.5),
            _make_note(1.0, 1.05),  # very short phrase
            _make_note(2.0, 2.5),
        ]
        energy, ts = self._make_energy_track(3.0)
        # Silence from 0.6-0.9 and 1.1-1.9
        energy[(ts >= 0.55) & (ts <= 0.95)] = 0.01
        energy[(ts >= 1.10) & (ts <= 1.95)] = 0.01
        phrases = detect_phrases_by_silence(
            notes, energy, ts, silence_threshold=0.10, silence_min_duration=0.25,
            min_phrase_duration=0.2,
        )
        # Middle phrase (0.05s) is below min_phrase_duration -> merged
        self.assertTrue(len(phrases) <= 2)
        total_notes = sum(len(p.notes) for p in phrases)
        self.assertEqual(total_notes, 3)

    def test_min_notes_filter(self):
        """Phrases with fewer notes than min_notes_in_phrase get merged."""
        notes = [
            _make_note(0.0, 0.5),
            _make_note(0.55, 1.0),
            _make_note(2.0, 2.3),  # solo note after silence
            _make_note(3.0, 3.5),
            _make_note(3.55, 4.0),
        ]
        energy, ts = self._make_energy_track(4.0)
        energy[(ts >= 1.05) & (ts <= 1.95)] = 0.01
        energy[(ts >= 2.35) & (ts <= 2.95)] = 0.01
        phrases = detect_phrases_by_silence(
            notes, energy, ts, silence_threshold=0.10, silence_min_duration=0.25,
            min_notes_in_phrase=2,
        )
        # Solo note phrase should be merged into a neighbor
        total_notes = sum(len(p.notes) for p in phrases)
        self.assertEqual(total_notes, 5)
        self.assertTrue(all(len(p.notes) >= 2 for p in phrases))

    def test_single_note_input(self):
        """Single note -> single phrase."""
        notes = [_make_note(0.0, 0.5)]
        energy, ts = self._make_energy_track(0.5)
        phrases = detect_phrases_by_silence(
            notes, energy, ts, silence_threshold=0.10, silence_min_duration=0.25,
        )
        self.assertEqual(len(phrases), 1)
        self.assertEqual(len(phrases[0].notes), 1)

    def test_empty_notes(self):
        """Empty note list -> empty phrases."""
        energy, ts = self._make_energy_track(1.0)
        phrases = detect_phrases_by_silence(
            [], energy, ts, silence_threshold=0.10, silence_min_duration=0.25,
        )
        self.assertEqual(len(phrases), 0)

    def test_all_silent_energy(self):
        """All energy below threshold -> still groups all notes into one phrase."""
        notes = [_make_note(0.0, 0.3), _make_note(0.5, 0.8)]
        n = 100
        ts = np.linspace(0, 1.0, n)
        energy = np.ones(n) * 0.01  # all below threshold
        phrases = detect_phrases_by_silence(
            notes, energy, ts, silence_threshold=0.10, silence_min_duration=0.25,
        )
        # Even with all-silent energy, notes still get grouped
        total_notes = sum(len(p.notes) for p in phrases)
        self.assertEqual(total_notes, 2)

    def test_threshold_zero_returns_single_phrase(self):
        """silence_threshold=0 disables splitting -> one phrase."""
        notes = [_make_note(0.0, 0.3), _make_note(1.0, 1.3)]
        energy, ts = self._make_energy_track(1.5)
        energy[(ts >= 0.4) & (ts <= 0.9)] = 0.01
        phrases = detect_phrases_by_silence(
            notes, energy, ts, silence_threshold=0.0, silence_min_duration=0.25,
        )
        self.assertEqual(len(phrases), 1)
        self.assertEqual(len(phrases[0].notes), 2)


if __name__ == "__main__":
    unittest.main()
