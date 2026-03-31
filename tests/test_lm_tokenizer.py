import unittest
from raga_pipeline.sequence import Note, tokenize_notes_for_lm


class TestTokenizeNotesForLM(unittest.TestCase):
    """Tests for the shared LM tokenizer."""

    def _make_note(self, start, end, midi, sargam="", pitch_class=-1, confidence=0.9):
        return Note(
            start=start, end=end, pitch_midi=midi,
            pitch_hz=440.0, confidence=confidence,
            sargam=sargam, pitch_class=pitch_class,
        )

    def test_basic_middle_octave(self):
        """Notes in the middle octave get bare sargam tokens."""
        # Tonic = C4 = MIDI 60, so Sa=60, Re=62, Ga=64
        notes = [
            self._make_note(0.0, 0.5, 60),   # Sa
            self._make_note(0.5, 1.0, 62),   # Re
            self._make_note(1.0, 1.5, 64),   # Ga
        ]
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(tokens, ["<BOS>", "Sa", "Re", "Ga"])

    def test_lower_octave_marking(self):
        """Notes below the reference octave get ' suffix."""
        notes = [
            self._make_note(0.0, 0.5, 60),   # Sa (middle)
            self._make_note(0.5, 1.0, 59),   # Ni (lower)
            self._make_note(1.0, 1.5, 57),   # Dha (lower)
        ]
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(tokens, ["<BOS>", "Sa", "Ni'", "Dha'"])

    def test_upper_octave_marking(self):
        """Notes above the reference octave get '' suffix."""
        # Sa one octave above = MIDI 72
        notes = [
            self._make_note(0.0, 0.5, 72),   # Sa (upper)
            self._make_note(0.5, 1.0, 74),   # Re (upper)
        ]
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(tokens, ["<BOS>", "Sa''", "Re''"])

    def test_phrase_boundaries(self):
        """Silence gaps > phrase_gap_sec insert <BOS> tokens."""
        notes = [
            self._make_note(0.0, 0.5, 60),   # Sa
            self._make_note(0.5, 1.0, 62),   # Re
            # 1.0s gap (> 0.25 default)
            self._make_note(2.0, 2.5, 64),   # Ga
            self._make_note(2.5, 3.0, 65),   # ma
        ]
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60, phrase_gap_sec=0.25)
        self.assertEqual(tokens, ["<BOS>", "Sa", "Re", "<BOS>", "Ga", "ma"])

    def test_empty_notes(self):
        """Empty note list returns empty token list."""
        tokens = tokenize_notes_for_lm([], tonic_midi=60)
        self.assertEqual(tokens, [])

    def test_komal_shuddha_encoding(self):
        """Komal/shuddha distinction preserved via sargam case."""
        notes = [
            self._make_note(0.0, 0.5, 61),   # re (komal)
            self._make_note(0.5, 1.0, 62),   # Re (shuddha)
        ]
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(tokens, ["<BOS>", "re", "Re"])

    def test_single_note(self):
        """Single note produces BOS + one token."""
        notes = [self._make_note(0.0, 0.5, 67)]  # Pa
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(tokens, ["<BOS>", "Pa"])

    def test_clipping_extreme_octaves(self):
        """Notes more than 1 octave away are clipped to the boundary octave."""
        notes = [self._make_note(0.0, 0.5, 36)]  # Sa, 2 octaves below -> clip to Sa'
        tokens = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(tokens, ["<BOS>", "Sa'"])

    def test_non_c_tonic_middle_octave(self):
        """Notes in the same musical octave as a non-C tonic stay in middle octave."""
        # Tonic = G4 = MIDI 67. The musical octave spans G4(67) to F#5(78).
        # All 12 sargam notes in this range should be bare (no octave marker).
        # Sa=67(G), re=68(Ab), Re=69(A), ga=70(Bb), Ga=71(B),
        # ma=72(C5), Ma=73(C#5), Pa=74(D5), dha=75(Eb5), Dha=76(E5),
        # ni=77(F5), Ni=78(F#5)
        notes = [
            self._make_note(0.0, 0.3, 67),   # Sa
            self._make_note(0.3, 0.6, 72),   # ma (C5, still middle octave)
            self._make_note(0.6, 0.9, 74),   # Pa (D5, still middle octave)
            self._make_note(0.9, 1.2, 78),   # Ni (F#5, still middle octave)
        ]
        tokens = tokenize_notes_for_lm(notes, tonic_midi=67)
        self.assertEqual(tokens, ["<BOS>", "Sa", "ma", "Pa", "Ni"])

    def test_non_c_tonic_octave_boundaries(self):
        """Upper/lower octave markers are correct for non-C tonic."""
        # Tonic = G4 = MIDI 67.
        # G5 = MIDI 79 -> Sa'' (upper octave)
        # F#4 = MIDI 66 -> Ni' (lower octave, one semitone below tonic)
        notes = [
            self._make_note(0.0, 0.3, 79),   # Sa (upper)
            self._make_note(0.3, 0.6, 66),   # Ni (lower) -- contiguous, no phrase break
        ]
        tokens = tokenize_notes_for_lm(notes, tonic_midi=67)
        self.assertEqual(tokens, ["<BOS>", "Sa''", "Ni'"])


if __name__ == "__main__":
    unittest.main()
