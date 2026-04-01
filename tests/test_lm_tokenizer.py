import unittest
from raga_pipeline.sequence import Note, tokenize_notes_for_lm


class TestTokenizeNotesForLM(unittest.TestCase):
    """Tests for the shared LM tokenizer (phrase-separated output)."""

    def _make_note(self, start, end, midi, sargam="", pitch_class=-1, confidence=0.9):
        return Note(
            start=start, end=end, pitch_midi=midi,
            pitch_hz=440.0, confidence=confidence,
            sargam=sargam, pitch_class=pitch_class,
        )

    def test_basic_middle_octave(self):
        """Notes in the middle octave get bare sargam tokens."""
        notes = [
            self._make_note(0.0, 0.5, 60),   # Sa
            self._make_note(0.5, 1.0, 62),   # Re
            self._make_note(1.0, 1.5, 64),   # Ga
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(phrases, [["<BOS>", "Sa", "Re", "Ga"]])

    def test_lower_octave_marking(self):
        """Notes below the reference octave get ' suffix."""
        notes = [
            self._make_note(0.0, 0.5, 60),   # Sa (middle)
            self._make_note(0.5, 1.0, 59),   # Ni (lower)
            self._make_note(1.0, 1.5, 57),   # Dha (lower)
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(phrases, [["<BOS>", "Sa", "Ni'", "Dha'"]])

    def test_upper_octave_marking(self):
        """Notes above the reference octave get '' suffix."""
        notes = [
            self._make_note(0.0, 0.5, 72),   # Sa (upper)
            self._make_note(0.5, 1.0, 74),   # Re (upper)
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(phrases, [["<BOS>", "Sa''", "Re''"]])

    def test_phrase_boundaries(self):
        """Silence gaps > phrase_gap_sec produce separate phrase lists."""
        notes = [
            self._make_note(0.0, 0.5, 60),   # Sa
            self._make_note(0.5, 1.0, 62),   # Re
            # 1.0s gap (> 0.25 default)
            self._make_note(2.0, 2.5, 64),   # Ga
            self._make_note(2.5, 3.0, 65),   # ma
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60, phrase_gap_sec=0.25)
        self.assertEqual(phrases, [
            ["<BOS>", "Sa", "Re"],
            ["<BOS>", "Ga", "ma"],
        ])

    def test_empty_notes(self):
        """Empty note list returns empty list."""
        phrases = tokenize_notes_for_lm([], tonic_midi=60)
        self.assertEqual(phrases, [])

    def test_komal_shuddha_encoding(self):
        """Komal/shuddha distinction preserved via sargam case."""
        notes = [
            self._make_note(0.0, 0.5, 61),   # re (komal)
            self._make_note(0.5, 1.0, 62),   # Re (shuddha)
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(phrases, [["<BOS>", "re", "Re"]])

    def test_single_note(self):
        """Single note produces one phrase with BOS + one token."""
        notes = [self._make_note(0.0, 0.5, 67)]  # Pa
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(phrases, [["<BOS>", "Pa"]])

    def test_clipping_extreme_octaves(self):
        """Notes more than 1 octave away are clipped to the boundary octave."""
        notes = [self._make_note(0.0, 0.5, 36)]  # Sa, 2 octaves below -> clip to Sa'
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(phrases, [["<BOS>", "Sa'"]])

    def test_non_c_tonic_middle_octave(self):
        """Notes in the same musical octave as a non-C tonic stay in middle octave."""
        notes = [
            self._make_note(0.0, 0.3, 67),   # Sa
            self._make_note(0.3, 0.6, 72),   # ma (C5, still middle octave)
            self._make_note(0.6, 0.9, 74),   # Pa (D5, still middle octave)
            self._make_note(0.9, 1.2, 78),   # Ni (F#5, still middle octave)
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=67)
        self.assertEqual(phrases, [["<BOS>", "Sa", "ma", "Pa", "Ni"]])

    def test_non_c_tonic_octave_boundaries(self):
        """Upper/lower octave markers are correct for non-C tonic."""
        notes = [
            self._make_note(0.0, 0.3, 79),   # Sa (upper)
            self._make_note(0.3, 0.6, 66),   # Ni (lower) -- contiguous, no phrase break
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=67)
        self.assertEqual(phrases, [["<BOS>", "Sa''", "Ni'"]])

    def test_multiple_phrases_each_start_with_bos(self):
        """Every phrase starts with <BOS>."""
        notes = [
            self._make_note(0.0, 0.2, 60),
            self._make_note(1.0, 1.2, 62),  # gap > 0.25
            self._make_note(2.5, 2.7, 64),  # gap > 0.25
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60, phrase_gap_sec=0.25)
        self.assertEqual(len(phrases), 3)
        for phrase in phrases:
            self.assertEqual(phrase[0], "<BOS>")


if __name__ == "__main__":
    unittest.main()
