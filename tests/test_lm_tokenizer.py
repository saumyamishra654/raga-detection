import unittest
from raga_pipeline.sequence import Note, tokenize_notes_for_lm


class TestTokenizeNotesForLM(unittest.TestCase):
    """Tests for the shared LM tokenizer (phrase-separated, direction off by default)."""

    def _make_note(self, start, end, midi, sargam="", pitch_class=-1, confidence=0.9):
        return Note(
            start=start, end=end, pitch_midi=midi,
            pitch_hz=440.0, confidence=confidence,
            sargam=sargam, pitch_class=pitch_class,
        )

    def test_basic_middle_octave(self):
        notes = [
            self._make_note(0.0, 0.5, 60),
            self._make_note(0.5, 1.0, 62),
            self._make_note(1.0, 1.5, 64),
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(phrases, [["<BOS>", "Sa", "Re", "Ga"]])

    def test_lower_octave_marking(self):
        notes = [
            self._make_note(0.0, 0.5, 60),
            self._make_note(0.5, 1.0, 59),
            self._make_note(1.0, 1.5, 57),
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(phrases, [["<BOS>", "Sa", "Ni'", "Dha'"]])

    def test_upper_octave_marking(self):
        notes = [
            self._make_note(0.0, 0.5, 72),
            self._make_note(0.5, 1.0, 74),
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(phrases, [["<BOS>", "Sa''", "Re''"]])

    def test_phrase_boundaries(self):
        notes = [
            self._make_note(0.0, 0.5, 60),
            self._make_note(0.5, 1.0, 62),
            self._make_note(2.0, 2.5, 64),
            self._make_note(2.5, 3.0, 65),
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60, phrase_gap_sec=0.25)
        self.assertEqual(phrases, [
            ["<BOS>", "Sa", "Re"],
            ["<BOS>", "Ga", "ma"],
        ])

    def test_empty_notes(self):
        phrases = tokenize_notes_for_lm([], tonic_midi=60)
        self.assertEqual(phrases, [])

    def test_komal_shuddha_encoding(self):
        notes = [
            self._make_note(0.0, 0.5, 61),
            self._make_note(0.5, 1.0, 62),
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(phrases, [["<BOS>", "re", "Re"]])

    def test_single_note(self):
        notes = [self._make_note(0.0, 0.5, 67)]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(phrases, [["<BOS>", "Pa"]])

    def test_clipping_extreme_octaves(self):
        notes = [self._make_note(0.0, 0.5, 36)]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60)
        self.assertEqual(phrases, [["<BOS>", "Sa'"]])

    def test_non_c_tonic_middle_octave(self):
        notes = [
            self._make_note(0.0, 0.3, 67),
            self._make_note(0.3, 0.6, 72),
            self._make_note(0.6, 0.9, 74),
            self._make_note(0.9, 1.2, 78),
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=67)
        self.assertEqual(phrases, [["<BOS>", "Sa", "ma", "Pa", "Ni"]])

    def test_non_c_tonic_octave_boundaries(self):
        notes = [
            self._make_note(0.0, 0.3, 79),
            self._make_note(0.3, 0.6, 66),
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=67)
        self.assertEqual(phrases, [["<BOS>", "Sa''", "Ni'"]])

    def test_multiple_phrases_each_start_with_bos(self):
        notes = [
            self._make_note(0.0, 0.2, 60),
            self._make_note(1.0, 1.2, 62),
            self._make_note(2.5, 2.7, 64),
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60, phrase_gap_sec=0.25)
        self.assertEqual(len(phrases), 3)
        for phrase in phrases:
            self.assertEqual(phrase[0], "<BOS>")

    def test_direction_opt_in(self):
        """Direction markers work when explicitly enabled."""
        notes = [
            self._make_note(0.0, 0.2, 60),
            self._make_note(0.2, 0.4, 64),
            self._make_note(0.4, 0.6, 62),
            self._make_note(0.6, 0.8, 62),
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60, include_direction=True)
        self.assertEqual(phrases, [["<BOS>", "Sa", "Ga/U", "Re/D", "Re/="]])

    def test_direction_resets_at_phrase_boundary(self):
        notes = [
            self._make_note(0.0, 0.2, 60),
            self._make_note(0.2, 0.4, 64),
            self._make_note(1.0, 1.2, 62),
            self._make_note(1.2, 1.4, 60),
        ]
        phrases = tokenize_notes_for_lm(notes, tonic_midi=60, phrase_gap_sec=0.25, include_direction=True)
        self.assertEqual(phrases, [
            ["<BOS>", "Sa", "Ga/U"],
            ["<BOS>", "Re", "Sa/D"],
        ])


if __name__ == "__main__":
    unittest.main()
