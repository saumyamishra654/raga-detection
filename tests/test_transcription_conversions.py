import unittest

import numpy as np

try:
    from raga_pipeline import transcription

    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


@unittest.skipUnless(IMPORT_OK, "transcription module unavailable in current environment")
class TranscriptionConversionTests(unittest.TestCase):
    def test_hz_midi_roundtrip(self) -> None:
        midi_values = np.array([48.0, 60.0, 69.0, 72.5], dtype=float)
        hz_values = np.array([transcription._midi_to_hz_scalar(m) for m in midi_values], dtype=float)
        recovered = transcription._hz_to_midi_array(hz_values)
        self.assertTrue(np.allclose(recovered, midi_values, atol=1e-9))

    def test_parse_note_with_and_without_octave(self) -> None:
        self.assertEqual(transcription._parse_note_to_midi("C4"), 60.0)
        self.assertEqual(transcription._parse_note_to_midi("A4"), 69.0)
        self.assertEqual(transcription._parse_note_to_midi("Bb3"), 58.0)
        self.assertEqual(transcription._parse_note_to_midi("C#"), 61.0)
        self.assertEqual(transcription._parse_note_to_midi("db"), 61.0)
        self.assertIsNone(transcription._parse_note_to_midi("not-a-note"))

    def test_resolve_tonic_supports_hz_and_note(self) -> None:
        self.assertAlmostEqual(transcription._resolve_tonic(440.0), 69.0, places=6)
        self.assertEqual(transcription._resolve_tonic("C#4"), 61.0)
        self.assertEqual(transcription._resolve_tonic("C#"), 61.0)


if __name__ == "__main__":
    unittest.main()
