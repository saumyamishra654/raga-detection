import unittest

try:
    from raga_pipeline import transcription

    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


@unittest.skipUnless(IMPORT_OK, "transcription module unavailable in current environment")
class TranscriptionRagaSnapContinuousTests(unittest.TestCase):
    def test_raga_mode_uses_continuous_nearest_allowed_note(self) -> None:
        snapped, _, _, keep = transcription._snap_pitch(  # type: ignore[attr-defined]
            pitch=61.49,
            tonic_midi=60.0,
            mode="raga",
            allowed_pcs=[0, 2],
            tolerance_cents=35.0,
        )
        self.assertTrue(keep)
        self.assertIsNotNone(snapped)
        self.assertAlmostEqual(float(snapped), 62.0, places=6)

    def test_raga_mode_rejects_candidates_farther_than_one_semitone(self) -> None:
        snapped, error, label, keep = transcription._snap_pitch(  # type: ignore[attr-defined]
            pitch=63.4,
            tonic_midi=60.0,
            mode="raga",
            allowed_pcs=[0],
            tolerance_cents=35.0,
        )
        self.assertFalse(keep)
        self.assertIsNone(snapped)
        self.assertEqual(error, 0.0)
        self.assertEqual(label, "")

    def test_chromatic_mode_remains_unchanged(self) -> None:
        snapped, _, _, keep = transcription._snap_pitch(  # type: ignore[attr-defined]
            pitch=61.49,
            tonic_midi=60.0,
            mode="chromatic",
            allowed_pcs=[0],
            tolerance_cents=35.0,
        )
        self.assertTrue(keep)
        self.assertIsNotNone(snapped)
        self.assertAlmostEqual(float(snapped), 61.0, places=6)


if __name__ == "__main__":
    unittest.main()
