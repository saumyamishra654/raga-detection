import unittest

try:
    from raga_pipeline.sequence import Note, merge_consecutive_notes

    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


def _note(start: float, end: float, pitch_midi: float, energy: float = 0.5) -> "Note":
    return Note(
        start=start,
        end=end,
        pitch_midi=pitch_midi,
        pitch_hz=440.0,
        confidence=1.0,
        energy=energy,
        sargam="re",
        pitch_class=2,
    )


@unittest.skipUnless(IMPORT_OK, "sequence module unavailable in current environment")
class MergeConsecutiveNotesTests(unittest.TestCase):
    def test_dropout_healing_merges_short_fragments(self) -> None:
        notes = [
            _note(77.720, 77.730, 62.0, 0.1708),
            _note(77.832, 77.842, 62.0, 0.4481),
            _note(77.976, 78.040, 62.0, 0.6971),
            _note(78.296, 78.914, 62.0, 0.7886),
            _note(79.064, 79.090, 62.0, 0.6960),
        ]

        merged = merge_consecutive_notes(
            notes,
            max_gap=0.1,
            pitch_tolerance=0.7,
            max_dropout_gap=0.18,
            dropout_fragment_duration=0.12,
        )

        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[0].start, 77.720, places=3)
        self.assertAlmostEqual(merged[0].end, 78.040, places=3)
        self.assertAlmostEqual(merged[1].start, 78.296, places=3)
        self.assertAlmostEqual(merged[1].end, 79.090, places=3)

    def test_phrase_level_collapse_merges_same_note_segments(self) -> None:
        notes = [
            _note(10.00, 10.05, 62.0),
            _note(10.40, 10.55, 62.0),
            _note(10.80, 11.00, 62.0),
        ]

        merged = merge_consecutive_notes(
            notes,
            max_gap=1.0,
            pitch_tolerance=0.7,
            max_dropout_gap=1.0,
            dropout_fragment_duration=1.0,
        )

        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0].start, 10.00, places=3)
        self.assertAlmostEqual(merged[0].end, 11.00, places=3)

    def test_long_non_fragment_gap_not_merged_by_dropout_rule(self) -> None:
        notes = [
            _note(1.00, 1.45, 62.0),
            _note(1.60, 2.10, 62.0),
        ]

        merged = merge_consecutive_notes(
            notes,
            max_gap=0.1,
            pitch_tolerance=0.7,
            max_dropout_gap=0.18,
            dropout_fragment_duration=0.12,
        )
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
