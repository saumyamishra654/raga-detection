import unittest

try:
    from raga_pipeline.raga import apply_raga_correction_to_notes
    from raga_pipeline.sequence import Note

    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


class _DummyRagaDb:
    def __init__(self) -> None:
        mask = tuple(1 if i == 0 else 0 for i in range(12))
        self.name_to_mask = {"testraga": mask}
        self.all_ragas = [{"names": ["TestRaga"], "mask": mask}]


def _mk_note(pitch_midi: float, sargam: str, pitch_class: int) -> "Note":
    return Note(
        start=0.0,
        end=0.5,
        pitch_midi=pitch_midi,
        pitch_hz=float(440.0 * (2.0 ** ((pitch_midi - 69.0) / 12.0))),
        confidence=1.0,
        energy=0.8,
        sargam=sargam,
        pitch_class=pitch_class,
    )


@unittest.skipUnless(IMPORT_OK, "raga pipeline imports unavailable in current environment")
class RagaCorrectionRoundingTests(unittest.TestCase):
    def test_kept_note_uses_snapped_pitch_not_raw_microtonal(self) -> None:
        db = _DummyRagaDb()
        micro_note = _mk_note(60.49, sargam="re", pitch_class=1)

        corrected, stats, _ = apply_raga_correction_to_notes(
            [micro_note],
            db,
            "TestRaga",
            tonic=0,
            max_distance=1.0,
            keep_impure=False,
        )

        self.assertEqual(len(corrected), 1)
        self.assertEqual(stats["discarded"], 0)
        self.assertEqual(stats["unchanged"], 1)
        self.assertAlmostEqual(corrected[0].pitch_midi, 60.0, places=6)
        self.assertEqual(corrected[0].pitch_class, 0)

    def test_corrected_note_refreshes_sargam_after_pitch_change(self) -> None:
        db = _DummyRagaDb()
        wrong_note = _mk_note(61.0, sargam="re", pitch_class=1)

        corrected, stats, _ = apply_raga_correction_to_notes(
            [wrong_note],
            db,
            "TestRaga",
            tonic=0,
            max_distance=1.0,
            keep_impure=False,
        )

        self.assertEqual(len(corrected), 1)
        self.assertEqual(stats["corrected"], 1)
        self.assertAlmostEqual(corrected[0].pitch_midi, 60.0, places=6)
        self.assertNotEqual(corrected[0].sargam, "re")


if __name__ == "__main__":
    unittest.main()
