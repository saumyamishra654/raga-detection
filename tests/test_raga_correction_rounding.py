import unittest

try:
    from raga_pipeline.raga import apply_raga_correction_to_notes
    from raga_pipeline.sequence import Note

    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


class _DummyRagaDb:
    def __init__(self, allowed_pcs: tuple[int, ...] = (0,)) -> None:
        allowed = {int(pc) % 12 for pc in allowed_pcs}
        mask = tuple(1 if i in allowed else 0 for i in range(12))
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
        self.assertEqual(stats["corrected"], 1)
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

    def test_continuous_distance_prefers_truly_nearest_allowed_note(self) -> None:
        # Allowed notes are 60 (Sa) and 62 (Re). 61.49 is much closer to 62.
        db = _DummyRagaDb((0, 2))
        note = _mk_note(61.49, sargam="re", pitch_class=1)

        corrected, stats, _ = apply_raga_correction_to_notes(
            [note],
            db,
            "TestRaga",
            tonic=0,
            max_distance=2.0,
            keep_impure=False,
        )

        self.assertEqual(len(corrected), 1)
        self.assertEqual(stats["discarded"], 0)
        self.assertEqual(stats["corrected"], 1)
        self.assertAlmostEqual(corrected[0].pitch_midi, 62.0, places=6)

    def test_tie_prefers_continuity_then_higher(self) -> None:
        db = _DummyRagaDb((0, 2))

        # Continuity path: second note at 61.0 is equidistant to 60 and 62;
        # previous corrected note near 60 should bias toward 60.
        notes = [
            _mk_note(60.05, sargam="Sa", pitch_class=0),
            _mk_note(61.0, sargam="?", pitch_class=1),
        ]
        corrected, _, _ = apply_raga_correction_to_notes(
            notes,
            db,
            "TestRaga",
            tonic=0,
            max_distance=2.0,
            keep_impure=False,
        )
        self.assertEqual(len(corrected), 2)
        self.assertAlmostEqual(corrected[0].pitch_midi, 60.0, places=6)
        self.assertAlmostEqual(corrected[1].pitch_midi, 60.0, places=6)

        # No continuity context: equal-distance tie should prefer higher note.
        single_tie, _, _ = apply_raga_correction_to_notes(
            [_mk_note(61.0, sargam="?", pitch_class=1)],
            db,
            "TestRaga",
            tonic=0,
            max_distance=2.0,
            keep_impure=False,
        )
        self.assertEqual(len(single_tie), 1)
        self.assertAlmostEqual(single_tie[0].pitch_midi, 62.0, places=6)

    def test_correction_uses_raw_pitch_when_available(self) -> None:
        db = _DummyRagaDb((0, 2))
        # Canonical pitch is already quantized at 61 (tie between 60/62),
        # but raw contour was 60.51 and should therefore snap to 60.
        note = _mk_note(61.0, sargam="?", pitch_class=1)
        note.raw_pitch_midi = 60.51

        corrected, stats, _ = apply_raga_correction_to_notes(
            [note],
            db,
            "TestRaga",
            tonic=0,
            max_distance=2.0,
            keep_impure=False,
        )

        self.assertEqual(len(corrected), 1)
        self.assertEqual(stats["discarded"], 0)
        self.assertEqual(stats["corrected"], 1)
        self.assertAlmostEqual(corrected[0].pitch_midi, 60.0, places=6)

    def test_keep_impure_far_note_keeps_raw_not_farther_snapped_pitch(self) -> None:
        db = _DummyRagaDb((0,))
        # Tiny dip around 61.49 was pre-snapped to 62.0 in transcription,
        # but in keep-impure mode the far note should stay at raw 61.49.
        note = _mk_note(62.0, sargam="re", pitch_class=2)
        note.raw_pitch_midi = 61.49

        corrected, stats, corrections = apply_raga_correction_to_notes(
            [note],
            db,
            "TestRaga",
            tonic=0,
            max_distance=1.0,
            keep_impure=True,
        )

        self.assertEqual(len(corrected), 1)
        self.assertEqual(stats["unchanged"], 1)
        self.assertEqual(corrections[0]["action"], "unchanged_far")
        self.assertAlmostEqual(float(corrections[0]["corrected_midi"]), 61.49, places=6)
        self.assertAlmostEqual(corrected[0].pitch_midi, 61.49, places=6)


if __name__ == "__main__":
    unittest.main()
