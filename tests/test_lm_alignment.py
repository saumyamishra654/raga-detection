import unittest

from raga_pipeline.language_model.alignment import (
    token_pitch_info,
    pitch_distance,
    build_substitution_map,
)


class TestTokenPitchInfo(unittest.TestCase):
    """Extract (pitch_class, octave) from sargam LM tokens."""

    def test_middle_octave(self):
        self.assertEqual(token_pitch_info("Sa"), (0, 0))
        self.assertEqual(token_pitch_info("Re"), (2, 0))
        self.assertEqual(token_pitch_info("re"), (1, 0))
        self.assertEqual(token_pitch_info("Ga"), (4, 0))
        self.assertEqual(token_pitch_info("Pa"), (7, 0))
        self.assertEqual(token_pitch_info("Ni"), (11, 0))

    def test_lower_octave(self):
        self.assertEqual(token_pitch_info("Sa'"), (0, -1))
        self.assertEqual(token_pitch_info("Ni'"), (11, -1))
        self.assertEqual(token_pitch_info("dha'"), (8, -1))

    def test_upper_octave(self):
        self.assertEqual(token_pitch_info("Sa''"), (0, 1))
        self.assertEqual(token_pitch_info("Re''"), (2, 1))

    def test_direction_suffix_stripped(self):
        self.assertEqual(token_pitch_info("Re/U"), (2, 0))
        self.assertEqual(token_pitch_info("Ga/D"), (4, 0))
        self.assertEqual(token_pitch_info("Sa/="), (0, 0))
        self.assertEqual(token_pitch_info("Ni'/D"), (11, -1))

    def test_bos_returns_none(self):
        self.assertIsNone(token_pitch_info("<BOS>"))

    def test_unknown_returns_none(self):
        self.assertIsNone(token_pitch_info("?3"))
        self.assertIsNone(token_pitch_info(""))


class TestPitchDistance(unittest.TestCase):
    """Circular pitch-class distance + octave penalty."""

    def test_same_token(self):
        self.assertEqual(pitch_distance(0, 0, 0, 0), 0)

    def test_adjacent_semitone(self):
        # Sa (0) to re (1) = 1 semitone
        self.assertEqual(pitch_distance(0, 0, 1, 0), 1)

    def test_circular_wrap(self):
        # Sa (0) to Ni (11) = 1 semitone (wraps around)
        self.assertEqual(pitch_distance(0, 0, 11, 0), 1)

    def test_tritone(self):
        # Sa (0) to Ma (6) = 6 semitones
        self.assertEqual(pitch_distance(0, 0, 6, 0), 6)

    def test_octave_penalty(self):
        # Same pitch class, different octave = 1
        self.assertEqual(pitch_distance(0, 0, 0, 1), 1)
        # Adjacent PC + octave = 2
        self.assertEqual(pitch_distance(0, 0, 1, 1), 2)


class TestSubstitutionMap(unittest.TestCase):
    """Build token -> [(target, distance)] map."""

    def test_small_vocabulary(self):
        vocab = {"<BOS>", "Sa", "re", "Re", "Ni", "Sa'"}
        smap = build_substitution_map(vocab, max_distance=2)

        # BOS should not appear as source or target
        self.assertNotIn("<BOS>", smap)

        # Sa (pc=0) -> re (pc=1, dist=1), Ni (pc=11, dist=1), Sa' (pc=0, oct=-1, dist=1)
        sa_targets = {t for t, d in smap.get("Sa", [])}
        self.assertIn("re", sa_targets)
        self.assertIn("Ni", sa_targets)
        self.assertIn("Sa'", sa_targets)

        # re (pc=1) -> Sa (dist=1), Re (dist=1)
        re_targets = {t for t, d in smap.get("re", [])}
        self.assertIn("Sa", re_targets)
        self.assertIn("Re", re_targets)

    def test_max_distance_filter(self):
        vocab = {"Sa", "Re", "Ga"}  # pc 0, 2, 4
        smap = build_substitution_map(vocab, max_distance=1)
        # Sa to Re is 2 semitones, should not be included at max_distance=1
        sa_targets = {t for t, d in smap.get("Sa", [])}
        self.assertNotIn("Re", sa_targets)


if __name__ == "__main__":
    unittest.main()
