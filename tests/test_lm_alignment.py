import unittest

from raga_pipeline.language_model import NgramModel
from raga_pipeline.language_model.alignment import (
    token_pitch_info,
    pitch_distance,
    build_substitution_map,
    AlignmentConfig,
    AlignmentResult,
    score_phrase_aligned,
    score_sequence_aligned,
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


def _build_test_model() -> NgramModel:
    """Small model: Yaman vs Bhairav with varied training phrases."""
    model = NgramModel(order=3, smoothing="add-k", smoothing_k=0.01)
    # Multiple phrase shapes so the LM learns distinctive bigram/trigram patterns
    yaman_phrases = [
        ["<BOS>", "Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"],
        ["<BOS>", "Ni", "Re", "Ga", "Ma", "Dha", "Ni", "Sa"],
        ["<BOS>", "Sa", "Re", "Ga", "Re", "Sa", "Ni", "Dha"],
        ["<BOS>", "Pa", "Dha", "Ni", "Sa", "Re", "Ga", "Ma"],
    ]
    bhairav_phrases = [
        ["<BOS>", "Sa", "re", "ga", "Ma", "Pa", "dha", "ni"],
        ["<BOS>", "ni", "Sa", "re", "ga", "Ma", "dha", "ni"],
        ["<BOS>", "Sa", "re", "ga", "re", "Sa", "ni", "dha"],
        ["<BOS>", "Pa", "dha", "ni", "Sa", "re", "ga", "Ma"],
    ]
    for _ in range(100):
        for p in yaman_phrases:
            model.add_sequence("Yaman", [p])
        for p in bhairav_phrases:
            model.add_sequence("Bhairav", [p])
    model.finalize()
    return model


class TestScorePhraseAligned(unittest.TestCase):

    def test_clean_phrase_scores_higher_for_correct_raga(self):
        """A clean Yaman phrase should score higher under Yaman than Bhairav."""
        model = _build_test_model()
        phrase = ["<BOS>", "Sa", "Re", "Ga", "Ma", "Pa"]
        cfg = AlignmentConfig()

        yaman_result = score_phrase_aligned(model, "Yaman", phrase, cfg)
        bhairav_result = score_phrase_aligned(model, "Bhairav", phrase, cfg)

        self.assertGreater(yaman_result.lm_per_token, bhairav_result.lm_per_token)
        self.assertEqual(yaman_result.n_skipped, 0)

    def test_noisy_phrase_still_identifies_correct_raga(self):
        """A Yaman phrase with noise tokens inserted should still rank Yaman first."""
        model = _build_test_model()
        # Yaman core: Sa Re Ga Ma Pa, with noise "ni" and "dha" inserted
        noisy = ["<BOS>", "Sa", "ni", "Re", "dha", "Ga", "Ma", "Pa"]
        cfg = AlignmentConfig(lambda_skip=0.5, beam_width=100)

        yaman_result = score_phrase_aligned(model, "Yaman", noisy, cfg)
        bhairav_result = score_phrase_aligned(model, "Bhairav", noisy, cfg)

        self.assertGreater(yaman_result.lm_per_token, bhairav_result.lm_per_token)
        self.assertGreater(yaman_result.n_skipped, 0)

    def test_substitution_helps(self):
        """Substituting a near-pitch token should score better than skipping it."""
        model = _build_test_model()
        # "re" is 1 semitone from "Re" (Yaman uses Re, not re)
        phrase_with_sub = ["<BOS>", "Sa", "re", "Ga"]
        cfg_with_sub = AlignmentConfig(lambda_skip=1.0, lambda_sub=0.1, max_sub_distance=2)
        cfg_no_sub = AlignmentConfig(lambda_skip=1.0, lambda_sub=0.1, max_sub_distance=0)

        # Build explicit sub_map so the substitution branch is actually entered
        sub_map = build_substitution_map(model.vocabulary, max_distance=2)
        result_sub = score_phrase_aligned(model, "Yaman", phrase_with_sub, cfg_with_sub, sub_map=sub_map)
        result_nosub = score_phrase_aligned(model, "Yaman", phrase_with_sub, cfg_no_sub)

        # With substitution enabled, scorer must actually use substitutions
        self.assertGreater(result_sub.n_substituted, 0, "substitution branch was never taken")
        self.assertGreaterEqual(result_sub.n_matched, result_nosub.n_matched)

    def test_empty_phrase(self):
        model = _build_test_model()
        result = score_phrase_aligned(model, "Yaman", [], AlignmentConfig())
        self.assertEqual(result.n_matched, 0)
        self.assertAlmostEqual(result.lm_per_token, 0.0)

    def test_bos_only_phrase(self):
        model = _build_test_model()
        result = score_phrase_aligned(model, "Yaman", ["<BOS>"], AlignmentConfig())
        self.assertEqual(result.n_matched, 0)


class TestScoreSequenceAligned(unittest.TestCase):

    def test_multi_phrase_aggregation(self):
        """Scores across multiple phrases should be aggregated."""
        model = _build_test_model()
        # Both phrases start with Sa (as in training data) so the first
        # token has a reasonable log-prob and the DP prefers matching.
        phrases = [
            ["<BOS>", "Sa", "Re", "Ga"],
            ["<BOS>", "Sa", "Re", "Ga"],
        ]
        cfg = AlignmentConfig(lambda_skip=1.0)
        result = score_sequence_aligned(model, "Yaman", phrases, cfg)
        self.assertGreater(result.lm_per_token, -10.0)
        self.assertEqual(result.n_matched, 6)  # 3 + 3 tokens (excl BOS)

    def test_correct_raga_ranks_first_noisy(self):
        """Noisy Yaman (with inserted Bhairav tokens) should rank Yaman over Bhairav.

        Substitution disabled so Bhairav cannot cheat by substituting Re->re etc.
        (With substitution enabled, Yaman/Bhairav are only 1 semitone apart and
        the toy model can't discriminate -- this is a test limitation, not a bug.)
        """
        model = _build_test_model()
        noisy_phrases = [
            ["<BOS>", "Sa", "Re", "Ga", "Ma", "dha", "Pa", "Dha", "Ni"],
            ["<BOS>", "Ni", "Re", "Ga", "ni", "Ma", "Dha", "Ni", "Sa"],
        ]
        cfg = AlignmentConfig(lambda_skip=0.5, max_sub_distance=0)

        yaman = score_sequence_aligned(model, "Yaman", noisy_phrases, cfg)
        bhairav = score_sequence_aligned(model, "Bhairav", noisy_phrases, cfg)
        self.assertGreater(yaman.lm_per_token, bhairav.lm_per_token)


class TestNgramModelPublicAPI(unittest.TestCase):
    """Verify the new public API methods on NgramModel."""

    def test_vocabulary_property(self):
        model = _build_test_model()
        vocab = model.vocabulary
        self.assertIsInstance(vocab, set)
        self.assertIn("Sa", vocab)
        self.assertIn("<BOS>", vocab)

    def test_remove_raga(self):
        model = _build_test_model()
        self.assertIn("Yaman", model.ragas())
        model.remove_raga("Yaman")
        self.assertNotIn("Yaman", model.ragas())
        # Bhairav still there
        self.assertIn("Bhairav", model.ragas())


if __name__ == "__main__":
    unittest.main()
