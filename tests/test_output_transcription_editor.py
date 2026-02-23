import unittest

try:
    from raga_pipeline.output import _generate_transcription_editor_section
    from raga_pipeline.sequence import Note, Phrase

    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


@unittest.skipUnless(IMPORT_OK, "output module unavailable in current environment")
class TranscriptionEditorSectionTests(unittest.TestCase):
    def test_editor_section_contains_expected_controls_and_hooks(self) -> None:
        notes = [
            Note(
                start=0.10,
                end=0.35,
                pitch_midi=60.0,
                pitch_hz=261.63,
                confidence=0.95,
                energy=0.2,
                sargam="Sa",
                pitch_class=0,
            ),
            Note(
                start=0.40,
                end=0.62,
                pitch_midi=62.0,
                pitch_hz=293.66,
                confidence=0.95,
                energy=0.3,
                sargam="Re",
                pitch_class=2,
            ),
        ]
        notes[0].raw_pitch_midi = 60.24
        notes[0].snapped_pitch_midi = 60.0
        notes[0].corrected_pitch_midi = 60.0
        notes[0].rendered_pitch_midi = 60.0
        phrases = [Phrase(notes=notes)]

        html = _generate_transcription_editor_section(notes, phrases, tonic=60)

        self.assertIn("Transcription Editor (Experimental)", html)
        self.assertIn("Save as New Version", html)
        self.assertIn("Delete Version", html)
        self.assertIn("Regenerate Report", html)
        self.assertIn("Set Selected Default", html)
        self.assertIn("Set Original Default", html)
        self.assertIn("Default: Original", html)
        self.assertIn("Create new version...", html)
        self.assertIn("Merge Checked Phrases", html)
        self.assertIn("Add Note From Range", html)
        self.assertIn("raga-transcription-selection", html)
        self.assertIn("/api/transcription-edits/", html)
        self.assertIn("function loadDefaultSelection()", html)
        self.assertIn("function setDefaultSelectionTo(", html)
        self.assertIn("function regenerateSelectedVersion()", html)
        self.assertIn("function saveCurrentVersion(", html)
        self.assertIn("target_version_id", html)
        self.assertIn("create_new_version", html)
        self.assertIn("default_selection", html)
        self.assertIn('"sargam": "Sa"', html)
        self.assertIn('"pitch_midi": 60.0', html)
        self.assertIn('"raw_pitch_midi": 60.24', html)
        self.assertIn('"snapped_pitch_midi": 60.0', html)
        self.assertIn('"corrected_pitch_midi": 60.0', html)


if __name__ == "__main__":
    unittest.main()
