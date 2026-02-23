import unittest

import numpy as np

try:
    from raga_pipeline.audio import PitchData
    from raga_pipeline.output import create_scrollable_pitch_plot_html
    from raga_pipeline.sequence import Note

    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


def _midi_to_hz(midi_vals: np.ndarray) -> np.ndarray:
    return 440.0 * (2.0 ** ((midi_vals - 69.0) / 12.0))


def _dummy_pitch_data() -> "PitchData":
    timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    midi_vals = np.array([60.0, 60.2, 60.1, 60.3, 60.0], dtype=float)
    pitch_hz = _midi_to_hz(midi_vals)
    confidence = np.full_like(timestamps, 0.95, dtype=float)
    voicing = np.ones_like(timestamps, dtype=bool)
    energy = np.array([0.05, 0.08, 0.12, 0.09, 0.03], dtype=float)
    return PitchData(
        timestamps=timestamps,
        pitch_hz=pitch_hz,
        confidence=confidence,
        voicing=voicing,
        valid_freqs=pitch_hz.copy(),
        midi_vals=midi_vals,
        energy=energy,
        frame_period=0.1,
        audio_path="dummy.wav",
    )


@unittest.skipUnless(IMPORT_OK, "output module unavailable in current environment")
class ScrollInspectorHtmlTests(unittest.TestCase):
    def test_scroll_plot_html_contains_inspector_elements_and_hooks(self) -> None:
        pitch_data = _dummy_pitch_data()
        notes = [
            Note(
                start=0.10,
                end=0.25,
                pitch_midi=60.0,
                pitch_hz=261.63,
                confidence=1.0,
                energy=0.09,
                sargam="Sa",
                pitch_class=0,
            )
        ]

        html = create_scrollable_pitch_plot_html(
            pitch_data,
            tonic=60,
            raga_name="Bhairavi",
            audio_element_ids=["audio-track-1"],
            overlay_energy=pitch_data.energy,
            overlay_timestamps=pitch_data.timestamps,
            overlay_label="Vocals RMS",
            transcription_notes=notes,
        )

        self.assertRegex(html, r'id="sp_[0-9a-f]{6}-point-marker"')
        self.assertRegex(html, r'id="sp_[0-9a-f]{6}-range-band"')
        self.assertRegex(html, r'id="sp_[0-9a-f]{6}-hover-tooltip"')
        self.assertRegex(html, r'id="sp_[0-9a-f]{6}-inspector"')
        self.assertRegex(html, r'id="sp_[0-9a-f]{6}-clear-selection"')

        self.assertIn("const transcriptionNotesRaw =", html)
        self.assertIn('"start": 0.1', html)
        self.assertIn('"end": 0.25', html)
        self.assertIn('"sargam": "Sa"', html)
        self.assertIn('"pitch_midi": 60.0', html)
        self.assertIn('"energy": 0.09', html)
        self.assertIn('draggable="false"', html)
        self.assertIn("function normalizeSargamLabel", html)
        self.assertIn("function noteBreakdownTableHtml", html)
        self.assertIn("function centsDistance", html)
        self.assertIn("function renderHoverTooltip", html)
        self.assertIn("max-height:\" + maxHeight + \"px; overflow-y:auto", html)
        self.assertIn(">Note</th>", html)
        self.assertIn(">Duration</th>", html)
        self.assertIn(">MIDI</th>", html)
        self.assertIn(">Dist to Corrected</th>", html)
        self.assertIn("addEventListener(\"dragstart\"", html)
        self.assertIn("addEventListener(\"mousedown\"", html)
        self.assertIn("addEventListener(\"mousemove\"", html)
        self.assertIn("addEventListener(\"mouseleave\"", html)
        self.assertIn("addEventListener(\"mouseup\"", html)
        self.assertIn("raga-scroll-selection-clear", html)

    def test_scroll_plot_html_serializes_empty_note_payload_when_missing(self) -> None:
        pitch_data = _dummy_pitch_data()
        html = create_scrollable_pitch_plot_html(
            pitch_data,
            tonic=60,
            raga_name="Bhairavi",
            audio_element_ids=["audio-track-1"],
            overlay_energy=pitch_data.energy,
            overlay_timestamps=pitch_data.timestamps,
            overlay_label="Vocals RMS",
            transcription_notes=None,
        )
        self.assertIn("const transcriptionNotesRaw = [];", html)


if __name__ == "__main__":
    unittest.main()
