import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from raga_pipeline import batch


class _FakeStdout:
    def read(self, _n: int) -> str:
        return ""


class _FakePopen:
    def __init__(self, cmd_capture: list[list[str]], cmd: list[str], **_kwargs) -> None:
        cmd_capture.append(list(cmd))
        self.stdout = _FakeStdout()
        self.returncode = 0

    def poll(self) -> int:
        return 0

    def wait(self) -> int:
        return self.returncode


class BatchGroundTruthMetadataTests(unittest.TestCase):
    def test_load_ground_truth_accepts_alias_headers_and_normalizes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            track_flute = (tmp_path / "setA" / "song_flute.wav").resolve()
            track_vocal = (tmp_path / "setB" / "song_vocal.mp3").resolve()
            tasks = [str(track_flute), str(track_vocal)]

            gt_csv = tmp_path / "ground_truth_v6.csv"
            gt_csv.write_text(
                "\n".join(
                    [
                        "Filename,Raga,Tonic,Instrument,Gender",
                        "song_flute.mp3,Yaman,C,Flute,M",
                        "song_vocal.wav,Bhairavi,D,Vocal,F",
                    ]
                ),
                encoding="utf-8",
            )

            payload = batch.load_ground_truth(str(gt_csv), tasks=tasks)
            self.assertIn(str(track_flute), payload)
            self.assertIn(str(track_vocal), payload)

            flute_row = payload[str(track_flute)]
            self.assertEqual(flute_row.get("raga"), "Yaman")
            self.assertEqual(flute_row.get("tonic"), "C")
            self.assertEqual(flute_row.get("source_type"), "instrumental")
            self.assertEqual(flute_row.get("instrument_type"), "bansuri")
            self.assertNotIn("vocalist_gender", flute_row)

            vocal_row = payload[str(track_vocal)]
            self.assertEqual(vocal_row.get("source_type"), "vocal")
            self.assertEqual(vocal_row.get("vocalist_gender"), "female")
            self.assertNotIn("instrument_type", vocal_row)

    def test_load_ground_truth_skips_ambiguous_stem_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tasks = [
                str((tmp_path / "a" / "dup.mp3").resolve()),
                str((tmp_path / "b" / "dup.wav").resolve()),
            ]
            gt_csv = tmp_path / "ground_truth_v6.csv"
            gt_csv.write_text(
                "\n".join(
                    [
                        "Filename,Raga,Tonic,Instrument,Gender",
                        "dup.mp3,Yaman,C,Flute,M",
                    ]
                ),
                encoding="utf-8",
            )

            payload = batch.load_ground_truth(str(gt_csv), tasks=tasks)
            self.assertEqual(payload, {})

    def test_match_csv_filename_to_task_falls_back_to_basename_match(self) -> None:
        matched, reason = batch._match_csv_filename_to_task(  # pylint: disable=protected-access
            "exact_name.mp3",
            {},
            {"exact_name.mp3": ["/tmp/exact_name.mp3"]},
        )
        self.assertEqual(matched, "/tmp/exact_name.mp3")
        self.assertIsNone(reason)

    def test_process_directory_detect_mode_never_emits_raga_or_tonic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "audio"
            input_dir.mkdir(parents=True, exist_ok=True)
            song_a = input_dir / "song_a.mp3"
            song_b = input_dir / "song_b.mp3"
            song_a.write_bytes(b"a")
            song_b.write_bytes(b"b")

            gt_csv = tmp_path / "ground_truth_v6.csv"
            gt_csv.write_text(
                "\n".join(
                    [
                        "Filename,Raga,Tonic,Instrument,Gender",
                        "song_a.mp3,Yaman,C,Flute,M",
                        "song_b.mp3,Bhairavi,D,Vocal,F",
                    ]
                ),
                encoding="utf-8",
            )

            output_dir = tmp_path / "batch_results"
            captured_cmds: list[list[str]] = []

            def _fake_popen(cmd, **kwargs):
                return _FakePopen(captured_cmds, cmd, **kwargs)

            with patch("raga_pipeline.batch.subprocess.Popen", side_effect=_fake_popen):
                batch.process_directory(
                    input_dir=str(input_dir),
                    ground_truth_path=str(gt_csv),
                    output_dir=str(output_dir),
                    mode="detect",
                    silent=True,
                )

            self.assertEqual(len(captured_cmds), 2)
            for cmd in captured_cmds:
                self.assertEqual(cmd[1], "detect")
                self.assertNotIn("--raga", cmd)
                self.assertNotIn("--tonic", cmd)

    def test_process_directory_analyze_mode_skips_missing_or_incomplete_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "audio"
            input_dir.mkdir(parents=True, exist_ok=True)
            song_valid = input_dir / "song_valid.mp3"
            song_missing_tonic = input_dir / "song_missing_tonic.mp3"
            song_no_row = input_dir / "song_no_row.mp3"
            song_valid.write_bytes(b"a")
            song_missing_tonic.write_bytes(b"b")
            song_no_row.write_bytes(b"c")

            gt_csv = tmp_path / "ground_truth_v6.csv"
            gt_csv.write_text(
                "\n".join(
                    [
                        "Filename,Raga,Tonic,Instrument,Gender",
                        "song_valid.mp3,Yaman,C,Flute,M",
                        "song_missing_tonic.mp3,Bhairavi,,Vocal,F",
                    ]
                ),
                encoding="utf-8",
            )

            output_dir = tmp_path / "batch_results"
            captured_cmds: list[list[str]] = []

            def _fake_popen(cmd, **kwargs):
                return _FakePopen(captured_cmds, cmd, **kwargs)

            with patch("raga_pipeline.batch.subprocess.Popen", side_effect=_fake_popen):
                batch.process_directory(
                    input_dir=str(input_dir),
                    ground_truth_path=str(gt_csv),
                    output_dir=str(output_dir),
                    mode="analyze",
                    silent=True,
                )

            self.assertEqual(len(captured_cmds), 1)
            cmd = captured_cmds[0]
            self.assertEqual(cmd[1], "analyze")
            self.assertIn("--raga", cmd)
            self.assertIn("Yaman", cmd)
            self.assertIn("--tonic", cmd)
            self.assertIn("C", cmd)
            self.assertIn("--source-type", cmd)
            self.assertIn("instrumental", cmd)
            self.assertIn("--instrument-type", cmd)
            self.assertIn("bansuri", cmd)


if __name__ == "__main__":
    unittest.main()
