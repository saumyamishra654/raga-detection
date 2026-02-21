import tempfile
import unittest
import os
from pathlib import Path

try:
    from raga_pipeline.cli_args import params_to_argv
    from raga_pipeline.cli_schema import get_mode_schema
    from raga_pipeline.config import build_cli_parser, parse_config_from_argv
    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


@unittest.skipUnless(IMPORT_OK, "raga_pipeline imports unavailable in current environment")
class ConfigSchemaTests(unittest.TestCase):
    def test_build_cli_parser_has_subcommands(self) -> None:
        parser = build_cli_parser()
        subparser_actions = [a for a in parser._actions if a.dest == "command"]
        self.assertTrue(subparser_actions)
        choices = subparser_actions[0].choices
        self.assertIn("preprocess", choices)
        self.assertIn("detect", choices)
        self.assertIn("analyze", choices)

    def test_parse_config_from_argv_detect(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "demo.wav"
            audio_path.write_bytes(b"RIFF")
            config = parse_config_from_argv(
                [
                    "detect",
                    "--audio",
                    str(audio_path),
                    "--output",
                    tmpdir,
                ]
            )
            self.assertEqual(config.mode, "detect")
            self.assertEqual(config.output_dir, str(Path(tmpdir).absolute()))
            self.assertEqual(config.audio_path, str(audio_path.absolute()))

    def test_schema_includes_actions_and_grouping(self) -> None:
        schema = get_mode_schema("analyze")
        self.assertEqual(schema["mode"], "analyze")
        by_name = {field["name"]: field for field in schema["fields"]}
        self.assertIn("audio", by_name)
        self.assertIn("tonic", by_name)
        self.assertEqual(by_name["no_smoothing"]["action"], "store_true")
        self.assertEqual(by_name["bias_rotation"]["action"], "store_false")
        self.assertIn(by_name["audio"]["group"], {"common", "advanced"})

    def test_preprocess_schema_includes_recording_fields(self) -> None:
        schema = get_mode_schema("preprocess")
        by_name = {field["name"]: field for field in schema["fields"]}
        self.assertIn("ingest", by_name)
        self.assertIn("record_mode", by_name)
        self.assertIn("tanpura_key", by_name)
        self.assertIn("recorded_audio", by_name)
        self.assertEqual(by_name["ingest"]["default"], "youtube")
        self.assertEqual(by_name["record_mode"]["default"], "song")

    def test_params_to_argv_roundtrip_shape(self) -> None:
        argv = params_to_argv(
            "analyze",
            params={
                "audio": "/tmp/song.wav",
                "output": "batch_results",
                "tonic": "C",
                "raga": "Bhairavi",
                "no_smoothing": True,
                "bias_rotation": False,
                "phrase_min_notes": 3,
            },
            extra_args=["--force"],
        )
        self.assertEqual(argv[0], "analyze")
        self.assertIn("--audio", argv)
        self.assertIn("--tonic", argv)
        self.assertIn("--raga", argv)
        self.assertIn("--no-smoothing", argv)
        self.assertIn("--bias-rotation", argv)
        self.assertIn("--force", argv)

    def test_params_to_argv_preprocess_record_mode(self) -> None:
        argv = params_to_argv(
            "preprocess",
            params={
                "ingest": "record",
                "record_mode": "tanpura_vocal",
                "tanpura_key": "A",
                "audio_dir": "/tmp/audio",
                "filename": "take_1",
                "recorded_audio": "/tmp/source.webm",
                "output": "batch_results",
            },
        )
        self.assertEqual(argv[0], "preprocess")
        self.assertIn("--ingest", argv)
        self.assertIn("record", argv)
        self.assertIn("--record-mode", argv)
        self.assertIn("tanpura_vocal", argv)
        self.assertIn("--tanpura-key", argv)
        self.assertIn("A", argv)
        self.assertIn("--recorded-audio", argv)

    def test_preprocess_youtube_requires_yt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "requires --yt"):
                parse_config_from_argv(
                    [
                        "preprocess",
                        "--audio-dir",
                        tmpdir,
                        "--filename",
                        "demo",
                    ]
                )

    def test_preprocess_record_without_yt_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorded_path = Path(tmpdir) / "recorded.wav"
            recorded_path.write_bytes(b"RIFF")
            config = parse_config_from_argv(
                [
                    "preprocess",
                    "--ingest",
                    "record",
                    "--record-mode",
                    "song",
                    "--recorded-audio",
                    str(recorded_path),
                    "--audio-dir",
                    tmpdir,
                    "--filename",
                    "demo",
                ]
            )
            self.assertEqual(config.mode, "preprocess")
            self.assertEqual(config.preprocess_ingest, "record")
            self.assertEqual(config.preprocess_record_mode, "song")
            self.assertEqual(config.preprocess_recorded_audio, os.path.abspath(str(recorded_path)))

    def test_tanpura_vocal_requires_tanpura_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "requires --tanpura-key"):
                parse_config_from_argv(
                    [
                        "preprocess",
                        "--ingest",
                        "record",
                        "--record-mode",
                        "tanpura_vocal",
                        "--audio-dir",
                        tmpdir,
                        "--filename",
                        "demo",
                    ]
                )

    def test_missing_recorded_audio_path_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "missing.wav"
            with self.assertRaises(FileNotFoundError):
                parse_config_from_argv(
                    [
                        "preprocess",
                        "--ingest",
                        "record",
                        "--recorded-audio",
                        str(missing_path),
                        "--audio-dir",
                        tmpdir,
                        "--filename",
                        "demo",
                    ]
                )

    def test_invalid_tanpura_key_rejected_by_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(SystemExit):
                parse_config_from_argv(
                    [
                        "preprocess",
                        "--ingest",
                        "record",
                        "--record-mode",
                        "tanpura_vocal",
                        "--tanpura-key",
                        "C#",
                        "--audio-dir",
                        tmpdir,
                        "--filename",
                        "demo",
                    ]
                )

    def test_detect_skip_separation_requires_tonic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "demo.wav"
            audio_path.write_bytes(b"RIFF")
            with self.assertRaisesRegex(ValueError, "requires --tonic"):
                parse_config_from_argv(
                    [
                        "detect",
                        "--audio",
                        str(audio_path),
                        "--skip-separation",
                    ]
                )

    def test_detect_skip_separation_with_tonic_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "demo.wav"
            audio_path.write_bytes(b"RIFF")
            config = parse_config_from_argv(
                [
                    "detect",
                    "--audio",
                    str(audio_path),
                    "--skip-separation",
                    "--tonic",
                    "C",
                ]
            )
            self.assertTrue(config.skip_separation)
            self.assertEqual(config.tonic_override, "C")
            self.assertEqual(config.melody_source, "composite")

    def test_detect_skip_separation_forces_composite_melody_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "demo.wav"
            audio_path.write_bytes(b"RIFF")
            config = parse_config_from_argv(
                [
                    "detect",
                    "--audio",
                    str(audio_path),
                    "--skip-separation",
                    "--tonic",
                    "A",
                    "--melody-source",
                    "separated",
                ]
            )
            self.assertEqual(config.melody_source, "composite")
            self.assertTrue(config.skip_separation_forced_composite)


if __name__ == "__main__":
    unittest.main()
