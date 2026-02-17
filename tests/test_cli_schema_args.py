import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
