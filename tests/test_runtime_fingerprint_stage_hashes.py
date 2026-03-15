import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import raga_pipeline.runtime_fingerprint as runtime_fp


class StageHashingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.tmp.name)
        self._write(
            "detect_only.py",
            "def detect_fn():\n"
            "    return 1\n",
        )
        self._write(
            "analyze_only.py",
            "def analyze_fn():\n"
            "    return 1\n",
        )
        self._write(
            "shared.py",
            "def shared_fn():\n"
            "    return 1\n",
        )
        self.stage_selectors = {
            "detect": [
                {"path": "detect_only.py", "type": "function", "name": "detect_fn"},
                {"path": "shared.py", "type": "function", "name": "shared_fn"},
            ],
            "analyze": [
                {"path": "analyze_only.py", "type": "function", "name": "analyze_fn"},
                {"path": "shared.py", "type": "function", "name": "shared_fn"},
            ],
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, rel_path: str, content: str) -> None:
        path = self.repo_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _build_hashes(self) -> dict[str, str]:
        with patch.object(runtime_fp, "STAGE_SELECTORS", self.stage_selectors):
            return runtime_fp.build_stage_hashes(self.repo_root)

    def test_stage_hashes_contain_detect_and_analyze(self) -> None:
        hashes = self._build_hashes()
        self.assertIn("detect", hashes)
        self.assertIn("analyze", hashes)
        self.assertTrue(hashes["detect"])
        self.assertTrue(hashes["analyze"])

    def test_detect_owned_change_updates_detect_hash_only(self) -> None:
        before = self._build_hashes()
        self._write(
            "detect_only.py",
            "def detect_fn():\n"
            "    return 2\n",
        )
        after = self._build_hashes()
        self.assertNotEqual(before["detect"], after["detect"])
        self.assertEqual(before["analyze"], after["analyze"])

    def test_analyze_owned_change_updates_analyze_hash_only(self) -> None:
        before = self._build_hashes()
        self._write(
            "analyze_only.py",
            "def analyze_fn():\n"
            "    return 2\n",
        )
        after = self._build_hashes()
        self.assertEqual(before["detect"], after["detect"])
        self.assertNotEqual(before["analyze"], after["analyze"])

    def test_shared_change_updates_both_hashes(self) -> None:
        before = self._build_hashes()
        self._write(
            "shared.py",
            "def shared_fn():\n"
            "    return 2\n",
        )
        after = self._build_hashes()
        self.assertNotEqual(before["detect"], after["detect"])
        self.assertNotEqual(before["analyze"], after["analyze"])


if __name__ == "__main__":
    unittest.main()
