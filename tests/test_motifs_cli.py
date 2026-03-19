import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from raga_pipeline import motifs


def _write_notes_csv(path: Path, rows: list[tuple[float, float, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["start", "end", "pitch_class", "sargam"],
        )
        writer.writeheader()
        for start, end, pitch_class, sargam in rows:
            writer.writerow(
                {
                    "start": f"{start:.3f}",
                    "end": f"{end:.3f}",
                    "pitch_class": pitch_class,
                    "sargam": sargam,
                }
            )


class MotifsCliTests(unittest.TestCase):
    def test_mine_accepts_alias_headers_and_prefers_edited_in_auto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"
            rec_dir = results_dir / "song_a"

            _write_notes_csv(
                rec_dir / "transcribed_notes.csv",
                [
                    (0.0, 0.2, 0, "Sa"),
                    (0.2, 0.5, 2, "Re"),
                    (0.5, 0.8, 4, "Ga"),
                ],
            )
            _write_notes_csv(
                rec_dir / "transcription_edits" / "transcription_edited.csv",
                [
                    (0.0, 0.2, 11, "Ni"),
                    (0.2, 0.5, 2, "Re"),
                    (0.5, 0.8, 4, "Ga"),
                ],
            )

            gt = tmp_path / "ground_truth_v6.csv"
            gt.write_text(
                "\n".join(
                    [
                        "Filename,Raga,Tonic,Instrument,Gender",
                        "song_a.wav,Yaman,C,Flute,M",
                    ]
                ),
                encoding="utf-8",
            )

            out_index = tmp_path / "motifs.json"
            result = motifs.mine_motifs(
                ground_truth=str(gt),
                results_dir=str(results_dir),
                index_out=str(out_index),
                transcription_source="auto",
                min_len=3,
                max_len=3,
                min_recording_support=1,
                quiet=True,
            )

            self.assertTrue(out_index.exists())
            row = result["summary_rows"][0]
            self.assertEqual(row["status"], "processed")
            self.assertEqual(row["transcription_source"], "edited")
            self.assertIn("transcription_edited.csv", row["transcription_file"])
            self.assertEqual(row["gender"], "male")
            self.assertEqual(row["instrument"], "bansuri")

            motif_strs = [m["motif_str"] for m in result["index"]["ragas"]["Yaman"]["motifs"]]
            self.assertIn("ni:11 re:2 ga:4", motif_strs)

    def test_mine_recording_support_filters_single_recording_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"

            _write_notes_csv(
                results_dir / "rec_1" / "transcribed_notes.csv",
                [
                    (0.0, 0.1, 0, "Sa"),
                    (0.1, 0.2, 2, "Re"),
                    (0.2, 0.3, 4, "Ga"),
                    (0.3, 0.4, 5, "Ma"),
                ],
            )
            _write_notes_csv(
                results_dir / "rec_2" / "transcribed_notes.csv",
                [
                    (0.0, 0.1, 0, "Sa"),
                    (0.1, 0.2, 2, "Re"),
                    (0.2, 0.3, 4, "Ga"),
                    (0.3, 0.4, 7, "Pa"),
                ],
            )

            gt = tmp_path / "gt.csv"
            gt.write_text(
                "\n".join(
                    [
                        "Filename,Raga,Tonic",
                        "rec_1.mp3,Yaman,C",
                        "rec_2.mp3,Yaman,C",
                    ]
                ),
                encoding="utf-8",
            )

            result = motifs.mine_motifs(
                ground_truth=str(gt),
                results_dir=str(results_dir),
                index_out=str(tmp_path / "index.json"),
                min_len=3,
                max_len=3,
                min_recording_support=2,
                quiet=True,
            )

            motif_entries = result["index"]["ragas"]["Yaman"]["motifs"]
            self.assertEqual(len(motif_entries), 1)
            self.assertEqual(motif_entries[0]["motif_str"], "sa:0 re:2 ga:4")
            self.assertEqual(motif_entries[0]["recording_support"], 2)

    def test_mine_missing_transcription_rows_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"
            _write_notes_csv(
                results_dir / "exists" / "transcribed_notes.csv",
                [
                    (0.0, 0.2, 0, "Sa"),
                    (0.2, 0.4, 2, "Re"),
                    (0.4, 0.6, 4, "Ga"),
                ],
            )

            gt = tmp_path / "gt.csv"
            gt.write_text(
                "\n".join(
                    [
                        "Filename,Raga,Tonic",
                        "exists.mp3,Yaman,C",
                        "missing.mp3,Yaman,C",
                    ]
                ),
                encoding="utf-8",
            )

            result = motifs.mine_motifs(
                ground_truth=str(gt),
                results_dir=str(results_dir),
                index_out=str(tmp_path / "index.json"),
                min_len=3,
                max_len=3,
                min_recording_support=1,
                quiet=True,
            )

            missing_rows = [row for row in result["summary_rows"] if row["status"] == "missing_transcription"]
            self.assertEqual(len(missing_rows), 1)
            self.assertIn("missing.mp3", result["missing_files"])

    def test_score_ranks_expected_raga_and_emits_phrase_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"

            _write_notes_csv(
                results_dir / "yam_1" / "transcribed_notes.csv",
                [
                    (0.0, 0.2, 0, "Sa"),
                    (0.2, 0.4, 2, "Re"),
                    (0.4, 0.6, 4, "Ga"),
                ],
            )
            _write_notes_csv(
                results_dir / "bha_1" / "transcribed_notes.csv",
                [
                    (0.0, 0.2, 0, "Sa"),
                    (0.2, 0.4, 1, "re"),
                    (0.4, 0.6, 3, "ga"),
                ],
            )

            gt = tmp_path / "gt.csv"
            gt.write_text(
                "\n".join(
                    [
                        "Filename,Raga,Tonic",
                        "yam_1.mp3,Yaman,C",
                        "bha_1.mp3,Bhairavi,C",
                    ]
                ),
                encoding="utf-8",
            )

            index_path = tmp_path / "motifs.json"
            motifs.mine_motifs(
                ground_truth=str(gt),
                results_dir=str(results_dir),
                index_out=str(index_path),
                min_len=3,
                max_len=3,
                min_recording_support=1,
                quiet=True,
            )

            query = tmp_path / "query.csv"
            _write_notes_csv(
                query,
                [
                    (0.0, 0.2, 0, "Sa"),
                    (0.2, 0.4, 2, "Re"),
                    (0.4, 0.6, 4, "Ga"),
                    (2.2, 2.4, 0, "Sa"),
                    (2.4, 2.6, 2, "Re"),
                    (2.6, 2.8, 4, "Ga"),
                ],
            )

            scored = motifs.score_transcription(
                index_path=str(index_path),
                transcription_path=str(query),
                top_k=2,
            )

            self.assertTrue(scored["ranked_ragas"])
            self.assertEqual(scored["ranked_ragas"][0]["raga"], "Yaman")
            self.assertGreater(scored["ranked_ragas"][0]["score"], 0.0)
            self.assertTrue(scored["phrase_overlays"])
            self.assertIn("Yaman", {item["raga"] for item in scored["phrase_overlays"]})

    def test_cli_main_mine_and_score_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"
            _write_notes_csv(
                results_dir / "rec_a" / "transcribed_notes.csv",
                [
                    (0.0, 0.2, 0, "Sa"),
                    (0.2, 0.4, 2, "Re"),
                    (0.4, 0.6, 4, "Ga"),
                ],
            )

            gt = tmp_path / "gt.csv"
            gt.write_text("Filename,Raga,Tonic\nrec_a.mp3,Yaman,C\n", encoding="utf-8")

            index_path = tmp_path / "index.json"
            summary_path = tmp_path / "summary.csv"
            exit_code = motifs.main(
                [
                    "mine",
                    "--ground-truth",
                    str(gt),
                    "--results-dir",
                    str(results_dir),
                    "--index-out",
                    str(index_path),
                    "--summary-out",
                    str(summary_path),
                    "--min-recording-support",
                    "1",
                    "--quiet",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(index_path.exists())
            self.assertTrue(summary_path.exists())

            query = tmp_path / "query.csv"
            _write_notes_csv(
                query,
                [
                    (0.0, 0.2, 0, "Sa"),
                    (0.2, 0.4, 2, "Re"),
                    (0.4, 0.6, 4, "Ga"),
                ],
            )

            with io.StringIO() as buffer, redirect_stdout(buffer):
                score_exit = motifs.main(
                    [
                        "score",
                        "--index",
                        str(index_path),
                        "--transcription",
                        str(query),
                        "--top-k",
                        "1",
                    ]
                )
                output = buffer.getvalue()

            self.assertEqual(score_exit, 0)
            self.assertIn('"ranked_ragas"', output)


if __name__ == "__main__":
    unittest.main()
