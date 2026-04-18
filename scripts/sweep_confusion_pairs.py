#!/usr/bin/env python3
"""Build confusion-pair diagnostics from a calibration progress CSV."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from raga_pipeline.config import find_default_raga_db_path  # noqa: E402
from raga_pipeline.raga import RagaDatabase, _normalize_raga_name  # noqa: E402


def split_ragas(cell: str) -> List[str]:
    return [part.strip() for part in str(cell or "").split(",") if part.strip()]


def raga_names_match(candidate_cell: str, gt_name: str) -> bool:
    gt_norm = gt_name.strip().lower().replace(" ", "")
    for alias in split_ragas(candidate_cell):
        if alias.strip().lower().replace(" ", "") == gt_norm:
            return True
    return False


def predicted_individual_raga(candidate_cell: str, gt_raga: str) -> str:
    aliases = split_ragas(candidate_cell)
    if not aliases:
        return ""
    if raga_names_match(candidate_cell, gt_raga):
        return gt_raga
    return aliases[0]


def load_scale_lookup(raga_db_path: str) -> Dict[str, Tuple[int, ...]]:
    raga_db = RagaDatabase(raga_db_path)
    lookup: Dict[str, Tuple[int, ...]] = {}
    for entry in raga_db.all_ragas:
        mask = tuple(int(x) for x in entry["mask"])
        for name in entry.get("names", []):
            lookup[_normalize_raga_name(name)] = mask
    return lookup


def same_scale(scale_lookup: Dict[str, Tuple[int, ...]], a: str, b: str) -> bool:
    return scale_lookup.get(_normalize_raga_name(a)) == scale_lookup.get(_normalize_raga_name(b))


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--progress-csv",
        default=str(REPO_ROOT / "results" / "saturation_calibration" / "progress.csv"),
    )
    parser.add_argument("--variant", default="clip_2_0")
    parser.add_argument("--raga-db", default=None)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "confusion_analysis"))
    parser.add_argument("--min-total", type=int, default=2)
    args = parser.parse_args()

    progress_path = Path(args.progress_csv)
    if not progress_path.exists():
        print(f"Missing progress CSV: {progress_path}", file=sys.stderr)
        return 2
    raga_db_path = args.raga_db or find_default_raga_db_path()
    if not raga_db_path:
        print("Could not locate raga DB", file=sys.stderr)
        return 2

    rows_all = list(csv.DictReader(progress_path.open("r", encoding="utf-8", newline="")))
    rows = [r for r in rows_all if r.get("variant") == args.variant and r.get("status") == "ok"]
    if not rows:
        print(f"No ok rows for variant {args.variant!r}", file=sys.stderr)
        return 2

    ragas = sorted({r["gt_raga"] for r in rows if r.get("gt_raga")})
    matrix_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    examples: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    correct = 0
    for row in rows:
        gt = row.get("gt_raga", "")
        pred = predicted_individual_raga(row.get("hist_top1_raga", ""), gt)
        if not gt or not pred:
            continue
        matrix_counts[(gt, pred)] += 1
        if gt == pred:
            correct += 1
        elif len(examples[(gt, pred)]) < 5:
            examples[(gt, pred)].append(row.get("filename", ""))
        if pred not in ragas:
            ragas.append(pred)
    ragas = sorted(ragas)

    matrix_rows: List[dict] = []
    for gt in ragas:
        out = {"gt_raga": gt}
        for pred in ragas:
            out[pred] = matrix_counts.get((gt, pred), 0)
        matrix_rows.append(out)

    pair_counts: Counter[Tuple[str, str]] = Counter()
    directional: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for i, a in enumerate(ragas):
        for b in ragas[i + 1 :]:
            a_to_b = matrix_counts.get((a, b), 0)
            b_to_a = matrix_counts.get((b, a), 0)
            total = a_to_b + b_to_a
            if total > 0:
                pair_counts[(a, b)] = total
                directional[(a, b)] = (a_to_b, b_to_a)

    scale_lookup = load_scale_lookup(raga_db_path)
    pair_rows: List[dict] = []
    for (a, b), total in pair_counts.most_common():
        if total < args.min_total:
            continue
        a_to_b, b_to_a = directional[(a, b)]
        sample_recordings = examples.get((a, b), []) + examples.get((b, a), [])
        pair_rows.append(
            {
                "raga_a": a,
                "raga_b": b,
                "same_scale": str(same_scale(scale_lookup, a, b)).lower(),
                "a_to_b": a_to_b,
                "b_to_a": b_to_a,
                "total": total,
                "sample_recordings": ";".join(sample_recordings[:5]),
                "distinguishing_feature": "",
            }
        )

    output_dir = Path(args.output_dir)
    matrix_path = output_dir / "confusion_matrix.csv"
    pairs_path = output_dir / "top_pairs.csv"
    summary_path = output_dir / "summary.csv"
    write_csv(matrix_path, ["gt_raga", *ragas], matrix_rows)
    write_csv(
        pairs_path,
        [
            "raga_a",
            "raga_b",
            "same_scale",
            "a_to_b",
            "b_to_a",
            "total",
            "sample_recordings",
            "distinguishing_feature",
        ],
        pair_rows,
    )

    n = len(rows)
    summary_rows = [
        {
            "variant": args.variant,
            "n": n,
            "correct": correct,
            "accuracy": round(correct / max(n, 1), 4),
            "errors": n - correct,
            "top_pair_coverage": round(sum(int(r["total"]) for r in pair_rows[:5]) / max(n - correct, 1), 4),
            "pair_rows": len(pair_rows),
        }
    ]
    write_csv(summary_path, list(summary_rows[0].keys()), summary_rows)

    print(f"Wrote {matrix_path}")
    print(f"Wrote {pairs_path}")
    print(f"Wrote {summary_path}")
    print(f"\nVariant: {args.variant}")
    print(f"Accuracy: {correct}/{n} = {correct / max(n, 1) * 100:.1f}%")
    print("Top confusion pairs:")
    for row in pair_rows[:10]:
        print(
            f"  {row['raga_a']} <-> {row['raga_b']}: {row['total']} "
            f"({row['a_to_b']}/{row['b_to_a']}), same_scale={row['same_scale']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
