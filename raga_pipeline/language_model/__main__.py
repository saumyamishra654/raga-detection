"""CLI entry point: python -m raga_pipeline.language_model train|score|evaluate"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from raga_pipeline.language_model import (
    evaluate_model,
    score_transcription,
    train_model,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-raga n-gram language models for raga detection.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train = subparsers.add_parser("train", help="Train n-gram models from a labeled corpus.")
    train.add_argument("--gt", required=True, help="Path to ground-truth CSV.")
    train.add_argument("--results-dir", required=True, help="Root directory with transcription CSVs.")
    train.add_argument("--output", default="raga_ngram_model.json", help="Output model JSON path.")
    train.add_argument("--order", type=int, default=5, help="Maximum n-gram order (default: 5).")
    train.add_argument("--smoothing", choices=["add-k", "kneser-ney"], default="add-k")
    train.add_argument("--smoothing-k", type=float, default=0.01)
    train.add_argument("--min-recordings", type=int, default=3)
    train.add_argument("--lambdas", default=None,
                       help="Interpolation weights, comma-separated highest-order-first.")
    train.add_argument("--quiet", action="store_true")

    # --- score ---
    score = subparsers.add_parser("score", help="Score a transcription against a trained model.")
    score.add_argument("--model", required=True)
    score.add_argument("--transcription", required=True)
    score.add_argument("--tonic", required=True)
    score.add_argument("--segments", action="store_true")
    score.add_argument("--segment-window", type=int, default=100)
    score.add_argument("--top-k", type=int, default=5)
    score.add_argument("--output", default=None)

    # --- evaluate ---
    evaluate = subparsers.add_parser("evaluate", help="Leave-one-out cross-validation.")
    evaluate.add_argument("--gt", required=True)
    evaluate.add_argument("--results-dir", required=True)
    evaluate.add_argument("--output", default="eval_results.csv")
    evaluate.add_argument("--order", type=int, default=5)
    evaluate.add_argument("--smoothing", choices=["add-k", "kneser-ney"], default="add-k")
    evaluate.add_argument("--smoothing-k", type=float, default=0.01)
    evaluate.add_argument("--min-recordings", type=int, default=3)
    evaluate.add_argument("--lambdas", default=None)
    evaluate.add_argument("--sweep-orders", default=None,
                          help="Comma-separated orders to sweep.")
    evaluate.add_argument("--quiet", action="store_true")

    return parser


def _parse_lambdas(raw: Optional[str], order: int) -> Optional[list[float]]:
    if raw is None:
        return None
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != order:
        raise ValueError(f"--lambdas must have {order} values, got {len(parts)}")
    return parts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        lambdas = _parse_lambdas(args.lambdas, args.order)
        train_model(
            ground_truth=args.gt, results_dir=args.results_dir,
            output=args.output, order=args.order, smoothing=args.smoothing,
            smoothing_k=args.smoothing_k, lambdas=lambdas,
            min_recordings=args.min_recordings, quiet=args.quiet)
        return 0

    if args.command == "score":
        result = score_transcription(
            model_path=args.model, transcription_path=args.transcription,
            tonic=args.tonic, segments=args.segments,
            segment_window=args.segment_window, top_k=args.top_k,
            output=args.output)
        if not args.output:
            print(json.dumps(result, indent=2))
        return 0

    if args.command == "evaluate":
        lambdas = _parse_lambdas(args.lambdas, args.order) if args.lambdas else None
        sweep = [int(x) for x in args.sweep_orders.split(",")] if args.sweep_orders else None
        evaluate_model(
            ground_truth=args.gt, results_dir=args.results_dir,
            output=args.output, order=args.order, smoothing=args.smoothing,
            smoothing_k=args.smoothing_k, lambdas=lambdas,
            min_recordings=args.min_recordings, sweep_orders=sweep,
            quiet=args.quiet)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
