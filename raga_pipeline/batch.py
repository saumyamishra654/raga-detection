import csv
import argparse
import json
import os
import sys
import subprocess
from datetime import datetime
from typing import Dict, Optional, Any


AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac"}


def _is_valid_audio_file(filename: str, valid_exts: set[str]) -> bool:
    """Filter out hidden/AppleDouble files and non-audio extensions."""
    if not filename or filename.startswith("."):
        return False
    return os.path.splitext(filename)[1].lower() in valid_exts


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _driver_path() -> str:
    return os.path.join(_project_root(), "driver.py")


def _default_progress_file(output_dir: str, input_dir: str) -> str:
    input_name = os.path.basename(os.path.normpath(input_dir)) or "batch"
    return os.path.join(output_dir, "logs", f"{input_name}_batch_progress.json")


def _load_progress(progress_file: str) -> Dict[str, Any]:
    if not os.path.exists(progress_file):
        return {"processed": [], "failed": [], "attempts": []}

    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed reading progress file {progress_file}: {e}")
        return {"processed": [], "failed": [], "attempts": []}

    processed = data.get("processed", [])
    failed = data.get("failed", [])
    attempts = data.get("attempts", [])
    if not isinstance(processed, list):
        processed = []
    if not isinstance(failed, list):
        failed = []
    if not isinstance(attempts, list):
        attempts = []
    return {"processed": processed, "failed": failed, "attempts": attempts}


def _save_progress(progress_file: str, progress: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(progress_file) or ".", exist_ok=True)
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def _append_metadata_args(cmd: list[str], gt_info: dict) -> None:
    """Append optional CSV metadata fields shared by detect/analyze modes."""
    metadata_fields = [
        ("instrument_type", "--instrument-type", "Instrument"),
        ("vocalist_gender", "--vocalist-gender", "Gender"),
        ("source_type", "--source-type", "Source"),
        ("melody_source", "--melody-source", "Melody Source"),
    ]
    for key, flag, label in metadata_fields:
        value = gt_info.get(key)
        if value:
            cmd.extend([flag, value])
            print(f"     + {label}: {value}")

def load_ground_truth(csv_path: str) -> Dict[str, dict]:
    """
    load ground truth data from annotated song db
    expected columns: filename, raga, tonic
    optional columns: instrument_type, vocalist_gender, source_type, melody_source
    returns dict: {filename: {data}}
    """
    ground_truth: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Ground truth file not found at {csv_path}")
        return ground_truth

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Map robust headers to actual column names
            fieldnames = reader.fieldnames or []
            if not fieldnames:
                print("Error: CSV has no headers.")
                return ground_truth

            name_col = next((h for h in fieldnames if 'file' in h.lower() or 'name' in h.lower()), None)
            
            if not name_col:
                print(f"Error: CSV must contain a filename column. Found: {reader.fieldnames}")
                return ground_truth
            
            # Helper to find column by partial name
            def find_col(partial_name):
                return next((h for h in fieldnames if partial_name in h.lower()), None)

            raga_col = find_col('raga')
            tonic_col = find_col('tonic')
            inst_col = find_col('instrument_type')
            gender_col = find_col('vocalist_gender')
            source_col = find_col('source_type')
            melody_col = find_col('melody_source')
            
            for row in reader:
                # clean filename (basename only)
                fname = os.path.basename(row[name_col]).strip()
                
                entry = {}
                if raga_col and row[raga_col]: entry['raga'] = row[raga_col].strip()
                if tonic_col and row[tonic_col]: entry['tonic'] = row[tonic_col].strip()
                if inst_col and row[inst_col]: entry['instrument_type'] = row[inst_col].strip()
                if gender_col and row[gender_col]: entry['vocalist_gender'] = row[gender_col].strip()
                if source_col and row[source_col]: entry['source_type'] = row[source_col].strip()
                if melody_col and row[melody_col]: entry['melody_source'] = row[melody_col].strip()
                
                ground_truth[fname] = entry
                
    except Exception as e:
        print(f"Error reading ground truth CSV: {e}")
        
    return ground_truth


def init_ground_truth_csv(input_dir: str, output_csv: str) -> None:
    """
    Initialize a ground truth CSV with files from input_dir.
    Sets defaults: source_type='vocal', melody_source='separated'.
    """
    files_to_process = []
    
    for root, _dirs, files in os.walk(input_dir):
        for file in files:
            if _is_valid_audio_file(file, AUDIO_EXTS):
                files_to_process.append(file)
                
    files_to_process.sort()
    
    if not files_to_process:
        print(f"No audio files found in {input_dir}")
        return

    print(f"Initializing CSV at {output_csv} with {len(files_to_process)} files.")
    
    fieldnames = ['filename', 'raga', 'tonic', 'source_type', 'vocalist_gender', 'instrument_type', 'melody_source']
    
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for fname in files_to_process:
                writer.writerow({
                    'filename': fname,
                    'raga': '',
                    'tonic': '',
                    'source_type': 'vocal',
                    'vocalist_gender': '',
                    'instrument_type': '',
                    'melody_source': 'separated'
                })
        print(f"Successfully created {output_csv}")
    except Exception as e:
        print(f"Error creating CSV: {e}")


def _build_pipeline_cmd(
    audio_path: str,
    output_dir: str,
    requested_mode: str,
    gt_info: dict,
) -> tuple[list[str], str]:
    """Build a driver invocation command and return (command, effective_mode)."""
    command = [
        sys.executable,
        _driver_path(),
        "detect",
        "--audio",
        audio_path,
        "--output",
        output_dir,
    ]
    effective_mode = "detect"

    if requested_mode == "auto" and gt_info:
        raga = gt_info.get("raga")
        tonic = gt_info.get("tonic")
        if raga and tonic:
            print(f"  -> Found Ground Truth: Raga='{raga}', Tonic='{tonic}'")
            print("  -> Running in ANALYZE mode")
            command = [
                sys.executable,
                _driver_path(),
                "analyze",
                "--audio",
                audio_path,
                "--output",
                output_dir,
                "--raga",
                raga,
                "--tonic",
                tonic,
            ]
            effective_mode = "analyze"
        else:
            print("  -> Ground truth entry found but missing Raga/Tonic. Falling back to DETECT.")
    elif requested_mode == "detect":
        print("  -> Forced DETECT mode")
    else:
        print("  -> No Ground Truth found")
        print("  -> Running in DETECT mode")

    if gt_info:
        _append_metadata_args(command, gt_info)

    return command, effective_mode


def _run_with_live_log(cmd: list[str], log_file: str, silent: bool) -> int:
    with open(log_file, "w", encoding="utf-8") as f_log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ,
        )
        if process.stdout is None:
            raise RuntimeError("Failed to capture subprocess stdout.")

        # Read char-by-char so '\r' progress updates stay visible in logs.
        while True:
            char = process.stdout.read(1)
            if not char and process.poll() is not None:
                break
            if char:
                f_log.write(char)
                if not silent:
                    sys.stdout.write(char)
                    sys.stdout.flush()

        process.wait()
        return int(process.returncode or 0)


def process_directory(
    input_dir: str,
    ground_truth_path: Optional[str] = None,
    output_dir: str = "results",
    mode: str = "auto",
    max_files: int = 0,
    progress_file: Optional[str] = None,
    silent: bool = False,
) -> Dict[str, int]:
    """
    Walk an input directory and run the pipeline on each audio file.

    Returns a summary dictionary with counts for processed/failed/remaining.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    if progress_file is None:
        progress_file = _default_progress_file(output_dir, input_dir)
    else:
        progress_file = os.path.abspath(progress_file)

    gt_data: Dict[str, dict] = {}
    if ground_truth_path:
        gt_data = load_ground_truth(ground_truth_path)
        print(f"Loaded {len(gt_data)} ground truth entries.")
    elif mode == "detect":
        print("Mode set to 'detect' with no ground truth CSV.")

    tasks: list[str] = []
    for root, _dirs, files in os.walk(input_dir):
        for file in files:
            if _is_valid_audio_file(file, AUDIO_EXTS):
                tasks.append(os.path.abspath(os.path.join(root, file)))
    tasks.sort()

    progress = _load_progress(progress_file)
    processed_set = set(str(path) for path in progress.get("processed", []))
    pending_tasks = [task for task in tasks if task not in processed_set]

    print(f"Found {len(tasks)} audio files total.")
    print(f"Already processed (from checkpoint): {len(processed_set & set(tasks))}")

    if max_files > 0:
        tasks_to_run = pending_tasks[:max_files]
        print(f"Processing up to {len(tasks_to_run)} files this run (--max-files={max_files}).")
    else:
        tasks_to_run = pending_tasks
        print(f"Processing all remaining files this run ({len(tasks_to_run)}).")

    run_successes = 0
    run_failures = 0

    for i, audio_path in enumerate(tasks_to_run):
        fname = os.path.basename(audio_path)
        print(f"\n[{i+1}/{len(tasks_to_run)}] Processing: {fname}")

        gt_info = gt_data.get(fname, {})
        cmd, effective_mode = _build_pipeline_cmd(
            audio_path=audio_path,
            output_dir=output_dir,
            requested_mode=mode,
            gt_info=gt_info,
        )

        try:
            log_file = os.path.join(output_dir, "logs", f"{fname}.log")
            returncode = _run_with_live_log(cmd, log_file, silent=silent)

            if returncode == 0:
                print(f"  [SUCCESS] Log: {log_file}")
                run_successes += 1
                if audio_path not in processed_set:
                    processed_set.add(audio_path)
                    progress.setdefault("processed", []).append(audio_path)
            else:
                print(f"  [FAILURE] Exit Code: {returncode}. Check Log: {log_file}")
                run_failures += 1
                progress.setdefault("failed", []).append(
                    {
                        "audio_path": audio_path,
                        "exit_code": returncode,
                        "mode": effective_mode,
                        "timestamp": _now_iso(),
                    }
                )

            progress.setdefault("attempts", []).append(
                {
                    "audio_path": audio_path,
                    "mode": effective_mode,
                    "status": "success" if returncode == 0 else "failure",
                    "exit_code": returncode,
                    "command": cmd,
                    "log_file": log_file,
                    "timestamp": _now_iso(),
                }
            )
            _save_progress(progress_file, progress)
        except Exception as e:
            print(f"  [ERROR] Execution failed: {e}")
            run_failures += 1
            progress.setdefault("failed", []).append(
                {
                    "audio_path": audio_path,
                    "exit_code": -1,
                    "mode": effective_mode,
                    "error": str(e),
                    "timestamp": _now_iso(),
                }
            )
            _save_progress(progress_file, progress)

    remaining = len([task for task in tasks if task not in processed_set])
    print("\nBatch summary:")
    print(f"  Processed this run: {run_successes}")
    print(f"  Failed this run: {run_failures}")
    print(f"  Remaining: {remaining}")
    print(f"  Progress file: {progress_file}")

    return {
        "processed_in_run": run_successes,
        "failed_in_run": run_failures,
        "remaining": remaining,
        "total": len(tasks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch process audio files for Raga Detection.")
    parser.add_argument("input_dir", help="Directory containing audio files")
    
    # Default paths relative to project root (assuming script is run from project root or inside module)
    # We resolve them relative to this file's location to be robust
    
    # Current file: raga_detection/raga_pipeline/batch.py
    # Ground truth: <input_dir>_gt.csv by default (stored alongside input_dir)
    # Output:       raga_detection/batch_results

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # raga-detection/
    default_gt = "__AUTO__"
    default_output = os.path.join(project_root, "batch_results")

    parser.add_argument("--ground-truth", "-g", default=default_gt, 
                        help=f"Path to ground truth CSV (default: {default_gt})")
    parser.add_argument("--output", "-o", default=default_output, 
                        help=f"Output directory (default: {default_output})")
    parser.add_argument("--mode", "-m", choices=['auto', 'detect'], default='auto',
                        help="Processing mode: 'auto' (checks ground truth) or 'detect' (force detection)")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Maximum files to process in this run (0 = all remaining)")
    parser.add_argument("--progress-file", default=None,
                        help="Path to checkpoint JSON. Default: <output>/logs/<input_dir_name>_batch_progress.json")
    parser.add_argument("--exit-99-on-remaining", action="store_true",
                        help="Exit with code 99 when files remain (for PBS resubmission loops)")
    parser.add_argument("--silent", "-s", action="store_true", help="Suppress output to console (log files are still saved)")
    parser.add_argument("--init-csv", action="store_true", help="Initialize a blank ground truth CSV for files in input_dir")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
        
    if args.ground_truth == "__AUTO__":
        input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
        args.ground_truth = os.path.join(args.input_dir, f"{input_dir_name}_gt.csv")

    if args.init_csv:
        init_ground_truth_csv(args.input_dir, args.ground_truth)
        return 0

    summary = process_directory(
        input_dir=args.input_dir,
        ground_truth_path=args.ground_truth,
        output_dir=args.output,
        mode=args.mode,
        max_files=max(0, int(args.max_files)),
        progress_file=args.progress_file,
        silent=args.silent,
    )

    if summary["failed_in_run"] > 0:
        return 1
    if args.exit_99_on_remaining and summary["remaining"] > 0:
        return 99
    return 0


if __name__ == "__main__":
    sys.exit(main())
