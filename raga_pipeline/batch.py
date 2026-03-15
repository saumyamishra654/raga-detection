import csv
import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Optional


AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac"}
VALID_SOURCE_TYPES = {"mixed", "vocal", "instrumental"}
VALID_MELODY_SOURCES = {"separated", "composite"}
VALID_INSTRUMENT_TYPES = {"autodetect", "sitar", "sarod", "bansuri", "slide_guitar"}

GENDER_ALIASES = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
}

INSTRUMENT_ALIASES = {
    "sitar": "sitar",
    "sarod": "sarod",
    "flute": "bansuri",
    "bansuri": "bansuri",
    "mohan veena": "slide_guitar",
    "slide guitar": "slide_guitar",
    "slide_guitar": "slide_guitar",
}


def _is_valid_audio_file(filename: str, valid_exts: set) -> bool:
    """Filter out hidden/AppleDouble files and non-audio extensions."""
    if not filename or filename.startswith("."):
        return False
    return os.path.splitext(filename)[1].lower() in valid_exts


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


def _normalize_header(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _find_column(fieldnames: list[str], aliases: list[str]) -> Optional[str]:
    normalized: Dict[str, str] = {}
    for name in fieldnames:
        key = _normalize_header(name)
        if key and key not in normalized:
            normalized[key] = name
    for alias in aliases:
        match = normalized.get(_normalize_header(alias))
        if match is not None:
            return match
    return None


def _clean_text(value: Optional[str]) -> str:
    return str(value or "").strip()


def _normalize_gender(value: str) -> Optional[str]:
    if not value:
        return None
    return GENDER_ALIASES.get(value.strip().lower())


def _normalize_source_type(value: str) -> Optional[str]:
    if not value:
        return None
    lowered = value.strip().lower()
    if lowered in {"voice", "vocal", "vocals"}:
        return "vocal"
    if lowered in {"instrument", "instrumental", "instruments"}:
        return "instrumental"
    if lowered in VALID_SOURCE_TYPES:
        return lowered
    return None


def _normalize_melody_source(value: str) -> Optional[str]:
    if not value:
        return None
    lowered = value.strip().lower()
    if lowered in VALID_MELODY_SOURCES:
        return lowered
    return None


def _normalize_instrument_type(value: str) -> Optional[str]:
    if not value:
        return None
    lowered = value.strip().lower()
    if lowered in VALID_INSTRUMENT_TYPES:
        return lowered
    if lowered in {"vocal", "voice", "vocals"}:
        return None
    if lowered in INSTRUMENT_ALIASES:
        return INSTRUMENT_ALIASES[lowered]
    return "autodetect"


def _derive_source_type(source_type_value: str, instrument_value: str) -> Optional[str]:
    explicit = _normalize_source_type(source_type_value)
    if explicit is not None:
        return explicit

    instrument_lower = instrument_value.strip().lower()
    if instrument_lower in {"vocal", "voice", "vocals"}:
        return "vocal"
    if instrument_lower:
        return "instrumental"
    return None


def _build_task_lookup(tasks: list[str]) -> tuple[Dict[str, list[str]], Dict[str, list[str]]]:
    stem_map: Dict[str, list[str]] = {}
    basename_map: Dict[str, list[str]] = {}
    for task in tasks:
        abs_task = os.path.abspath(task)
        base = os.path.basename(abs_task)
        stem_key = Path(base).stem.lower()
        base_key = base.lower()
        stem_map.setdefault(stem_key, []).append(abs_task)
        basename_map.setdefault(base_key, []).append(abs_task)
    return stem_map, basename_map


def _match_csv_filename_to_task(
    csv_filename: str,
    stem_map: Dict[str, list[str]],
    basename_map: Dict[str, list[str]],
) -> tuple[Optional[str], Optional[str]]:
    token = _clean_text(csv_filename)
    if not token:
        return None, "empty filename token"

    base = os.path.basename(token)
    stem = Path(base).stem if Path(base).suffix else base
    stem_key = stem.lower()
    stem_matches = stem_map.get(stem_key, [])
    if len(stem_matches) == 1:
        return stem_matches[0], None
    if len(stem_matches) > 1:
        return None, f"ambiguous stem match '{token}' -> {len(stem_matches)} files"

    base_matches = basename_map.get(base.lower(), [])
    if len(base_matches) == 1:
        return base_matches[0], None
    if len(base_matches) > 1:
        return None, f"ambiguous basename match '{token}' -> {len(base_matches)} files"

    return None, f"no file match for '{token}'"


def load_ground_truth(csv_path: str, tasks: Optional[list[str]] = None) -> Dict[str, dict]:
    """
    Load ground truth data and normalize metadata columns.

    Supported aliases include:
    - filename: filename, file, name (also handles 'Filename')
    - raga: raga
    - tonic: tonic
    - instrument: instrument_type or instrument
    - gender: vocalist_gender or gender
    - source: source_type
    - melody: melody_source

    When tasks are provided, rows are mapped to absolute audio file paths using
    stem-first matching, then basename fallback.
    """
    ground_truth: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Ground truth file not found at {csv_path}")
        return ground_truth

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if not fieldnames:
                print("Error: CSV has no headers.")
                return ground_truth

            name_col = _find_column(fieldnames, ["filename", "file", "audio", "name"])
            if not name_col:
                print(f"Error: CSV must contain a filename column. Found: {reader.fieldnames}")
                return ground_truth

            raga_col = _find_column(fieldnames, ["raga"])
            tonic_col = _find_column(fieldnames, ["tonic"])
            inst_col = _find_column(fieldnames, ["instrument_type", "instrument"])
            gender_col = _find_column(fieldnames, ["vocalist_gender", "gender"])
            source_col = _find_column(fieldnames, ["source_type", "source"])
            melody_col = _find_column(fieldnames, ["melody_source", "melody"])

            stem_map: Dict[str, list[str]] = {}
            basename_map: Dict[str, list[str]] = {}
            if tasks:
                stem_map, basename_map = _build_task_lookup(tasks)

            for row_idx, row in enumerate(reader, start=2):
                filename_token = _clean_text(row.get(name_col))
                if not filename_token:
                    print(f"[GT] Row {row_idx}: empty filename; skipping")
                    continue

                key: Optional[str]
                if tasks:
                    matched, reason = _match_csv_filename_to_task(filename_token, stem_map, basename_map)
                    if matched is None:
                        print(f"[GT] Row {row_idx}: {reason}; skipping")
                        continue
                    key = matched
                else:
                    key = os.path.basename(filename_token).lower()

                entry: Dict[str, str] = {}
                raga = _clean_text(row.get(raga_col)) if raga_col else ""
                tonic = _clean_text(row.get(tonic_col)) if tonic_col else ""
                if raga:
                    entry["raga"] = raga
                if tonic:
                    entry["tonic"] = tonic

                instrument_value = _clean_text(row.get(inst_col)) if inst_col else ""
                source_value = _clean_text(row.get(source_col)) if source_col else ""
                melody_value = _clean_text(row.get(melody_col)) if melody_col else ""
                gender_value = _clean_text(row.get(gender_col)) if gender_col else ""

                source_type = _derive_source_type(source_value, instrument_value)
                if source_type:
                    entry["source_type"] = source_type

                melody_source = _normalize_melody_source(melody_value)
                if melody_source:
                    entry["melody_source"] = melody_source

                if source_type == "vocal":
                    gender = _normalize_gender(gender_value)
                    if gender:
                        entry["vocalist_gender"] = gender
                else:
                    instrument_type = _normalize_instrument_type(instrument_value)
                    if instrument_type:
                        entry["instrument_type"] = instrument_type

                ground_truth[key] = entry
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

def process_directory(
    input_dir: str,
    ground_truth_path: Optional[str] = None,
    output_dir: str = "results",
    mode: str = "detect",
    silent: bool = False,
) -> None:
    """Walk an input directory and run the pipeline on each audio file."""
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"detect", "analyze"}:
        raise ValueError("mode must be 'detect' or 'analyze'.")
    if normalized_mode == "analyze" and not str(ground_truth_path or "").strip():
        raise ValueError("ground_truth_path is required when mode='analyze'.")

    tasks: list[str] = []
    for root, _dirs, files in os.walk(input_dir):
        for file in files:
            if _is_valid_audio_file(file, AUDIO_EXTS):
                tasks.append(os.path.join(root, file))
    tasks.sort()

    print(f"Found {len(tasks)} audio files to process.")

    gt_data: Dict[str, dict] = {}
    if ground_truth_path:
        gt_data = load_ground_truth(ground_truth_path, tasks=tasks)
        print(f"Loaded {len(gt_data)} matched ground truth entries.")
    elif normalized_mode == "detect":
        print("Mode set to 'detect' with no ground truth CSV.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pipeline_script = os.path.join(project_root, "run_pipeline.sh")
    processed_count = 0
    skipped_count = 0

    for i, audio_path in enumerate(tasks):
        fname = os.path.basename(audio_path)
        audio_abs = os.path.abspath(audio_path)
        print(f"\n[{i+1}/{len(tasks)}] Processing: {fname}")

        gt_info = gt_data.get(audio_abs, {})
        if normalized_mode == "detect":
            print("  -> Running in DETECT mode")
            cmd = [pipeline_script, "detect", "--audio", audio_path, "--output", output_dir]
        else:
            if not gt_info:
                print("  -> [SKIP] No matched ground truth row for analyze mode.")
                skipped_count += 1
                continue
            raga = gt_info.get("raga")
            tonic = gt_info.get("tonic")
            if not raga or not tonic:
                print("  -> [SKIP] Missing raga/tonic in ground truth for analyze mode.")
                skipped_count += 1
                continue
            print(f"  -> Running in ANALYZE mode with Raga='{raga}', Tonic='{tonic}'")
            cmd = [
                pipeline_script,
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

        if gt_info:
            _append_metadata_args(cmd, gt_info)

        try:
            log_dir = os.path.join(output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{fname}.log")

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

                # Read char-by-char so tqdm-style '\r' progress lines are preserved.
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
                returncode = process.returncode

            if returncode == 0:
                print(f"  [SUCCESS] Log: {log_file}")
                processed_count += 1
            else:
                print(f"  [FAILURE] Exit Code: {returncode}. Check Log: {log_file}")
                skipped_count += 1
        except Exception as e:
            print(f"  [ERROR] Execution failed: {e}")
            skipped_count += 1

    print(f"\nBatch complete. Processed: {processed_count}, Skipped: {skipped_count}, Total: {len(tasks)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process audio files for Raga Detection.")
    parser.add_argument("input_dir", help="Directory containing audio files")
    
    # Default paths relative to project root (assuming script is run from project root or inside module)
    # We resolve them relative to this file's location to be robust
    
    # Current file: raga_detection/raga_pipeline/batch.py
    # Ground truth: optional for detect, required for analyze
    # Output:       raga_detection/batch_results

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # raga-detection/
    default_output = os.path.join(project_root, "batch_results")

    parser.add_argument("--ground-truth", "-g", default=None,
                        help="Path to ground truth CSV (required for mode='analyze').")
    parser.add_argument("--output", "-o", default=default_output, 
                        help=f"Output directory (default: {default_output})")
    parser.add_argument("--mode", "-m", choices=['detect', 'analyze'], default='detect',
                        help="Processing mode: 'detect' or 'analyze'.")
    parser.add_argument("--silent", "-s", action="store_true", help="Suppress output to console (log files are still saved)")
    parser.add_argument("--init-csv", action="store_true", help="Initialize a blank ground truth CSV for files in input_dir")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
        
    if args.init_csv:
        target_csv = args.ground_truth
        if not target_csv:
            input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
            target_csv = os.path.join(args.input_dir, f"{input_dir_name}_gt.csv")
            print(f"No --ground-truth provided; initializing CSV at {target_csv}")
        assert target_csv is not None
        init_ground_truth_csv(args.input_dir, target_csv)
    else:
        try:
            process_directory(args.input_dir, args.ground_truth, args.output, args.mode, args.silent)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
