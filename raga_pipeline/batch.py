import csv
import os
import subprocess
import argparse
import sys
from typing import Dict, Optional, Tuple

def load_ground_truth(csv_path: str) -> Dict[str, dict]:
    """
    load ground truth data from annotated song db
    expected columns: filename, raga, tonic
    optional columns: instrument_type, vocalist_gender, source_type, melody_source
    returns dict: {filename: {data}}
    """
    ground_truth = {}
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

def init_ground_truth_csv(input_dir: str, output_csv: str):
    """
    Initialize a ground truth CSV with files from input_dir.
    Sets defaults: source_type='vocal', melody_source='separated'.
    """
    valid_exts = {'.mp3', '.wav', '.m4a', '.flac'}
    files_to_process = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
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

def process_directory(input_dir: str, ground_truth_path: Optional[str] = None, output_dir: str = "results", mode: str = "auto", silent: bool = False):
    """
    walk through directory and process audio files.
    """
    # list of allowed filetypes
    valid_exts = {'.mp3', '.wav', '.m4a', '.flac'}
    
    # load ground truth
    gt_data = {}
    if ground_truth_path and mode == "auto":
        gt_data = load_ground_truth(ground_truth_path)
        print(f"Loaded {len(gt_data)} ground truth entries.")
    elif mode == "detect":
        print("Mode set to 'detect'. Ignoring ground truth.")

    # walk directory
    tasks = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                full_path = os.path.join(root, file)
                tasks.append(full_path)

    print(f"Found {len(tasks)} audio files to process.")
    
    # Process each file
    for i, audio_path in enumerate(tasks):
        fname = os.path.basename(audio_path)
        print(f"\n[{i+1}/{len(tasks)}] Processing: {fname}")
        
        # Check for ground truth
        gt_info = gt_data.get(fname)
        
        # Build command to use run_pipeline.sh
        # We need the absolute path to run_pipeline.sh
        # Assuming raga_pipeline/batch.py -> ../run_pipeline.sh
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        pipeline_script = os.path.join(project_root, "run_pipeline.sh")
        
        base_cmd = [pipeline_script]
        
        # Determine mode
        if mode == "auto" and gt_info:
            raga = gt_info.get('raga')
            tonic = gt_info.get('tonic')
            
            if raga and tonic:
                print(f"  -> Found Ground Truth: Raga='{raga}', Tonic='{tonic}'")
                print("  -> Running in ANALYZE mode")
                
                cmd = base_cmd + [
                    "analyze",
                    "--audio", audio_path,
                    "--output", output_dir,
                    "--raga", raga,
                    "--tonic", tonic
                ]
                
                # Add optional metadata args if present in CSV
                if gt_info.get('instrument_type'):
                    cmd.extend(["--instrument-type", gt_info['instrument_type']])
                    print(f"     + Instrument: {gt_info['instrument_type']}")
                    
                if gt_info.get('vocalist_gender'):
                    cmd.extend(["--vocalist-gender", gt_info['vocalist_gender']])
                    print(f"     + Gender: {gt_info['vocalist_gender']}")
                    
                if gt_info.get('source_type'):
                    cmd.extend(["--source-type", gt_info['source_type']])
                    print(f"     + Source: {gt_info['source_type']}")
                    
                if gt_info.get('melody_source'):
                    cmd.extend(["--melody-source", gt_info['melody_source']])
                    print(f"     + Melody Source: {gt_info['melody_source']}")

            else:
                 print("  -> Ground truth entry found but missing Raga/Tonic. Falling back to DETECT.")
                 cmd = base_cmd + [
                    "detect",
                    "--audio", audio_path,
                    "--output", output_dir
                ]
        else:
            if mode == "detect":
                 print("  -> Forced DETECT mode")
            else:
                 print("  -> No Ground Truth found")
                 print("  -> Running in DETECT mode")
            
            cmd = base_cmd + [
                "detect",
                "--audio", audio_path,
                "--output", output_dir
            ]
            
            # Forward CSV metadata even in DETECT mode so detect/analyze flags share the same hints.
            if mode == "detect" and gt_info:
                if gt_info.get('instrument_type'):
                    cmd.extend(["--instrument-type", gt_info['instrument_type']])
                    print(f"     + Instrument: {gt_info['instrument_type']}")
                    
                if gt_info.get('vocalist_gender'):
                    cmd.extend(["--vocalist-gender", gt_info['vocalist_gender']])
                    print(f"     + Gender: {gt_info['vocalist_gender']}")
                    
                if gt_info.get('source_type'):
                    cmd.extend(["--source-type", gt_info['source_type']])
                    print(f"     + Source: {gt_info['source_type']}")
                    
                if gt_info.get('melody_source'):
                    cmd.extend(["--melody-source", gt_info['melody_source']])
                    print(f"     + Melody Source: {gt_info['melody_source']}")


        # Run pipeline
        try:
            log_dir = os.path.join(output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{fname}.log")
            
            # Use Popen to stream output
            with open(log_file, "w") as f_log:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1  # Line buffered
                )

                if process.stdout is None:
                    raise RuntimeError("Failed to capture subprocess stdout.")
                
                # Read char by char so tqdm-style progress lines using '\r' render correctly.
                
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
            else:
                print(f"  [FAILURE] Exit Code: {returncode}. Check Log: {log_file}")
                
        except Exception as e:
            print(f"  [ERROR] Execution failed: {e}")

if __name__ == "__main__":
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
    else:
        process_directory(args.input_dir, args.ground_truth, args.output, args.mode, args.silent)
