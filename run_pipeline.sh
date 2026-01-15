#!/bin/bash
# Raga Detection Pipeline Runner
# Automatically activates the 'raga' conda environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Source conda initialization
source /opt/miniconda3/etc/profile.d/conda.sh

# Activate the raga environment
conda activate raga

# Run the pipeline with all arguments passed to this script
python "$SCRIPT_DIR/driver.py" "$@"
