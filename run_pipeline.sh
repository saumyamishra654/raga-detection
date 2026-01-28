#!/bin/bash
# wrapper for driver.py
# activates the 'raga' conda environment

# get pwd
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# conda init
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate raga

# run pipeline all arguments passed to script
python "$SCRIPT_DIR/driver.py" "$@"
