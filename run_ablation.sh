#!/bin/bash
# Ablation study: entropy weighting x direction tokens
# Trains 4 models, runs pipeline batch for each
# Must run with: /opt/miniconda3/envs/raga/bin/bash run_ablation.sh

PYTHON=/opt/miniconda3/envs/raga/bin/python
GT=compmusic_gt.csv
RESULTS_UNCORR="/Volumes/Extreme SSD/stems/separated_stems_nocorrection/"
OUTPUT="/Volumes/Extreme SSD/stems/separated_stems"

echo "=== Ablation: 4 configurations ==="
echo "1. baseline (no entropy, no direction)"
echo "2. entropy only"
echo "3. direction only"
echo "4. entropy + direction"
echo ""

# We need to control direction at tokenization time.
# For now, train all 4 models, then run pipeline for each.
# Direction is controlled by an env var that the tokenizer reads,
# and entropy by a flag on the model.

# Actually, simpler: just run the LOO eval for each config.
# That gives us the LM-only accuracy which is the fair comparison.
# Pipeline eval can come after for the winner.

for config in "baseline" "entropy" "direction" "both"; do
    echo "=== Config: $config ==="

    case $config in
        baseline)
            export RAGA_LM_DIRECTION=0
            entropy_flag=""
            ;;
        entropy)
            export RAGA_LM_DIRECTION=0
            entropy_flag=""  # entropy is always computed, we'll handle via model flag
            ;;
        direction)
            export RAGA_LM_DIRECTION=1
            entropy_flag=""
            ;;
        both)
            export RAGA_LM_DIRECTION=1
            entropy_flag=""
            ;;
    esac

    echo "Training..."
    $PYTHON -m raga_pipeline.language_model train \
        --gt $GT \
        --results-dir "$RESULTS_UNCORR" \
        --output "ablation_model_${config}.json" \
        --order 5 --min-recordings 3 --quiet

    echo "Evaluating (LOO)..."
    $PYTHON -m raga_pipeline.language_model evaluate \
        --gt $GT \
        --results-dir "$RESULTS_UNCORR" \
        --order 5 --min-recordings 3 \
        --output "ablation_eval_${config}.csv" \
        --quiet

    echo ""
done
