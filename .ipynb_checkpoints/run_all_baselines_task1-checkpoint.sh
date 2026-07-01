#!/bin/bash
# Run all baseline experiments for Task 1
# Must be executed from la-proteina-main/ directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine project root (script is in baselines/, so go up one level)
if [ -f "$SCRIPT_DIR/run_baseline_task1.py" ]; then
    # Script is in the same directory as run_baseline_task1.py
    PROJECT_ROOT="$SCRIPT_DIR"
else
    # Script is in baselines/, run_baseline_task1.py is in parent
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

cd "$PROJECT_ROOT" || exit 1

# Set DATA_PATH environment variable
export DATA_PATH=./data

NUM_PAIRS=50
OUTPUT_DIR="baselines/task1_unconditional"

# 7 points for Task 1: small noise neighborhoods
NOISE_SCALES=(0.00 0.05 0.10 0.15 0.20 0.25 0.30)

for scale in "${NOISE_SCALES[@]}"; do
    echo "=========================================="
    echo "Running with noise_scale=$scale"
    echo "=========================================="
    python run_baseline_task1.py \
        --noise_scale $scale \
        --num_pairs $NUM_PAIRS \
        --output_dir $OUTPUT_DIR
done

echo "All experiments completed!"