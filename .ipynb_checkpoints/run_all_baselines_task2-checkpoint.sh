#!/bin/bash
# Run all baseline experiments for Task 2 (Conditional)
# Must be executed from la-proteina-main/ directory

cd "$(dirname "$0")" || exit 1

export DATA_PATH=./data

NUM_PAIRS=50
OUTPUT_DIR="baselines/task2_conditional_default"

# Motif parameters (from successful test run)
MOTIF_NAME="1BCF_AA_single"
MOTIF_PDB="./motifs/1bcf_aa.pdb"
CONTIG_STRING="A1-9/31/A41-158"
MOTIF_MIN_LENGTH=158
MOTIF_MAX_LENGTH=158

# 10 points for Task 2
NOISE_SCALES=(0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.70 1.0)

for scale in "${NOISE_SCALES[@]}"; do
    echo "=========================================="
    echo "Running with noise_scale=$scale"
    echo "=========================================="
    python run_baseline_task2.py \
        --noise_scale $scale \
        --num_pairs $NUM_PAIRS \
        --output_dir $OUTPUT_DIR \
        --motif_name $MOTIF_NAME \
        --motif_pdb $MOTIF_PDB \
        --contig_string $CONTIG_STRING \
        --motif_min_length $MOTIF_MIN_LENGTH \
        --motif_max_length $MOTIF_MAX_LENGTH
done

echo "All experiments completed!"