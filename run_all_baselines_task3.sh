#!/bin/bash
# Run all baseline experiments for Task 3 (Conditional with LARGE noise)
# Must be executed from la-proteina-main/ directory

cd "$(dirname "$0")" || exit 1

export DATA_PATH=./data

NUM_PAIRS=50
OUTPUT_DIR="baselines/task3_conditional_large_sim_eps"

# Motif parameters (same as Task 2)
MOTIF_NAME="1BCF_AA_single"
MOTIF_PDB="./motifs/1bcf_aa.pdb"
CONTIG_STRING="A1-9/31/A41-158"
MOTIF_MIN_LENGTH=158
MOTIF_MAX_LENGTH=158

# LARGE noise scales for Task 3
NOISE_SCALES=(2.0 3.0 5.0 10.0 20.0 50.0 100.0)

for scale in "${NOISE_SCALES[@]}"; do
    echo "=========================================="
    echo "Running with noise_scale=$scale"
    echo "=========================================="
    python run_baseline_task3.py \
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