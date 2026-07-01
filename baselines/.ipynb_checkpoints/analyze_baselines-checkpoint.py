"""
Analysis of baseline experiment results.
Calculates RMSD and Sequence Identity between structures A and B.

Usage (from la-proteina-main/ directory):
    python baselines/analyze_baselines.py
"""

import os
import sys

# Change working directory to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
os.chdir(PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one


def calculate_rmsd(pdb1_path, pdb2_path):
    """Calculate C-alpha RMSD between two structures after superposition."""
    parser = PDBParser(QUIET=True)
    struct1 = parser.get_structure("struct1", str(pdb1_path))
    struct2 = parser.get_structure("struct2", str(pdb2_path))

    ca_coords1 = []
    ca_coords2 = []

    for res1, res2 in zip(struct1.get_residues(), struct2.get_residues()):
        if res1.id[0] == ' ' and res2.id[0] == ' ':
            if 'CA' in res1 and 'CA' in res2:
                ca_coords1.append(res1['CA'].get_vector().get_array())
                ca_coords2.append(res2['CA'].get_vector().get_array())

    ca_coords1 = np.array(ca_coords1)
    ca_coords2 = np.array(ca_coords2)

    if len(ca_coords1) == 0 or len(ca_coords1) != len(ca_coords2):
        return float('nan')

    # Kabsch alignment (center + rotate)
    ca_coords1_centered = ca_coords1 - ca_coords1.mean(axis=0)
    ca_coords2_centered = ca_coords2 - ca_coords2.mean(axis=0)

    # Compute optimal rotation using SVD
    H = ca_coords1_centered.T @ ca_coords2_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])
    R = Vt.T @ sign_matrix @ U.T

    ca_coords1_aligned = (R @ ca_coords1_centered.T).T

    # Calculate RMSD
    rmsd = np.sqrt(np.mean((ca_coords1_aligned - ca_coords2_centered) ** 2))
    return rmsd


def calculate_sequence_identity(pdb1_path, pdb2_path):
    """Calculate sequence identity between two structures."""
    parser = PDBParser(QUIET=True)
    struct1 = parser.get_structure("struct1", str(pdb1_path))
    struct2 = parser.get_structure("struct2", str(pdb2_path))

    seq1 = []
    seq2 = []

    for res1, res2 in zip(struct1.get_residues(), struct2.get_residues()):
        if res1.id[0] == ' ' and res2.id[0] == ' ':
            try:
                seq1.append(three_to_one(res1.resname))
                seq2.append(three_to_one(res2.resname))
            except KeyError:
                seq1.append('X')
                seq2.append('X')

    seq1 = ''.join(seq1)
    seq2 = ''.join(seq2)

    if len(seq1) == 0:
        return 0.0

    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    identity = matches / len(seq1)
    return identity


def analyze_experiment(exp_dir):
    """Analyze results of one experiment (one noise_scale)."""
    exp_dir = Path(exp_dir)

    # Find all pathA.pdb files
    pdb_files = sorted(exp_dir.glob("*_pathA.pdb"))

    results = []
    for pdb_A in pdb_files:
        pdb_B = pdb_A.with_name(pdb_A.name.replace("_pathA.pdb", "_pathB.pdb"))

        if not pdb_B.exists():
            continue

        rmsd = calculate_rmsd(pdb_A, pdb_B)
        seq_id = calculate_sequence_identity(pdb_A, pdb_B)

        results.append({
            'pair': pdb_A.parent.name,
            'rmsd': rmsd,
            'seq_identity': seq_id
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze baseline experiment results"
    )
    parser.add_argument(
        "--baselines_dir",
        type=str,
        default="baselines/task1_unconditional",
        help="Directory with baseline results",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="baselines/baselines_task1_analysis.csv",
        help="Output CSV file",
    )
    args = parser.parse_args()

    baselines_dir = Path(args.baselines_dir)

    all_results = []

    for exp_dir in sorted(baselines_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        print(f"\nAnalyzing experiment: {exp_dir.name}")
        df = analyze_experiment(exp_dir)

        if df.empty:
            print(f"  No valid pairs found")
            continue

        df['experiment'] = exp_dir.name

        # Extract noise_scale from directory name (format: noise_0.10_...)
        try:
            noise_part = exp_dir.name.split('_')[1]
            noise_scale = float(noise_part)
        except (IndexError, ValueError):
            print(f"  Could not extract noise_scale from {exp_dir.name}")
            continue

        df['noise_scale'] = noise_scale
        all_results.append(df)

        # Print statistics
        valid_rmsd = df['rmsd'].dropna()
        print(f"  Number of pairs: {len(df)}")
        print(f"  Mean RMSD: {valid_rmsd.mean():.2f} +/- {valid_rmsd.std():.2f} A")
        print(f"  Mean Seq Identity: {df['seq_identity'].mean()*100:.2f} +/- {df['seq_identity'].std()*100:.2f}%")

    if not all_results:
        print("No results found!")
        return

    # Save results
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(args.output_csv, index=False)

    print(f"\nResults saved in: {args.output_csv}")

    # Print summary table
    summary = all_df.groupby('noise_scale').agg({
        'rmsd': ['mean', 'std', 'median'],
        'seq_identity': ['mean', 'std', 'median']
    }).round(3)

    print("\nSummary table:")
    print(summary)


if __name__ == "__main__":
    main()