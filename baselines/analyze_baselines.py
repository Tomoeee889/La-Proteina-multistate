"""
Universal analysis script for all baseline experiments.
Calculates RMSD, sequence identity, and (for conditional tasks) motif/scaffold breakdown.

Usage (from la-proteina-main/):
    python baselines/analyze_baselines.py --baselines_dir baselines/task1_unconditional_default
    python baselines/analyze_baselines.py --baselines_dir baselines/task2_conditional_default --conditional --motif_indices "1-9,41-158"
"""

import argparse
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist

# Compatible import for three_to_one across Biopython versions
try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    try:
        from Bio.SeqUtils import seq1 as three_to_one
    except ImportError:
        from Bio.Data.PDBData import protein_letters_3to1 as _p3to1
        def three_to_one(resname):
            resname = resname.strip()
            if len(resname) != 3:
                raise KeyError(resname)
            return _p3to1.get(resname, 'X')

warnings.filterwarnings('ignore')


def kabsch_rmsd(P, Q):
    """
    Calculate RMSD after Kabsch alignment.
    P, Q: numpy arrays of shape (N, 3)
    """
    assert P.shape == Q.shape
    n = P.shape[0]
    if n == 0:
        return float('nan')

    p = P - P.mean(axis=0)
    q = Q - Q.mean(axis=0)

    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T

    p_aligned = (R @ p.T).T
    rmsd = np.sqrt(np.mean(np.sum((p_aligned - q) ** 2, axis=1)))
    return rmsd


def extract_ca_and_seq(pdb_path, motif_indices=None):
    """Extract Cα coordinates and sequence from PDB."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("struct", str(pdb_path))
    except Exception as e:
        print(f"  Error parsing {pdb_path}: {e}")
        return None

    ca_coords = []
    sequence = []

    for residue in structure.get_residues():
        if residue.id[0] != ' ':
            continue
        if 'CA' in residue:
            ca_coords.append(residue['CA'].get_vector().get_array())
            try:
                sequence.append(three_to_one(residue.resname))
            except (KeyError, ValueError):
                sequence.append('X')

    if len(ca_coords) == 0:
        return None

    ca_coords = np.array(ca_coords)
    sequence = ''.join(sequence)

    result = {
        'ca': ca_coords,
        'sequence': sequence,
        'length': len(sequence),
    }

    if motif_indices is not None:
        motif_mask = np.array([i in motif_indices for i in range(len(ca_coords))])
        result['motif_ca'] = ca_coords[motif_mask] if motif_mask.any() else np.array([]).reshape(0, 3)
        result['scaffold_ca'] = ca_coords[~motif_mask] if (~motif_mask).any() else np.array([]).reshape(0, 3)
        result['motif_sequence'] = ''.join([sequence[i] for i in range(len(sequence)) if i in motif_indices])
        result['scaffold_sequence'] = ''.join([sequence[i] for i in range(len(sequence)) if i not in motif_indices])

    return result


def analyze_pair(pdb_A, pdb_B, motif_indices=None):
    """Analyze one pair of structures (path A vs path B)."""
    data_A = extract_ca_and_seq(pdb_A, motif_indices)
    data_B = extract_ca_and_seq(pdb_B, motif_indices)

    if data_A is None or data_B is None:
        return None

    if data_A['length'] != data_B['length']:
        return None

    results = {
        'length': data_A['length'],
        'full_rmsd': kabsch_rmsd(data_A['ca'], data_B['ca']),
        'seq_identity': sum(a == b for a, b in zip(data_A['sequence'], data_B['sequence'])) / data_A['length'],
    }

    if motif_indices is not None:
        if len(data_A['motif_ca']) == len(data_B['motif_ca']) and len(data_A['motif_ca']) > 0:
            results['motif_rmsd'] = kabsch_rmsd(data_A['motif_ca'], data_B['motif_ca'])
            if len(data_A['motif_sequence']) > 0:
                results['motif_seq_identity'] = sum(
                    a == b for a, b in zip(data_A['motif_sequence'], data_B['motif_sequence'])
                ) / len(data_A['motif_sequence'])
        else:
            results['motif_rmsd'] = float('nan')
            results['motif_seq_identity'] = float('nan')

        if len(data_A['scaffold_ca']) == len(data_B['scaffold_ca']) and len(data_A['scaffold_ca']) > 0:
            results['scaffold_rmsd'] = kabsch_rmsd(data_A['scaffold_ca'], data_B['scaffold_ca'])
            if len(data_A['scaffold_sequence']) > 0:
                results['scaffold_seq_identity'] = sum(
                    a == b for a, b in zip(data_A['scaffold_sequence'], data_B['scaffold_sequence'])
                ) / len(data_A['scaffold_sequence'])
        else:
            results['scaffold_rmsd'] = float('nan')
            results['scaffold_seq_identity'] = float('nan')

    return results


def parse_noise_scale(dirname):
    """Extract noise_scale from directory name."""
    m = re.match(r'noise_([\d.]+)_job_(\d+)_n_(\d+)_id_(\d+)', dirname)
    if m:
        return float(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return None, None, None, None


def analyze_experiment(exp_dir, motif_indices=None):
    """Analyze all pairs in one experiment directory."""
    exp_dir = Path(exp_dir)
    pdb_files = sorted(exp_dir.glob("*_pathA.pdb"))

    results = []
    for pdb_A in pdb_files:
        pdb_B = pdb_A.with_name(pdb_A.name.replace("_pathA.pdb", "_pathB.pdb"))
        if not pdb_B.exists():
            continue

        pair_results = analyze_pair(pdb_A, pdb_B, motif_indices)
        if pair_results is None:
            continue

        pair_results['pair'] = pdb_A.parent.name
        results.append(pair_results)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Analyze baseline experiments")
    parser.add_argument("--baselines_dir", type=str, required=True)
    parser.add_argument("--conditional", action="store_true", help="Enable motif/scaffold breakdown")
    parser.add_argument("--motif_indices", type=str, default=None,
                        help="Comma-separated 1-based residue indices for motif (e.g., '1-9,41-158')")
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    baselines_dir = Path(args.baselines_dir)

    motif_indices = None
    if args.conditional and args.motif_indices:
        motif_indices = set()
        for part in args.motif_indices.split(','):
            if '-' in part:
                start, end = part.split('-')
                for i in range(int(start) - 1, int(end)):
                    motif_indices.add(i)
            else:
                motif_indices.add(int(part) - 1)

    all_results = []

    for exp_dir in sorted(baselines_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        noise_scale, job_id, n, exp_id = parse_noise_scale(exp_dir.name)
        if noise_scale is None:
            print(f"Skipping {exp_dir.name} (cannot parse noise_scale)")
            continue

        print(f"\nAnalyzing: {exp_dir.name} (noise_scale={noise_scale})")
        df = analyze_experiment(exp_dir, motif_indices)

        if df.empty:
            print("  No valid pairs found")
            continue

        df['experiment'] = exp_dir.name
        df['noise_scale'] = noise_scale
        df['job_id'] = job_id
        df['n_residues'] = n
        all_results.append(df)

        print(f"  Pairs: {len(df)}")
        for col in ['full_rmsd', 'scaffold_rmsd', 'motif_rmsd', 'seq_identity', 'scaffold_seq_identity', 'motif_seq_identity']:
            if col in df.columns:
                valid = df[col].dropna()
                if len(valid) > 0:
                    print(f"  {col}: {valid.mean():.3f} ± {valid.std():.3f}")

    if not all_results:
        print("No results found!")
        return

    all_df = pd.concat(all_results, ignore_index=True)

    if args.output_csv:
        output_csv = args.output_csv
    else:
        output_csv = str(baselines_dir / "analysis_results.csv")
    all_df.to_csv(output_csv, index=False)
    print(f"\nRaw results saved: {output_csv}")

    agg_dict = {
        'full_rmsd': ['mean', 'std', 'median', 'count'],
        'seq_identity': ['mean', 'std', 'median'],
    }
    if args.conditional:
        agg_dict.update({
            'motif_rmsd': ['mean', 'std', 'median'],
            'scaffold_rmsd': ['mean', 'std', 'median'],
            'motif_seq_identity': ['mean', 'std', 'median'],
            'scaffold_seq_identity': ['mean', 'std', 'median'],
        })

    summary = all_df.groupby('noise_scale').agg(agg_dict).round(3)
    print("\n" + "=" * 80)
    print("SUMMARY BY NOISE SCALE")
    print("=" * 80)
    print(summary.to_string())

    summary_csv = str(baselines_dir / "analysis_summary.csv")
    summary.to_csv(summary_csv)
    print(f"\nSummary saved: {summary_csv}")


if __name__ == "__main__":
    main()