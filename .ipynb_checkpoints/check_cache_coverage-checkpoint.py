#!/usr/bin/env python3
"""
check_cache_coverage.py

Проверяет, какие последовательности из baselines/ уже есть в кэше
BioEmu эмбеддингов (~/.bioemu_embeds_cache/), а каких не хватает.

Если недостающие есть — сохраняет их в FASTA для последующей
генерации MSA через colabfold_search.

Использование:
  python check_cache_coverage.py --baselines_dir baselines
"""

import os
import re
import sys
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'UNK': 'X', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}

NOISE_PATTERN = re.compile(r"noise_(\d+\.\d+)_job_\d+_n_\d+_id_\d+")


def shahexencode(s):
    return hashlib.sha256(s.encode()).hexdigest()


def get_sequence_from_pdb(pdb_path):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    model = structure[0]
    chain = list(model.get_chains())[0]
    seq = []
    for residue in chain.get_residues():
        if residue.id[0] == ' ':
            resname = residue.resname
            seq.append(THREE_TO_ONE.get(resname, 'X'))
    return ''.join(seq)


def main():
    parser = argparse.ArgumentParser(
        description="Check which baseline sequences are cached in BioEmu embeds cache"
    )
    parser.add_argument("--baselines_dir", default="baselines")
    parser.add_argument(
        "--cache_dir",
        default=os.path.expanduser("~/.bioemu_embeds_cache"),
    )
    parser.add_argument("--output_fasta", default="missing_sequences.fasta")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    baselines_dir = Path(args.baselines_dir)

    # Собираем все хэши в кэше
    cached_hashes = set()
    if cache_dir.exists():
        for f in cache_dir.glob("*_single.npy"):
            h = f.stem.replace("_single", "")
            cached_hashes.add(h)
    print(f"Cache: {len(cached_hashes)} sequences cached in {cache_dir}")

    if not baselines_dir.is_dir():
        print(f"ERROR: {baselines_dir} not found")
        sys.exit(1)

    # Сканируем baselines
    total = 0
    cached = 0
    missing = 0
    missing_info = []  # (task_mode, sample_dir, seq_type, full_seq)

    for task_mode_dir in sorted(baselines_dir.iterdir()):
        if not task_mode_dir.is_dir():
            continue

        for sample_dir in sorted(task_mode_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            if not NOISE_PATTERN.match(sample_dir.name):
                continue

            for seq_label, suffix in [("A", "_pathA.pdb"), ("B", "_pathB.pdb")]:
                pdb = sample_dir / f"{sample_dir.name}{suffix}"
                if not pdb.exists():
                    pdb = sample_dir / ("pathA.pdb" if seq_label == "A" else "pathB.pdb")
                if not pdb.exists():
                    continue

                try:
                    seq = get_sequence_from_pdb(pdb)
                    if len(seq) == 0:
                        continue
                except Exception:
                    continue

                h = shahexencode(seq)
                total += 1
                if h in cached_hashes:
                    cached += 1
                else:
                    missing += 1
                    missing_info.append((task_mode_dir.name, sample_dir.name, seq_label, seq))

    print(f"\nBaselines sequences: {total}")
    if total > 0:
        print(f"  Cached:   {cached} ({cached/total*100:.1f}%)")
        print(f"  Missing:  {missing} ({missing/total*100:.1f}%)")

    if missing > 0:
        print(f"\nFirst 10 missing:")
        for tm, sd, st, seq in missing_info[:10]:
            print(f"  {tm}/{sd}/seq{st}: {seq[:60]}...")

        # Сохранить недостающие в FASTA (уникальные последовательности)
        seen_seqs = set()
        unique_missing = []
        for tm, sd, st, seq in missing_info:
            if seq not in seen_seqs:
                seen_seqs.add(seq)
                unique_missing.append((tm, sd, st, seq))

        fasta_path = args.output_fasta
        with open(fasta_path, "w") as f:
            for i, (tm, sd, st, seq) in enumerate(unique_missing):
                seq_id = f"seq_{i:04d}_{tm}_{sd}_seq{st}"
                f.write(f">{seq_id}\n{seq}\n")
        print(f"\n{len(unique_missing)} unique missing sequences saved to: {fasta_path}")
        print(f"(out of {missing} total missing, {missing - len(unique_missing)} duplicates)")
    else:
        print("\nAll sequences are cached! BioEmu should work without colabfold.")


if __name__ == "__main__":
    main()
