#!/usr/bin/env python3
import hashlib
import re
from pathlib import Path
from Bio.PDB import PDBParser

NOISE_PATTERN = re.compile(r"noise_(\d+\.\d+)_job_\d+_n_\d+_id_\d+")

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X", "MSE": "M", "SEC": "U", "PYL": "O",
}

parser = PDBParser(QUIET=True)
baselines = Path('baselines')
sequences = {}

for task_mode_dir in sorted(baselines.iterdir()):
    if not task_mode_dir.is_dir(): continue
    for sample_dir in sorted(task_mode_dir.iterdir()):
        if not sample_dir.is_dir(): continue
        if not NOISE_PATTERN.match(sample_dir.name): continue

        for suffix in ["_pathA.pdb", "_pathB.pdb"]:
            pdb = sample_dir / f"{sample_dir.name}{suffix}"
            if not pdb.exists():
                pdb = sample_dir / ("pathA.pdb" if "pathA" in suffix else "pathB.pdb")
            if not pdb.exists(): continue

            try:
                structure = parser.get_structure("p", str(pdb))
                chain = list(structure[0].get_chains())[0]
                seq = "".join([THREE_TO_ONE.get(res.resname, "X") for res in chain.get_residues() if res.id[0] == " "])
                if len(seq) > 0:
                    h = hashlib.sha256(seq.encode()).hexdigest()
                    sequences[h] = seq
            except Exception:
                continue

with open("all_sequences.fasta", "w") as f:
    for h, seq in sequences.items():
        f.write(f">{h}\n{seq}\n")

print(f"✅ Создан файл all_sequences.fasta с {len(sequences)} уникальными последовательностями.")