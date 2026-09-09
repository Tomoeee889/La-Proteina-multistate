#!/usr/bin/env python3
"""
precompute_embeddings.py (Final Local Mode)

Предгенерация AF2-эмбеддингов для всех последовательностей из baselines/.
ИСПОЛЬЗУЕТ ТОЛЬКО ЛОКАЛЬНЫЙ ПОИСК (colabfold_search). 
Никаких обращений к api.colabfold.com. Гарантирует совместимость с BioEmu.

Использование:
  tmux new -s bioemu_run
  conda activate bioemu_env
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python precompute_embeddings.py --baselines_dir baselines > run.log 2>&1
  # (Отключиться: Ctrl+B, затем D. Подключиться: tmux attach -t bioemu_run)
"""

import os
import re
import sys
import time
import hashlib
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

NOISE_PATTERN = re.compile(r"noise_(\d+\.\d+)_job_\d+_n_\d+_id_\d+")

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X", "MSE": "M", "SEC": "U", "PYL": "O",
}


def get_sequence_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", str(pdb_path))
    model = structure[0]
    chain = list(model.get_chains())[0]
    seq = []
    for res in chain.get_residues():
        if res.id[0] == " ":
            seq.append(THREE_TO_ONE.get(res.resname, "X"))
    return "".join(seq)


def find_colabfold_binary():
    """Находит colabfold_batch — предпочитает BioEmu-версию."""
    bioemu_cf = Path.home() / ".bioemu_colabfold" / "bin" / "colabfold_batch"
    if bioemu_cf.exists():
        return str(bioemu_cf)
    which = shutil.which("colabfold_batch")
    if which:
        return which
    return None


def scan_all_sequences(baselines_dir):
    """Сканирует baselines/ и возвращает dict {hash: sequence}."""
    baselines_dir = Path(baselines_dir)
    sequences = {}

    for task_mode_dir in sorted(baselines_dir.iterdir()):
        if not task_mode_dir.is_dir():
            continue

        for sample_dir in sorted(task_mode_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            if not NOISE_PATTERN.match(sample_dir.name):
                continue

            for suffix in ["_pathA.pdb", "_pathB.pdb"]:
                pdb = sample_dir / f"{sample_dir.name}{suffix}"
                if not pdb.exists():
                    pdb = sample_dir / ("pathA.pdb" if "pathA" in suffix else "pathB.pdb")
                if not pdb.exists():
                    continue

                try:
                    seq = get_sequence_from_pdb(pdb)
                    if len(seq) == 0:
                        continue
                except Exception:
                    continue

                h = hashlib.sha256(seq.encode()).hexdigest()
                if h not in sequences:
                    sequences[h] = seq

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Precompute AF2 embeddings for BioEmu (Local Mode)")
    parser.add_argument("--baselines_dir", default="baselines")
    parser.add_argument(
        "--cache_dir",
        default=str(Path.home() / ".bioemu_embeds_cache"),
    )
    parser.add_argument(
        "--db_dir",
        default="/home/domain/data/aristowi/colabfold_db",
        help="Path to local ColabFold databases"
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    db_dir = Path(args.db_dir)
    if not db_dir.exists():
        print(f"ERROR: Local ColabFold database directory not found: {db_dir}", flush=True)
        print("Please download databases first or specify correct --db_dir", flush=True)
        sys.exit(1)

    cf_binary = find_colabfold_binary()
    if not cf_binary:
        print("ERROR: colabfold_batch not found", flush=True)
        sys.exit(1)
    print(f"ColabFold binary: {cf_binary}", flush=True)

    cf_search = shutil.which("colabfold_search")
    if not cf_search:
        print("ERROR: colabfold_search not found in PATH", flush=True)
        print("Please install mmseqs2: conda install -c conda-forge mmseqs2", flush=True)
        sys.exit(1)
    print(f"ColabFold search binary: {cf_search}", flush=True)
    print(f"Using local databases from: {db_dir}", flush=True)

    print(f"Scanning {args.baselines_dir} ...", flush=True)
    all_seqs = scan_all_sequences(args.baselines_dir)
    print(f"Total unique sequences: {len(all_seqs)}", flush=True)

    cached = 0
    uncached = []
    for h, seq in all_seqs.items():
        single_file = cache_dir / f"{h}_single.npy"
        pair_file = cache_dir / f"{h}_pair.npy"
        if single_file.exists() and pair_file.exists():
            cached += 1
        else:
            uncached.append((h, seq))

    print(f"Already cached: {cached}", flush=True)
    print(f"Need to compute: {len(uncached)}", flush=True)

    if not uncached:
        print("All sequences are cached! Nothing to do.", flush=True)
        return

    failed_logs_dir = cache_dir / "failed_logs"
    failed_logs_dir.mkdir(exist_ok=True)

    done = 0
    failed = 0
    max_retries = 3
    retry_delay = 30

    for i, (h, seq) in enumerate(uncached):
        single_file = cache_dir / f"{h}_single.npy"
        pair_file = cache_dir / f"{h}_pair.npy"
        
        # Повторная проверка кэша на случай перезапуска скрипта
        if single_file.exists() and pair_file.exists():
            done += 1
            continue

        print(f"\n[{i+1}/{len(uncached)}] hash={h[:12]}... len={len(seq)}", flush=True)

        success = False
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                print(f"  Retry {attempt}/{max_retries} (waiting {retry_delay}s)...", flush=True)
                time.sleep(retry_delay)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                fasta_file = tmpdir / f"{h}.fasta"
                with open(fasta_file, "w") as f:
                    f.write(f">{h}\n{seq}\n")

                res_dir = tmpdir / "results"
                res_dir.mkdir()

                # === ШАГ A: Локальный поиск MSA ===
                msa_dir = tmpdir / "msa_results"
                msa_dir.mkdir()
                
                # db-load-mode 0 критичен для NFS-дисков, чтобы не пытаться загрузить 100ГБ в RAM
                search_cmd = [
                    cf_search,
                    str(fasta_file),
                    str(db_dir),
                    str(msa_dir),
                    "--db-load-mode", "0",
                    "--threads", "8"
                ]
                
                try:
                    print(f"  Running colabfold_search...", end=" ", flush=True)
                    search_result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=3600)
                    if search_result.returncode != 0:
                        print(f"FAILED (returncode={search_result.returncode})", flush=True)
                        log_path = failed_logs_dir / f"{h}_attempt{attempt}_search_stderr.txt"
                        log_path.write_text(search_result.stderr)
                        print(f"  search stderr: {search_result.stderr[:500]}", flush=True)
                        continue
                    print("done", flush=True)
                except subprocess.TimeoutExpired:
                    print("TIMEOUT (1 hour)", flush=True)
                    continue
                except Exception as e:
                    print(f"ERROR: {e}", flush=True)
                    continue

                a3m_file = msa_dir / f"{h}.a3m"
                if not a3m_file.exists():
                    print(f"  ERROR: .a3m file not created by colabfold_search", flush=True)
                    print(f"  Files in msa_dir: {list(msa_dir.iterdir())}", flush=True)
                    continue

                # === ШАГ B: Генерация эмбеддингов через colabfold_batch ===
                # --msa-mode single_sequence запрещает любые обращения к интернету
                cmd = [
                    cf_binary,
                    str(a3m_file),
                    str(res_dir),
                    "--msa-mode", "single_sequence",
                    "--num-models", "1",
                    "--model-order", "3",
                    "--model-type", "alphafold2",
                    "--num-recycle", "0",
                    "--save-single-representations",
                    "--save-pair-representations",
                ]

                try:
                    print(f"  Running colabfold_batch...", end=" ", flush=True)
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                    if result.returncode != 0:
                        print(f"FAILED (returncode={result.returncode})", flush=True)
                        log_path = failed_logs_dir / f"{h}_attempt{attempt}_batch_stderr.txt"
                        log_path.write_text(result.stderr)
                        cf_log = res_dir / "log.txt"
                        if cf_log.exists():
                            shutil.copy(str(cf_log), str(failed_logs_dir / f"{h}_attempt{attempt}_colabfold_log.txt"))
                        print(f"  batch stderr: {result.stderr[:500]}", flush=True)
                        continue
                    print("done", flush=True)
                except subprocess.TimeoutExpired:
                    print("TIMEOUT (1 hour)", flush=True)
                    cf_log = res_dir / "log.txt"
                    if cf_log.exists():
                        shutil.copy(str(cf_log), str(failed_logs_dir / f"{h}_attempt{attempt}_timeout_log.txt"))
                    continue
                except Exception as e:
                    print(f"ERROR: {e}", flush=True)
                    continue

                # === ШАГ C: Проверка и сохранение результатов ===
                single_files = list(res_dir.glob(f"*{h}*single_repr*evo*model_3*.npy"))
                pair_files = list(res_dir.glob(f"*{h}*pair_repr*evo*model_3*.npy"))

                if not single_files or not pair_files:
                    single_files = list(res_dir.glob(f"*single_repr*evo*model_3*.npy"))
                    pair_files = list(res_dir.glob(f"*pair_repr*evo*model_3*.npy"))

                if not single_files or not pair_files:
                    print(f"  ERROR: output .npy files not found in {res_dir}", flush=True)
                    print(f"  Files in res_dir: {list(res_dir.iterdir())}", flush=True)
                    continue

                shutil.copy(str(single_files[0]), str(single_file))
                shutil.copy(str(pair_files[0]), str(pair_file))

                fasta_cache = cache_dir / f"{h}.fasta"
                shutil.copy(str(fasta_file), str(fasta_cache))

                done += 1
                success = True
                print(f"  Cached: {single_file.name}, {pair_file.name}", flush=True)
                break

        if not success:
            failed += 1
            print(f"  FAILED after {max_retries} attempts", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Precompute complete:", flush=True)
    print(f"  Done:    {done}", flush=True)
    print(f"  Failed:  {failed}", flush=True)
    print(f"  Total cached now: {cached + done}", flush=True)
    print(f"{'=' * 60}", flush=True)

    if failed > 0:
        print(f"\n{failed} sequences failed. Re-run this script to retry ONLY the failed ones.", flush=True)
    else:
        print(f"\nAll done! Now run validate_baselines_bioemu.py — all embeddings are cached.", flush=True)


if __name__ == "__main__":
    main()