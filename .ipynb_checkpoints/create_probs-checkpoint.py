import numpy
import os
import subprocess

INFERENCE_DIR = "/home/domain/aristowi/la-proteina-main/inference/inference_ucond_notri"
MPNN_SCRIPT = "/home/domain/aristowi/ProteinMPNN/protein_mpnn_run.py"
PARSED_A_DIR = "/home/domain/aristowi/la-proteina-main/mpnn_input/pdbs_A"
PARSED_B_DIR = "/home/domain/aristowi/la-proteina-main/mpnn_input/pdbs_B"
OUTPUT_DIR = "/home/domain/aristowi/la-proteina-main/mpnn_probs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "2"  # GPU 0
env["OMP_NUM_THREADS"] = "8"      # количество CPU ядер

for sample_name in sorted(os.listdir(INFERENCE_DIR)):
    sample_dir = os.path.join(INFERENCE_DIR, sample_name)
    if not os.path.isdir(sample_dir):
        continue

    out_A = os.path.join(OUTPUT_DIR, sample_name + "_A")
    out_B = os.path.join(OUTPUT_DIR, sample_name + "_B")
    os.makedirs(out_A, exist_ok=True)
    os.makedirs(out_B, exist_ok=True)

    pdb_a = os.path.join(PARSED_A_DIR, sample_name + "_pathA.pdb")
    pdb_b = os.path.join(PARSED_B_DIR, sample_name + "_pathB.pdb")

    if not os.path.exists(pdb_a) or not os.path.exists(pdb_b):
        continue

    for pdb, out in [(pdb_a, out_A), (pdb_b, out_B)]:
        subprocess.run([
            "python", MPNN_SCRIPT,
            "--pdb_path", pdb,
            "--out_folder", out,
            "--unconditional_probs_only", "1",
            "--save_probs", "1",
        ], env=env)

print("Готово")