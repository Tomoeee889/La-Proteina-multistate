"""
Рисует ОДИН график для unconditional генерации (Task 1), ТОЛЬКО для режима
sim_eps (common random numbers / CRN): RMSD между A и B на одной оси Y,
sequence identity между A и B на второй оси Y (twin axis), оба — как функция
noise_scale.

Сначала пробует взять уже посчитанные данные из baselines/all_pairs_raw.csv
(если вы уже запускали make_all_plots.py) — это быстро. Если файла нет,
считает всё заново прямо из PDB (self-contained, ничего больше не нужно).

Запуск (из la-proteina-main/):
    python plot_unconditional_combined.py
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASELINES_DIR = Path("baselines")
OUT_PATH = BASELINES_DIR / "plots" / "task1_unconditional_sim_eps_rmsd_and_seqid.png"

# Только sim_eps — поправьте путь, если у вас папка называется иначе
TASK1_DIRS = {
    "sim_eps": BASELINES_DIR / "task1_unconditional_sim_eps",
}

RAW_CSV = BASELINES_DIR / "all_pairs_raw.csv"


# ----------------------------------------------------------------------
# Расчёт метрик из PDB (используется только если RAW_CSV не найден)
# ----------------------------------------------------------------------

def _get_three_to_one():
    try:
        from Bio.PDB.Polypeptide import three_to_one
        return three_to_one
    except ImportError:
        pass
    try:
        from Bio.SeqUtils import seq1
        return seq1
    except ImportError:
        from Bio.Data.PDBData import protein_letters_3to1 as _p3to1
        return lambda resname: _p3to1.get(resname.strip(), "X")


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    p, q = P - P.mean(axis=0), Q - Q.mean(axis=0)
    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    p_aligned = (R @ p.T).T
    return float(np.sqrt(np.mean(np.sum((p_aligned - q) ** 2, axis=1))))


def extract_ca_and_seq(pdb_path: Path, three_to_one):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))

    ca_coords, sequence = [], []
    for residue in structure.get_residues():
        if residue.id[0] != " " or "CA" not in residue:
            continue
        ca_coords.append(residue["CA"].get_vector().get_array())
        try:
            sequence.append(three_to_one(residue.resname))
        except (KeyError, ValueError):
            sequence.append("X")

    if not ca_coords:
        return None
    return np.array(ca_coords), "".join(sequence)


def parse_noise_scale(dirname: str):
    m = re.match(r"noise_([\d.]+)_job_(\d+)_n_(\d+)_id_(\d+)", dirname)
    return float(m.group(1)) if m else None


def compute_from_pdbs(mode: str, exp_dir: Path) -> pd.DataFrame:
    three_to_one = _get_three_to_one()
    rows = []

    for pair_dir in sorted(exp_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        noise_scale = parse_noise_scale(pair_dir.name)
        if noise_scale is None:
            continue

        pdb_A_list = list(pair_dir.glob("*_pathA.pdb"))
        if not pdb_A_list:
            continue
        pdb_A = pdb_A_list[0]
        pdb_B = pdb_A.with_name(pdb_A.name.replace("_pathA.pdb", "_pathB.pdb"))
        if not pdb_B.exists():
            continue

        res_A = extract_ca_and_seq(pdb_A, three_to_one)
        res_B = extract_ca_and_seq(pdb_B, three_to_one)
        if res_A is None or res_B is None:
            continue
        ca_A, seq_A = res_A
        ca_B, seq_B = res_B
        if len(seq_A) != len(seq_B):
            print(f"  [WARN] length mismatch in {pair_dir.name}, skipping")
            continue

        rows.append({
            "noise_scale": noise_scale,
            "full_rmsd": kabsch_rmsd(ca_A, ca_B),
            "seq_identity": sum(a == b for a, b in zip(seq_A, seq_B)) / len(seq_A),
            "mode": mode,
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Загрузка данных
# ----------------------------------------------------------------------

def load_task1_data() -> pd.DataFrame:
    if RAW_CSV.exists():
        print(f"Загружаю уже посчитанные данные из {RAW_CSV} ...")
        df = pd.read_csv(RAW_CSV)
        df = df[(df["task"] == "Task1_unconditional") & (df["mode"] == "sim_eps")].copy()
        if not df.empty:
            return df[["noise_scale", "full_rmsd", "seq_identity", "mode"]]
        print("  ...в CSV нет Task1_unconditional/sim_eps, считаю заново из PDB.")

    print("Считаю метрики заново прямо из PDB-файлов...")
    parts = []
    for mode, exp_dir in TASK1_DIRS.items():
        if not exp_dir.exists():
            print(f"  [SKIP] {exp_dir} не найдена")
            continue
        print(f"  Обрабатываю {exp_dir} (mode={mode}) ...")
        df = compute_from_pdbs(mode, exp_dir)
        print(f"    -> {len(df)} пар")
        parts.append(df)

    if not parts or all(p.empty for p in parts):
        raise RuntimeError(
            "Не нашёл данных ни в RAW_CSV, ни в TASK1_DIRS. "
            "Проверьте пути в TASK1_DIRS в начале скрипта."
        )
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------
# График
# ----------------------------------------------------------------------

def plot(df: pd.DataFrame, out_path: Path):
    df = df[df["mode"] == "sim_eps"].copy()
    assert not df.empty, "Нет данных с mode == 'sim_eps' — проверьте TASK1_DIRS / RAW_CSV"

    summary = (
        df.groupby("noise_scale")[["full_rmsd", "seq_identity"]]
        .agg(["mean", "std", "count"])
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index().sort_values("noise_scale")

    color_rmsd = "#E63946"
    color_seq = "#2E86AB"

    fig, ax_rmsd = plt.subplots(figsize=(9, 6))
    ax_seq = ax_rmsd.twinx()

    # RMSD — сплошная линия, левая ось
    ax_rmsd.errorbar(
        summary["noise_scale"], summary["full_rmsd_mean"], yerr=summary["full_rmsd_std"],
        marker="o", markersize=9, linewidth=2.5, capsize=4,
        color=color_rmsd, linestyle="-", label="RMSD",
    )

    # Sequence identity — пунктирная линия, правая ось
    ax_seq.errorbar(
        summary["noise_scale"], summary["seq_identity_mean"], yerr=summary["seq_identity_std"],
        marker="s", markersize=9, linewidth=2.5, capsize=4,
        color=color_seq, linestyle="--", label="Sequence identity",
    )

    ax_rmsd.set_xlabel("Noise scale (σ)", fontsize=14)
    ax_rmsd.set_ylabel("CA RMSD(A, B), Å", fontsize=14, color=color_rmsd)
    ax_rmsd.tick_params(axis="y", labelcolor=color_rmsd)

    ax_seq.set_ylabel("Sequence identity(A, B)", fontsize=14, color=color_seq)
    ax_seq.tick_params(axis="y", labelcolor=color_seq)
    ax_seq.set_ylim(0, 1.05)

    ax_rmsd.set_title(
        "Task 1 (unconditional, sim_eps / CRN): structure vs sequence divergence",
        fontsize=14, fontweight="bold",
    )
    ax_rmsd.grid(True, alpha=0.3)

    lines1, labels1 = ax_rmsd.get_legend_handles_labels()
    lines2, labels2 = ax_seq.get_legend_handles_labels()
    ax_rmsd.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Сохранено: {out_path} (+ .pdf)")

    print("\nСводная таблица:")
    print(summary[["noise_scale", "full_rmsd_mean", "full_rmsd_std",
                    "seq_identity_mean", "seq_identity_std", "full_rmsd_count"]]
          .round(3).to_string(index=False))


def main():
    df = load_task1_data()
    plot(df, OUT_PATH)


if __name__ == "__main__":
    main()
