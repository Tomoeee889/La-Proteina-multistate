"""
Единый скрипт: считает метрики (RMSD, sequence identity, а для conditional
задач ещё и разбивку motif/scaffold) по всем 6 бейзлайн-экспериментам и
рисует все графики. Не зависит от того, что и как было прогнано раньше
(analyze_baselines.py / aggregate_results.py) — всё считается заново с нуля
прямо из PDB-файлов.

Запуск (из la-proteina-main/):
    python make_all_plots.py

Если структура директорий/имена экспериментов у вас отличаются от дефолтных —
поправьте EXPERIMENTS и MOTIF_RANGES ниже.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# three_to_one совместимо с любой версией Biopython
# ----------------------------------------------------------------------
try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    try:
        from Bio.SeqUtils import seq1 as three_to_one
    except ImportError:
        from Bio.Data.PDBData import protein_letters_3to1 as _p3to1

        def three_to_one(resname):
            resname = resname.strip()
            return _p3to1.get(resname, "X")


# ----------------------------------------------------------------------
# Конфигурация — поправьте под себя, если что-то отличается
# ----------------------------------------------------------------------

BASELINES_DIR = Path("baselines")
OUT_DIR = BASELINES_DIR / "plots"

# (имя_папки_в_baselines, метка_задачи, метка_режима)
EXPERIMENTS = [
    ("task1_unconditional_default", "Task1_unconditional", "default"),
    ("task1_unconditional_sim_eps", "Task1_unconditional", "sim_eps"),
    ("task2_conditional_default", "Task2_conditional_small", "default"),
    ("task2_conditional_sim_eps", "Task2_conditional_small", "sim_eps"),
    ("task3_conditional_large_default", "Task3_conditional_large", "default"),
    ("task3_conditional_large_sim_eps", "Task3_conditional_large", "sim_eps"),
]

# Задачи, для которых считаем разбивку motif/scaffold (conditional)
CONDITIONAL_TASKS = {"Task2_conditional_small", "Task3_conditional_large"}

# Индексы мотива, 1-based, inclusive — из contig_string "A1-9/31/A41-158"
MOTIF_RANGES_1BASED = [(1, 9), (41, 158)]


def motif_index_set(ranges_1based):
    s = set()
    for start, end in ranges_1based:
        for i in range(start - 1, end):  # переводим в 0-based
            s.add(i)
    return s


MOTIF_INDICES = motif_index_set(MOTIF_RANGES_1BASED)


# ----------------------------------------------------------------------
# Геометрия / метрики
# ----------------------------------------------------------------------

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD между двумя облаками точек (N, 3) после оптимального совмещения (Kabsch)."""
    assert P.shape == Q.shape
    n = P.shape[0]
    if n == 0:
        return float("nan")

    p = P - P.mean(axis=0)
    q = Q - Q.mean(axis=0)

    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T

    p_aligned = (R @ p.T).T
    return float(np.sqrt(np.mean(np.sum((p_aligned - q) ** 2, axis=1))))


def extract_ca_and_seq(pdb_path: Path):
    """Извлекает Cα координаты и последовательность (1-буквенный код) из PDB."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("s", str(pdb_path))
    except Exception as e:
        print(f"  [ERROR] parsing {pdb_path}: {e}")
        return None

    ca_coords, sequence = [], []
    for residue in structure.get_residues():
        if residue.id[0] != " ":
            continue
        if "CA" in residue:
            ca_coords.append(residue["CA"].get_vector().get_array())
            try:
                sequence.append(three_to_one(residue.resname))
            except (KeyError, ValueError):
                sequence.append("X")

    if len(ca_coords) == 0:
        return None

    return {
        "ca": np.array(ca_coords),
        "sequence": "".join(sequence),
        "length": len(sequence),
    }


def analyze_pair(pdb_A: Path, pdb_B: Path, use_motif: bool):
    """Считает full/motif/scaffold RMSD и sequence identity для одной пары A/B."""
    data_A = extract_ca_and_seq(pdb_A)
    data_B = extract_ca_and_seq(pdb_B)
    if data_A is None or data_B is None:
        return None
    if data_A["length"] != data_B["length"]:
        print(f"  [WARN] length mismatch {pdb_A.name}={data_A['length']} vs {pdb_B.name}={data_B['length']}, skipping pair")
        return None

    n = data_A["length"]
    seq_A, seq_B = data_A["sequence"], data_B["sequence"]

    result = {
        "length": n,
        "full_rmsd": kabsch_rmsd(data_A["ca"], data_B["ca"]),
        "seq_identity": sum(a == b for a, b in zip(seq_A, seq_B)) / n,
    }

    if use_motif:
        motif_idx = np.array([i in MOTIF_INDICES and i < n for i in range(n)])
        scaffold_idx = ~motif_idx

        if motif_idx.any():
            result["motif_rmsd"] = kabsch_rmsd(data_A["ca"][motif_idx], data_B["ca"][motif_idx])
            seq_A_motif = "".join(c for c, m in zip(seq_A, motif_idx) if m)
            seq_B_motif = "".join(c for c, m in zip(seq_B, motif_idx) if m)
            result["motif_seq_identity"] = sum(a == b for a, b in zip(seq_A_motif, seq_B_motif)) / len(seq_A_motif)
        else:
            result["motif_rmsd"] = float("nan")
            result["motif_seq_identity"] = float("nan")

        if scaffold_idx.any():
            result["scaffold_rmsd"] = kabsch_rmsd(data_A["ca"][scaffold_idx], data_B["ca"][scaffold_idx])
            seq_A_scaf = "".join(c for c, m in zip(seq_A, scaffold_idx) if m)
            seq_B_scaf = "".join(c for c, m in zip(seq_B, scaffold_idx) if m)
            result["scaffold_seq_identity"] = sum(a == b for a, b in zip(seq_A_scaf, seq_B_scaf)) / len(seq_A_scaf)
        else:
            result["scaffold_rmsd"] = float("nan")
            result["scaffold_seq_identity"] = float("nan")

    return result


def parse_noise_scale(dirname: str):
    m = re.match(r"noise_([\d.]+)_job_(\d+)_n_(\d+)_id_(\d+)", dirname)
    if m:
        return float(m.group(1))
    return None


def analyze_experiment_dir(exp_dir: Path, use_motif: bool) -> pd.DataFrame:
    """Проходит по всем поддиректориям с парами A/B и считает метрики."""
    rows = []
    for pair_dir in sorted(exp_dir.iterdir()):
        if not pair_dir.is_dir():
            continue

        noise_scale = parse_noise_scale(pair_dir.name)
        if noise_scale is None:
            continue

        pdb_files_A = list(pair_dir.glob("*_pathA.pdb"))
        if not pdb_files_A:
            continue
        pdb_A = pdb_files_A[0]
        pdb_B = pdb_A.with_name(pdb_A.name.replace("_pathA.pdb", "_pathB.pdb"))
        if not pdb_B.exists():
            continue

        res = analyze_pair(pdb_A, pdb_B, use_motif)
        if res is None:
            continue

        res["noise_scale"] = noise_scale
        res["pair"] = pair_dir.name
        rows.append(res)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Основной сбор данных по всем 6 экспериментам
# ----------------------------------------------------------------------

def collect_all_experiments() -> pd.DataFrame:
    all_rows = []

    for dir_name, task, mode in EXPERIMENTS:
        exp_dir = BASELINES_DIR / dir_name
        if not exp_dir.exists():
            print(f"[SKIP] {exp_dir} not found, skipping")
            continue

        use_motif = task in CONDITIONAL_TASKS
        print(f"\n>>> Analyzing {dir_name} (task={task}, mode={mode}, motif_breakdown={use_motif})")

        df = analyze_experiment_dir(exp_dir, use_motif)
        if df.empty:
            print(f"  [WARN] no valid pairs found in {exp_dir}")
            continue

        df["task"] = task
        df["mode"] = mode
        n_scales = df["noise_scale"].nunique()
        print(f"  Found {len(df)} pairs across {n_scales} noise_scale values")
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(
            "No data found in any experiment directory! "
            "Check that BASELINES_DIR / EXPERIMENTS match your actual folder names."
        )

    return pd.concat(all_rows, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует per-pair метрики в mean/std/median/count по (task, mode, noise_scale)."""
    metric_cols = [c for c in [
        "full_rmsd", "seq_identity", "motif_rmsd", "scaffold_rmsd",
        "motif_seq_identity", "scaffold_seq_identity",
    ] if c in df.columns]

    agg = df.groupby(["task", "mode", "noise_scale"])[metric_cols].agg(["mean", "std", "median", "count"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()


# ----------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "font.family": "DejaVu Sans",
})

C_DEFAULT = "#2E86AB"
C_SIM_EPS = "#E63946"
MODE_STYLE = {"default": (C_DEFAULT, "o"), "sim_eps": (C_SIM_EPS, "s")}


def plot_task1(summary: pd.DataFrame, out_dir: Path):
    task1 = summary[summary["task"] == "Task1_unconditional"]
    if task1.empty:
        print("[SKIP plot] no Task1_unconditional data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for mode, (color, marker) in MODE_STYLE.items():
        sub = task1[task1["mode"] == mode].sort_values("noise_scale")
        if sub.empty:
            continue

        ax1.plot(sub["noise_scale"], sub["full_rmsd_mean"], marker=marker,
                 linewidth=2.5, markersize=9, color=color, label=mode)
        ax1.fill_between(sub["noise_scale"],
                          sub["full_rmsd_mean"] - sub["full_rmsd_std"],
                          sub["full_rmsd_mean"] + sub["full_rmsd_std"],
                          alpha=0.15, color=color)

        ax2.plot(sub["noise_scale"], sub["seq_identity_mean"], marker=marker,
                  linewidth=2.5, markersize=9, color=color, label=mode)
        ax2.fill_between(sub["noise_scale"],
                          sub["seq_identity_mean"] - sub["seq_identity_std"],
                          sub["seq_identity_mean"] + sub["seq_identity_std"],
                          alpha=0.15, color=color)

    ax1.set_xlabel("Noise Scale")
    ax1.set_ylabel("Full RMSD (Å)")
    ax1.set_title("Task 1: Unconditional — RMSD")
    ax1.legend(title="Mode", loc="best")

    ax2.set_xlabel("Noise Scale")
    ax2.set_ylabel("Sequence Identity")
    ax2.set_title("Task 1: Unconditional — Sequence")
    ax2.legend(title="Mode", loc="best")
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    path = out_dir / "plot_task1_unconditional.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


def plot_task23(summary: pd.DataFrame, out_dir: Path):
    task23 = summary[summary["task"].isin(["Task2_conditional_small", "Task3_conditional_large"])]
    if task23.empty:
        print("[SKIP plot] no Task2/Task3 data")
        return
    if "scaffold_rmsd_mean" not in task23.columns:
        print("[SKIP plot] scaffold_rmsd not available (should not happen with this script)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for mode, (color, marker) in MODE_STYLE.items():
        sub = task23[task23["mode"] == mode].sort_values("noise_scale")
        if sub.empty:
            continue

        ax1.semilogy(sub["noise_scale"], np.maximum(sub["scaffold_rmsd_mean"], 1e-3),
                      marker=marker, linewidth=2.5, markersize=9, color=color, label=mode)
        ax1.fill_between(sub["noise_scale"],
                          np.maximum(sub["scaffold_rmsd_mean"] - sub["scaffold_rmsd_std"], 1e-4),
                          sub["scaffold_rmsd_mean"] + sub["scaffold_rmsd_std"],
                          alpha=0.15, color=color)

        ax2.plot(sub["noise_scale"], sub["scaffold_seq_identity_mean"],
                  marker=marker, linewidth=2.5, markersize=9, color=color, label=mode)
        ax2.fill_between(sub["noise_scale"],
                          sub["scaffold_seq_identity_mean"] - sub["scaffold_seq_identity_std"],
                          sub["scaffold_seq_identity_mean"] + sub["scaffold_seq_identity_std"],
                          alpha=0.15, color=color)

    ax1.set_xlabel("Noise Scale")
    ax1.set_ylabel("Scaffold RMSD (Å, log scale)")
    ax1.set_title("Task 2+3: Conditional — Scaffold RMSD")
    ax1.axvline(x=1.5, color="gray", linestyle="--", alpha=0.4, label="Task 2 / Task 3 boundary")
    ax1.legend(title="Mode", loc="best")

    ax2.set_xlabel("Noise Scale")
    ax2.set_ylabel("Scaffold Sequence Identity")
    ax2.set_title("Task 2+3: Conditional — Scaffold Sequence")
    ax2.axvline(x=1.5, color="gray", linestyle="--", alpha=0.4)
    ax2.legend(title="Mode", loc="best")
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    path = out_dir / "plot_task23_conditional.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


def plot_comparison_all(summary: pd.DataFrame, out_dir: Path):
    task1 = summary[(summary["task"] == "Task1_unconditional") & (summary["mode"] == "sim_eps")].sort_values("noise_scale")
    task23 = summary[summary["task"].isin(["Task2_conditional_small", "Task3_conditional_large"]) & (summary["mode"] == "sim_eps")].sort_values("noise_scale")

    if task1.empty and task23.empty:
        print("[SKIP plot] no sim_eps data for comparison plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    if not task1.empty:
        ax.plot(task1["noise_scale"], task1["full_rmsd_mean"], marker="o",
                linewidth=3, markersize=10, color=C_DEFAULT,
                label="Task 1: Unconditional (full RMSD)")

    if not task23.empty and "scaffold_rmsd_mean" in task23.columns:
        ax.plot(task23["noise_scale"], task23["scaffold_rmsd_mean"], marker="s",
                linewidth=3, markersize=10, color=C_SIM_EPS,
                label="Task 2+3: Conditional (scaffold RMSD)")

    if not task23.empty and "motif_rmsd_mean" in task23.columns:
        ax.plot(task23["noise_scale"], task23["motif_rmsd_mean"], marker="^",
                linewidth=2, markersize=8, color="#F18F01", linestyle="--",
                label="Task 2+3: Motif RMSD (fixed)")

    ax.set_xlabel("Noise Scale", fontsize=15)
    ax.set_ylabel("RMSD (Å)", fontsize=15)
    ax.set_title("Sensitivity to Initial Conditions: All Tasks (CRN / sim_eps mode)", fontsize=16, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)

    fig.tight_layout()
    path = out_dir / "plot_comparison_all.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


def plot_paradox(summary: pd.DataFrame, out_dir: Path):
    task23_sim = summary[
        summary["task"].isin(["Task2_conditional_small", "Task3_conditional_large"])
        & (summary["mode"] == "sim_eps")
    ].sort_values("noise_scale")

    if task23_sim.empty or "scaffold_rmsd_mean" not in task23_sim.columns:
        print("[SKIP plot] no data for structure-vs-sequence paradox plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(task23_sim["noise_scale"], task23_sim["scaffold_rmsd_mean"], marker="o",
            linewidth=3, markersize=10, color=C_SIM_EPS, label="Scaffold RMSD (structure)")

    ax2 = ax.twinx()
    ax2.plot(task23_sim["noise_scale"], task23_sim["scaffold_seq_identity_mean"], marker="s",
              linewidth=3, markersize=10, color=C_DEFAULT, label="Scaffold Seq Identity (sequence)")

    ax.set_xlabel("Noise Scale", fontsize=15)
    ax.set_ylabel("Scaffold RMSD (Å)", fontsize=15, color=C_SIM_EPS)
    ax2.set_ylabel("Scaffold Sequence Identity", fontsize=15, color=C_DEFAULT)
    ax.tick_params(axis="y", labelcolor=C_SIM_EPS)
    ax2.tick_params(axis="y", labelcolor=C_DEFAULT)
    ax2.set_ylim(0, 1.05)

    ax.set_title("Structure vs Sequence Divergence", fontsize=16, fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=11)

    fig.tight_layout()
    path = out_dir / "plot_paradox_structure_vs_sequence.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


def plot_default_vs_sim_eps(summary: pd.DataFrame, out_dir: Path):
    task1 = summary[summary["task"] == "Task1_unconditional"]
    task23 = summary[summary["task"].isin(["Task2_conditional_small", "Task3_conditional_large"])]

    if task1.empty and task23.empty:
        print("[SKIP plot] no data for default_vs_sim_eps plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels = {"default": "default (SDE noise dominates)", "sim_eps": "sim_eps (clean signal, CRN)"}
    for mode, (color, _) in MODE_STYLE.items():
        sub = task1[task1["mode"] == mode].sort_values("noise_scale")
        if not sub.empty:
            ax1.plot(sub["noise_scale"], sub["full_rmsd_mean"], marker="o",
                      linewidth=2.5, markersize=9, color=color, label=labels[mode])

    ax1.set_xlabel("Noise Scale")
    ax1.set_ylabel("Full RMSD (Å)")
    ax1.set_title("Task 1: Why CRN (sim_eps) is Critical")
    ax1.legend(loc="best")

    for mode, (color, _) in MODE_STYLE.items():
        sub = task23[task23["mode"] == mode].sort_values("noise_scale")
        if not sub.empty and "scaffold_rmsd_mean" in sub.columns:
            ax2.semilogy(sub["noise_scale"], np.maximum(sub["scaffold_rmsd_mean"], 1e-3),
                          marker="o", linewidth=2.5, markersize=9, color=color, label=labels[mode])

    ax2.set_xlabel("Noise Scale")
    ax2.set_ylabel("Scaffold RMSD (Å, log scale)")
    ax2.set_title("Task 2+3: Conditional — default vs sim_eps")
    ax2.legend(loc="best")

    fig.tight_layout()
    path = out_dir / "plot_default_vs_sim_eps.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Collecting data from all experiments...")
    print("=" * 70)
    raw_df = collect_all_experiments()

    raw_csv = BASELINES_DIR / "all_pairs_raw.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"\nSaved raw per-pair data: {raw_csv}")

    summary = summarize(raw_df)
    summary_csv = BASELINES_DIR / "all_experiments_comparison.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved aggregated summary: {summary_csv}")

    print("\n" + "=" * 70)
    print("Building plots...")
    print("=" * 70)
    plot_task1(summary, OUT_DIR)
    plot_task23(summary, OUT_DIR)
    plot_comparison_all(summary, OUT_DIR)
    plot_paradox(summary, OUT_DIR)
    plot_default_vs_sim_eps(summary, OUT_DIR)

    print(f"\nГотово! Все графики в {OUT_DIR}/")


if __name__ == "__main__":
    main()