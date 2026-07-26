#!/usr/bin/env python3
"""
plot_baselines.py

Сканирует baselines/ и рисует два простых графика:
  1. CA-RMSD между путями A и B vs noise_scale
  2. Sequence identity между A и B vs noise_scale

Каждый график — 3 сабплота (task1, task2, task3),
на каждом — две линии: default vs sim_eps.

Использование:
  python plot_baselines.py [--baselines_dir baselines] [--output_dir baselines/plots]
"""

import os
import re
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Шрифты ──
plt.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]


# ===========================================================================
#  PDB PARSING — извлекаем CA-координаты и последовательность
# ===========================================================================

def parse_pdb_ca(pdb_path):
    """Парсит PDB, возвращает (ca_coords [N,3], sequence [N]).

    Берёт только CA-атомы из ATOM-записей.
    """
    ca_coords = []
    seq = []

    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    resname = line[17:20].strip()
                    ca_coords.append([x, y, z])
                    seq.append(three_to_one.get(resname, "X"))

    return np.array(ca_coords), np.array(seq)


# ===========================================================================
#  RMSD — с выравниванием Кабша
# ===========================================================================

def kabsch_rmsd(P, Q):
    """RMSD между двумя наборами точек P и Q [N,3] после оптимального
    выравнивания (Кабш). Предполагается, что центры масс уже совпадают."""
    # Центрируем
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    # Kabsch
    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    P_rot = P_c @ R.T
    rmsd = np.sqrt(np.mean(np.sum((P_rot - Q_c) ** 2, axis=1)))
    return rmsd


def ca_rmsd_from_pdb(pdb_a, pdb_b):
    """CA-RMSD между двумя PDB-файлами."""
    ca_a, seq_a = parse_pdb_ca(pdb_a)
    ca_b, seq_b = parse_pdb_ca(pdb_b)

    n = min(len(ca_a), len(ca_b))
    if n == 0:
        return None, None

    rmsd = kabsch_rmsd(ca_a[:n], ca_b[:n])
    return rmsd, n


# ===========================================================================
#  SEQUENCE IDENTITY
# ===========================================================================

def sequence_identity(seq_a, seq_b):
    """Процент идентичных остатков (Hamming). 
    Берёт минимальную длину."""
    n = min(len(seq_a), len(seq_b))
    if n == 0:
        return None
    matches = np.sum(seq_a[:n] == seq_b[:n])
    return matches / n * 100.0


def seq_identity_from_pdb(pdb_a, pdb_b):
    """Sequence identity между двумя PDB-файлами."""
    _, seq_a = parse_pdb_ca(pdb_a)
    _, seq_b = parse_pdb_ca(pdb_b)
    return sequence_identity(seq_a, seq_b)


# ===========================================================================
#  СКАНИРОВАНИЕ ДИРЕКТОРИЙ
# ===========================================================================

# Паттерн имени папки: noise_0.30_job_0_n_128_id_0
NOISE_PATTERN = re.compile(r"noise_(\d+\.\d+)_job_\d+_n_\d+_id_\d+")


def scan_baselines(baselines_dir):
    """Сканирует baselines/ и возвращает структуру:
    
    results[task][mode][noise_scale] = list of (rmsd, seq_identity)
    
    task: "task1", "task2", "task3"
    mode: "default", "sim_eps"
    noise_scale: float
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    if not os.path.isdir(baselines_dir):
        print(f"ERROR: {baselines_dir} not found")
        sys.exit(1)

    # Ищем папки вида task1_default, task2_sim_eps, и т.д.
    for entry in sorted(os.listdir(baselines_dir)):
        entry_path = os.path.join(baselines_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        # Парсим task_mode
        parts = entry.split("_")
        if len(parts) < 2:
            continue

        # task1_default → task="task1", mode="default"
        # task2_sim_eps → task="task2", mode="sim_eps"
        task = parts[0]  # "task1"
        mode = "_".join(parts[1:])  # "default" или "sim_eps"

        if not task.startswith("task"):
            continue

        # Сканируем подпапки с шумами
        for sample_dir in sorted(os.listdir(entry_path)):
            sample_path = os.path.join(entry_path, sample_dir)
            if not os.path.isdir(sample_path):
                continue

            m = NOISE_PATTERN.match(sample_dir)
            if not m:
                continue

            noise_scale = float(m.group(1))

            # Ищем pathA.pdb и pathB.pdb
            pdb_a = os.path.join(sample_path, f"{sample_dir}_pathA.pdb")
            pdb_b = os.path.join(sample_path, f"{sample_dir}_pathB.pdb")

            # Если нет pathB — пропускаем (не dual-path)
            if not os.path.isfile(pdb_a) or not os.path.isfile(pdb_b):
                continue

            # Считаем
            rmsd, n = ca_rmsd_from_pdb(pdb_a, pdb_b)
            seq_id = seq_identity_from_pdb(pdb_a, pdb_b)

            if rmsd is not None and seq_id is not None:
                results[task][mode][noise_scale].append((rmsd, seq_id))

    return results


# ===========================================================================
#  ПЛОТЫ
# ===========================================================================

TASKS = ["task1", "task2", "task3"]
TASK_TITLES = {
    "task1": "Task 1: Unconditional",
    "task2": "Task 2: Motif (low noise)",
    "task3": "Task 3: Motif (high noise)",
}
MODE_COLORS = {
    "default": "#0279EE",   # синий
    "sim_eps": "#FF9400",   # оранжевый
}
MODE_LABELS = {
    "default": "default (indep. RNG)",
    "sim_eps": "sim_eps (shared RNG)",
}

# Для общего графика: цвет по таску, стиль линии по режиму
TASK_COLORS = {
    "task1": "#0279EE",   # синий
    "task2": "#75A025",   # зелёный
    "task3": "#FD9BED",   # розовый
}
MODE_LINESTYLES = {
    "default": "-",    # сплошная
    "sim_eps": "--",   # пунктир
}
TASK_MARKERS = {
    "task1": "o",
    "task2": "s",
    "task3": "^",
}


def aggregate(values):
    """Возвращает (mean, std) из списка (rmsd, seq_id)."""
    if not values:
        return None, None, 0
    rmsds = [v[0] for v in values]
    seq_ids = [v[1] for v in values]
    return (
        np.mean(rmsds),
        np.std(rmsds),
        np.mean(seq_ids),
        np.std(seq_ids),
        len(values),
    )


def plot_metric(results, metric_idx, ylabel, title, output_path):
    """
    metric_idx: 0 = RMSD, 1 = seq_identity
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for ax_idx, task in enumerate(TASKS):
        ax = axes[ax_idx]
        ax.set_title(TASK_TITLES[task], fontsize=11)
        ax.set_xlabel("Noise scale")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        has_data = False
        for mode in ["default", "sim_eps"]:
            if task not in results or mode not in results[task]:
                continue

            noise_data = results[task][mode]
            if not noise_data:
                continue

            scales = sorted(noise_data.keys())
            means = []
            stds = []
            for s in scales:
                vals = noise_data[s]
                metric_vals = [v[metric_idx] for v in vals]
                means.append(np.mean(metric_vals))
                stds.append(np.std(metric_vals))

            ax.errorbar(
                scales, means, yerr=stds,
                marker="o", markersize=5, capsize=3,
                color=MODE_COLORS[mode],
                label=MODE_LABELS[mode],
                linewidth=1.5,
            )
            has_data = True

        if has_data:
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_scatter(results, output_path):
    """Дополнительный scatter-плот: RMSD vs seq_identity, каждый режим своим цветом."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("RMSD vs Sequence Identity (per sample)", fontsize=14, fontweight="bold", y=1.02)

    for ax_idx, task in enumerate(TASKS):
        ax = axes[ax_idx]
        ax.set_title(TASK_TITLES[task], fontsize=11)
        ax.set_xlabel("Sequence identity (%)")
        ax.set_ylabel("CA-RMSD (Å)")
        ax.grid(True, alpha=0.3)

        for mode in ["default", "sim_eps"]:
            if task not in results or mode not in results[task]:
                continue

            all_rmsds = []
            all_seqs = []
            for noise_scale, vals in results[task][mode].items():
                for rmsd, seq_id in vals:
                    all_rmsds.append(rmsd)
                    all_seqs.append(seq_id)

            if all_rmsds:
                ax.scatter(
                    all_seqs, all_rmsds,
                    s=15, alpha=0.5,
                    color=MODE_COLORS[mode],
                    label=MODE_LABELS[mode],
                )

        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ===========================================================================
#  ОБЩИЙ ГРАФИК — все таски на одних осях
# ===========================================================================

def plot_overview(results, output_path):
    """Один общий график: 2 панели (RMSD и seq identity) + scatter.
    Все таски на одних осях, цвет=таск, стиль линии=режим.
    X-axis логарифмический (т.к. task3 уходит в 2–100)."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Overview: All tasks — paths A vs B divergence",
                 fontsize=14, fontweight="bold", y=1.02)

    # --- Панель 1: CA-RMSD vs noise ---
    ax = axes[0]
    ax.set_title("CA-RMSD vs noise scale", fontsize=11)
    ax.set_xlabel("Noise scale (log)")
    ax.set_ylabel("CA-RMSD (Å)")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.grid(True, alpha=0.3, which="both")

    for task in TASKS:
        if task not in results:
            continue
        for mode in ["default", "sim_eps"]:
            if mode not in results[task]:
                continue
            noise_data = results[task][mode]
            if not noise_data:
                continue
            scales = sorted(noise_data.keys())
            means = [np.mean([v[0] for v in noise_data[s]]) for s in scales]
            stds = [np.std([v[0] for v in noise_data[s]]) for s in scales]
            label = f"{task} {mode}"
            ax.errorbar(
                scales, means, yerr=stds,
                marker=TASK_MARKERS[task], markersize=5, capsize=3,
                color=TASK_COLORS[task],
                linestyle=MODE_LINESTYLES[mode],
                linewidth=1.5, alpha=0.85,
                label=label,
            )
    ax.legend(fontsize=7, loc="upper left")

    # --- Панель 2: Seq identity vs noise ---
    ax = axes[1]
    ax.set_title("Sequence identity vs noise scale", fontsize=11)
    ax.set_xlabel("Noise scale (log)")
    ax.set_ylabel("Sequence identity (%)")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.grid(True, alpha=0.3, which="both")

    for task in TASKS:
        if task not in results:
            continue
        for mode in ["default", "sim_eps"]:
            if mode not in results[task]:
                continue
            noise_data = results[task][mode]
            if not noise_data:
                continue
            scales = sorted(noise_data.keys())
            means = [np.mean([v[1] for v in noise_data[s]]) for s in scales]
            stds = [np.std([v[1] for v in noise_data[s]]) for s in scales]
            label = f"{task} {mode}"
            ax.errorbar(
                scales, means, yerr=stds,
                marker=TASK_MARKERS[task], markersize=5, capsize=3,
                color=TASK_COLORS[task],
                linestyle=MODE_LINESTYLES[mode],
                linewidth=1.5, alpha=0.85,
                label=label,
            )
    ax.legend(fontsize=7, loc="lower left")

    # --- Панель 3: Scatter RMSD vs seq identity (все точки) ---
    ax = axes[2]
    ax.set_title("RMSD vs sequence identity (all samples)", fontsize=11)
    ax.set_xlabel("Sequence identity (%)")
    ax.set_ylabel("CA-RMSD (Å)")
    ax.grid(True, alpha=0.3)

    for task in TASKS:
        if task not in results:
            continue
        for mode in ["default", "sim_eps"]:
            if mode not in results[task]:
                continue
            all_rmsds = []
            all_seqs = []
            for noise_scale, vals in results[task][mode].items():
                for rmsd, seq_id in vals:
                    all_rmsds.append(rmsd)
                    all_seqs.append(seq_id)
            if all_rmsds:
                ax.scatter(
                    all_seqs, all_rmsds,
                    s=12, alpha=0.35,
                    color=TASK_COLORS[task],
                    marker=TASK_MARKERS[task],
                    label=f"{task} {mode}",
                )
    ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot baseline results")
    parser.add_argument(
        "--baselines_dir", type=str, default="baselines",
        help="Directory with task1_default/, task2_sim_eps/, etc.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="baselines/plots",
        help="Where to save plots",
    )
    args = parser.parse_args()

    print(f"Scanning {args.baselines_dir} ...")
    results = scan_baselines(args.baselines_dir)

    # Печать сводки
    print("\n=== Summary ===")
    for task in TASKS:
        for mode in ["default", "sim_eps"]:
            if task in results and mode in results[task]:
                noise_data = results[task][mode]
                total = sum(len(v) for v in noise_data.values())
                scales = sorted(noise_data.keys())
                print(f"  {task}/{mode}: {total} samples, scales={scales}")

    # Создаём выходную папку
    os.makedirs(args.output_dir, exist_ok=True)

    # График 1: CA-RMSD vs noise_scale
    print("\nPlotting CA-RMSD ...")
    plot_metric(
        results,
        metric_idx=0,
        ylabel="CA-RMSD (Å)",
        title="CA-RMSD between paths A and B vs noise scale",
        output_path=os.path.join(args.output_dir, "rmsd_vs_noise.png"),
    )

    # График 2: Sequence identity vs noise_scale
    print("Plotting sequence identity ...")
    plot_metric(
        results,
        metric_idx=1,
        ylabel="Sequence identity (%)",
        title="Sequence identity between paths A and B vs noise scale",
        output_path=os.path.join(args.output_dir, "seq_identity_vs_noise.png"),
    )

    # График 3: Scatter RMSD vs seq_identity
    print("Plotting scatter ...")
    plot_scatter(
        results,
        output_path=os.path.join(args.output_dir, "rmsd_vs_seqid_scatter.png"),
    )

    # График 4: Общий обзор — все таски на одних осях
    print("Plotting overview ...")
    plot_overview(
        results,
        output_path=os.path.join(args.output_dir, "overview_all_tasks.png"),
    )

    # Сохраняем числовые результаты в CSV
    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("task,mode,noise_scale,n_samples,rmsd_mean,rmsd_std,seqid_mean,seqid_std\n")
        for task in TASKS:
            for mode in ["default", "sim_eps"]:
                if task not in results or mode not in results[task]:
                    continue
                for noise_scale in sorted(results[task][mode].keys()):
                    vals = results[task][mode][noise_scale]
                    rmsds = [v[0] for v in vals]
                    seqids = [v[1] for v in vals]
                    f.write(
                        f"{task},{mode},{noise_scale},{len(vals)},"
                        f"{np.mean(rmsds):.4f},{np.std(rmsds):.4f},"
                        f"{np.mean(seqids):.2f},{np.std(seqids):.2f}\n"
                    )
    print(f"\n  Saved: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
