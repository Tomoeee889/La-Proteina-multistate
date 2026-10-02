#!/usr/bin/env python3
"""
Под счёт количества состояний в результатах BioEmu.

Считает:
1. n_states_xtc — количество кадров в samples.xtc
2. n_raw_npz — количество сырых сэмплов в batch_*.npz

Группирует по режимам, если может их определить:
- task1_default
- task1_sim_eps
- task2_default
- task2_sim_eps
- task3_default
- task3_sim_eps

Если имя папки не содержит task..., скрипт пытается восстановить режим
через mpnn_metrics_all.csv по SHA256-хэшу последовательности.
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==================== НАСТРОЙКИ ====================

BIOEMU_DIR = Path("/home/domain/aristowi/la-proteina-main/bioemu_baselines")

# CSV от предыдущего шага ProteinMPNN. Нужен, если папки BioEmu называются хэшами.
MPNN_CSV = Path("/home/domain/aristowi/la-proteina-main/mpnn_metrics_all.csv")

OUT_CSV = Path("/home/domain/aristowi/la-proteina-main/bioemu_states_summary.csv")

HIST_PLOT = Path("/home/domain/aristowi/la-proteina-main/bioemu_states_histogram.png")
BOX_PLOT = Path("/home/domain/aristowi/la-proteina-main/bioemu_states_boxplot.png")

# В какой группировке строить статистику и графики:
# "regime" — task1_default, task1_sim_eps, task2_default и т.д.
# "task"   — task1, task2, task3
GROUP_BY = "task"

# ==================== ЗАВИСИМОСТИ ====================

try:
    import MDAnalysis as mda
    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False
    print("WARNING: MDAnalysis не установлен. Буду считать только .npz, без .xtc states.")


# ==================== ХЕЛПЕРЫ ====================

def sha256_sequence(seq: str) -> str:
    """Считает SHA256 от последовательности, как это обычно делает ColabFold."""
    seq = seq.strip().upper()
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()


def read_fasta_sequence(fasta_path: Path) -> str:
    """Читает первую последовательность из FASTA."""
    seq_parts = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_parts:
                    break
                continue
            seq_parts.append(line)
    return "".join(seq_parts)


def infer_regime_from_text(text: str):
    """
    Пытается вытащить режим из имени папки/файла.
    Примеры:
      task1_default
      task2_sim_eps
      task3_default/noise_.../...
    """
    text = str(text)

    # Полный режим
    m = re.search(r"(task\d+_(?:default|sim_eps))", text)
    if m:
        return m.group(1)

    # Только task1/task2/task3
    m = re.search(r"(task\d+)", text)
    if m:
        return m.group(1)

    return None


def regime_to_task(regime: str) -> str:
    """
    task1_default -> task1
    task2_sim_eps -> task2
    unknown -> unknown
    """
    if not regime:
        return "unknown"

    m = re.match(r"(task\d+)", regime)
    if m:
        return m.group(1)

    return regime


def load_sequence_to_regime_mapping(csv_path: Path):
    """
    Строит mapping:
      SHA256(sequence) -> regime/task

    Используется, если папки BioEmu называются хэшами,
    а не именами вида task1_default/...
    """
    mapping = {}

    if not csv_path.exists():
        print(f"WARNING: CSV не найден: {csv_path}")
        print("Буду пытаться определять режим только по имени папки.")
        return mapping

    df = pd.read_csv(csv_path)

    # Если есть колонка is_best, берём только лучшие последовательности
    if "is_best" in df.columns:
        mask = df["is_best"].astype(str).str.lower().isin(["true", "1", "yes"])
        if mask.any():
            df = df[mask]

    required_cols = {"sequence"}
    if not required_cols.issubset(set(df.columns)):
        print(f"WARNING: В CSV нет колонки sequence: {csv_path}")
        return mapping

    for _, row in df.iterrows():
        seq = str(row.get("sequence", "")).strip().upper()
        if not seq:
            continue

        seq_hash = sha256_sequence(seq)

        regime = None

        # Колонка task из process_mpnn_all.py
        if "task" in row and pd.notna(row["task"]):
            regime = str(row["task"])

        # Или попробуем из unique_backbone_name
        if not regime and "unique_backbone_name" in row:
            regime = infer_regime_from_text(row["unique_backbone_name"])

        # Или из backbone_name
        if not regime and "backbone_name" in row:
            regime = infer_regime_from_text(row["backbone_name"])

        if regime:
            mapping[seq_hash] = regime

    print(f"Загружено маппингов последовательность -> режим: {len(mapping)}")
    return mapping


def count_raw_npz_samples(folder: Path) -> int:
    """
    Суммарное количество сырых сэмплов во всех batch_*.npz.
    """
    total = 0

    for npz_file in folder.glob("batch_*.npz"):
        try:
            data = np.load(npz_file, allow_pickle=True)
            if "pos" in data:
                total += int(data["pos"].shape[0])
        except Exception:
            pass

    return total


def count_xtc_states(folder: Path):
    """
    Количество состояний в samples.xtc.
    Возвращает (n_states, error_message).
    """
    xtc_path = folder / "samples.xtc"
    topo_path = folder / "topology.pdb"

    if not xtc_path.exists():
        return None, "samples.xtc not found"

    if not topo_path.exists():
        return None, "topology.pdb not found"

    if not HAS_MDANALYSIS:
        return None, "MDAnalysis not installed"

    try:
        u = mda.Universe(str(topo_path), str(xtc_path))
        n_states = len(u.trajectory)
        del u
        return n_states, None
    except Exception as e:
        return None, str(e)


# ==================== ОСНОВНОЙ ПРОЦЕСС ====================

def main():
    if not BIOEMU_DIR.exists():
        raise FileNotFoundError(f"Папка не найдена: {BIOEMU_DIR}")

    print("=" * 70)
    print("ПОДСЧЁТ СОСТОЯНИЙ BIOEMU")
    print("=" * 70)
    print(f"Папка с результатами: {BIOEMU_DIR}")
    print(f"CSV для маппинга режимов: {MPNN_CSV}")
    print(f"Группировка: {GROUP_BY}")
    print("=" * 70)

    seq_to_regime = load_sequence_to_regime_mapping(MPNN_CSV)

    folders = sorted([p for p in BIOEMU_DIR.iterdir() if p.is_dir()])
    print(f"Найдено папок с результатами: {len(folders)}")

    rows = []

    for folder in tqdm(folders, desc="Считаю состояния"):
        folder_name = folder.name

        # 1. Пробуем определить режим из имени папки
        regime = infer_regime_from_text(folder_name)

        # 2. Если не получилось, читаем sequence.fasta и ищем по хэшу
        sequence = ""
        seq_file = folder / "sequence.fasta"
        if seq_file.exists():
            sequence = read_fasta_sequence(seq_file)

        if not regime and sequence:
            seq_hash = sha256_sequence(sequence)
            regime = seq_to_regime.get(seq_hash)

        # 3. Если имя папки — это 64-символьный хэш, пробуем напрямую
        if not regime and len(folder_name) == 64:
            regime = seq_to_regime.get(folder_name)

        if not regime:
            regime = "unknown"

        task = regime_to_task(regime)

        # 4. Сырые сэмплы в .npz
        n_raw_npz = count_raw_npz_samples(folder)

        # 5. Финальные состояния в .xtc
        n_states_xtc, xtc_error = count_xtc_states(folder)

        rows.append(
            {
                "folder": folder_name,
                "path": str(folder),
                "regime": regime,
                "task": task,
                "n_raw_npz": n_raw_npz,
                "n_states_xtc": n_states_xtc,
                "has_xtc": int(n_states_xtc is not None),
                "xtc_error": xtc_error or "",
                "sequence_length": len(sequence) if sequence else None,
            }
        )

    df = pd.DataFrame(rows)

    # Сохраняем подробный CSV
    df.to_csv(OUT_CSV, index=False)
    print(f"\nПодробный CSV сохранён: {OUT_CSV}")

    # ==================== СТАТИСТИКА ====================

    print("\n" + "=" * 70)
    print("ОБЩАЯ СТАТИСТИКА")
    print("=" * 70)

    print(f"Всего папок: {len(df)}")
    print(f"Папок с samples.xtc: {df['has_xtc'].sum()}")
    print(f"Папок без samples.xtc: {(df['has_xtc'] == 0).sum()}")

    if df["n_states_xtc"].notna().any():
        print("\nСтатистика по финальным состояниям в .xtc:")
        print(df["n_states_xtc"].describe())
    else:
        print("\nФинальные состояния в .xtc не считались или MDAnalysis не установлен.")

    print("\nСтатистика по сырым сэмплам в .npz:")
    print(df["n_raw_npz"].describe())

    # ==================== СТАТИСТИКА ПО РЕЖИМАМ ====================

    group_col = GROUP_BY
    if group_col not in df.columns:
        group_col = "regime"

    print("\n" + "=" * 70)
    print(f"СТАТИСТИКА ПО ГРУППАМ: {group_col}")
    print("=" * 70)

    summary = (
        df.groupby(group_col)
        .agg(
            n_folders=("folder", "count"),
            n_with_xtc=("has_xtc", "sum"),
            mean_states_xtc=("n_states_xtc", "mean"),
            min_states_xtc=("n_states_xtc", "min"),
            max_states_xtc=("n_states_xtc", "max"),
            median_states_xtc=("n_states_xtc", "median"),
            mean_raw_npz=("n_raw_npz", "mean"),
            min_raw_npz=("n_raw_npz", "min"),
            max_raw_npz=("n_raw_npz", "max"),
        )
        .reset_index()
    )

    # Округляем для красоты
    for col in summary.columns:
        if summary[col].dtype == float:
            summary[col] = summary[col].round(3)

    print(summary.to_string(index=False))

    # Доля папок, где хотя бы 90 состояний
    if df["n_states_xtc"].notna().any():
        df["ge_90"] = df["n_states_xtc"] >= 90
        df["ge_95"] = df["n_states_xtc"] >= 95
        df["eq_100"] = df["n_states_xtc"] == 100

        frac_summary = (
            df.groupby(group_col)[["ge_90", "ge_95", "eq_100"]]
            .mean()
            .round(3)
            .reset_index()
        )

        print("\n" + "=" * 70)
        print("ДОЛЯ ХОРОШИХ АНСАМБЛЕЙ ПО ГРУППАМ")
        print("=" * 70)
        print("ge_90  — доля папок, где >= 90 состояний")
        print("ge_95  — доля папок, где >= 95 состояний")
        print("eq_100 — доля папок, где ровно 100 состояний")
        print()
        print(frac_summary.to_string(index=False))

    # ==================== ГРАФИКИ ====================

    # Для графиков используем xtc, если есть; иначе npz
    if df["n_states_xtc"].notna().any():
        plot_col = "n_states_xtc"
        plot_title = "Распределение количества состояний в samples.xtc"
    else:
        plot_col = "n_raw_npz"
        plot_title = "Распределение количества сырых сэмплов в .npz"
        print("\nWARNING: нет данных по .xtc, строю график по .npz.")

    plot_df = df[df[plot_col].notna()].copy()
    plot_df[plot_col] = plot_df[plot_col].astype(int)

    if plot_df.empty:
        print("\nНедостаточно данных для построения графиков.")
        return

    plt.figure(figsize=(12, 7))
    sns.histplot(
        data=plot_df,
        x=plot_col,
        hue=group_col,
        multiple="dodge",
        shrink=0.8,
        common_norm=False,
        palette="deep",
    )
    plt.title(plot_title, fontsize=14)
    plt.xlabel("Количество состояний / сэмплов")
    plt.ylabel("Число последовательностей")
    plt.legend(title=group_col)
    plt.tight_layout()
    plt.savefig(HIST_PLOT, dpi=200)
    plt.close()
    print(f"\nГистограмма сохранена: {HIST_PLOT}")

    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=plot_df,
        x=group_col,
        y=plot_col,
        palette="deep",
    )
    plt.title(f"Boxplot: {plot_title} по {group_col}", fontsize=14)
    plt.xlabel(group_col)
    plt.ylabel("Количество состояний / сэмплов")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(BOX_PLOT, dpi=200)
    plt.close()
    print(f"Boxplot сохранён: {BOX_PLOT}")

    print("\n" + "=" * 70)
    print("ГОТОВО")
    print("=" * 70)
    print(f"CSV:        {OUT_CSV}")
    print(f"Histogram:  {HIST_PLOT}")
    print(f"Boxplot:    {BOX_PLOT}")


if __name__ == "__main__":
    main()
