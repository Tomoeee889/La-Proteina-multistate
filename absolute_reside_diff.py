"""
Считает АБСОЛЮТНОЕ число отличающихся остатков (не %, а именно штуки) между
path A и path B, по каждому noise_scale, для всех 6 экспериментов
(task1/2/3 x default/sim_eps). Для conditional задач (task2/task3) отдельно
считает разбивку: сколько отличий в motif-части и сколько в scaffold-части.

Сначала пробует взять уже посчитанные PDB-парсинги из baselines/all_pairs_raw.csv
(там есть колонка length + seq_identity, из них точное число отличий
восстанавливается однозначно: n_diff = length - round(seq_identity * length)).
Если файла нет — считает всё заново прямо из PDB (self-contained).

Запуск (из la-proteina-main/):
    python absolute_residue_differences.py
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASELINES_DIR = Path("baselines")
RAW_CSV = BASELINES_DIR / "all_pairs_raw.csv"
OUT_CSV = BASELINES_DIR / "residue_differences_summary.csv"
OUT_CSV_RAW = BASELINES_DIR / "residue_differences_raw.csv"

EXPERIMENTS = [
    ("task1_unconditional_default", "Task1_unconditional", "default"),
    ("task1_unconditional_sim_eps", "Task1_unconditional", "sim_eps"),
    ("task2_conditional_default", "Task2_conditional_small", "default"),
    ("task2_conditional_sim_eps", "Task2_conditional_small", "sim_eps"),
    ("task3_conditional_large_default", "Task3_conditional_large", "default"),
    ("task3_conditional_large_sim_eps", "Task3_conditional_large", "sim_eps"),
]

CONDITIONAL_TASKS = {"Task2_conditional_small", "Task3_conditional_large"}
MOTIF_RANGES_1BASED = [(1, 9), (41, 158)]


def motif_index_set(ranges_1based):
    s = set()
    for start, end in ranges_1based:
        for i in range(start - 1, end):
            s.add(i)
    return s


MOTIF_INDICES = motif_index_set(MOTIF_RANGES_1BASED)


# ----------------------------------------------------------------------
# Self-contained расчёт напрямую из PDB (точный подсчёт отличий, без округлений)
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


def extract_seq(pdb_path: Path, three_to_one) -> str:
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    seq = []
    for residue in structure.get_residues():
        if residue.id[0] != " " or "CA" not in residue:
            continue
        try:
            seq.append(three_to_one(residue.resname))
        except (KeyError, ValueError):
            seq.append("X")
    return "".join(seq)


def parse_noise_scale(dirname: str):
    m = re.match(r"noise_([\d.]+)_job_(\d+)_n_(\d+)_id_(\d+)", dirname)
    return float(m.group(1)) if m else None


def n_diff_count(seq_A: str, seq_B: str, idx_mask=None) -> int:
    """Точное число позиций, где остатки отличаются (опционально только в подмножестве idx_mask)."""
    if idx_mask is None:
        return sum(a != b for a, b in zip(seq_A, seq_B))
    return sum(
        a != b for a, b, m in zip(seq_A, seq_B, idx_mask) if m
    )


def compute_from_pdbs(task: str, mode: str, exp_dir: Path, use_motif: bool) -> pd.DataFrame:
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

        seq_A = extract_seq(pdb_A, three_to_one)
        seq_B = extract_seq(pdb_B, three_to_one)
        if len(seq_A) != len(seq_B) or len(seq_A) == 0:
            continue

        n = len(seq_A)
        row = {
            "task": task,
            "mode": mode,
            "noise_scale": noise_scale,
            "pair": pair_dir.name,
            "length": n,
            "n_diff_full": n_diff_count(seq_A, seq_B),
        }

        if use_motif:
            motif_mask = [i in MOTIF_INDICES and i < n for i in range(n)]
            scaffold_mask = [not m for m in motif_mask]
            n_motif = sum(motif_mask)
            n_scaffold = sum(scaffold_mask)
            row["length_motif"] = n_motif
            row["length_scaffold"] = n_scaffold
            row["n_diff_motif"] = n_diff_count(seq_A, seq_B, motif_mask) if n_motif else np.nan
            row["n_diff_scaffold"] = n_diff_count(seq_A, seq_B, scaffold_mask) if n_scaffold else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Загрузка из уже готового all_pairs_raw.csv (быстрый путь)
# ----------------------------------------------------------------------

def load_from_raw_csv() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    required = {"task", "mode", "noise_scale", "length", "seq_identity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{RAW_CSV} не содержит нужные колонки: {missing}")

    # Точное число отличий восстанавливается однозначно из identity, т.к.
    # seq_identity = n_matches / length с точными целыми n_matches/length
    df["n_diff_full"] = (df["length"] * (1 - df["seq_identity"])).round().astype(int)

    if "motif_seq_identity" in df.columns and "length_motif" in df.columns:
        df["n_diff_motif"] = (df["length_motif"] * (1 - df["motif_seq_identity"])).round()
    if "scaffold_seq_identity" in df.columns and "length_scaffold" in df.columns:
        df["n_diff_scaffold"] = (df["length_scaffold"] * (1 - df["scaffold_seq_identity"])).round()

    return df


# ----------------------------------------------------------------------
# Основной сбор
# ----------------------------------------------------------------------

def collect_all() -> pd.DataFrame:
    if RAW_CSV.exists():
        print(f"Пробую быстрый путь: читаю {RAW_CSV} ...")
        try:
            df = load_from_raw_csv()
            if "length_motif" in df.columns:
                print("  Нашёл разбивку motif/scaffold — использую её.")
            else:
                print("  [WARN] В all_pairs_raw.csv нет разбивки motif/scaffold "
                      "(length_motif/length_scaffold) — будет только n_diff_full.")
            return df
        except Exception as e:
            print(f"  [WARN] Не получилось: {e}. Считаю заново из PDB.")

    print("Считаю всё заново прямо из PDB-файлов (не нашёл all_pairs_raw.csv)...")
    all_rows = []
    for dir_name, task, mode in EXPERIMENTS:
        exp_dir = BASELINES_DIR / dir_name
        if not exp_dir.exists():
            print(f"  [SKIP] {exp_dir} не найдена")
            continue
        use_motif = task in CONDITIONAL_TASKS
        print(f"  Обрабатываю {dir_name} ...")
        df = compute_from_pdbs(task, mode, exp_dir, use_motif)
        print(f"    -> {len(df)} пар")
        all_rows.append(df)

    if not all_rows or all(d.empty for d in all_rows):
        raise RuntimeError("Не нашёл данных ни в all_pairs_raw.csv, ни в директориях экспериментов.")

    return pd.concat(all_rows, ignore_index=True)


# ----------------------------------------------------------------------
# Агрегация и вывод
# ----------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in ["n_diff_full", "n_diff_motif", "n_diff_scaffold"] if c in df.columns]
    agg = df.groupby(["task", "mode", "noise_scale"])[metric_cols].agg(["mean", "std", "min", "max", "median"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()

    # Средняя длина белка/мотива/скаффолда — для контекста (сколько это в %)
    if "length" in df.columns:
        len_mean = df.groupby(["task", "mode", "noise_scale"])["length"].mean().rename("length_mean")
        agg = agg.merge(len_mean.reset_index(), on=["task", "mode", "noise_scale"])

    return agg.sort_values(["task", "mode", "noise_scale"])


def print_report(summary: pd.DataFrame):
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    for task in summary["task"].unique():
        print(f"\n{'=' * 90}\n{task}\n{'=' * 90}")
        task_df = summary[summary["task"] == task]

        for mode in task_df["mode"].unique():
            sub = task_df[task_df["mode"] == mode].sort_values("noise_scale")
            print(f"\n--- mode = {mode} ---")

            cols_full = ["noise_scale", "length_mean", "n_diff_full_mean", "n_diff_full_std",
                         "n_diff_full_min", "n_diff_full_max"]
            cols_full = [c for c in cols_full if c in sub.columns]
            print("[full sequence]")
            print(sub[cols_full].round(2).to_string(index=False))

            if "n_diff_motif_mean" in sub.columns:
                print("\n[motif part, зафиксированный участок]")
                cols_motif = ["noise_scale", "n_diff_motif_mean", "n_diff_motif_std",
                              "n_diff_motif_min", "n_diff_motif_max"]
                cols_motif = [c for c in cols_motif if c in sub.columns]
                print(sub[cols_motif].round(2).to_string(index=False))

            if "n_diff_scaffold_mean" in sub.columns:
                print("\n[scaffold part, генерируемый заново участок]")
                cols_scaf = ["noise_scale", "n_diff_scaffold_mean", "n_diff_scaffold_std",
                             "n_diff_scaffold_min", "n_diff_scaffold_max"]
                cols_scaf = [c for c in cols_scaf if c in sub.columns]
                print(sub[cols_scaf].round(2).to_string(index=False))


def main():
    raw = collect_all()
    raw.to_csv(OUT_CSV_RAW, index=False)
    print(f"\nСохранил построчные (per-pair) данные: {OUT_CSV_RAW}")

    summary = summarize(raw)
    summary.to_csv(OUT_CSV, index=False)
    print(f"Сохранил агрегированную таблицу: {OUT_CSV}")

    print_report(summary)


if __name__ == "__main__":
    main()
