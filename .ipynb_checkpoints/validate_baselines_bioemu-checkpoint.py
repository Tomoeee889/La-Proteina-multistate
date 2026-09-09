#!/usr/bin/env python3
"""
validate_baselines_bioemu.py

Валидация baseline-структур через BioEmu.

Для каждой пары (pathA.pdb, pathB.pdb) из baselines/:
  1. Извлекает последовательности из обоих PDB
  2. Запускает BioEmu на каждой последовательности
  3. Считает CA-RMSD и TM-score каждого кадра к СВОЕМУ референсу
     (seqA → pathA, seqB → pathB)
  4. Записывает результаты в results.csv

Чекпойнты: status.csv отслеживает статус каждой задачи.
Возобновление: при рестарте начинает с первой строки status=pending.

Использование:
  python validate_baselines_bioemu.py \\
      --baselines_dir baselines \\
      --output_dir bioemu_baselines \\
      --task all --mode all \\
      --num_samples 120 --threshold 3.5

  # Dry-run (сформировать список задач в status.csv):
  python validate_baselines_bioemu.py --task all --mode all --max_runs 0

  # Лимит запусков за вызов:
  python validate_baselines_bioemu.py --task 1 --mode default --max_runs 50
"""

import os
import re
import sys
import argparse
import logging
import datetime
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── BioEmu импортируем НА ВЕРХНЕМ УРОВНЕ (как в рабочем running_bioemu.py) ──
print("Loading BioEmu model (this may take a minute) ...", flush=True)
from bioemu import sample as bioemu_sample
print("BioEmu model loaded.", flush=True)

# BioPython для извлечения последовательности
from Bio.PDB import PDBParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ===========================================================================
#  КОНСТАНТЫ
# ===========================================================================

NOISE_PATTERN = re.compile(r"noise_(\d+\.\d+)_job_\d+_n_\d+_id_\d+")

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X", "MSE": "M", "SEC": "U", "PYL": "O",
}

STATUS_COLUMNS = [
    "task", "mode", "noise_scale", "pair_id", "seq_type",
    "status", "timestamp", "error",
]
RESULTS_COLUMNS = [
    "task", "mode", "noise_scale", "pair_id", "seq_type",
    "coverage", "rmsd_best", "rmsd_mean", "tm_best", "tm_mean",
    "num_samples",
]


# ===========================================================================
#  PDB PARSING — извлечение последовательности
# ===========================================================================

def get_sequence_from_pdb(pdb_path):
    """Извлекает последовательность из PDB (первая цепь первой модели)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    model = structure[0]
    chain = list(model.get_chains())[0]
    seq = []
    for residue in chain.get_residues():
        if residue.id[0] == " ":  # пропускаем гетероатомы
            resname = residue.resname
            seq.append(THREE_TO_ONE.get(resname, "X"))
    return "".join(seq)


# ===========================================================================
#  СКАНИРОВАНИЕ BASELINES
# ===========================================================================

def scan_baselines(baselines_dir, tasks, modes, num_pairs, seed):
    """Сканирует baselines/ и формирует список задач.

    Возвращает список dict:
      {task, mode, noise_scale, pair_id, seq_type, pdb_a, pdb_b}
    """
    baselines_dir = Path(baselines_dir)
    if not baselines_dir.is_dir():
        logger.error(f"baselines_dir not found: {baselines_dir}")
        sys.exit(1)

    all_tasks = []

    for task in tasks:
        for mode in modes:
            task_mode_dir = baselines_dir / f"task{task}_{mode}"
            if not task_mode_dir.is_dir():
                logger.warning(f"Directory not found, skipping: {task_mode_dir}")
                continue

            pairs_by_scale = defaultdict(list)

            for sample_dir in sorted(task_mode_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue

                m = NOISE_PATTERN.match(sample_dir.name)
                if not m:
                    continue

                noise_scale = float(m.group(1))

                pdb_a = sample_dir / f"{sample_dir.name}_pathA.pdb"
                if not pdb_a.exists():
                    pdb_a = sample_dir / "pathA.pdb"
                pdb_b = sample_dir / f"{sample_dir.name}_pathB.pdb"
                if not pdb_b.exists():
                    pdb_b = sample_dir / "pathB.pdb"

                if not pdb_a.exists() or not pdb_b.exists():
                    continue

                pairs_by_scale[noise_scale].append({
                    "pair_id": sample_dir.name,
                    "pdb_a": str(pdb_a),
                    "pdb_b": str(pdb_b),
                })

            for noise_scale in sorted(pairs_by_scale.keys()):
                pairs = pairs_by_scale[noise_scale]
                n_select = min(num_pairs, len(pairs))

                # Детерминированный seed (hashlib вместо рандомизированного hash())
                h = int(hashlib.md5(f"{task}_{mode}_{noise_scale}".encode()).hexdigest(), 16)
                local_rng = np.random.RandomState(seed + h % (2**31))
                selected_idx = local_rng.choice(len(pairs), size=n_select, replace=False)
                selected = [pairs[i] for i in sorted(selected_idx)]

                for pair in selected:
                    for seq_type in ["A", "B"]:
                        all_tasks.append({
                            "task": f"task{task}",
                            "mode": mode,
                            "noise_scale": noise_scale,
                            "pair_id": pair["pair_id"],
                            "seq_type": seq_type,
                            "pdb_a": pair["pdb_a"],
                            "pdb_b": pair["pdb_b"],
                        })

    return all_tasks


# ===========================================================================
#  ЧЕКПОЙНТЫ
# ===========================================================================

def load_status(output_dir):
    """Загружает status.csv. Возвращает DataFrame (или пустой)."""
    status_path = Path(output_dir) / "status.csv"
    if status_path.exists():
        df = pd.read_csv(status_path)
        for col in STATUS_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=STATUS_COLUMNS)


def save_status(output_dir, df):
    """Сохраняет status.csv."""
    status_path = Path(output_dir) / "status.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df[STATUS_COLUMNS].to_csv(status_path, index=False)


def update_status_row(status_df, task_info, status, error=""):
    """Обновляет или добавляет строку в status_df."""
    mask = (
        (status_df["task"] == task_info["task"]) &
        (status_df["mode"] == task_info["mode"]) &
        (status_df["noise_scale"] == task_info["noise_scale"]) &
        (status_df["pair_id"] == task_info["pair_id"]) &
        (status_df["seq_type"] == task_info["seq_type"])
    )

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    if mask.any():
        idx = mask.idxmax()
        status_df.loc[idx, "status"] = status
        status_df.loc[idx, "timestamp"] = timestamp
        status_df.loc[idx, "error"] = error
    else:
        new_row = {
            "task": task_info["task"],
            "mode": task_info["mode"],
            "noise_scale": task_info["noise_scale"],
            "pair_id": task_info["pair_id"],
            "seq_type": task_info["seq_type"],
            "status": status,
            "timestamp": timestamp,
            "error": error,
        }
        status_df = pd.concat(
            [status_df, pd.DataFrame([new_row])],
            ignore_index=True,
        )

    return status_df


def is_done(status_df, task_info):
    """Проверяет, завершена ли задача (status=done)."""
    if status_df.empty:
        return False
    mask = (
        (status_df["task"] == task_info["task"]) &
        (status_df["mode"] == task_info["mode"]) &
        (status_df["noise_scale"] == task_info["noise_scale"]) &
        (status_df["pair_id"] == task_info["pair_id"]) &
        (status_df["seq_type"] == task_info["seq_type"])
    )
    if not mask.any():
        return False
    return status_df.loc[mask.idxmax(), "status"] == "done"


def find_first_unfinished_index(status_df, all_tasks):
    """Находит индекс в all_tasks первой НЕ завершённой задачи.
    Не завершённая = status=pending, status=failed, или отсутствует в status_df.

    Ищет в порядке status.csv (как они были записаны), а не в порядке all_tasks.
    Это гарантирует, что при возобновлении скрипт начнёт с той же задачи,
    на которой остановился в прошлый раз.

    Возвращает None, если все done."""
    if status_df.empty:
        return 0  # все новые

    # Проходим по status.csv в порядке записей
    for _, row in status_df.iterrows():
        if row["status"] in ("pending", "failed"):
            # Нашли первую незавершённую — ищем её индекс в all_tasks
            for i, task_info in enumerate(all_tasks):
                if (
                    task_info["task"] == row["task"]
                    and task_info["mode"] == row["mode"]
                    and task_info["noise_scale"] == row["noise_scale"]
                    and task_info["pair_id"] == row["pair_id"]
                    and task_info["seq_type"] == row["seq_type"]
                ):
                    return i

    # Если в status.csv все done — проверяем, есть ли задачи, которых нет в status_df
    for i, task_info in enumerate(all_tasks):
        mask = (
            (status_df["task"] == task_info["task"]) &
            (status_df["mode"] == task_info["mode"]) &
            (status_df["noise_scale"] == task_info["noise_scale"]) &
            (status_df["pair_id"] == task_info["pair_id"]) &
            (status_df["seq_type"] == task_info["seq_type"])
        )
        if not mask.any():
            return i

    return None


# ===========================================================================
#  BIOEMU ГЕНЕРАЦИЯ
# ===========================================================================

def run_bioemu(sequence, output_dir, num_samples, batch_size):
    """Запускает BioEmu на последовательности."""
    bioemu_sample.main(
        sequence=sequence,
        num_samples=num_samples,
        output_dir=str(output_dir),
        batch_size_100=batch_size,
        model_name="bioemu-v1.2",
    )


# ===========================================================================
#  TM-SCORE (Zhang Group алгоритм)
# ===========================================================================

def kabsch_align(P, Q):
    """Выравнивание Кабша. P, Q: [N, 3]."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = Q.mean(axis=0) - P.mean(axis=0) @ R.T
    return R, t


def tm_score(P, Q):
    """TM-score между двумя CA-массивами [N, 3]."""
    L = min(len(P), len(Q))
    if L == 0:
        return 0.0
    Lnorm = L
    if Lnorm < 21:
        d0 = 0.5
    else:
        d0 = 1.24 * (Lnorm - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)
    R, t = kabsch_align(P[:L], Q[:L])
    P_aligned = P[:L] @ R.T + t
    di = np.sqrt(np.sum((P_aligned - Q[:L]) ** 2, axis=1))
    tm = np.sum(1.0 / (1.0 + (di / d0) ** 2)) / Lnorm
    return float(tm)


def tm_score_traj(traj_ca, ref_ca):
    """TM-score для каждого кадра к референсу. [n_frames] → [n_frames]."""
    n_frames = traj_ca.shape[0]
    scores = np.zeros(n_frames)
    for i in range(n_frames):
        scores[i] = tm_score(traj_ca[i], ref_ca)
    return scores


# ===========================================================================
#  RMSD + TM-SCORE РАСЧЁТ (mdtraj)
# ===========================================================================

def calc_metrics(ensemble_dir, ref_pdb, threshold):
    """Считает CA-RMSD и TM-score каждого кадра к ref_pdb.

    Ансамбль сравнивается только со СВОИМ референсом:
      seqA → pathA, seqB → pathB.
    """
    import mdtraj as md

    ensemble_dir = Path(ensemble_dir)
    xtc = ensemble_dir / "samples.xtc"
    top = ensemble_dir / "topology.pdb"

    if not xtc.exists() or not top.exists():
        raise FileNotFoundError(f"BioEmu ensemble not found: {xtc} / {top}")

    traj = md.load(str(xtc), top=str(top))
    ref = md.load(ref_pdb)

    ca_traj = [a.index for a in traj.topology.atoms if a.name == "CA"]
    ca_ref = [a.index for a in ref.topology.atoms if a.name == "CA"]
    L = min(len(ca_traj), len(ca_ref))
    if L == 0:
        raise ValueError("No CA atoms found")

    # RMSD
    rmsd = md.rmsd(traj, ref, atom_indices=ca_traj[:L], ref_atom_indices=ca_ref[:L])
    rmsd = rmsd * 10.0  # nm → Å

    # TM-score
    traj_ca = traj.xyz[:, ca_traj[:L], :] * 10.0  # nm → Å
    ref_ca = ref.xyz[0, ca_ref[:L], :] * 10.0
    tm = tm_score_traj(traj_ca, ref_ca)

    coverage = float(np.mean(rmsd < threshold))

    return {
        "coverage": round(coverage, 4),
        "rmsd_best": round(float(rmsd.min()), 3),
        "rmsd_mean": round(float(rmsd.mean()), 3),
        "tm_best": round(float(tm.max()), 4),
        "tm_mean": round(float(tm.mean()), 4),
        "num_samples": len(rmsd),
    }


# ===========================================================================
#  RESULTS CSV
# ===========================================================================

def save_results(output_dir, status_df, all_tasks):
    """Собирает results.csv из всех задач со status=done."""
    results_path = Path(output_dir) / "results.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    done_tasks = [t for t in all_tasks if is_done(status_df, t)]

    for task_info in done_tasks:
        ensemble_dir = (
            Path(output_dir) / "ensembles"
            / f"{task_info['task']}_{task_info['mode']}"
            / task_info["pair_id"]
            / f"seq{task_info['seq_type']}"
        )
        ref_pdb = task_info["pdb_a"] if task_info["seq_type"] == "A" else task_info["pdb_b"]

        try:
            metrics = calc_metrics(ensemble_dir, ref_pdb, threshold=2.0)
            row = {
                "task": task_info["task"],
                "mode": task_info["mode"],
                "noise_scale": task_info["noise_scale"],
                "pair_id": task_info["pair_id"],
                "seq_type": task_info["seq_type"],
                **metrics,
            }
            rows.append(row)
        except Exception as e:
            logger.warning(
                f"Could not compute metrics for "
                f"{task_info['task']}_{task_info['mode']}/"
                f"{task_info['pair_id']}/seq{task_info['seq_type']}: {e}"
            )

    df = pd.DataFrame(rows, columns=RESULTS_COLUMNS)
    if not df.empty:
        df = df.sort_values(
            ["task", "mode", "noise_scale", "pair_id", "seq_type"]
        ).reset_index(drop=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Results saved: {results_path} ({len(df)} rows)")


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BioEmu validation для baseline-экспериментов"
    )
    parser.add_argument("--baselines_dir", type=str, default="baselines")
    parser.add_argument("--output_dir", type=str, default="bioemu_baselines")
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--num_pairs", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--max_runs", type=int, default=-1,
                        help="Max new runs (-1=unlimited, 0=dry-run)")
    args = parser.parse_args()

    tasks = [1, 2, 3] if args.task == "all" else [int(args.task)]
    modes = ["default", "sim_eps"] if args.mode == "all" else [args.mode]

    print(f"Tasks: {tasks}, Modes: {modes}", flush=True)
    print(f"Num pairs/scale: {args.num_pairs}, Num samples: {args.num_samples}", flush=True)
    print(f"Threshold: {args.threshold} Å, Seed: {args.seed}", flush=True)

    # ── Сканирование baselines ──
    print(f"Scanning {args.baselines_dir} ...", flush=True)
    all_tasks = scan_baselines(
        args.baselines_dir, tasks, modes, args.num_pairs, args.seed
    )
    print(f"Total tasks: {len(all_tasks)}", flush=True)

    if not all_tasks:
        print("ERROR: No tasks found. Check --baselines_dir and --task/--mode.", flush=True)
        sys.exit(1)

    by_task_mode = defaultdict(int)
    for t in all_tasks:
        by_task_mode[f"{t['task']}_{t['mode']}"] += 1
    for key in sorted(by_task_mode.keys()):
        print(f"  {key}: {by_task_mode[key]} tasks", flush=True)

    # ── Загрузка чекпойнтов ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_df = load_status(args.output_dir)

    done_count = 0
    if not status_df.empty:
        done_count = (status_df["status"] == "done").sum()
    print(f"Checkpoint: {done_count} tasks already done", flush=True)

    # ── Dry-run ──
    if args.max_runs == 0:
        print("Dry-run mode: writing status.csv, not running BioEmu", flush=True)
        for task_info in all_tasks:
            if not is_done(status_df, task_info):
                status_df = update_status_row(status_df, task_info, "pending")
        save_status(args.output_dir, status_df)
        print(f"Status written: {args.output_dir}/status.csv", flush=True)
        print(f"Pending: {(status_df['status'] == 'pending').sum()}", flush=True)
        print(f"Done: {(status_df['status'] == 'done').sum()}", flush=True)
        return

    # ── Находим первую незавершённую задачу ──
    start_idx = find_first_unfinished_index(status_df, all_tasks)
    if start_idx is not None:
        start_task = all_tasks[start_idx]
        # Определяем статус для печати
        if not status_df.empty:
            mask = (
                (status_df["task"] == start_task["task"]) &
                (status_df["mode"] == start_task["mode"]) &
                (status_df["noise_scale"] == start_task["noise_scale"]) &
                (status_df["pair_id"] == start_task["pair_id"]) &
                (status_df["seq_type"] == start_task["seq_type"])
            )
            if mask.any():
                prev_status = status_df.loc[mask.idxmax(), "status"]
            else:
                prev_status = "new"
        else:
            prev_status = "new"
        print(f"\nStarting from first unfinished (prev_status={prev_status}): "
              f"{start_task['task']}_{start_task['mode']}/"
              f"{start_task['pair_id']}/seq{start_task['seq_type']} "
              f"(index {start_idx}/{len(all_tasks)})", flush=True)
    else:
        print("\nAll tasks are done!", flush=True)
        print("Building results.csv ...", flush=True)
        save_results(args.output_dir, status_df, all_tasks)
        print("Done.", flush=True)
        return

    # ── Основной цикл: начинаем с первой незавершённой ──
    new_runs = 0
    failed = 0

    for i in range(start_idx, len(all_tasks)):
        task_info = all_tasks[i]

        # Пропускаем только завершённые (done). pending и failed — ретрай.
        if is_done(status_df, task_info):
            continue

        # Лимит запусков
        if args.max_runs > 0 and new_runs >= args.max_runs:
            print(f"Reached max_runs={args.max_runs}, stopping.", flush=True)
            break

        task_label = (
            f"{task_info['task']}_{task_info['mode']}/"
            f"{task_info['pair_id']}/seq{task_info['seq_type']} "
            f"(noise={task_info['noise_scale']:.2f})"
        )
        print(f"\n[{new_runs + 1}] {task_label}", flush=True)

        # Путь к ансамблю
        ensemble_dir = (
            output_dir / "ensembles"
            / f"{task_info['task']}_{task_info['mode']}"
            / task_info["pair_id"]
            / f"seq{task_info['seq_type']}"
        )
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Пропускаем если topology.pdb уже есть
        if (ensemble_dir / "topology.pdb").exists():
            print(f"  Already has topology.pdb, marking as done", flush=True)
            status_df = update_status_row(status_df, task_info, "done")
            save_status(args.output_dir, status_df)
            continue

        # ── Извлекаем последовательность ──
        pdb_path = task_info["pdb_a"] if task_info["seq_type"] == "A" else task_info["pdb_b"]
        try:
            sequence = get_sequence_from_pdb(pdb_path)
            if len(sequence) == 0:
                raise ValueError("empty sequence")
        except Exception as e:
            print(f"  Failed to extract sequence: {e}", flush=True)
            status_df = update_status_row(
                status_df, task_info, "failed", f"seq_extract: {str(e)[:100]}"
            )
            save_status(args.output_dir, status_df)
            failed += 1
            new_runs += 1
            continue

        # ── Запуск BioEmu ──
        try:
            print(f"  BioEmu: len={len(sequence)}, samples={args.num_samples} ...",
                  end=" ", flush=True)
            run_bioemu(
                sequence=sequence,
                output_dir=ensemble_dir,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
            )
            print("done", flush=True)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            status_df = update_status_row(
                status_df, task_info, "failed", f"bioemu: {str(e)[:100]}"
            )
            save_status(args.output_dir, status_df)
            failed += 1
            new_runs += 1
            continue

        # ── Метрики (RMSD + TM-score к СВОЕМУ референсу) ──
        ref_pdb = task_info["pdb_a"] if task_info["seq_type"] == "A" else task_info["pdb_b"]
        try:
            metrics = calc_metrics(ensemble_dir, ref_pdb, threshold=args.threshold)
            print(
                f"  Metrics: cov={metrics['coverage']}, "
                f"rmsd_best={metrics['rmsd_best']}, "
                f"rmsd_mean={metrics['rmsd_mean']}, "
                f"tm_best={metrics['tm_best']}",
                flush=True,
            )
        except Exception as e:
            print(f"  Metrics failed: {e}", flush=True)
            status_df = update_status_row(
                status_df, task_info, "failed", f"metrics: {str(e)[:100]}"
            )
            save_status(args.output_dir, status_df)
            failed += 1
            new_runs += 1
            continue

        # ── Сохраняем статус ──
        status_df = update_status_row(status_df, task_info, "done")
        save_status(args.output_dir, status_df)
        new_runs += 1

    # ── Итоги ──
    print(f"\n{'=' * 60}", flush=True)
    print(f"Session complete:", flush=True)
    print(f"  New runs:    {new_runs}", flush=True)
    print(f"  Failed:      {failed}", flush=True)
    print(f"  Total done:  {(status_df['status'] == 'done').sum()}", flush=True)
    print(f"  Total tasks: {len(all_tasks)}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # ── Сохраняем results.csv ──
    print("Building results.csv ...", flush=True)
    save_results(args.output_dir, status_df, all_tasks)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
