#!/usr/bin/env python3
"""
diagnose_one_ensemble.py

Диагностика одного готового BioEmu-ансамбля:
- RMSD между pathA и pathB (насколько состояния разные?)
- Распределение RMSD кадров к СВОЕМУ референсу
- TM-score распределение
- Проверка: какие кадры близки к порогу, бимодальность
- Сохраняет гистограмму RMSD

Логика: seqA ансамбль сравнивается с pathA, seqB — с pathB.
Опционально можно показать и второй референс для контекста.

Использование:
  python diagnose_one_ensemble.py \
    --ensemble_dir bioemu_baselines/ensembles/task1_default/noise_0.00_job_0_n_120_id_1/seqA \
    --ref baselines/task1_default/noise_0.00_job_0_n_120_id_1/..._pathA.pdb \
    --ref_other baselines/task1_default/noise_0.00_job_0_n_120_id_1/..._pathB.pdb \
    --output diagnostic_task1_noise0_id1_seqA.png
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none'


# ── Kabsch + TM-score ──

def kabsch_align(P, Q):
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
    n_frames = traj_ca.shape[0]
    scores = np.zeros(n_frames)
    for i in range(n_frames):
        scores[i] = tm_score(traj_ca[i], ref_ca)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Diagnose one BioEmu ensemble")
    parser.add_argument("--ensemble_dir", required=True)
    parser.add_argument("--ref", required=True, help="СВОЙ референс (pathA для seqA, pathB для seqB)")
    parser.add_argument("--ref_other", default=None, help="Опц.: второй референс для контекста")
    parser.add_argument("--output", default="diagnostic_report.png")
    args = parser.parse_args()

    ensemble_dir = Path(args.ensemble_dir)
    xtc = ensemble_dir / "samples.xtc"
    top = ensemble_dir / "topology.pdb"

    if not xtc.exists() or not top.exists():
        print(f"ERROR: {xtc} or {top} not found")
        sys.exit(1)

    # Загружаем
    traj = md.load(str(xtc), top=str(top))
    ref = md.load(args.ref)
    ref_other = md.load(args.ref_other) if args.ref_other else None

    print(f"Ensemble: {len(traj)} frames, {traj.n_atoms} atoms")
    print(f"Ref (own): {ref.n_atoms} atoms")
    if ref_other is not None:
        print(f"Ref (other): {ref_other.n_atoms} atoms")

    # CA-атомы
    ca_traj = [a.index for a in traj.topology.atoms if a.name == "CA"]
    ca_ref = [a.index for a in ref.topology.atoms if a.name == "CA"]
    L = min(len(ca_traj), len(ca_ref))
    print(f"CA atoms: traj={len(ca_traj)}, ref={len(ca_ref)}, L={L}")

    # ── 1. RMSD между pathA и pathB (если есть второй референс) ──
    if ref_other is not None:
        ca_other = [a.index for a in ref_other.topology.atoms if a.name == "CA"]
        L2 = min(len(ca_ref), len(ca_other))
        rmsd_ab = md.rmsd(ref, ref_other, atom_indices=ca_ref[:L2], ref_atom_indices=ca_other[:L2])[0] * 10.0
        tm_ab = tm_score(ref.xyz[0, ca_ref[:L2], :] * 10.0, ref_other.xyz[0, ca_other[:L2], :] * 10.0)
        print(f"\n=== pathA vs pathB (контекст) ===")
        print(f"  CA-RMSD: {rmsd_ab:.3f} Å")
        print(f"  TM-score: {tm_ab:.4f}")
        print(f"  (Если <3 Å — это вообще не 'два состояния', а одна конформация)")

    # ── 2. RMSD кадров к СВОЕМУ референсу ──
    rmsd = md.rmsd(traj, ref, atom_indices=ca_traj[:L], ref_atom_indices=ca_ref[:L]) * 10.0

    # ── 3. TM-score ──
    traj_ca = traj.xyz[:, ca_traj[:L], :] * 10.0
    ref_ca = ref.xyz[0, ca_ref[:L], :] * 10.0
    tm = tm_score_traj(traj_ca, ref_ca)

    # ── 4. Статистика ──
    print(f"\n=== RMSD to own ref ===")
    print(f"  min={rmsd.min():.3f}, max={rmsd.max():.3f}, mean={rmsd.mean():.3f}, median={np.median(rmsd):.3f}")
    for thr in [2.0, 3.0, 5.0, 8.0]:
        print(f"  <{thr} Å: {np.mean(rmsd < thr)*100:.1f}% ({(rmsd < thr).sum()}/{len(rmsd)})")

    print(f"\n=== TM-score to own ref ===")
    print(f"  max={tm.max():.4f}, mean={tm.mean():.4f}")
    for thr in [0.5, 0.7]:
        print(f"  >{thr}: {np.mean(tm > thr)*100:.1f}% ({(tm > thr).sum()}/{len(tm)})")

    # ── 5. Гистограмма ──
    n_plots = 2 if ref_other is None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # RMSD histogram
    ax = axes[0]
    ax.hist(rmsd, bins=20, color="#0279EE", alpha=0.7, label=f"to own ref (best={rmsd.min():.2f} Å)")
    ax.axvline(2.0, color="red", linestyle="--", label="threshold 2.0 Å")
    ax.axvline(5.0, color="red", linestyle=":", alpha=0.5, label="threshold 5.0 Å")
    ax.set_xlabel("CA-RMSD (Å)")
    ax.set_ylabel("Count")
    ax.set_title("RMSD distribution to own ref")
    ax.legend(fontsize=9)

    # TM-score histogram
    ax = axes[1]
    ax.hist(tm, bins=20, color="#75A025", alpha=0.7, label=f"to own ref (best={tm.max():.3f})")
    ax.axvline(0.5, color="red", linestyle="--", label="TM=0.5 (same fold)")
    ax.axvline(0.7, color="red", linestyle=":", alpha=0.5, label="TM=0.7")
    ax.set_xlabel("TM-score")
    ax.set_ylabel("Count")
    ax.set_title("TM-score distribution to own ref")
    ax.legend(fontsize=9)

    # Scatter: RMSD vs TM-score
    if ref_other is not None:
        ax = axes[2]
        ax.scatter(rmsd, tm, c="#0279EE", alpha=0.6, s=30)
        ax.axvline(2.0, color="red", linestyle="--", alpha=0.5)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("RMSD to own ref (Å)")
        ax.set_ylabel("TM-score to own ref")
        ax.set_title("RMSD vs TM-score (each frame)")
    else:
        # Если нет второго референса — scatter RMSD vs TM
        ax = axes[1] if n_plots > 1 else axes[0]
        # Заменим: добавим scatter в третьей панели если есть
        pass

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nHistogram saved: {args.output}")


if __name__ == "__main__":
    main()
