# calc_metrics_pymol_correct.py
import os
import mdtraj as md
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import tempfile
import pymol
from pymol import cmd

# Запуск PyMOL в headless режиме (без GUI)
pymol.finish_launching(['pymol', '-cq'])

def load_ensemble(seq_dir):
    """Загружает ансамбль BioEmu (xtc+top или npz+top)"""
    xtc = seq_dir / "samples.xtc"
    top = seq_dir / "topology.pdb"
    if xtc.exists() and top.exists():
        return md.load(str(xtc), top=str(top))
    
    npz = sorted(seq_dir.glob("batch_*.npz"))
    if npz and top.exists():
        data = np.load(npz[0], allow_pickle=True)
        coords = None
        for k in ['coords', 'positions', 'xyz', 'arr_0', 'data']:
            if k in data: coords = data[k]; break
        if coords is None and len(data) > 0:
            coords = list(data.values())[0]
        if coords is not None and coords.ndim == 3:
            return md.Trajectory(coords, md.load_topology(str(top)))
    return None

def find_refs(seq_id, ref_base, exp_name):
    """Поиск референсных PDB (pathA и pathB)"""
    # seq_id обычно выглядит как "job_0_n_120_id_0"
    ref_dir = Path(ref_base) / exp_name / seq_id
    
    ref_A = ref_dir / "pathA.pdb"
    ref_B = ref_dir / "pathB.pdb"
    
    # Альтернативные имена, если вдруг файлы называются иначе
    if not ref_A.exists():
        for p in [f"{seq_id}_pathA.pdb", "state1.pdb", "open.pdb"]:
            if (ref_dir / p).exists(): ref_A = ref_dir / p; break
    if not ref_B.exists():
        for p in [f"{seq_id}_pathB.pdb", "state2.pdb", "closed.pdb"]:
            if (ref_dir / p).exists(): ref_B = ref_dir / p; break
            
    return ref_A, ref_B

def calc_rmsd_pymol_fit(frame_pdb, ref_pdb):
    """Честный RMSD через PyMOL fit (без отбрасывания атомов, алгоритм Кабша)"""
    try:
        cmd.load(str(frame_pdb), "mob", quiet=1)
        cmd.load(str(ref_pdb), "ref", quiet=1)
        
        # Проверяем, что число CA-атомов совпадает (защита от ошибок)
        mob_ca = cmd.count_atoms("mob and name CA")
        ref_ca = cmd.count_atoms("ref and name CA")
        
        if mob_ca != ref_ca or mob_ca == 0:
            cmd.delete("all")
            return 999.0, 0
        
        # fit делает жесткое наложение по всем атомам
        rmsd_val = cmd.fit("mob and name CA", "ref and name CA", quiet=1)
        
        cmd.delete("all")
        return rmsd_val, mob_ca
    except Exception:
        cmd.delete("all")
        return 999.0, 0

def main(ensemble_base, ref_base, output_csv, threshold=3.0):
    results = []
    ensemble_base = Path(ensemble_base)
    
    # Временная папка для сохранения кадров траектории в PDB
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        exp_dirs = [d for d in ensemble_base.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for exp_dir in tqdm(exp_dirs, desc="Experiments"):
            exp_name = exp_dir.name
            seq_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
            
            for seq_dir in tqdm(seq_dirs, desc=f"Sequences ({exp_name})", leave=False):
                seq_id = seq_dir.name
                ref_A, ref_B = find_refs(seq_id, ref_base, exp_name)
                
                if not ref_A.exists() or not ref_B.exists():
                    results.append({"experiment": exp_name, "id": seq_id, "status": "MISSING_REFS"})
                    continue
                
                traj = load_ensemble(seq_dir)
                if traj is None or len(traj) == 0:
                    results.append({"experiment": exp_name, "id": seq_id, "status": "NO_ENSEMBLE"})
                    continue
                
                rmsd_A_list, rmsd_B_list = [], []
                atoms_used = 0
                
                # Считаем RMSD для КАЖДОГО кадра ансамбля
                for i, frame in enumerate(traj):
                    frame_pdb = tmp_path / f"frame_{i}.pdb"
                    frame.save_pdb(str(frame_pdb))
                    
                    rA, nA = calc_rmsd_pymol_fit(str(frame_pdb), str(ref_A))
                    rB, nB = calc_rmsd_pymol_fit(str(frame_pdb), str(ref_B))
                    
                    if rA < 999 and rB < 999:
                        rmsd_A_list.append(rA)
                        rmsd_B_list.append(rB)
                        if atoms_used == 0:
                            atoms_used = nA
                
                if not rmsd_A_list:
                    results.append({"experiment": exp_name, "id": seq_id, "status": "RMSD_FAIL"})
                    continue
                
                rmsd_A = np.array(rmsd_A_list)
                rmsd_B = np.array(rmsd_B_list)
                
                # Coverage = доля кадров, попавших в порог
                coverage_A = np.mean(rmsd_A < threshold)
                coverage_B = np.mean(rmsd_B < threshold)
                bimodal_score = min(coverage_A, coverage_B)
                
                results.append({
                    "experiment": exp_name, "id": seq_id, "status": "OK",
                    "coverage_A": round(float(coverage_A), 3),
                    "coverage_B": round(float(coverage_B), 3),
                    "bimodal_score": round(float(bimodal_score), 3),
                    "rmsd_A_best": round(float(rmsd_A.min()), 2),
                    "rmsd_B_best": round(float(rmsd_B.min()), 2),
                    "rmsd_A_mean": round(float(rmsd_A.mean()), 2),
                    "rmsd_B_mean": round(float(rmsd_B.mean()), 2),
                    "num_samples": len(rmsd_A),
                    "atoms_used": atoms_used
                })
    
    # Сохранение и сортировка результатов
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("bimodal_score", ascending=False, na_position="last")
    
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Результаты сохранены: {output_csv}")
    total_ok = (df["status"] == "OK").sum() if not df.empty else 0
    finalists = (df["bimodal_score"] >= 0.15).sum() if not df.empty else 0
    
    if not df.empty and "atoms_used" in df.columns:
        avg_atoms = df["atoms_used"].mean()
        print(f"📊 Среднее число CA-атомов для выравнивания: {avg_atoms:.0f}")
        if avg_atoms < 100:
            print(f"⚠️ Внимание: мало атомов! Проверь нумерацию в PDB файлах.")
            
    print(f"📊 Успешно обработано: {total_ok}")
    print(f"🏆 Финалисты (bimodal_score ≥ 0.15): {finalists}")
    print(f"{'='*70}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Расчёт метрик multi-state дизайна из ансамблей BioEmu (через PyMOL fit)")
    ap.add_argument("--ensemble_base", default="bioemu_ensembles_from_pdb", help="Директория с ансамблями BioEmu")
    ap.add_argument("--ref_base", required=True, help="Директория с референсными PDB (inference)")
    ap.add_argument("--output_csv", default="bioemu_finalists_mlpon.csv", help="Выходной CSV файл")
    ap.add_argument("--threshold", type=float, default=3.0, help="RMSD порог для coverage (Å)")
    args = ap.parse_args()
    
    main(args.ensemble_base, args.ref_base, args.output_csv, args.threshold)