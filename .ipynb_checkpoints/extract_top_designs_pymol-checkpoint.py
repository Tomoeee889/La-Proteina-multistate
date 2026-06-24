# extract_top_designs_pymol.py
import os
import mdtraj as md
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil
import pymol
from pymol import cmd

# Запуск PyMOL в headless режиме
pymol.finish_launching(['pymol', '-cq'])

def load_ensemble(seq_dir):
    """Загружает ансамбль BioEmu"""
    xtc = seq_dir / "samples.xtc"
    top = seq_dir / "topology.pdb"
    if xtc.exists() and top.exists():
        return md.load(str(xtc), top=str(top))
    
    npz = sorted(seq_dir.glob("batch_*.npz"))
    if npz and top.exists():
        data = np.load(npz[0], allow_pickle=True)
        coords = list(data.values())[0] if len(data) > 0 else None
        if coords is not None and coords.ndim == 3:
            return md.Trajectory(coords, md.load_topology(str(top)))
    return None

def find_refs(seq_id, ref_base, exp_name):
    """Поиск референсных PDB"""
    sample_name = seq_id.split("_seq")[0]
    ref_dir = Path(ref_base) / exp_name / sample_name
    
    ref_A = ref_dir / "pathA.pdb"
    ref_B = ref_dir / "pathB.pdb"
    
    if not ref_A.exists():
        for p in [f"{sample_name}_pathA.pdb", "state1.pdb", "open.pdb"]:
            if (ref_dir / p).exists(): ref_A = ref_dir / p; break
    if not ref_B.exists():
        for p in [f"{sample_name}_pathB.pdb", "state2.pdb", "closed.pdb"]:
            if (ref_dir / p).exists(): ref_B = ref_dir / p; break
            
    return ref_A, ref_B

def calc_rmsd_fit(frame_pdb, ref_pdb):
    """Честный RMSD через PyMOL fit (без отбрасывания атомов)"""
    try:
        cmd.load(str(frame_pdb), "mob", quiet=1)
        cmd.load(str(ref_pdb), "ref", quiet=1)
        
        # Проверяем число CA-атомов
        mob_ca = cmd.count_atoms("mob and name CA")
        ref_ca = cmd.count_atoms("ref and name CA")
        
        if mob_ca != ref_ca:
            print(f"   ⚠️ Несоответствие CA: {mob_ca} vs {ref_ca}")
            cmd.delete("all")
            return 999.0, 0
        
        # fit — жесткое наложение по всем атомам (алгоритм Кабша)
        rmsd_val = cmd.fit("mob and name CA", "ref and name CA", quiet=1)
        
        cmd.delete("all")
        return rmsd_val, mob_ca
    except Exception as e:
        cmd.delete("all")
        return 999.0, 0

def extract_for_sequence(seq_dir, ref_A, ref_B, output_dir, exp_name, seq_id, top_structures=3):
    """Извлекает топ-N структур для одной последовательности"""
    traj = load_ensemble(seq_dir)
    if traj is None:
        print(f"   ❌ Не удалось загрузить ансамбль")
        return None
    
    # Временная папка для кадров
    tmp_dir = Path(output_dir) / "tmp_frames"
    tmp_dir.mkdir(exist_ok=True)
    
    rmsd_A_list = []
    rmsd_B_list = []
    atoms_used = 0
    
    # Считаем RMSD для каждого кадра
    for i, frame in enumerate(traj):
        frame_pdb = tmp_dir / f"frame_{i:03d}.pdb"
        frame.save_pdb(str(frame_pdb))
        
        rA, nA = calc_rmsd_fit(str(frame_pdb), str(ref_A))
        rB, nB = calc_rmsd_fit(str(frame_pdb), str(ref_B))
        
        if rA < 999 and rB < 999:
            rmsd_A_list.append((i, rA))
            rmsd_B_list.append((i, rB))
            if atoms_used == 0:
                atoms_used = nA
    
    if not rmsd_A_list:
        print(f"   ❌ Не удалось рассчитать RMSD")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None
    
    # Сортируем по RMSD (меньше = лучше)
    rmsd_A_list.sort(key=lambda x: x[1])
    rmsd_B_list.sort(key=lambda x: x[1])
    
    best_A = rmsd_A_list[:top_structures]
    best_B = rmsd_B_list[:top_structures]
    
    # 🔑 Сохраняем с учётом имени эксперимента
    save_dir = Path(output_dir) / exp_name / seq_id
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_A = []
    for rank, (idx, rmsd) in enumerate(best_A, 1):
        src = tmp_dir / f"frame_{idx:03d}.pdb"
        dst = save_dir / f"stateA_rank{rank}_rmsd{rmsd:.2f}A.pdb"
        if src.exists():
            shutil.copy(src, dst)
            saved_A.append((rank, rmsd, dst))
    
    saved_B = []
    for rank, (idx, rmsd) in enumerate(best_B, 1):
        src = tmp_dir / f"frame_{idx:03d}.pdb"
        dst = save_dir / f"stateB_rank{rank}_rmsd{rmsd:.2f}A.pdb"
        if src.exists():
            shutil.copy(src, dst)
            saved_B.append((rank, rmsd, dst))
    
    # Удаляем временные файлы
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return saved_A, saved_B, atoms_used

def main(results_csv, output_dir, ensemble_base, ref_base, top_n=3, top_structures=3, specific_ids=None):
    """Основная функция"""
    # Читаем результаты
    df = pd.read_csv(results_csv)
    
    # Выбираем последовательности для обработки
    if specific_ids:
        # Конкретные ID через запятую
        ids_list = [sid.strip() for sid in specific_ids.split(",")]
        selected = df[df["id"].isin(ids_list)].copy()
        print(f"\n📊 Выбрано {len(selected)} конкретных последовательностей")
    else:
        # Топ-N по bimodal_score
        df_sorted = df.sort_values("bimodal_score", ascending=False)
        selected = df_sorted.head(top_n).copy()
        print(f"\n📊 Выбрано топ-{top_n} последовательностей по bimodal_score")
    
    if len(selected) == 0:
        print("❌ Нет последовательностей для обработки")
        return
    
    print(f"\n{'='*70}")
    print("Последовательности для обработки:")
    print(f"{'='*70}")
    for idx, row in selected.iterrows():
        print(f"  • {row['id']} | {row['experiment']} | bimodal={row['bimodal_score']:.3f}")
    print(f"{'='*70}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble_base = Path(ensemble_base)
    
    total_processed = 0
    
    for idx, row in tqdm(selected.iterrows(), desc="Processing", total=len(selected)):
        seq_id = row["id"]
        exp_name = row["experiment"]
        bimodal = row["bimodal_score"]
        
        seq_dir = ensemble_base / exp_name / seq_id
        if not seq_dir.exists():
            print(f"⚠️ Не найдена папка: {seq_dir}")
            continue
        
        ref_A, ref_B = find_refs(seq_id, ref_base, exp_name)
        if not ref_A.exists() or not ref_B.exists():
            print(f"⚠️ Нет референсов для {seq_id} в {ref_A.parent}")
            continue
        
        result = extract_for_sequence(seq_dir, ref_A, ref_B, output_dir, exp_name, seq_id, top_structures)
        
        if result:
            saved_A, saved_B, atoms_used = result
            total_processed += 1
            
            print(f"✅ {seq_id} | Ref: {exp_name} | Bimodal: {bimodal:.3f} | CA atoms: {atoms_used}")
            if saved_A:
                print(f"   State A: {saved_A[0][2].name} (RMSD={saved_A[0][1]:.2f}Å)")
            if saved_B:
                print(f"   State B: {saved_B[0][2].name} (RMSD={saved_B[0][1]:.2f}Å)")
    
    print(f"\n{'='*70}")
    print(f"✅ Готово! Обработано последовательностей: {total_processed}")
    print(f"📁 Результаты в: {output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Извлечение лучших структур из ансамблей BioEmu для топ-последовательностей")
    ap.add_argument("--results_csv", required=True, help="CSV с результатами метрик")
    ap.add_argument("--ensemble_base", default="bioemu_ensembles", help="Директория с ансамблями")
    ap.add_argument("--ref_base", required=True, help="Директория с референсами")
    ap.add_argument("--output_dir", default="top_ensembles", help="Папка для извлечённых структур")
    ap.add_argument("--top_n", type=int, default=3, help="Сколько лучших последовательностей взять (по bimodal_score)")
    ap.add_argument("--top_structures", type=int, default=3, help="Сколько структур сохранить для каждого состояния")
    ap.add_argument("--specific_ids", type=str, default=None, help="Конкретные ID через запятую (например: 'seq00,seq01,seq02')")
    args = ap.parse_args()
    
    main(args.results_csv, args.output_dir, args.ensemble_base, args.ref_base, 
         args.top_n, args.top_structures, args.specific_ids)