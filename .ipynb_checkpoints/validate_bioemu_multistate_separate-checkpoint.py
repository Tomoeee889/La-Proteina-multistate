# validate_bioemu_multistate_separate.py
import os
import torch
import mdtraj as md
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from bioemu import sample as bioemu_sample
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def calc_ca_rmsd(ref_pdb, gen_pdb):
    """Быстрый расчёт CA-RMSD через mdtraj"""
    t_ref = md.load(ref_pdb)
    t_gen = md.load(gen_pdb)
    idx_ref = [a.index for a in t_ref.topology.atoms if a.name == "CA"]
    idx_gen = [a.index for a in t_gen.topology.atoms if a.name == "CA"]
    L = min(len(idx_ref), len(idx_gen))
    rmsd = md.rmsd(t_gen, t_ref, atom_indices=idx_gen[:L], ref_atom_indices=idx_ref[:L])
    return rmsd[0]

def find_ref_pdbs(seq_id, ref_base_dir, experiment_name):
    """Ищет pathA.pdb и pathB.pdb для данного seq_id"""
    # seq_id формата: job_0_n_120_id_0_seq00 → sample_name = job_0_n_120_id_0
    sample_name = seq_id.split("_seq")[0]
    ref_dir = Path(ref_base_dir) / experiment_name / sample_name
    
    ref_A = ref_dir / "pathA.pdb"
    ref_B = ref_dir / "pathB.pdb"
    
    # Пробуем альтернативные паттерны
    if not ref_A.exists():
        for pattern in [f"{sample_name}_pathA.pdb", "state1.pdb", "open.pdb"]:
            alt = ref_dir / pattern
            if alt.exists():
                ref_A = alt
                break
    
    if not ref_B.exists():
        for pattern in [f"{sample_name}_pathB.pdb", "state2.pdb", "closed.pdb"]:
            alt = ref_dir / pattern
            if alt.exists():
                ref_B = alt
                break
    
    return ref_A, ref_B

def run_validation_separate(fasta_files, ref_base_dir, output_dir, num_samples=120, batch_size=10, threshold=3.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Запуск валидации на {device}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for fasta_path in fasta_files:
        fasta_path = Path(fasta_path)
        if not fasta_path.exists():
            logger.warning(f"⚠️ FASTA не найден: {fasta_path}")
            continue
        
        # Извлекаем имя эксперимента из имени файла
        # inference_ucond_tri_result_mlpoff_0505_sequences.fasta → inference_ucond_tri_result_mlpoff_0505
        experiment_name = fasta_path.stem.replace("_sequences", "")
        logger.info(f"\n{'='*70}\nОбработка: {experiment_name}\n{'='*70}")
        
        records = list(SeqIO.parse(fasta_path, "fasta"))
        results = []
        
        for rec in tqdm(records, desc=f"BioEmu + Metrics ({experiment_name})"):
            seq_id = rec.id
            sequence = str(rec.seq)
            out_dir = Path(f"bioemu_ensembles/{experiment_name}/{seq_id}")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Ищем референсы
            ref_A, ref_B = find_ref_pdbs(seq_id, ref_base_dir, experiment_name)
            
            if not ref_A.exists() or not ref_B.exists():
                logger.warning(f"⚠️ Референсы не найдены для {seq_id} в {ref_A.parent}")
                results.append({
                    "experiment": experiment_name,
                    "id": seq_id,
                    "status": "MISSING_REFS"
                })
                continue
            
            try:
                # 1. Генерация ансамбля BioEmu
                bioemu_sample.main(
                    sequence=sequence,
                    num_samples=num_samples,
                    output_dir=str(out_dir),
                    batch_size_100=batch_size,
                    model_name="bioemu-v1.2",
                )
                
                # 2. Расчёт метрик
                rmsd_A_list, rmsd_B_list = [], []
                for i in range(num_samples):
                    gen_pdb = out_dir / f"{seq_id}_sample_{i:03d}.pdb"
                    if not gen_pdb.exists():
                        continue
                    rmsd_A_list.append(calc_ca_rmsd(str(ref_A), str(gen_pdb)))
                    rmsd_B_list.append(calc_ca_rmsd(str(ref_B), str(gen_pdb)))
                
                if not rmsd_A_list:
                    raise RuntimeError("Не удалось загрузить PDB-сэмплы")
                
                rmsd_A = np.array(rmsd_A_list)
                rmsd_B = np.array(rmsd_B_list)
                
                coverage_A = np.mean(rmsd_A < threshold)
                coverage_B = np.mean(rmsd_B < threshold)
                bimodal_score = min(coverage_A, coverage_B)
                
                results.append({
                    "experiment": experiment_name,
                    "id": seq_id,
                    "status": "OK",
                    "coverage_A": round(float(coverage_A), 3),
                    "coverage_B": round(float(coverage_B), 3),
                    "bimodal_score": round(float(bimodal_score), 3),
                    "rmsd_A_best": round(float(rmsd_A.min()), 2),
                    "rmsd_B_best": round(float(rmsd_B.min()), 2),
                    "num_samples": num_samples
                })
                
            except Exception as e:
                logger.error(f"❌ Ошибка на {seq_id}: {e}")
                results.append({
                    "experiment": experiment_name,
                    "id": seq_id,
                    "status": f"FAIL: {str(e)[:50]}"
                })
        
        # Сохраняем отдельный CSV для каждого эксперимента
        df = pd.DataFrame(results)
        df = df.sort_values("bimodal_score", ascending=False, na_position="last").reset_index(drop=True)
        csv_path = Path(output_dir) / f"{experiment_name}_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"✅ Результаты сохранены: {csv_path}")
        
        ok_count = (df["status"] == "OK").sum()
        finalist_count = (df["bimodal_score"] >= 0.15).sum()
        logger.info(f"📊 Финалисты (bimodal_score ≥ 0.15): {finalist_count} из {ok_count} успешных")
        
        all_results.extend(results)
    
    # Общий CSV со всеми экспериментами
    df_all = pd.DataFrame(all_results)
    df_all = df_all.sort_values("bimodal_score", ascending=False, na_position="last").reset_index(drop=True)
    all_csv = Path(output_dir) / "all_experiments_results.csv"
    df_all.to_csv(all_csv, index=False)
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ Общий CSV сохранён: {all_csv}")
    logger.info(f"📊 Всего финалистов: {(df_all['bimodal_score'] >= 0.15).sum()} из {len(df_all)}")
    logger.info(f"{'='*70}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioEmu validation для multi-state дизайнов (раздельные эксперименты)")
    parser.add_argument("--fasta_files", nargs="+", required=True, 
                        help="Список FASTA файлов (например, mpnn_output_fullseq_separate/*.fasta)")
    parser.add_argument("--ref_base_dir", required=True, 
                        help="Базовая директория с inference (например, /home/.../inference)")
    parser.add_argument("--output_dir", default="bioemu_results_separate",
                        help="Директория для выходных CSV")
    parser.add_argument("--num_samples", type=int, default=120,
                        help="Количество сэмплов BioEmu на последовательность")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Батч-размер для GPU")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="RMSD порог для 'попадания' (Å)")
    args = parser.parse_args()
    
    run_validation_separate(args.fasta_files, args.ref_base_dir, args.output_dir, 
                            args.num_samples, args.batch_size, args.threshold)