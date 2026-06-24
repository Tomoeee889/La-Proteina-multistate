# run_bioemu_from_pdb.py
from bioemu import sample
from Bio import SeqIO
from Bio.PDB import PDBParser
from pathlib import Path
import argparse

# Словарь конвертации трёхбуквенных кодов в однобуквенные
THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # Нестандартные/неизвестные → 'X'
    'UNK': 'X', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}

def get_sequence_from_pdb(pdb_path):
    """Извлекает последовательность из PDB-файла"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    
    # Берем первую модель и первую цепь
    model = structure[0]
    chain = list(model.get_chains())[0]
    
    # Извлекаем последовательность из остатков
    seq = []
    for residue in chain.get_residues():
        # Пропускаем гетероатомы (воду, лиганды)
        if residue.id[0] == ' ':
            resname = residue.resname
            seq.append(THREE_TO_ONE.get(resname, 'X'))
    
    return ''.join(seq)

def run_bioemu_from_pdb(pdb_dir, output_base, num_samples=200, batch_size=30):
    """Запускает BioEmu на последовательностях из PDB-файлов"""
    pdb_dir = Path(pdb_dir)
    
    # Находим все эксперименты
    experiments = [d for d in pdb_dir.iterdir() if d.is_dir()]
    
    for exp_dir in experiments:
        exp_name = exp_dir.name
        print(f'\n🧬 Эксперимент: {exp_name}')
        
        # Находим все job-папки
        job_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
        print(f"Всего job-папок: {len(job_dirs)}")
        
        for i, job_dir in enumerate(job_dirs, 1):
            job_id = job_dir.name
            
            # Ищем PDB-файл (берем pathA.pdb или первый доступный PDB)
            pdb_file = job_dir / "pathA.pdb"
            if not pdb_file.exists():
                # Пробуем найти любой PDB
                pdb_files = list(job_dir.glob("*.pdb"))
                if pdb_files:
                    pdb_file = pdb_files[0]
                else:
                    print(f"[{i}/{len(job_dirs)}] ⚠️ {job_id} — нет PDB-файлов")
                    continue
            
            # Извлекаем последовательность
            try:
                sequence = get_sequence_from_pdb(pdb_file)
                if len(sequence) == 0:
                    print(f"[{i}/{len(job_dirs)}] ⚠️ {job_id} — пустая последовательность")
                    continue
            except Exception as e:
                print(f"[{i}/{len(job_dirs)}] ❌ {job_id} — ошибка чтения PDB: {e}")
                continue
            
            # Создаем выходную директорию
            out_dir = Path(output_base) / exp_name / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Проверяем, есть ли уже результат
            if (out_dir / "topology.pdb").exists():
                print(f"[{i}/{len(job_dirs)}] ⏭️ {job_id} — уже готово")
                continue
            
            print(f"[{i}/{len(job_dirs)}] 🔄 {job_id} (len={len(sequence)})", end=" ... ", flush=True)
            
            try:
                sample.main(
                    sequence=sequence,
                    num_samples=num_samples,
                    output_dir=str(out_dir),
                    batch_size_100=batch_size,
                    model_name='bioemu-v1.2',
                )
                print("✅")
            except Exception as e:
                print(f"❌ {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск BioEmu на последовательностях из PDB-файлов La-Proteina")
    parser.add_argument("--pdb_dir", required=True, help="Директория с результатами La-Proteina")
    parser.add_argument("--output_base", default="bioemu_ensembles_from_pdb", help="Базовая директория для выходных ансамблей")
    parser.add_argument("--num_samples", type=int, default=200, help="Количество сэмплов BioEmu")
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size для BioEmu")
    
    args = parser.parse_args()
    
    run_bioemu_from_pdb(args.pdb_dir, args.output_base, args.num_samples, args.batch_size)