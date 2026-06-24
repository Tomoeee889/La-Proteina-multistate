# collect_fullseq_multistate_seqs_separate.py
# mamba run -n laproteina_env python collect_fullseq_multistate_seqs_separate.py

import os
import argparse

# === КОНФИГУРАЦИЯ ===
BASE_WORK_DIR = "/home/domain/aristowi/la-proteina-main/mpnn_fullseq_separate"
OUTPUT_BASE_DIR = "/home/domain/aristowi/la-proteina-main/mpnn_output_fullseq_separate"

# === АРГУМЕНТЫ ===
arg_parser = argparse.ArgumentParser(description="Collect full-sequence multistate sequences (separate outputs)")
arg_parser.add_argument("--base_work_dir", default=BASE_WORK_DIR, help="Базовая рабочая директория с результатами")
arg_parser.add_argument("--output_base_dir", default=OUTPUT_BASE_DIR, help="Базовая директория для выходных FASTA")
args = arg_parser.parse_args()

BASE_WORK_DIR = args.base_work_dir
OUTPUT_BASE_DIR = args.output_base_dir

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# === ОБРАБОТКА КАЖДОЙ ДИРЕКТОРИИ ОТДЕЛЬНО ===
for dir_name in sorted(os.listdir(BASE_WORK_DIR)):
    work_dir = os.path.join(BASE_WORK_DIR, dir_name)
    
    if not os.path.isdir(work_dir):
        continue
    
    print(f"\n{'='*70}")
    print(f"Сбор результатов: {dir_name}")
    print(f"{'='*70}\n")
    
    all_sequences = {}
    processed_dirs = 0
    skipped_dirs = 0
    
    for sample_name in sorted(os.listdir(work_dir)):
        sample_dir = os.path.join(work_dir, sample_name)
        
        if not os.path.isdir(sample_dir):
            continue
        
        msd_fasta = os.path.join(sample_dir, "outs", "msd.fasta")
        
        if not os.path.exists(msd_fasta):
            print(f"[SKIP] {sample_name} — нет msd.fasta")
            skipped_dirs += 1
            continue
        
        with open(msd_fasta) as f:
            lines = f.readlines()
        
        # Парсим FASTA
        entries = []
        current_header = None
        for line in lines:
            line = line.strip()
            if line.startswith(">"):
                current_header = line
            elif current_header and line:
                entries.append((current_header, line))
                current_header = None
        
        # Пропускаем первую запись (оригинальная структура)
        designed = entries[1:]
        
        print(f"[OK] {sample_name}: {len(designed)} последовательностей")
        processed_dirs += 1
        
        for i, (header, seq) in enumerate(designed):
            # Берём только chain A (до "/"), chain B идентичен
            seq_a = seq.split("/")[0]
            
            # Извлекаем score из header
            score = ""
            for part in header.split(","):
                if "score=" in part:
                    score = part.strip()
                    break
            
            name = f"{sample_name}_seq{i:02d}"
            all_sequences[name] = (seq_a, score)
    
    # Сохраняем в отдельный файл для этой директории
    output_fa = os.path.join(OUTPUT_BASE_DIR, f"{dir_name}_sequences.fasta")
    
    with open(output_fa, "w") as f:
        for name, (seq, score) in all_sequences.items():
            f.write(f">{name} {score}\n{seq}\n")
    
    print(f"\n=== ИТОГ для {dir_name} ===")
    print(f"Обработано директорий: {processed_dirs}")
    print(f"Пропущено: {skipped_dirs}")
    print(f"Всего последовательностей: {len(all_sequences)}")
    print(f"Сохранено в {output_fa}")
    
    # Статистика
    lengths = [len(s) for s, _ in all_sequences.values()]
    if lengths:
        print(f"Длины: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)//len(lengths)}")
    
    # Превью первых 3
    print("\nПервые 3 записи:")
    for i, (name, (seq, score)) in enumerate(all_sequences.items()):
        if i >= 3:
            break
        print(f"  >{name} {score}")
        print(f"  {seq[:60]}...")

print(f"\n{'='*70}")
print(f"Все результаты собраны!")
print(f"Выходные файлы в: {OUTPUT_BASE_DIR}/")
for f in sorted(os.listdir(OUTPUT_BASE_DIR)):
    if f.endswith(".fasta"):
        print(f"  - {f}")
print(f"{'='*70}")