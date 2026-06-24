# collect_multistate_seqs.py
# mamba run -n laproteina_env python collect_multistate_seqs.py

import os
import argparse

# === КОНФИГУРАЦИЯ ===
WORK_DIR  = "/home/domain/aristowi/la-proteina-main/mpnn_kuhlman"
OUTPUT_FA = "/home/domain/aristowi/la-proteina-main/mpnn_output_tied/multistate_sequences_scaffold.fasta"
PREFIX_FILTER = "motif"  # Собирать только из папок с этим префиксом

# === АРГУМЕНТЫ КОМАНДНОЙ СТРОКИ ===
arg_parser = argparse.ArgumentParser(description="Collect multistate sequences from MPNN output")
arg_parser.add_argument("--work_dir", default=WORK_DIR, help="Директория с результатами MPNN")
arg_parser.add_argument("--output", default=OUTPUT_FA, help="Выходной FASTA файл")
arg_parser.add_argument("--prefix", default=PREFIX_FILTER, help="Префикс директорий для сбора (по умолчанию: inpainting_)")
arg_parser.add_argument("--no-filter", action="store_true", help="Отключить фильтрацию по префиксу")
args = arg_parser.parse_args()

WORK_DIR = args.work_dir
OUTPUT_FA = args.output
PREFIX_FILTER = None if args.no_filter else args.prefix

os.makedirs(os.path.dirname(OUTPUT_FA), exist_ok=True)

all_sequences = {}
processed_dirs = 0
skipped_dirs = 0

for sample_name in sorted(os.listdir(WORK_DIR)):
    # === ФИЛЬТР ПО ПРЕФИКСУ ===
    if PREFIX_FILTER and not PREFIX_FILTER in sample_name:
        skipped_dirs += 1
        continue
    
    sample_dir = os.path.join(WORK_DIR, sample_name)
    
    # Пропускаем если это не директория
    if not os.path.isdir(sample_dir):
        continue
    
    msd_fasta = os.path.join(sample_dir, "outs", "msd.fasta")

    if not os.path.exists(msd_fasta):
        print(f"[SKIP] {sample_name} — нет msd.fasta")
        continue

    with open(msd_fasta) as f:
        lines = f.readlines()

    # Парсим FASTA вручную
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

# Сохраняем
with open(OUTPUT_FA, "w") as f:
    for name, (seq, score) in all_sequences.items():
        f.write(f">{name} {score}\n{seq}\n")

print(f"\n=== ИТОГ ===")
print(f"Обработано директорий: {processed_dirs}")
print(f"Пропущено (фильтр):    {skipped_dirs}")
print(f"Всего последовательностей: {len(all_sequences)}")
print(f"Сохранено в {OUTPUT_FA}")

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