#!/bin/bash

# Переходим в папку с ProteinMPNN
cd ~/ProteinMPNN

# Базовая папка с бэкбонами
BASELINES_DIR=~/la-proteina-main/baselines

# Папка для результатов
OUTPUT_DIR=~/la-proteina-main/mpnn_results

# Создаём выходную папку
mkdir -p "$OUTPUT_DIR"

# Счётчик обработанных файлов
counter=0
total=$(find "$BASELINES_DIR" -type f -name "*.pdb" ! -path "*/.ipynb_checkpoints/*" | wc -l)

echo "Найдено $total PDB файлов для обработки"
echo "Результаты будут сохранены в: $OUTPUT_DIR"
echo "========================================="

# Находим все .pdb файлы, исключая .ipynb_checkpoints
find "$BASELINES_DIR" -type f -name "*.pdb" ! -path "*/.ipynb_checkpoints/*" | while read pdb_file; do
    counter=$((counter + 1))
    
    # Получаем относительный путь от baselines
    rel_path=${pdb_file#$BASELINES_DIR/}
    
    # Извлекаем имя файла без расширения
    filename=$(basename "$pdb_file" .pdb)
    
    # Создаём выходную папку с той же структурой
    out_subdir="$OUTPUT_DIR/$(dirname "$rel_path")/$filename"
    mkdir -p "$out_subdir"
    
    echo "[$counter/$total] Обработка: $rel_path"
    
    # Запускаем ProteinMPNN
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python protein_mpnn_run.py \
        --pdb_path "$pdb_file" \
        --out_folder "$out_subdir" \
        --num_seq_per_target 8 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1 2>&1 | tail -5
    
    echo "  -> Сохранено в: $out_subdir"
    echo ""
done

echo "========================================="
echo "Готово! Все $total бэкбонов обработаны."
echo "Результаты в: $OUTPUT_DIR"