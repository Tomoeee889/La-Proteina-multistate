#!/bin/bash

# ==================== ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ====================
export JAX_DEFAULT_MATMUL_PRECISION=float32
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1

# ==================== ПУТИ ====================
MSA_DIR="baselines_proteinmpnn"
OUT_DIR="bioemu_baselines"

# ==================== НАСТРОЙКИ ====================
NUM_SAMPLES=100
BATCH_SIZE_100=100  # <-- ОПТИМАЛЬНО ДЛЯ RTX A4500 (20GB) И ДЛИНЫ 120

# ==================== СЧЕТЧИКИ ====================
total=$(ls -1 "$MSA_DIR"/*.a3m 2>/dev/null | wc -l)
counter=0

echo "================================================================"
echo "ЗАПУСК BIOEMU (РЕЖИМ АНСАМБЛЯ: $NUM_SAMPLES СЭМПЛОВ)"
echo "================================================================"
echo "Входная папка (MSA): $MSA_DIR"
echo "Выходная папка:      $OUT_DIR"
echo "Всего .a3m файлов:   $total"
echo "Параметр batch_size_100: $BATCH_SIZE_100"
echo "================================================================"
echo ""

# ==================== ЦИКЛ ПО ФАЙЛАМ ====================
for a3m_file in "$MSA_DIR"/*.a3m; do
    counter=$((counter + 1))
    
    seq_name=$(basename "$a3m_file" .a3m)
    seq_out="$OUT_DIR/$seq_name"
    
    if [ -d "$seq_out" ] && [ "$(ls -A "$seq_out"/*.npz 2>/dev/null)" ]; then
        num_done=$(ls -1 "$seq_out"/*.npz 2>/dev/null | wc -l)
        if [ "$num_done" -ge "$NUM_SAMPLES" ]; then
            echo "[$counter/$total] ✅ Пропуск (уже готово $NUM_SAMPLES сэмплов): $seq_name"
            continue
        else
            echo "[$counter/$total] 🔄 Докачка (готово $num_done из $NUM_SAMPLES): $seq_name"
        fi
    fi
    
    echo "[$counter/$total] 🚀 Запуск BioEmu для: $seq_name"
    
    python -m bioemu.sample \
        --sequence "$a3m_file" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size_100 "$BATCH_SIZE_100" \
        --output_dir "$seq_out" \
        --filter_samples=True 2>&1 | tail -n 3
    
    echo "[$counter/$total] 🏁 Завершена итерация для: $seq_name"
    echo "--------------------------------------------------------"
done

echo "================================================================"
echo "ВСЕ ГОТОВО! Обработано $counter из $total."
echo "================================================================"
