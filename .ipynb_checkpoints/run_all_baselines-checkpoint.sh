#!/bin/bash
###############################################################################
#  run_baselines.sh
#  Запуск ОДНОГО таска в ОДНОМ режиме за раз.
#  Меняй параметры вверху и запускай скрипт сколько нужно раз.
#
#  Примеры:
#    TASK=1  MODE=sim_eps  SCALES="0.30 0.40 0.70 1.0"  → добить task1 sim_eps
#    TASK=2  MODE=default  SCALES="0.00 0.05 ... 1.0"   → весь task2 default
#    TASK=3  MODE=sim_eps  SCALES="2.0 3.0 5.0 100.0"   → часть task3 sim_eps
###############################################################################

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  РЕДАКТИРУЙ ТОЛЬКО ЭТИ 4 ПАРАМЕТРА                                       │
#  └─────────────────────────────────────────────────────────────────────────┘

TASK=3               # 1, 2 или 3
MODE="sim_eps"        # "default" или "sim_eps"
SCALES="2.0 3.0 5.0 10.0 20.0 50.0 100.0"   # noise scales через пробел
NUM_PAIRS=50          # сколько пар на каждый noise_scale

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  ДАЛЬШЕ НИЧЕГО НЕ МЕНЯЙ                                                │
#  └─────────────────────────────────────────────────────────────────────────┘

# ── Motif параметры (нужны только для task 2 и 3) ──
MOTIF_NAME="1BCF_AA_single"
MOTIF_PDB="./motifs/1bcf_aa.pdb"
CONTIG_STRING="A1-9/31/A41-158"
MOTIF_MIN_LENGTH=158
MOTIF_MAX_LENGTH=158

# ── Параметры режимов ──
if [ "$MODE" = "default" ]; then
    DECODE_MODE="independent"
    SIM_EPS="false"     # независимый RNG
elif [ "$MODE" = "sim_eps" ]; then
    DECODE_MODE="independent"
    SIM_EPS="true"      # общий RNG
else
    echo "ERROR: MODE must be 'default' or 'sim_eps'"
    exit 1
fi

# ── Определяем корень проекта ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
export DATA_PATH=./data

# ── Имя скрипта и выходная директория ──
SCRIPT="run_baseline_task${TASK}.py"
OUTPUT_DIR="baselines/task${TASK}_${MODE}"

echo "============================================"
echo "  TASK=${TASK}  MODE=${MODE}"
echo "  decode_mode=${DECODE_MODE}  sim_eps=${SIM_EPS}"
echo "  scales: ${SCALES}"
echo "  output: ${OUTPUT_DIR}"
echo "============================================"

# ── Motif аргументы для task 2 и 3 ──
EXTRA_ARGS=""
if [ "$TASK" = "2" ] || [ "$TASK" = "3" ]; then
    EXTRA_ARGS="--motif_name $MOTIF_NAME --motif_pdb $MOTIF_PDB --contig_string $CONTIG_STRING --motif_min_length $MOTIF_MIN_LENGTH --motif_max_length $MOTIF_MAX_LENGTH"
fi

# ── Цикл по noise scales ──
FAILED=0
DONE=0
TOTAL=$(echo $SCALES | wc -w)

for scale in $SCALES; do
    DONE=$((DONE + 1))
    echo ""
    echo "----------------------------------------------------------"
    echo "  [${DONE}/${TOTAL}] noise_scale=${scale}"
    echo "----------------------------------------------------------"

    python "$SCRIPT" \
        --noise_scale "$scale" \
        --num_pairs "$NUM_PAIRS" \
        --output_dir "$OUTPUT_DIR" \
        --decode_mode "$DECODE_MODE" \
        --sim_eps "$SIM_EPS" \
        $EXTRA_ARGS

    rc=$?
    if [ $rc -eq 0 ]; then
        echo "[OK] noise=${scale}"
    else
        echo "[FAIL] noise=${scale} (exit ${rc})"
        FAILED=$((FAILED + 1))
    fi

    # Пауза для очистки GPU между запусками
    if [ $DONE -lt $TOTAL ]; then
        echo "  GPU cleanup pause 10s..."
        sleep 10
    fi
done

echo ""
echo "============================================"
echo "  DONE: ${DONE} runs, ${FAILED} failed"
echo "  Results: ${OUTPUT_DIR}/"
echo "============================================"
