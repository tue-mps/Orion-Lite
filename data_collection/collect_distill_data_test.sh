#!/bin/bash
# Collect a small distillation subset for pipeline testing.
#
# Default behavior:
#   - validation split: 10 samples
#   - training split:   20 samples
#
# Usage:
#   bash data_collection/collect_distill_data_test.sh [all|val|train]
#
# Environment variables (override defaults):
#   ORION_CKPT    : path to Orion.pth checkpoint
#   DATA_ROOT     : path to bench2drive data directory
#   INFO_ROOT     : path to directory containing b2d_infos_*.pkl files
#   OUT_DIR       : root output directory for .npz files
#   GPU           : GPU id to run on (default: 0)
#   NUM_WORKERS   : dataloader workers (default: 4)
#   VAL_SAMPLES   : number of val samples (default: 10)
#   TRAIN_SAMPLES : number of train samples (default: 20)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPLIT="${1:-all}"

ORION_CKPT="${ORION_CKPT:-${REPO_ROOT}/ckpts/Orion.pth}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../Orion/data/bench2drive}"
INFO_ROOT="${INFO_ROOT:-${REPO_ROOT}/../Orion/data/infos}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/distill_data_test}"
GPU="${GPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_SAMPLES="${VAL_SAMPLES:-10}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-20}"

CONFIG="${CONFIG:-${REPO_ROOT}/adzoo/orion/configs/orion_stage3_infer.py}"
RUNNER="${REPO_ROOT}/data_collection/run_orion_collect.py"

echo "[collect_distill_data_test] REPO_ROOT      : ${REPO_ROOT}"
echo "[collect_distill_data_test] ORION_CKPT     : ${ORION_CKPT}"
echo "[collect_distill_data_test] DATA_ROOT      : ${DATA_ROOT}"
echo "[collect_distill_data_test] INFO_ROOT      : ${INFO_ROOT}"
echo "[collect_distill_data_test] OUT_DIR        : ${OUT_DIR}"
echo "[collect_distill_data_test] SPLIT          : ${SPLIT}"
echo "[collect_distill_data_test] VAL_SAMPLES    : ${VAL_SAMPLES}"
echo "[collect_distill_data_test] TRAIN_SAMPLES  : ${TRAIN_SAMPLES}"
echo ""

[[ -f "${ORION_CKPT}" ]] || { echo "ERROR: ORION_CKPT not found: ${ORION_CKPT}"; exit 1; }
[[ -d "${DATA_ROOT}" ]]  || { echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}";  exit 1; }
[[ -d "${INFO_ROOT}" ]]  || { echo "ERROR: INFO_ROOT not found: ${INFO_ROOT}";  exit 1; }
[[ -f "${CONFIG}" ]]     || { echo "ERROR: Config not found: ${CONFIG}";         exit 1; }
[[ -f "${RUNNER}" ]]     || { echo "ERROR: Runner not found: ${RUNNER}";         exit 1; }

mkdir -p "${OUT_DIR}/train" "${OUT_DIR}/val"

run_split() {
    local split_name="$1"
    local max_samples="$2"
    local save_dir="${OUT_DIR}/${split_name}"

    echo "============================================================"
    echo "[collect_distill_data_test] Collecting split=${split_name} max_samples=${max_samples}"
    echo "  save_dir -> ${save_dir}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${GPU}" python "${RUNNER}" \
        --config      "${CONFIG}" \
        --checkpoint  "${ORION_CKPT}" \
        --save-dir    "${save_dir}" \
        --split       "${split_name}" \
        --max-samples "${max_samples}" \
        --gpu         "${GPU}" \
        --num-workers "${NUM_WORKERS}" \
        --data-root   "${DATA_ROOT}" \
        --info-root   "${INFO_ROOT}"

    local count
    count=$(find "${save_dir}" -name "*.npz" | wc -l)
    echo "[collect_distill_data_test] Done ${split_name}: ${count} .npz files in ${save_dir}"
}

case "${SPLIT}" in
    train) run_split train "${TRAIN_SAMPLES}" ;;
    val)   run_split val   "${VAL_SAMPLES}"   ;;
    all)   run_split val "${VAL_SAMPLES}"; run_split train "${TRAIN_SAMPLES}" ;;
    *)
        echo "Invalid split '${SPLIT}'. Use train, val, or all."
        exit 1 ;;
esac

echo ""
echo "[collect_distill_data_test] Collection complete."
echo "  train: $(find "${OUT_DIR}/train" -name "*.npz" 2>/dev/null | wc -l) files"
echo "  val:   $(find "${OUT_DIR}/val"   -name "*.npz" 2>/dev/null | wc -l) files"
