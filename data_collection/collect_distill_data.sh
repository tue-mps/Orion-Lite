#!/bin/bash
# Collect distillation data by running Orion teacher inference with OrionCollect.
#
# OrionCollect is a detector subclass that has the save-tensors logic built in.
# No patching of orion.py is needed.
#
# Usage:
#   bash data_collection/collect_distill_data.sh [train|val|all]  (default: all)
#
# Environment variables (override defaults below):
#   ORION_CKPT   : path to Orion.pth checkpoint
#   DATA_ROOT    : path to bench2drive data directory
#   INFO_ROOT    : path to directory containing b2d_infos_*.pkl files
#   OUT_DIR      : root output directory for .npz files
#   GPU          : GPU id to run on (default: 0)
#   NUM_WORKERS  : dataloader workers (default: 4)
#
# Example:
#   ORION_CKPT=/path/to/Orion.pth bash data_collection/collect_distill_data.sh val

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPLIT="${1:-all}"

ORION_CKPT="${ORION_CKPT:-${REPO_ROOT}/ckpts/Orion.pth}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../Orion/data/bench2drive}"
INFO_ROOT="${INFO_ROOT:-${REPO_ROOT}/../Orion/data/infos}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/distill_data}"
GPU="${GPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

CONFIG="${CONFIG:-${REPO_ROOT}/adzoo/orion/configs/orion_stage3_infer.py}"
RUNNER="${REPO_ROOT}/data_collection/run_orion_collect.py"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
echo "[collect_distill_data] REPO_ROOT  : ${REPO_ROOT}"
echo "[collect_distill_data] ORION_CKPT : ${ORION_CKPT}"
echo "[collect_distill_data] DATA_ROOT  : ${DATA_ROOT}"
echo "[collect_distill_data] INFO_ROOT  : ${INFO_ROOT}"
echo "[collect_distill_data] OUT_DIR    : ${OUT_DIR}"
echo "[collect_distill_data] SPLIT      : ${SPLIT}"
echo ""

[[ -f "${ORION_CKPT}" ]] || { echo "ERROR: ORION_CKPT not found: ${ORION_CKPT}"; exit 1; }
[[ -d "${DATA_ROOT}" ]]  || { echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}";  exit 1; }
[[ -d "${INFO_ROOT}" ]]  || { echo "ERROR: INFO_ROOT not found: ${INFO_ROOT}";  exit 1; }
[[ -f "${CONFIG}" ]]     || { echo "ERROR: Config not found: ${CONFIG}";         exit 1; }
[[ -f "${RUNNER}" ]]     || { echo "ERROR: Runner not found: ${RUNNER}";         exit 1; }

mkdir -p "${OUT_DIR}/train" "${OUT_DIR}/val"

# ---------------------------------------------------------------------------
# Helper: run one split
# ---------------------------------------------------------------------------
run_split() {
    local split_name="$1"
    local save_dir="${OUT_DIR}/${split_name}"

    echo "============================================================"
    echo "[collect_distill_data] Collecting split=${split_name}"
    echo "  save_dir -> ${save_dir}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${GPU}" python "${RUNNER}" \
        --config      "${CONFIG}" \
        --checkpoint  "${ORION_CKPT}" \
        --save-dir    "${save_dir}" \
        --split       "${split_name}" \
        --gpu         "${GPU}" \
        --num-workers "${NUM_WORKERS}" \
        --data-root   "${DATA_ROOT}" \
        --info-root   "${INFO_ROOT}"

    local count
    count=$(find "${save_dir}" -name "*.npz" | wc -l)
    echo "[collect_distill_data] Done ${split_name}: ${count} .npz files in ${save_dir}"
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
case "${SPLIT}" in
    train) run_split train ;;
    val)   run_split val   ;;
    all)   run_split val; run_split train ;;
    *)
        echo "Invalid split '${SPLIT}'. Use train, val, or all."
        exit 1 ;;
esac

echo ""
echo "[collect_distill_data] Collection complete."
echo "  train: $(find "${OUT_DIR}/train" -name "*.npz" 2>/dev/null | wc -l) files"
echo "  val:   $(find "${OUT_DIR}/val"   -name "*.npz" 2>/dev/null | wc -l) files"
