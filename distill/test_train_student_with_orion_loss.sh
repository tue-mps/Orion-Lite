#!/bin/bash
# Smoke-test the Orion-loss student distillation training on distill_data_test.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TRAIN_DIR="${TRAIN_DIR:-${REPO_ROOT}/distill_data_test/train}"
VAL_DIR="${VAL_DIR:-${REPO_ROOT}/distill_data_test/val}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/distill/test_runs}"
RUN_NAME="${RUN_NAME:-train_student_with_orion_loss_test}"
ORION_CKPT="${ORION_CKPT:-${REPO_ROOT}/ckpts/Orion.pth}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
USE_FEATURE_MIMIC_LOSS="${USE_FEATURE_MIMIC_LOSS:-1}"
WITH_BOUND_LOSS="${WITH_BOUND_LOSS:-1}"
USE_COL_LOSS="${USE_COL_LOSS:-1}"
MIMIC_LOSS="${MIMIC_LOSS:-l1}"

echo "[test_train_student_with_orion_loss] REPO_ROOT               : ${REPO_ROOT}"
echo "[test_train_student_with_orion_loss] TRAIN_DIR               : ${TRAIN_DIR}"
echo "[test_train_student_with_orion_loss] VAL_DIR                 : ${VAL_DIR}"
echo "[test_train_student_with_orion_loss] OUT_DIR                 : ${OUT_DIR}"
echo "[test_train_student_with_orion_loss] RUN_NAME                : ${RUN_NAME}"
echo "[test_train_student_with_orion_loss] ORION_CKPT              : ${ORION_CKPT}"
echo "[test_train_student_with_orion_loss] GPU                     : ${GPU}"
echo "[test_train_student_with_orion_loss] EPOCHS                  : ${EPOCHS}"
echo "[test_train_student_with_orion_loss] BATCH_SIZE              : ${BATCH_SIZE}"
echo "[test_train_student_with_orion_loss] VAL_BATCH_SIZE          : ${VAL_BATCH_SIZE}"
echo "[test_train_student_with_orion_loss] NUM_WORKERS             : ${NUM_WORKERS}"
echo "[test_train_student_with_orion_loss] USE_FEATURE_MIMIC_LOSS  : ${USE_FEATURE_MIMIC_LOSS}"
echo "[test_train_student_with_orion_loss] WITH_BOUND_LOSS         : ${WITH_BOUND_LOSS}"
echo "[test_train_student_with_orion_loss] USE_COL_LOSS            : ${USE_COL_LOSS}"
echo "[test_train_student_with_orion_loss] MIMIC_LOSS              : ${MIMIC_LOSS}"
echo ""

[[ -d "${TRAIN_DIR}" ]] || { echo "ERROR: TRAIN_DIR not found: ${TRAIN_DIR}"; exit 1; }
[[ -d "${VAL_DIR}" ]]   || { echo "ERROR: VAL_DIR not found: ${VAL_DIR}"; exit 1; }

mkdir -p "${OUT_DIR}"

ARGS=(
    --train-dir "${TRAIN_DIR}"
    --val-dir "${VAL_DIR}"
    --out-dir "${OUT_DIR}"
    --run-name "${RUN_NAME}"
    --device "cuda:0"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --val-batch-size "${VAL_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --mimic-loss "${MIMIC_LOSS}"
)

if [[ -f "${ORION_CKPT}" ]]; then
    ARGS+=(--orion-ckpt "${ORION_CKPT}")
fi

if [[ "${USE_FEATURE_MIMIC_LOSS}" == "1" ]]; then
    ARGS+=(--use-feature-mimic-loss)
fi

if [[ "${WITH_BOUND_LOSS}" == "1" ]]; then
    ARGS+=(--with-bound-loss)
fi

if [[ "${USE_COL_LOSS}" == "1" ]]; then
    ARGS+=(--use-col-loss)
fi

CUDA_VISIBLE_DEVICES="${GPU}" python "${REPO_ROOT}/distill/train_student_with_orion_loss.py" "${ARGS[@]}"
