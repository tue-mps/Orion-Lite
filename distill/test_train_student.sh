#!/bin/bash
# Smoke-test the plain student distillation training on distill_data_test.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TRAIN_DIR="${TRAIN_DIR:-${REPO_ROOT}/distill_data_test/train}"
VAL_DIR="${VAL_DIR:-${REPO_ROOT}/distill_data_test/val}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/distill/test_runs/train_student}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOSS="${LOSS:-mse}"

echo "[test_train_student] REPO_ROOT      : ${REPO_ROOT}"
echo "[test_train_student] TRAIN_DIR      : ${TRAIN_DIR}"
echo "[test_train_student] VAL_DIR        : ${VAL_DIR}"
echo "[test_train_student] CHECKPOINT_DIR : ${CHECKPOINT_DIR}"
echo "[test_train_student] GPU            : ${GPU}"
echo "[test_train_student] EPOCHS         : ${EPOCHS}"
echo "[test_train_student] BATCH_SIZE     : ${BATCH_SIZE}"
echo "[test_train_student] NUM_WORKERS    : ${NUM_WORKERS}"
echo "[test_train_student] LOSS           : ${LOSS}"
echo ""

[[ -d "${TRAIN_DIR}" ]] || { echo "ERROR: TRAIN_DIR not found: ${TRAIN_DIR}"; exit 1; }
[[ -d "${VAL_DIR}" ]]   || { echo "ERROR: VAL_DIR not found: ${VAL_DIR}"; exit 1; }

mkdir -p "${CHECKPOINT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python "${REPO_ROOT}/distill/train_student.py" \
    --train_dir "${TRAIN_DIR}" \
    --val_dir "${VAL_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --loss "${LOSS}"
