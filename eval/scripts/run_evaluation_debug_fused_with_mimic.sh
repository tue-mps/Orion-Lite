#!/bin/bash
# Single-GPU sequential Bench2Drive evaluation for the fused 6-layer mimic model.
# This mirrors the old A100 debug script as closely as possible, but uses the
# Orion-Lite fused config/checkpoint flow.

set -euo pipefail

ORION_LITE_ROOT="${ORION_LITE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
B2D_ROOT="${B2D_ROOT:-/mnt/adas7tb/jgu/Bench2Drive}"

export ORION_LITE_ROOT
export B2D_ROOT
export PYTHONPATH="${ORION_LITE_ROOT}:${B2D_ROOT}:${PYTHONPATH:-}"
# Debug runs are usually used for clean apples-to-apples comparison.
export RESUME="${RESUME:-False}"

# Defaults for the released 6-layer mimic student.
export ORION_STUDENT_INPUT_DIM="${ORION_STUDENT_INPUT_DIM:-4096}"
export ORION_STUDENT_HIDDEN_DIM="${ORION_STUDENT_HIDDEN_DIM:-1024}"
export ORION_STUDENT_OUTPUT_DIM="${ORION_STUDENT_OUTPUT_DIM:-4096}"
export ORION_STUDENT_NUM_LAYERS="${ORION_STUDENT_NUM_LAYERS:-6}"
export ORION_STUDENT_NUM_HEADS="${ORION_STUDENT_NUM_HEADS:-16}"
export ORION_STUDENT_DROPOUT="${ORION_STUDENT_DROPOUT:-0.1}"
export ORION_STUDENT_WITH_BOUND_LOSS="${ORION_STUDENT_WITH_BOUND_LOSS:-True}"
export ORION_STUDENT_USE_COL_LOSS="${ORION_STUDENT_USE_COL_LOSS:-True}"

BASE_PORT="${BASE_PORT:-30000}"
BASE_TM_PORT="${BASE_TM_PORT:-50000}"
IS_BENCH2DRIVE="${IS_BENCH2DRIVE:-True}"
BASE_ROUTES="${BASE_ROUTES:-leaderboard/data/bench2drive220}"
TEAM_AGENT="${TEAM_AGENT:-${ORION_LITE_ROOT}/team_code/orion_b2d_agent.py}"
CONFIG_PATH="${CONFIG_PATH:-${ORION_LITE_ROOT}/adzoo/orion/configs/orion_distill_new_agent_fused.py}"
CKPT_PATH="${CKPT_PATH:-${ORION_LITE_ROOT}/eval/fused_ckpts/orion_student_with_mimic_loss.pth}"
RUN_TAG="${RUN_TAG:-orion_distilled_6l_with_mimic_fused_debug}"
BASE_CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT:-eval_mimic_fused}"
SAVE_PATH="${SAVE_PATH:-./eval_mimic_fused/}"
PLANNER_TYPE="${PLANNER_TYPE:-only_traj}"
GPU_RANK="${GPU_RANK:-0}"

PORT="${PORT:-${BASE_PORT}}"
TM_PORT="${TM_PORT:-${BASE_TM_PORT}}"
ROUTES="${ROUTES:-${BASE_ROUTES}.xml}"
CHECKPOINT_ENDPOINT="${CHECKPOINT_ENDPOINT:-${BASE_CHECKPOINT_ENDPOINT}.json}"
TEAM_CONFIG="${TEAM_CONFIG:-${CONFIG_PATH}+${CKPT_PATH}+${RUN_TAG}}"

assert_paths() {
  [[ -f "${TEAM_AGENT}" ]]  || { echo "Missing TEAM_AGENT: ${TEAM_AGENT}"; exit 2; }
  [[ -f "${CONFIG_PATH}" ]] || { echo "Missing CONFIG_PATH: ${CONFIG_PATH}"; exit 2; }
  [[ -f "${CKPT_PATH}" ]]   || { echo "Missing CKPT_PATH: ${CKPT_PATH}"; exit 2; }
  if [[ "${ROUTES}" = /* ]]; then
    [[ -f "${ROUTES}" ]] || { echo "Missing ROUTES xml: ${ROUTES}"; exit 2; }
  else
    [[ -f "${B2D_ROOT}/${ROUTES}" ]] || { echo "Missing ROUTES xml: ${B2D_ROOT}/${ROUTES}"; exit 2; }
  fi
}

print_settings() {
  echo "[debug_fused_mimic] ORION_LITE_ROOT : ${ORION_LITE_ROOT}"
  echo "[debug_fused_mimic] B2D_ROOT        : ${B2D_ROOT}"
  echo "[debug_fused_mimic] CONFIG_PATH     : ${CONFIG_PATH}"
  echo "[debug_fused_mimic] CKPT_PATH       : ${CKPT_PATH}"
  echo "[debug_fused_mimic] TEAM_AGENT      : ${TEAM_AGENT}"
  echo "[debug_fused_mimic] ROUTES          : ${ROUTES}"
  echo "[debug_fused_mimic] CHECKPOINT_JSON : ${CHECKPOINT_ENDPOINT}"
  echo "[debug_fused_mimic] SAVE_PATH       : ${SAVE_PATH}"
  echo "[debug_fused_mimic] PLANNER_TYPE    : ${PLANNER_TYPE}"
  echo "[debug_fused_mimic] GPU_RANK        : ${GPU_RANK}"
  echo "[debug_fused_mimic] RESUME          : ${RESUME}"
  echo "[debug_fused_mimic] TEAM_CONFIG     : ${TEAM_CONFIG}"
}

cd "${B2D_ROOT}"
assert_paths
mkdir -p "${SAVE_PATH}"
print_settings

bash leaderboard/scripts/run_evaluation.sh \
  "${PORT}" "${TM_PORT}" "${IS_BENCH2DRIVE}" \
  "${ROUTES}" "${TEAM_AGENT}" "${TEAM_CONFIG}" \
  "${CHECKPOINT_ENDPOINT}" "${SAVE_PATH}" "${PLANNER_TYPE}" "${GPU_RANK}"
