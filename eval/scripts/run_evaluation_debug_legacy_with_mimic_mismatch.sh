#!/bin/bash
# Single-GPU sequential Bench2Drive evaluation for the legacy two-checkpoint
# Orion mimic setup, focused on the currently mismatched routes only.
#
# "Legacy" here means:
#   - config builds OrionDistilledNew
#   - config loads student_model_path=.../last.pt
#   - agent then loads Orion.pth as the outer checkpoint
#
# We intentionally use the current Bench2Drive runner on this machine because
# the backup runner hardcodes an old A100 CARLA path that no longer exists.

set -euo pipefail

ORION_LITE_ROOT="${ORION_LITE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
ORION_ROOT="${ORION_ROOT:-/mnt/adas7tb/jgu/Orion}"
B2D_ROOT="${B2D_ROOT:-/mnt/adas7tb/jgu/Bench2Drive}"
RUN_EVAL_SCRIPT="${RUN_EVAL_SCRIPT:-${B2D_ROOT}/leaderboard/scripts/run_evaluation.sh}"

export ORION_LITE_ROOT
export ORION_ROOT
export B2D_ROOT
export PYTHONPATH="${ORION_ROOT}:${B2D_ROOT}:${PYTHONPATH:-}"
export RESUME="${RESUME:-False}"

BASE_PORT="${BASE_PORT:-30000}"
BASE_TM_PORT="${BASE_TM_PORT:-50000}"
IS_BENCH2DRIVE="${IS_BENCH2DRIVE:-True}"
TEAM_AGENT="${TEAM_AGENT:-${ORION_ROOT}/team_code/orion_b2d_agent.py}"
CONFIG_PATH="${CONFIG_PATH:-${ORION_ROOT}/adzoo/orion/configs/orion_distill_new_agent_with_mimic.py}"
CKPT_PATH="${CKPT_PATH:-${ORION_ROOT}/ckpts/Orion.pth}"
RUN_TAG="${RUN_TAG:-orion_mimic_legacy_diff}"
ROUTES="${ROUTES:-${ORION_LITE_ROOT}/eval/route_diffs/orion_distilled_6l_with_mimic_fused_vs_eval_mimic/mismatch_routes.xml}"
CHECKPOINT_ENDPOINT="${CHECKPOINT_ENDPOINT:-eval_mimic_legacy_diff.json}"
SAVE_PATH="${SAVE_PATH:-./eval_mimic_legacy_diff/}"
PLANNER_TYPE="${PLANNER_TYPE:-only_traj}"
GPU_RANK="${GPU_RANK:-0}"

PORT="${PORT:-${BASE_PORT}}"
TM_PORT="${TM_PORT:-${BASE_TM_PORT}}"
TEAM_CONFIG="${TEAM_CONFIG:-${CONFIG_PATH}+${CKPT_PATH}+${RUN_TAG}}"

assert_paths() {
  [[ -f "${RUN_EVAL_SCRIPT}" ]] || { echo "Missing RUN_EVAL_SCRIPT: ${RUN_EVAL_SCRIPT}"; exit 2; }
  [[ -f "${TEAM_AGENT}" ]]      || { echo "Missing TEAM_AGENT: ${TEAM_AGENT}"; exit 2; }
  [[ -f "${CONFIG_PATH}" ]]     || { echo "Missing CONFIG_PATH: ${CONFIG_PATH}"; exit 2; }
  [[ -f "${CKPT_PATH}" ]]       || { echo "Missing CKPT_PATH: ${CKPT_PATH}"; exit 2; }
  [[ -f "${ROUTES}" ]]          || { echo "Missing ROUTES xml: ${ROUTES}"; exit 2; }
}

print_settings() {
  echo "[legacy_mimic_diff] ORION_ROOT       : ${ORION_ROOT}"
  echo "[legacy_mimic_diff] B2D_ROOT         : ${B2D_ROOT}"
  echo "[legacy_mimic_diff] RUN_EVAL_SCRIPT  : ${RUN_EVAL_SCRIPT}"
  echo "[legacy_mimic_diff] TEAM_AGENT       : ${TEAM_AGENT}"
  echo "[legacy_mimic_diff] CONFIG_PATH      : ${CONFIG_PATH}"
  echo "[legacy_mimic_diff] CKPT_PATH        : ${CKPT_PATH}"
  echo "[legacy_mimic_diff] ROUTES           : ${ROUTES}"
  echo "[legacy_mimic_diff] CHECKPOINT_JSON  : ${CHECKPOINT_ENDPOINT}"
  echo "[legacy_mimic_diff] SAVE_PATH        : ${SAVE_PATH}"
  echo "[legacy_mimic_diff] PLANNER_TYPE     : ${PLANNER_TYPE}"
  echo "[legacy_mimic_diff] GPU_RANK         : ${GPU_RANK}"
  echo "[legacy_mimic_diff] RESUME           : ${RESUME}"
  echo "[legacy_mimic_diff] TEAM_CONFIG      : ${TEAM_CONFIG}"
}

cd "${B2D_ROOT}"
assert_paths
mkdir -p "${SAVE_PATH}"
print_settings

bash "${RUN_EVAL_SCRIPT}" \
  "${PORT}" "${TM_PORT}" "${IS_BENCH2DRIVE}" \
  "${ROUTES}" "${TEAM_AGENT}" "${TEAM_CONFIG}" \
  "${CHECKPOINT_ENDPOINT}" "${SAVE_PATH}" "${PLANNER_TYPE}" "${GPU_RANK}"
