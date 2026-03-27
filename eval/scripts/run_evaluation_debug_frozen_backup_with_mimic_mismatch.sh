#!/bin/bash
# Frozen-backup single-GPU mismatch evaluation for the old two-checkpoint mimic path.
#
# This script does NOT edit anything inside project_backup. Instead it:
#   - uses backup Bench2Drive + backup Orion code on PYTHONPATH
#   - uses a local config shim under Orion-Lite for updated absolute paths
#   - uses the current local CARLA install

set -euo pipefail

ORION_LITE_ROOT="${ORION_LITE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BACKUP_ORION_ROOT="${BACKUP_ORION_ROOT:-/mnt/adas7tb/jgu/project_backup/Orion}"
BACKUP_B2D_ROOT="${BACKUP_B2D_ROOT:-/mnt/adas7tb/jgu/project_backup/Bench2Drive}"
ORION_RUNTIME_ROOT="${ORION_RUNTIME_ROOT:-${BACKUP_B2D_ROOT}/tools/Orion}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/adas7tb/jgu/Bench2Drive}"

export ORION_LITE_ROOT
export BACKUP_ORION_ROOT
export BACKUP_B2D_ROOT
export ORION_RUNTIME_ROOT

export CARLA_ROOT="${CARLA_ROOT:-/mnt/adas7tb/jgu/carla}"
export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export SCENARIO_RUNNER_ROOT="${BACKUP_B2D_ROOT}/scenario_runner"
export LEADERBOARD_ROOT="${BACKUP_B2D_ROOT}/leaderboard"
export CHALLENGE_TRACK_CODENAME="${CHALLENGE_TRACK_CODENAME:-SENSORS}"
export RESUME="${RESUME:-False}"
export IS_BENCH2DRIVE="${IS_BENCH2DRIVE:-True}"
export PLANNER_TYPE="${PLANNER_TYPE:-only_traj}"
export RECORD_PATH="${RECORD_PATH:-}"
export ROUTES_SUBSET="${ROUTES_SUBSET:-}"

export PYTHONPATH="${BACKUP_ORION_ROOT}:${BACKUP_B2D_ROOT}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${BACKUP_B2D_ROOT}/leaderboard:${BACKUP_B2D_ROOT}/leaderboard/team_code:${BACKUP_B2D_ROOT}/scenario_runner:${PYTHONPATH:-}"

BASE_PORT="${BASE_PORT:-30300}"
BASE_TM_PORT="${BASE_TM_PORT:-50300}"
GPU_RANK="${GPU_RANK:-0}"

CONFIG_PATH="${CONFIG_PATH:-${ORION_LITE_ROOT}/eval/configs/orion_distill_new_agent_with_mimic_backup_local.py}"
TEAM_AGENT="${TEAM_AGENT:-${BACKUP_B2D_ROOT}/leaderboard/team_code/orion_b2d_agent.py}"
CKPT_PATH="${CKPT_PATH:-${ORION_RUNTIME_ROOT}/ckpts/Orion.pth}"
RUN_TAG="${RUN_TAG:-orion_mimic_frozen_backup_diff}"
ROUTES="${ROUTES:-${ORION_LITE_ROOT}/eval/route_diffs/orion_distilled_6l_with_mimic_fused_vs_eval_mimic/mismatch_routes.xml}"
CHECKPOINT_ENDPOINT="${CHECKPOINT_ENDPOINT:-${OUTPUT_ROOT}/eval_mimic_frozen_backup_diff.json}"
SAVE_PATH="${SAVE_PATH:-${OUTPUT_ROOT}/eval_mimic_frozen_backup_diff/}"
TEAM_CONFIG="${TEAM_CONFIG:-${CONFIG_PATH}+${CKPT_PATH}+${RUN_TAG}}"

export PORT="${PORT:-${BASE_PORT}}"
export TM_PORT="${TM_PORT:-${BASE_TM_PORT}}"
export DEBUG_CHALLENGE="${DEBUG_CHALLENGE:-0}"
export REPETITIONS="${REPETITIONS:-1}"
export ROUTES
export TEAM_AGENT
export TEAM_CONFIG
export CHECKPOINT_ENDPOINT
export SAVE_PATH
export GPU_RANK

assert_paths() {
  [[ -d "${CARLA_ROOT}" ]]         || { echo "Missing CARLA_ROOT: ${CARLA_ROOT}"; exit 2; }
  [[ -f "${TEAM_AGENT}" ]]         || { echo "Missing TEAM_AGENT: ${TEAM_AGENT}"; exit 2; }
  [[ -f "${CONFIG_PATH}" ]]        || { echo "Missing CONFIG_PATH: ${CONFIG_PATH}"; exit 2; }
  [[ -f "${CKPT_PATH}" ]]          || { echo "Missing CKPT_PATH: ${CKPT_PATH}"; exit 2; }
  [[ -f "${ROUTES}" ]]             || { echo "Missing ROUTES xml: ${ROUTES}"; exit 2; }
  [[ -f "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" ]] || {
    echo "Missing backup leaderboard evaluator under: ${LEADERBOARD_ROOT}"; exit 2;
  }
}

print_settings() {
  echo "[frozen_backup_mimic] BACKUP_ORION_ROOT : ${BACKUP_ORION_ROOT}"
  echo "[frozen_backup_mimic] BACKUP_B2D_ROOT   : ${BACKUP_B2D_ROOT}"
  echo "[frozen_backup_mimic] ORION_RUNTIME_ROOT: ${ORION_RUNTIME_ROOT}"
  echo "[frozen_backup_mimic] CARLA_ROOT        : ${CARLA_ROOT}"
  echo "[frozen_backup_mimic] TEAM_AGENT        : ${TEAM_AGENT}"
  echo "[frozen_backup_mimic] CONFIG_PATH       : ${CONFIG_PATH}"
  echo "[frozen_backup_mimic] CKPT_PATH         : ${CKPT_PATH}"
  echo "[frozen_backup_mimic] ROUTES            : ${ROUTES}"
  echo "[frozen_backup_mimic] CHECKPOINT_JSON   : ${CHECKPOINT_ENDPOINT}"
  echo "[frozen_backup_mimic] SAVE_PATH         : ${SAVE_PATH}"
  echo "[frozen_backup_mimic] GPU_RANK          : ${GPU_RANK}"
  echo "[frozen_backup_mimic] PORT/TM_PORT      : ${PORT}/${TM_PORT}"
  echo "[frozen_backup_mimic] RESUME            : ${RESUME}"
  echo "[frozen_backup_mimic] TEAM_CONFIG       : ${TEAM_CONFIG}"
}

assert_paths
mkdir -p "${SAVE_PATH}"
print_settings

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export GPU_RANK_IN_VISIBLE_SET="${GPU_RANK_IN_VISIBLE_SET:-0}"
  echo "[frozen_backup_mimic] Using pre-set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, GPU_RANK_IN_VISIBLE_SET=${GPU_RANK_IN_VISIBLE_SET}"
else
  export CUDA_VISIBLE_DEVICES="${GPU_RANK}"
  echo "[frozen_backup_mimic] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

cd "${BACKUP_B2D_ROOT}"

python "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
  --routes="${ROUTES}" \
  --routes-subset="${ROUTES_SUBSET}" \
  --repetitions="${REPETITIONS}" \
  --track="${CHALLENGE_TRACK_CODENAME}" \
  --checkpoint="${CHECKPOINT_ENDPOINT}" \
  --agent="${TEAM_AGENT}" \
  --agent-config="${TEAM_CONFIG}" \
  --debug="${DEBUG_CHALLENGE}" \
  --record="${RECORD_PATH}" \
  --resume="${RESUME}" \
  --port="${PORT}" \
  --traffic-manager-port="${TM_PORT}" \
  --gpu-rank="${GPU_RANK}"
