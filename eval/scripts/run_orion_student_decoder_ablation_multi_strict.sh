#!/bin/bash
# Evaluation script for Orion-Lite decoder-layer ablation models on Bench2Drive220.
#
# Usage:
#   bash eval/scripts/run_orion_student_decoder_ablation_multi_strict.sh \
#       <decoder_layers> <mimic> [preflight|full|postprocess|all]
#
#   <decoder_layers>  : 2 | 4 | 8 | 16
#   <mimic>           : no_mimic | with_mimic
#   [mode]            : preflight | full | postprocess | all  (default: full)
#
# Examples:
#   # Evaluate 2-layer decoder, no mimic loss, full B2D220
#   bash eval/scripts/run_orion_student_decoder_ablation_multi_strict.sh 2 no_mimic full
#
#   # Evaluate 8-layer decoder, with mimic loss, all stages
#   bash eval/scripts/run_orion_student_decoder_ablation_multi_strict.sh 8 with_mimic all
#
# Required environment variables (or edit defaults below):
#   ORION_ROOT   : path to Orion repository
#   B2D_ROOT     : path to Bench2Drive repository
#   CKPT_PATH    : merged full OrionStudentPlanner checkpoint
#                  (use evaluate_distill_result.py or export_distill_ckpt.py first)

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash $0 <decoder_layers> <no_mimic|with_mimic> [preflight|full|postprocess|all]"
  exit 1
fi

DECODER_LAYERS="$1"
MIMIC_MODE="$2"
MODE="${3:-full}"

# Validate arguments
case "${DECODER_LAYERS}" in
  2|4|8|16) ;;
  *) echo "Invalid decoder_layers '${DECODER_LAYERS}'. Use 2, 4, 8, or 16."; exit 1 ;;
esac
case "${MIMIC_MODE}" in
  no_mimic|with_mimic) ;;
  *) echo "Invalid mimic mode '${MIMIC_MODE}'. Use no_mimic or with_mimic."; exit 1 ;;
esac

ORION_LITE_ROOT="${ORION_LITE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
B2D_ROOT="${B2D_ROOT:-/mnt/adas7tb/jgu/Bench2Drive}"
ORION_LITE_ROOT="${ORION_LITE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

export B2D_ROOT
export PYTHONPATH="${ORION_LITE_ROOT}:${B2D_ROOT}:${PYTHONPATH:-}"
export ORION_SAVE_IN_SUBDIR="${ORION_SAVE_IN_SUBDIR:-True}"
export RESUME="${RESUME:-True}"

USER_BASE_PORT="${BASE_PORT:-}"
USER_BASE_TM_PORT="${BASE_TM_PORT:-}"
IS_BENCH2DRIVE="${IS_BENCH2DRIVE:-True}"
BASE_ROUTES="${BASE_ROUTES:-leaderboard/data/bench2drive220}"
DEV10_ROUTES="${DEV10_ROUTES:-leaderboard/data/drivetransformer_bench2drive_dev10.xml}"
TEAM_AGENT="${TEAM_AGENT:-${ORION_LITE_ROOT}/team_code/orion_b2d_agent_strict.py}"
BASE_CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT:-eval_bench2drive220}"
PLANNER_TYPE="${PLANNER_TYPE:-only_traj}"
TASK_NUM="${TASK_NUM:-4}"
GPU_RANK_LIST=(${GPU_RANK_LIST:-0 1 2 3})
TASK_LIST=(${TASK_LIST:-0 1 2 3})

# Derive config and checkpoint paths from arguments
ALGO="orion_student_decoder${DECODER_LAYERS}l_${MIMIC_MODE}_strict"
CONFIG_PATH="${ORION_LITE_ROOT}/eval/configs/orion_student_decoder${DECODER_LAYERS}l_${MIMIC_MODE}_b2d_agent_eval.py"

_DISTILL_SUBDIR="ablation_decoder_layers_${MIMIC_MODE}/decoder${DECODER_LAYERS}l"
MERGED_STEM="${_DISTILL_SUBDIR//\//__}"
CKPT_PATH="${CKPT_PATH:-${ORION_LITE_ROOT}/eval/merged_ckpts/${MERGED_STEM}.pth}"

BASE_PORT="${USER_BASE_PORT:-31000}"
BASE_TM_PORT="${USER_BASE_TM_PORT:-51000}"

if [[ "${#GPU_RANK_LIST[@]}" -ne "${TASK_NUM}" ]]; then
  echo "GPU_RANK_LIST length (${#GPU_RANK_LIST[@]}) must equal TASK_NUM (${TASK_NUM})."; exit 2
fi
if [[ "${#TASK_LIST[@]}" -ne "${TASK_NUM}" ]]; then
  echo "TASK_LIST length (${#TASK_LIST[@]}) must equal TASK_NUM (${TASK_NUM})."; exit 2
fi

TEAM_CONFIG="${CONFIG_PATH}+${CKPT_PATH}+${ALGO}_multi"
RESULT_DIR="${ALGO}_b2d_${PLANNER_TYPE}"
SAVE_PATH="${SAVE_PATH:-./eval_bench2drive220_${ALGO}_${PLANNER_TYPE}}"

PRECHECK_LOG="preflight_${ALGO}_${PLANNER_TYPE}.log"
SPLIT_FLAG="${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"

assert_paths() {
  if [[ "${CKPT_PATH}" == *.pt ]]; then
    echo "CKPT_PATH looks like a raw distill checkpoint: ${CKPT_PATH}"
    echo "Closed-loop strict evaluation expects a merged full OrionStudentPlanner checkpoint (.pth)."
    echo "Run eval/scripts/evaluate_distill_result.py or eval/scripts/export_distill_ckpt.py first."
    exit 3
  fi
  [[ -f "${TEAM_AGENT}" ]]    || { echo "Missing TEAM_AGENT: ${TEAM_AGENT}"; exit 3; }
  [[ -f "${CONFIG_PATH}" ]]   || { echo "Missing config: ${CONFIG_PATH}"; exit 3; }
  [[ -f "${CKPT_PATH}" ]]     || { echo "Missing checkpoint: ${CKPT_PATH}"; exit 3; }
  [[ -f "${BASE_ROUTES}.xml" ]] || { echo "Missing routes xml: ${BASE_ROUTES}.xml"; exit 3; }
  [[ -f "${DEV10_ROUTES}" ]]  || { echo "Missing dev10 routes xml: ${DEV10_ROUTES}"; exit 3; }
}

print_settings() {
  echo "[${ALGO}] ORION_LITE_ROOT : ${ORION_LITE_ROOT}"
  echo "[${ALGO}] B2D_ROOT        : ${B2D_ROOT}"
  echo "[${ALGO}] CONFIG_PATH     : ${CONFIG_PATH}"
  echo "[${ALGO}] CKPT_PATH       : ${CKPT_PATH}"
  echo "[${ALGO}] TEAM_AGENT      : ${TEAM_AGENT}"
  echo "[${ALGO}] RESULT_DIR      : ${RESULT_DIR}"
  echo "[${ALGO}] SAVE_PATH       : ${SAVE_PATH}"
  echo "[${ALGO}] MODE            : ${MODE}"
}

run_preflight() {
  echo "[${ALGO}] Preflight on Dev10 routes..."
  local preflight_json="${RESULT_DIR}/${BASE_CHECKPOINT_ENDPOINT}_dev10.json"
  mkdir -p "${RESULT_DIR}"
  bash -e leaderboard/scripts/run_evaluation.sh \
    "${BASE_PORT}" "${BASE_TM_PORT}" "${IS_BENCH2DRIVE}" \
    "${DEV10_ROUTES}" "${TEAM_AGENT}" "${TEAM_CONFIG}" \
    "${preflight_json}" "${SAVE_PATH}" "${PLANNER_TYPE}" "${GPU_RANK_LIST[0]}" \
    > "${PRECHECK_LOG}" 2>&1

  [[ -f "${preflight_json}" ]] || { echo "Preflight failed: missing ${preflight_json}"; exit 4; }

  if rg -n "Missing ego future trajectory" "${PRECHECK_LOG}" >/dev/null 2>&1; then
    echo "Preflight safety gate failed: strict-agent fallback warning found."; exit 5
  fi
  echo "[${ALGO}] Preflight passed."
}

run_full() {
  echo "[${ALGO}] Full B2D220 run (${TASK_NUM} tasks)..."
  mkdir -p "${RESULT_DIR}"

  if [[ ! -f "${SPLIT_FLAG}" ]]; then
    python tools/split_xml.py "${BASE_ROUTES}" "${TASK_NUM}" "${ALGO}" "${PLANNER_TYPE}"
    touch "${SPLIT_FLAG}"
  fi

  for ((i=0; i<TASK_NUM; i++)); do
    local port=$((BASE_PORT + i * 150))
    local tm_port=$((BASE_TM_PORT + i * 150))
    local routes="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.xml"
    local checkpoint_endpoint="${RESULT_DIR}/${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
    local gpu_rank="${GPU_RANK_LIST[$i]}"
    local log_file="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.log"

    bash -e leaderboard/scripts/run_evaluation.sh \
      "${port}" "${tm_port}" "${IS_BENCH2DRIVE}" \
      "${routes}" "${TEAM_AGENT}" "${TEAM_CONFIG}" \
      "${checkpoint_endpoint}" "${SAVE_PATH}" "${PLANNER_TYPE}" "${gpu_rank}" \
      > "${log_file}" 2>&1 &
    sleep 5
  done

  wait

  for task_idx in "${TASK_LIST[@]}"; do
    local out_json="${RESULT_DIR}/${BASE_CHECKPOINT_ENDPOINT}_${task_idx}.json"
    [[ -f "${out_json}" ]] || { echo "Missing split result json: ${out_json}"; exit 6; }
  done
  echo "[${ALGO}] Full run finished."
}

run_postprocess() {
  echo "[${ALGO}] Post-processing metrics..."
  local merged_json="${RESULT_DIR}/merged.json"
  local ability_log="${RESULT_DIR}/ability.log"
  local eff_log="${RESULT_DIR}/efficiency_smoothness.log"
  local ability_port=$((BASE_PORT + 900))
  local summary_json="${RESULT_DIR}/summary.json"

  python tools/merge_route_json.py -f "${RESULT_DIR}"
  python tools/ability_benchmark.py -f "${BASE_ROUTES}.xml" -r "${merged_json}" -p "${ability_port}" | tee "${ability_log}"
  python tools/efficiency_smoothness_benchmark.py -f "${merged_json}" -m "${SAVE_PATH}" | tee "${eff_log}"

  ALGO="${ALGO}" RESULT_DIR="${RESULT_DIR}" python - <<'PY'
import json, os, re
algo        = os.environ["ALGO"]
result_dir  = os.environ["RESULT_DIR"]
merged_json = os.path.join(result_dir, "merged.json")
ability_json = merged_json.replace(".json", "_ability.json")
eff_log     = os.path.join(result_dir, "efficiency_smoothness.log")
summary_json = os.path.join(result_dir, "summary.json")

with open(merged_json) as f:  merged  = json.load(f)
with open(ability_json) as f: ability = json.load(f)
with open(eff_log) as f:      eff_text = f.read()

def extract_float(name):
    m = re.search(rf"{name}=([0-9]*\.?[0-9]+)", eff_text)
    return float(m.group(1)) if m else None

summary = {
    "experiment": algo,
    "driving_score":        merged.get("driving score"),
    "success_rate":         merged.get("success rate"),
    "eval_num":             merged.get("eval num"),
    "ability_mean":         ability.get("mean"),
    "ability_overtaking":   ability.get("Overtaking"),
    "ability_merging":      ability.get("Merging"),
    "ability_emergency_brake": ability.get("Emergency_Brake"),
    "ability_give_way":     ability.get("Give_Way"),
    "ability_traffic_signs":ability.get("Traffic_Signs"),
    "driving_efficiency":   extract_float("Driving Efficiency"),
    "driving_smoothness":   extract_float("Driving Smoothness"),
    "crashed_routes":       len(ability.get("crashed", [])),
}
with open(summary_json, "w") as f: json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY
  echo "[${ALGO}] Wrote summary: ${RESULT_DIR}/summary.json"
}

cd "${B2D_ROOT}"
print_settings
assert_paths

case "${MODE}" in
  preflight)   run_preflight ;;
  full)        run_full ;;
  postprocess) run_postprocess ;;
  all)         run_preflight; run_full; run_postprocess ;;
  *)
    echo "Invalid mode '${MODE}'. Use preflight, full, postprocess, or all."
    exit 1 ;;
esac
