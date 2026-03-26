#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_orion_student_exp7_exp8_multi_strict.sh <exp7|exp8> [preflight|full|postprocess|all]"
  exit 1
fi

EXP_NAME="$1"
MODE="${2:-full}"

ORION_LITE_ROOT="${ORION_LITE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
B2D_ROOT="${B2D_ROOT:-/mnt/adas7tb/jgu/Bench2Drive}"

export B2D_ROOT
export PYTHONPATH="${ORION_LITE_ROOT}:${B2D_ROOT}:${PYTHONPATH:-}"
export ORION_SAVE_IN_SUBDIR="${ORION_SAVE_IN_SUBDIR:-True}"
export RESUME="${RESUME:-False}"

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

if [[ "${#GPU_RANK_LIST[@]}" -ne "${TASK_NUM}" ]]; then
  echo "GPU_RANK_LIST length (${#GPU_RANK_LIST[@]}) must equal TASK_NUM (${TASK_NUM})."
  exit 2
fi

if [[ "${#TASK_LIST[@]}" -ne "${TASK_NUM}" ]]; then
  echo "TASK_LIST length (${#TASK_LIST[@]}) must equal TASK_NUM (${TASK_NUM})."
  exit 2
fi

case "${EXP_NAME}" in
  exp7)
    ALGO="orion_student_exp7_stage2_strict"
    CONFIG_PATH="${ORION_LITE_ROOT}/eval/configs/orion_student_exp7_stage2_b2d_agent_eval.py"
    CKPT_PATH="${CKPT_PATH:-}"  # Required: set CKPT_PATH to exp7 stage2 checkpoint
    BASE_PORT="${USER_BASE_PORT:-30500}"
    BASE_TM_PORT="${USER_BASE_TM_PORT:-50500}"
    ;;
  exp8)
    ALGO="orion_student_exp8_stage2_strict"
    CONFIG_PATH="${ORION_LITE_ROOT}/eval/configs/orion_student_exp8_stage2_b2d_agent_eval.py"
    CKPT_PATH="${CKPT_PATH:-}"  # Required: set CKPT_PATH to exp8 stage2 checkpoint
    BASE_PORT="${USER_BASE_PORT:-30700}"
    BASE_TM_PORT="${USER_BASE_TM_PORT:-50700}"
    ;;
  *)
    echo "Invalid experiment '${EXP_NAME}'. Use exp7 or exp8."
    exit 1
    ;;
esac

TEAM_CONFIG="${CONFIG_PATH}+${CKPT_PATH}+${EXP_NAME}_stage2_strict_multi"
RESULT_DIR="${ALGO}_b2d_${PLANNER_TYPE}"
SAVE_PATH="${SAVE_PATH:-./eval_bench2drive220_${ALGO}_${PLANNER_TYPE}}"

PRECHECK_LOG="preflight_${EXP_NAME}_${PLANNER_TYPE}.log"
SPLIT_FLAG="${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"

assert_paths() {
  [[ -f "${TEAM_AGENT}" ]] || { echo "Missing TEAM_AGENT: ${TEAM_AGENT}"; exit 3; }
  [[ -f "${CONFIG_PATH}" ]] || { echo "Missing config: ${CONFIG_PATH}"; exit 3; }
  [[ -f "${CKPT_PATH}" ]] || { echo "Missing checkpoint: ${CKPT_PATH}"; exit 3; }
  [[ -f "${BASE_ROUTES}.xml" ]] || { echo "Missing routes xml: ${BASE_ROUTES}.xml"; exit 3; }
  [[ -f "${DEV10_ROUTES}" ]] || { echo "Missing dev10 routes xml: ${DEV10_ROUTES}"; exit 3; }
}

run_preflight() {
  echo "[${EXP_NAME}] Preflight on Dev10 routes..."
  local preflight_json="${RESULT_DIR}/${BASE_CHECKPOINT_ENDPOINT}_dev10.json"
  mkdir -p "${RESULT_DIR}"
  bash -e leaderboard/scripts/run_evaluation.sh \
    "${BASE_PORT}" "${BASE_TM_PORT}" "${IS_BENCH2DRIVE}" \
    "${DEV10_ROUTES}" "${TEAM_AGENT}" "${TEAM_CONFIG}" \
    "${preflight_json}" "${SAVE_PATH}" "${PLANNER_TYPE}" "${GPU_RANK_LIST[0]}" \
    > "${PRECHECK_LOG}" 2>&1

  if [[ ! -f "${preflight_json}" ]]; then
    echo "Preflight failed: missing result json ${preflight_json}"
    exit 4
  fi

  if rg -n "Missing ego future trajectory" "${PRECHECK_LOG}" >/dev/null 2>&1; then
    echo "Preflight safety gate failed: strict-agent fallback warning found."
    exit 5
  fi

  echo "[${EXP_NAME}] Preflight passed."
}

run_full() {
  echo "[${EXP_NAME}] Full B2D220 run (${TASK_NUM} tasks)..."
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
  echo "[${EXP_NAME}] Full run finished."
}

run_postprocess() {
  echo "[${EXP_NAME}] Post-processing metrics..."
  local merged_json="${RESULT_DIR}/merged.json"
  local ability_log="${RESULT_DIR}/ability.log"
  local eff_log="${RESULT_DIR}/efficiency_smoothness.log"
  local ability_port=$((BASE_PORT + 900))
  local summary_json="${RESULT_DIR}/summary.json"

  python tools/merge_route_json.py -f "${RESULT_DIR}"
  python tools/ability_benchmark.py -f "${BASE_ROUTES}.xml" -r "${merged_json}" -p "${ability_port}" | tee "${ability_log}"
  python tools/efficiency_smoothness_benchmark.py -f "${merged_json}" -m "${SAVE_PATH}" | tee "${eff_log}"

  EXP_NAME="${EXP_NAME}" RESULT_DIR="${RESULT_DIR}" python - <<'PY'
import json
import os
import re
import sys

exp_name = os.environ["EXP_NAME"]
result_dir = os.environ["RESULT_DIR"]
merged_json = os.path.join(result_dir, "merged.json")
ability_json = merged_json.replace(".json", "_ability.json")
eff_log = os.path.join(result_dir, "efficiency_smoothness.log")
summary_json = os.path.join(result_dir, "summary.json")

with open(merged_json, "r") as f:
    merged = json.load(f)
with open(ability_json, "r") as f:
    ability = json.load(f)
with open(eff_log, "r") as f:
    eff_text = f.read()

def extract_float(name):
    m = re.search(rf"{name}=([0-9]*\\.?[0-9]+)", eff_text)
    return float(m.group(1)) if m else None

summary = {
    "experiment": exp_name,
    "driving_score": merged.get("driving score"),
    "success_rate": merged.get("success rate"),
    "eval_num": merged.get("eval num"),
    "ability_mean": ability.get("mean"),
    "ability_overtaking": ability.get("Overtaking"),
    "ability_merging": ability.get("Merging"),
    "ability_emergency_brake": ability.get("Emergency_Brake"),
    "ability_give_way": ability.get("Give_Way"),
    "ability_traffic_signs": ability.get("Traffic_Signs"),
    "driving_efficiency": extract_float("Driving Efficiency"),
    "driving_smoothness": extract_float("Driving Smoothness"),
    "crashed_routes": len(ability.get("crashed", [])),
}

with open(summary_json, "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
PY
  echo "[${EXP_NAME}] Wrote summary: ${summary_json}"

  local exp7_dir="orion_student_exp7_stage2_strict_b2d_${PLANNER_TYPE}"
  local exp8_dir="orion_student_exp8_stage2_strict_b2d_${PLANNER_TYPE}"
  local exp7_summary="${exp7_dir}/summary.json"
  local exp8_summary="${exp8_dir}/summary.json"
  local compare_md="results/exp7_exp8_b2d220_compare_${PLANNER_TYPE}.md"

  if [[ -f "${exp7_summary}" && -f "${exp8_summary}" ]]; then
    mkdir -p results
    EXP7_SUMMARY="${exp7_summary}" EXP8_SUMMARY="${exp8_summary}" COMPARE_MD="${compare_md}" python - <<'PY'
import json
import os

exp7_path = os.environ["EXP7_SUMMARY"]
exp8_path = os.environ["EXP8_SUMMARY"]
out_path = os.environ["COMPARE_MD"]

with open(exp7_path, "r") as f:
    exp7 = json.load(f)
with open(exp8_path, "r") as f:
    exp8 = json.load(f)

rows = [
    ("Driving Score", "driving_score"),
    ("Success Rate", "success_rate"),
    ("Eval Num", "eval_num"),
    ("Ability Mean", "ability_mean"),
    ("Overtaking", "ability_overtaking"),
    ("Merging", "ability_merging"),
    ("Emergency Brake", "ability_emergency_brake"),
    ("Give Way", "ability_give_way"),
    ("Traffic Signs", "ability_traffic_signs"),
    ("Driving Efficiency", "driving_efficiency"),
    ("Driving Smoothness", "driving_smoothness"),
    ("Crashed Routes", "crashed_routes"),
]

lines = []
lines.append("| Metric | exp7 | exp8 |")
lines.append("|---|---:|---:|")
for name, key in rows:
    v7 = exp7.get(key)
    v8 = exp8.get(key)
    lines.append(f"| {name} | {v7} | {v8} |")

with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
PY
    echo "[compare] Wrote comparison table: ${compare_md}"
  else
    echo "[compare] Need both summaries before comparison table can be generated."
  fi
}

cd "${B2D_ROOT}"
assert_paths

case "${MODE}" in
  preflight)
    run_preflight
    ;;
  full)
    run_full
    ;;
  postprocess)
    run_postprocess
    ;;
  all)
    run_preflight
    run_full
    run_postprocess
    ;;
  *)
    echo "Invalid mode '${MODE}'. Use preflight, full, postprocess, or all."
    exit 1
    ;;
esac
