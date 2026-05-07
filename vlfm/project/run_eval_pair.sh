#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/project"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/results}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${PROJECT_DIR}/artifacts}"
REPORT_PATH="${REPORT_PATH:-${PROJECT_DIR}/REPORT.md}"
SPLIT="${SPLIT:-val_mini}"
EPISODES="${EPISODES:-1}"
VIDEO_OPTION="${VIDEO_OPTION:-[\"disk\"]}"
REUSE_OUTPUTS="${REUSE_OUTPUTS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-}"

mkdir -p "${RESULTS_DIR}" "${ARTIFACTS_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-vlfm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache-vlfm}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

run_plan() {
  local plan_name="$1"
  local script_path="$2"
  local output_dir="${RESULTS_DIR}/${plan_name}"
  local log_dir="${output_dir}/episode_stats"
  local video_dir="${output_dir}/videos"
  local extra_args=(
    habitat.task.measurements.frontier_exploration_map.draw_goal_aabbs=False
  )

  if [[ -n "${MAX_EPISODE_STEPS}" ]]; then
    extra_args+=("habitat.environment.max_episode_steps=${MAX_EPISODE_STEPS}")
    extra_args+=("habitat.task.measurements.frontier_exploration_map.max_episode_steps=${MAX_EPISODE_STEPS}")
  fi

  if [[ "${REUSE_OUTPUTS}" == "1" && -d "${log_dir}" && -n "$(find "${log_dir}" -maxdepth 1 -name '*.json' -print -quit)" ]]; then
    echo "Reusing existing outputs for ${plan_name} from ${output_dir}"
    return
  fi

  mkdir -p "${log_dir}" "${video_dir}"

  RUN_NAME="${plan_name}" \
  OUTPUT_DIR="${output_dir}" \
  LOG_DIR="${log_dir}" \
  VIDEO_DIR="${video_dir}" \
  EVAL_SPLIT="${SPLIT}" \
  TEST_EPISODE_COUNT="${EPISODES}" \
  VIDEO_OPTION="${VIDEO_OPTION}" \
  "${script_path}" \
  "${extra_args[@]}"
}

cd "${ROOT_DIR}"

run_plan "plan_a_dense" "${ROOT_DIR}/scripts/eval_dense_value_policy.sh"
run_plan "plan_b_frontier" "${ROOT_DIR}/scripts/eval_frontier_policy.sh"

python "${PROJECT_DIR}/summarize_results.py" \
  --project-dir "${PROJECT_DIR}" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --report-path "${REPORT_PATH}" \
  --dense-dir "${RESULTS_DIR}/plan_a_dense" \
  --frontier-dir "${RESULTS_DIR}/plan_b_frontier"
