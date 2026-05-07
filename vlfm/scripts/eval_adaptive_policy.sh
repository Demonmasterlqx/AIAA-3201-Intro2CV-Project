#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-part2_adaptive_hybrid}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/${RUN_NAME}}"
VIDEO_DIR="${VIDEO_DIR:-${OUTPUT_DIR}/videos}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/episode_stats}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
TEST_EPISODE_COUNT="${TEST_EPISODE_COUNT:--1}"
VIDEO_OPTION="${VIDEO_OPTION:-[\"disk\"]}"

mkdir -p "${VIDEO_DIR}" "${LOG_DIR}"
export ZSOS_LOG_DIR="${LOG_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-vlfm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache-vlfm}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cd "${ROOT_DIR}"
python -m vlfm.run \
  --config-name experiments/vlfm_adaptive_objectnav_hm3d \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir="${ROOT_DIR}/data/dummy_policy.pth" \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.split="${EVAL_SPLIT}" \
  habitat_baselines.test_episode_count="${TEST_EPISODE_COUNT}" \
  habitat_baselines.video_dir="${VIDEO_DIR}" \
  habitat_baselines.eval.video_option="${VIDEO_OPTION}" \
  habitat.task.measurements.frontier_exploration_map.draw_goal_aabbs=False \
  "$@"
