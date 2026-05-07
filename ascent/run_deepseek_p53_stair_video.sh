#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/home/sim6g/code/aiaa3201_cv_project/ascent"
ASCENT_DIR="${ROOT}/ascent"
RUN_ID="deepseek_p53_upstairs_success_ep0_video_$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${ASCENT_DIR}/results/video_deepseek_stair_success/${RUN_ID}"
VIDEO_DIR="${ASCENT_DIR}/debug/video_deepseek_stair_success/${RUN_ID}"

mkdir -p "${RESULT_DIR}" "${VIDEO_DIR}"

source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav

set -a
source "${ROOT}/ascent_qwenvl_objectpatch/.env"
set +a
export DEEPSEEK_API_BASE_URL="${DEEPSEEK_API_BASE_URL:-${DEEPSEEK_BASE_URL:-}}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-${DEEPSEEK_API_BASE_URL:-}}"

cd "${ASCENT_DIR}"

export QWEN2_5_PORT=14181
export BLIP2ITM_PORT=14182
export SAM_PORT=14183
export GROUNDING_DINO_PORT=14184
export RAM_PORT=14185
export DFINE_PORT=14186
export QWEN2_5_VL_PORT=14187

{
  echo "run_id=${RUN_ID}"
  echo "result_dir=${RESULT_DIR}"
  echo "video_dir=${VIDEO_DIR}"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "backend=DeepSeek native ASCENT-Single planner + BLIP2 cosine double-check"
  echo "scene=p53SfW6mjZe"
  echo "intended_episode=first p53SfW6mjZe episode, matching DeepSeek 20ep upstairs success candidate"
  echo
} | tee "${RESULT_DIR}/run_metadata.txt"

python -u -m ascent.run \
  --config-name=eval_ascent_hm3d.yaml \
  'habitat.dataset.content_scenes=[p53SfW6mjZe]' \
  habitat.environment.iterator_options.max_scene_repeat_episodes=-1 \
  habitat.environment.iterator_options.num_episode_sample=1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat.environment.iterator_options.group_by_scene=False \
  habitat_baselines.test_episode_count=1 \
  'habitat_baselines.eval.video_option=[disk]' \
  "habitat_baselines.video_dir=${VIDEO_DIR}" \
  2>&1 | tee "${RESULT_DIR}/run.log"

{
  echo
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "videos:"
  find "${VIDEO_DIR}" -maxdepth 1 -type f -name '*.mp4' -printf '%p %s bytes\n' | sort
} | tee -a "${RESULT_DIR}/run_metadata.txt"
