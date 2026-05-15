#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/home/sim6g/anaconda3/envs/zson_official/bin/python}"
DATA_ROOT="${PROJECT_ROOT}/data"
RUN_NAME="${RUN_NAME:-plan_a_smoke_hm3d}"

cd "${REPO_ROOT}"

CMD=(
  "${PYTHON_BIN}" tools/plan_a_eval.py
  --exp-config configs/experiments/plan_a_objectnav_hm3d.yaml
  PLAN_A.RUN_NAME "${RUN_NAME}"
  PLAN_A.MAX_EPISODES 1
  TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 60
  TASK_CONFIG.DATASET.DATA_PATH "${DATA_ROOT}/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz"
  TASK_CONFIG.DATASET.SCENES_DIR "${DATA_ROOT}/scene_datasets"
)
CMD+=("$@")

printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
