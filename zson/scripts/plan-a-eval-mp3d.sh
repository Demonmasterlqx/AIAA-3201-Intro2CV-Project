#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/home/sim6g/anaconda3/envs/zson_official/bin/python}"
DATA_ROOT="${PROJECT_ROOT}/data"
RUN_NAME="${RUN_NAME:-plan_a_mp3d_val}"

cd "${REPO_ROOT}"

CMD=(
  "${PYTHON_BIN}" tools/plan_a_eval.py
  --exp-config configs/experiments/plan_a_objectnav_mp3d.yaml
  PLAN_A.RUN_NAME "${RUN_NAME}"
  TASK_CONFIG.DATASET.DATA_PATH "${DATA_ROOT}/MatterPort3D/objectnav/mp3d/v1/{split}/{split}.json.gz"
  TASK_CONFIG.DATASET.SCENES_DIR "${DATA_ROOT}/MatterPort3D"
)
CMD+=("$@")

printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
