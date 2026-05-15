#!/usr/bin/env bash
set -euo pipefail

# Full HM3D val run for the best 100ep Local-Qwen selective-hybrid variant.
# Planner remains local_qwen; Qwen-VL is only used as a selective final-check verifier.
export RUN_NAME=${RUN_NAME:-final1000_variant_boundary_tvambig_blipfallback_v1}
export RESULT_DIR=${RESULT_DIR:-results/localqwen_1000ep/${RUN_NAME}}
export EPISODE_COUNT=${EPISODE_COUNT:-1000}
export SPLIT_SEED=${SPLIT_SEED:-3201}
export FINAL_CHECK_MODE=${FINAL_CHECK_MODE:-selective_hybrid_boundary_gate_only}
export HYBRID_ACCEPT_ON_QWEN_REJECT_IF_BLIP_HIGH=${HYBRID_ACCEPT_ON_QWEN_REJECT_IF_BLIP_HIGH:-true}
export FRONTIER_STUCK_RECOVERY_ENABLED=${FRONTIER_STUCK_RECOVERY_ENABLED:-false}
export STAIR_FLOOR_BUDGET_ENABLED=${STAIR_FLOOR_BUDGET_ENABLED:-false}
export QWEN_APPROACH_RECOVERY_ENABLED=${QWEN_APPROACH_RECOVERY_ENABLED:-false}
export SESSION=${SESSION:-localqwen_1000ep_boundary_tvfallback_v1}
export DATASET_ROOT=${DATASET_ROOT:-data/datasets/objectnav/hm3d/localqwen_1000ep_fixed/v2/val}
export EPISODES_MANIFEST=${EPISODES_MANIFEST:-data/datasets/objectnav/hm3d/localqwen_1000ep_fixed/episodes_1000_localqwen_fixed.json}
export VIDEO_OPTION=${VIDEO_OPTION:-[]}

REPO=${REPO:-/data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep}
SOURCE_ROOT=${SOURCE_ROOT:-/data/home/sim6g/code/aiaa3201_cv_project/data/ascent_imported/datasets/objectnav/hm3d/v2/val}
FULL_VAL_SCENES=(
  4ok3usBNeis 5cdEh9F2hJL 6s7QHgap2fW 7MXmsvcQjpJ BAbdmeyTvMZ CrMo8WxCyVb
  DYehNKdT76V Dd4bFSTQ8gi GLAQ4DNUx5U HY1NcmCgn3n LT9Jq6dN3Ea MHPLjHsuG27
  Nfvxx8J5NCo QaLdnwvtxbs TEEsavR23oF VBzV5z6i1WS XB4GS9ShBRE a8BtkwhxdRV
  bCPU9suPUw9 bxsVRursffK cvZr5TUy5C5 eF36g7L6Z9M h1zeeAwLh9Z k1cupFYWXJ6
  mL8ThkuaVTM mv2HUxq3B53 p53SfW6mjZe q3zU7Yy5E5s q5QZSEeHe5g qyAac8rV8Zk
  svBbv1Pavdk wcojb4TFT35 y9hTuugGdiq yr17PDCnDDW ziup5kvtCCR zt1RVoi7PcG
)

cd "$REPO"
if [ "${USE_EXISTING_SPLIT:-false}" != "true" ]; then
  python3 scripts/make_fixed_100ep_split.py \
    --source-root "$SOURCE_ROOT" \
    --sample-size "$EPISODE_COUNT" \
    --seed "$SPLIT_SEED" \
    --output-root "$DATASET_ROOT" \
    --manifest "$EPISODES_MANIFEST" \
    --scenes "${FULL_VAL_SCENES[@]}"
fi
export USE_EXISTING_SPLIT=true

exec "$(dirname "$0")/run_localqwen_100ep_common.sh"