#!/usr/bin/env bash
set -euo pipefail

OPT_REPO=${OPT_REPO:-/data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep}
BASE_REPO=${BASE_REPO:-/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent}
ASCENT_DEPS_REPO=${ASCENT_DEPS_REPO:-/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent}
ENV_FILE=${ENV_FILE:-/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent_qwenvl_objectpatch/.env}
PARSER=${PARSER:-$OPT_REPO/scripts/parse_100ep_failures.py}
SPLIT_MANIFEST=${SPLIT_MANIFEST:-$OPT_REPO/data/part3_200ep_fixed/episodes_200_part3_fixed.json}
if [ -z "${DATA_PATH:-}" ]; then
  DATA_PATH="$OPT_REPO/data/datasets/objectnav/hm3d/part3_200ep_fixed/v1/{split}/{split}.json.gz"
fi
SCENES_DIR=${SCENES_DIR:-$BASE_REPO/data/scene_datasets}
BASE_PORT=${BASE_PORT:-14181}
CONTENT_SCENES='[4ok3usBNeis,5cdEh9F2hJL,6s7QHgap2fW,DYehNKdT76V,Dd4bFSTQ8gi,Nfvxx8J5NCo,QaLdnwvtxbs,TEEsavR23oF,XB4GS9ShBRE,bxsVRursffK,cvZr5TUy5C5,mL8ThkuaVTM,mv2HUxq3B53,p53SfW6mjZe,q3zU7Yy5E5s,qyAac8rV8Zk,svBbv1Pavdk,wcojb4TFT35,ziup5kvtCCR,zt1RVoi7PcG]'
RESULT_ROOT=${RESULT_ROOT:-$OPT_REPO/results/part3_fixed200_final}
VARIANT_NAME=${VARIANT_NAME:-deepseek_selective_hybrid_samepart3_fixed200_v3}
BASELINE_NAME=${BASELINE_NAME:-part3_baseline_deepseek_fixed200_v3}
VIDEO_OPTION=${VIDEO_OPTION:-[]}

source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
export DEEPSEEK_API_BASE_URL="${DEEPSEEK_API_BASE_URL:-${DEEPSEEK_BASE_URL:-}}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-${DEEPSEEK_API_BASE_URL:-}}"
export QWEN2_5_PORT=$((BASE_PORT))
export BLIP2ITM_PORT=$((BASE_PORT + 1))
export SAM_PORT=$((BASE_PORT + 2))
export GROUNDING_DINO_PORT=$((BASE_PORT + 3))
export RAM_PORT=$((BASE_PORT + 4))
export DFINE_PORT=$((BASE_PORT + 5))
export QWEN2_5_VL_PORT=$((BASE_PORT + 6))

prepare_opt_repo() {
  cd "$OPT_REPO"
  if [ ! -e pretrained_weights ]; then
    ln -s "$ASCENT_DEPS_REPO/pretrained_weights" pretrained_weights
  fi
  mkdir -p third_party/places365
  if [ ! -f third_party/places365/categories_places365.txt ]; then
    cp "$ASCENT_DEPS_REPO/third_party/places365/categories_places365.txt" third_party/places365/categories_places365.txt
  fi
}

write_common_metadata() {
  local repo=$1
  local result_dir=$2
  mkdir -p "$result_dir"
  cp "$SPLIT_MANIFEST" "$result_dir/episodes_fixed.json"
  (
    cd "$repo"
    git rev-parse HEAD > "$result_dir/commit.txt"
    git status --short > "$result_dir/git_status_short.txt"
  )
  cat > "$result_dir/protocol.txt" <<PROTO
Part3 fixed 200ep final comparison protocol:
- fixed split manifest=$SPLIT_MANIFEST
- habitat.dataset.data_path=$DATA_PATH
- habitat.dataset.scenes_dir=$SCENES_DIR
- content_scenes=$CONTENT_SCENES
- habitat.environment.iterator_options.max_scene_repeat_episodes=-1
- habitat.environment.iterator_options.num_episode_sample=200
- habitat.environment.iterator_options.shuffle=True
- habitat.environment.iterator_options.group_by_scene=False
- habitat_baselines.test_episode_count=200
- habitat_baselines.eval.video_option=$VIDEO_OPTION
PROTO
}

write_variant_command() {
  local result_dir=$RESULT_ROOT/$VARIANT_NAME
  write_common_metadata "$OPT_REPO" "$result_dir"
  cat > "$result_dir/run_command.sh" <<RUNEOF
#!/usr/bin/env bash
set -euo pipefail
source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
export DEEPSEEK_API_BASE_URL="\${DEEPSEEK_API_BASE_URL:-\${DEEPSEEK_BASE_URL:-}}"
export DEEPSEEK_BASE_URL="\${DEEPSEEK_BASE_URL:-\${DEEPSEEK_API_BASE_URL:-}}"
export ASCENT_DEPS_REPO="$ASCENT_DEPS_REPO"
cd "$OPT_REPO"
if [ ! -e pretrained_weights ]; then
  ln -s "\$ASCENT_DEPS_REPO/pretrained_weights" pretrained_weights
fi
mkdir -p third_party/places365
if [ ! -f third_party/places365/categories_places365.txt ]; then
  cp "\$ASCENT_DEPS_REPO/third_party/places365/categories_places365.txt" third_party/places365/categories_places365.txt
fi
export PYTHONPATH="$OPT_REPO:$ASCENT_DEPS_REPO:$ASCENT_DEPS_REPO/third_party/D-FINE:$ASCENT_DEPS_REPO/third_party/GroundingDINO:$ASCENT_DEPS_REPO/third_party/MobileSAM:$ASCENT_DEPS_REPO/third_party/recognize-anything:$ASCENT_DEPS_REPO/third_party/habitat-lab/habitat-lab:$ASCENT_DEPS_REPO/third_party/habitat-lab/habitat-baselines:\${PYTHONPATH:-}"
export QWEN2_5_PORT=$QWEN2_5_PORT
export BLIP2ITM_PORT=$BLIP2ITM_PORT
export SAM_PORT=$SAM_PORT
export GROUNDING_DINO_PORT=$GROUNDING_DINO_PORT
export RAM_PORT=$RAM_PORT
export DFINE_PORT=$DFINE_PORT
export QWEN2_5_VL_PORT=$QWEN2_5_VL_PORT
python -u -m ascent.run \\
  --config-name=eval_ascent_hm3d.yaml \\
  "habitat.dataset.data_path='$DATA_PATH'" \\
  "habitat.dataset.scenes_dir=$SCENES_DIR" \\
  "habitat.dataset.content_scenes=$CONTENT_SCENES" \\
  habitat.environment.iterator_options.max_scene_repeat_episodes=-1 \\
  habitat.environment.iterator_options.num_episode_sample=200 \\
  habitat.environment.iterator_options.shuffle=True \\
  habitat.environment.iterator_options.group_by_scene=False \\
  habitat_baselines.test_episode_count=200 \\
  habitat_baselines.video_dir="$result_dir/videos" \\
  "habitat_baselines.eval.video_option=$VIDEO_OPTION" \\
  habitat_baselines.rl.policy.main_agent.llm.backend=deepseek_api \\
  habitat_baselines.rl.policy.main_agent.llm.model_name=deepseek-chat \\
  habitat_baselines.rl.policy.main_agent.pointnav_policy_path=/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/vlfm/data/pointnav_weights.pth \\
  habitat_baselines.rl.policy.main_agent.final_check.mode=selective_hybrid_boundary_gate_only \\
  habitat_baselines.rl.policy.main_agent.final_check.hybrid_accept_on_qwen_reject_if_blip_high=true \\
  habitat_baselines.rl.policy.main_agent.frontier.stuck_recovery_enabled=false \\
  habitat_baselines.rl.policy.main_agent.stair.floor_budget_enabled=false \\
  habitat_baselines.rl.policy.main_agent.final_check.qwen_approach_recovery_enabled=false
RUNEOF
  chmod +x "$result_dir/run_command.sh"
}

write_baseline_command() {
  local result_dir=$RESULT_ROOT/$BASELINE_NAME
  write_common_metadata "$BASE_REPO" "$result_dir"
  cat > "$result_dir/run_command.sh" <<RUNEOF
#!/usr/bin/env bash
set -euo pipefail
source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
export DEEPSEEK_API_BASE_URL="\${DEEPSEEK_API_BASE_URL:-\${DEEPSEEK_BASE_URL:-}}"
export DEEPSEEK_BASE_URL="\${DEEPSEEK_BASE_URL:-\${DEEPSEEK_API_BASE_URL:-}}"
cd "$BASE_REPO"
export QWEN2_5_PORT=$QWEN2_5_PORT
export BLIP2ITM_PORT=$BLIP2ITM_PORT
export SAM_PORT=$SAM_PORT
export GROUNDING_DINO_PORT=$GROUNDING_DINO_PORT
export RAM_PORT=$RAM_PORT
export DFINE_PORT=$DFINE_PORT
export QWEN2_5_VL_PORT=$QWEN2_5_VL_PORT
python -u -m ascent.run \\
  --config-name=eval_ascent_hm3d.yaml \\
  "habitat.dataset.data_path='$DATA_PATH'" \\
  "habitat.dataset.scenes_dir=$SCENES_DIR" \\
  "habitat.dataset.content_scenes=$CONTENT_SCENES" \\
  habitat.environment.iterator_options.max_scene_repeat_episodes=-1 \\
  habitat.environment.iterator_options.num_episode_sample=200 \\
  habitat.environment.iterator_options.shuffle=True \\
  habitat.environment.iterator_options.group_by_scene=False \\
  habitat_baselines.test_episode_count=200 \\
  habitat_baselines.video_dir="$result_dir/videos" \\
  "habitat_baselines.eval.video_option=$VIDEO_OPTION"
RUNEOF
  chmod +x "$result_dir/run_command.sh"
}

run_one() {
  local name=$1
  local result_dir=$RESULT_ROOT/$name
  echo "===== START $name $(date -Is) ====="
  set +e
  bash "$result_dir/run_command.sh" > "$result_dir/run.log" 2>&1
  local code=$?
  set -e
  echo "EXIT_CODE:$code" >> "$result_dir/run.log"
  python "$PARSER" "$result_dir" >> "$result_dir/parse.log" 2>&1 || true
  echo "===== END $name code=$code $(date -Is) ====="
  return "$code"
}

main() {
  prepare_opt_repo
  mkdir -p "$RESULT_ROOT"
  write_variant_command
  write_baseline_command
  local failed=0
  run_one "$VARIANT_NAME" || failed=1
  run_one "$BASELINE_NAME" || failed=1
  echo "All requested fixed-200 experiments attempted. Result root: $RESULT_ROOT"
  return "$failed"
}

main "$@"
