#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep}
RUN_NAME=${RUN_NAME:-final1000_variant_boundary_tvambig_blipfallback_samepart3_v1}
RESULT_DIR=${RESULT_DIR:-results/localqwen_1000ep/${RUN_NAME}}
SESSION=${SESSION:-localqwen_1000ep_boundary_tvfallback_samepart3_v1}
BASE_PORT=${BASE_PORT:-14181}
VIDEO_OPTION=${VIDEO_OPTION:-[]}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}
ASCENT_DEPS_REPO=${ASCENT_DEPS_REPO:-/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent}
ENV_FILE=${ENV_FILE:-/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent_qwenvl_objectpatch/.env}
if [ -z "${OLD_DATA_PATH:-}" ]; then
  OLD_DATA_PATH='/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/data/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz'
fi
if [ -z "${OLD_SCENES_DIR:-}" ]; then
  OLD_SCENES_DIR='/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/data/scene_datasets'
fi
CONTENT_SCENES='[4ok3usBNeis,5cdEh9F2hJL,6s7QHgap2fW,DYehNKdT76V,Dd4bFSTQ8gi,Nfvxx8J5NCo,QaLdnwvtxbs,TEEsavR23oF,XB4GS9ShBRE,bxsVRursffK,cvZr5TUy5C5,mL8ThkuaVTM,mv2HUxq3B53,p53SfW6mjZe,q3zU7Yy5E5s,qyAac8rV8Zk,svBbv1Pavdk,wcojb4TFT35,ziup5kvtCCR,zt1RVoi7PcG]'

source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
export DEEPSEEK_API_BASE_URL="${DEEPSEEK_API_BASE_URL:-${DEEPSEEK_BASE_URL:-}}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-${DEEPSEEK_API_BASE_URL:-}}"

cd "$REPO"
mkdir -p "$RESULT_DIR"
if [ ! -e pretrained_weights ]; then
  ln -s "$ASCENT_DEPS_REPO/pretrained_weights" pretrained_weights
fi
mkdir -p third_party/places365
if [ ! -f third_party/places365/categories_places365.txt ]; then
  cp "$ASCENT_DEPS_REPO/third_party/places365/categories_places365.txt" third_party/places365/categories_places365.txt
fi
export PYTHONPATH="$REPO:$ASCENT_DEPS_REPO:$ASCENT_DEPS_REPO/third_party/D-FINE:$ASCENT_DEPS_REPO/third_party/GroundingDINO:$ASCENT_DEPS_REPO/third_party/MobileSAM:$ASCENT_DEPS_REPO/third_party/recognize-anything:$ASCENT_DEPS_REPO/third_party/habitat-lab/habitat-lab:$ASCENT_DEPS_REPO/third_party/habitat-lab/habitat-baselines:${PYTHONPATH:-}"

git rev-parse HEAD > "$RESULT_DIR/commit.txt"
git status --short > "$RESULT_DIR/git_status_short.txt"
cat > "$RESULT_DIR/protocol.txt" <<PROTO
Same protocol as Part 3 1000ep rerun v2/v4:
- content_scenes=${CONTENT_SCENES}
- habitat.dataset.data_path=${OLD_DATA_PATH}
- habitat.dataset.scenes_dir=${OLD_SCENES_DIR}
- habitat.environment.iterator_options.max_scene_repeat_episodes=-1
- habitat.environment.iterator_options.num_episode_sample=1000
- habitat.environment.iterator_options.shuffle=True
- habitat.environment.iterator_options.group_by_scene=False
- habitat_baselines.test_episode_count=1000
- habitat_baselines.eval.video_option=[]
Variant changes only final-check/stop gate; planner remains local_qwen.
PROTO

cat > "$RESULT_DIR/run_command.sh" <<RUNEOF
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
cd "$REPO"
export ASCENT_DEPS_REPO="$ASCENT_DEPS_REPO"
if [ ! -e pretrained_weights ]; then
  ln -s "\$ASCENT_DEPS_REPO/pretrained_weights" pretrained_weights
fi
mkdir -p third_party/places365
if [ ! -f third_party/places365/categories_places365.txt ]; then
  cp "\$ASCENT_DEPS_REPO/third_party/places365/categories_places365.txt" third_party/places365/categories_places365.txt
fi
export PYTHONPATH="$REPO:$ASCENT_DEPS_REPO:$ASCENT_DEPS_REPO/third_party/D-FINE:$ASCENT_DEPS_REPO/third_party/GroundingDINO:$ASCENT_DEPS_REPO/third_party/MobileSAM:$ASCENT_DEPS_REPO/third_party/recognize-anything:$ASCENT_DEPS_REPO/third_party/habitat-lab/habitat-lab:$ASCENT_DEPS_REPO/third_party/habitat-lab/habitat-baselines:\${PYTHONPATH:-}"
export QWEN2_5_PORT=$((BASE_PORT))
export BLIP2ITM_PORT=$((BASE_PORT + 1))
export SAM_PORT=$((BASE_PORT + 2))
export GROUNDING_DINO_PORT=$((BASE_PORT + 3))
export RAM_PORT=$((BASE_PORT + 4))
export DFINE_PORT=$((BASE_PORT + 5))
export QWEN2_5_VL_PORT=$((BASE_PORT + 6))
EXTRA_OVERRIDES='$EXTRA_OVERRIDES'
python -u -m ascent.run \
  --config-name=eval_ascent_hm3d.yaml \
  "habitat.dataset.data_path='$OLD_DATA_PATH'" \
  "habitat.dataset.scenes_dir=$OLD_SCENES_DIR" \
  "habitat.dataset.content_scenes=$CONTENT_SCENES" \
  habitat.environment.iterator_options.max_scene_repeat_episodes=-1 \
  habitat.environment.iterator_options.num_episode_sample=1000 \
  habitat.environment.iterator_options.shuffle=True \
  habitat.environment.iterator_options.group_by_scene=False \
  habitat_baselines.test_episode_count=1000 \
  habitat_baselines.video_dir="$RESULT_DIR/videos" \
  "habitat_baselines.eval.video_option=$VIDEO_OPTION" \
  habitat_baselines.rl.policy.main_agent.llm.backend=local_qwen \
  habitat_baselines.rl.policy.main_agent.pointnav_policy_path=/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/vlfm/data/pointnav_weights.pth \
  habitat_baselines.rl.policy.main_agent.final_check.mode=selective_hybrid_boundary_gate_only \
  habitat_baselines.rl.policy.main_agent.final_check.hybrid_accept_on_qwen_reject_if_blip_high=true \
  habitat_baselines.rl.policy.main_agent.frontier.stuck_recovery_enabled=false \
  habitat_baselines.rl.policy.main_agent.stair.floor_budget_enabled=false \
  habitat_baselines.rl.policy.main_agent.final_check.qwen_approach_recovery_enabled=false \
  \$EXTRA_OVERRIDES
RUNEOF
chmod +x "$RESULT_DIR/run_command.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Killing existing tmux session $SESSION"
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" \
  "cd '$REPO' && bash '$RESULT_DIR/run_command.sh' > '$RESULT_DIR/run.log' 2>&1; code=\$?; echo EXIT_CODE:\$code >> '$RESULT_DIR/run.log'; python scripts/parse_100ep_failures.py '$RESULT_DIR' >> '$RESULT_DIR/parse.log' 2>&1 || true; exit \$code"

echo "started tmux session: $SESSION"
echo "result dir: $REPO/$RESULT_DIR"
sleep 3
tmux has-session -t "$SESSION"
tail -n 80 "$RESULT_DIR/run.log" || true