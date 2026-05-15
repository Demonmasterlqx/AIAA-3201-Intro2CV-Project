#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep}
RUN_NAME=${RUN_NAME:?RUN_NAME is required}
FINAL_CHECK_MODE=${FINAL_CHECK_MODE:-blip2}
FRONTIER_STUCK_RECOVERY_ENABLED=${FRONTIER_STUCK_RECOVERY_ENABLED:-false}
STAIR_FLOOR_BUDGET_ENABLED=${STAIR_FLOOR_BUDGET_ENABLED:-false}
QWEN_APPROACH_RECOVERY_ENABLED=${QWEN_APPROACH_RECOVERY_ENABLED:-false}
HYBRID_ACCEPT_ON_QWEN_REJECT_IF_BLIP_HIGH=${HYBRID_ACCEPT_ON_QWEN_REJECT_IF_BLIP_HIGH:-true}
EPISODE_COUNT=${EPISODE_COUNT:-100}
SPLIT_SEED=${SPLIT_SEED:-3201}
SESSION=${SESSION:-localqwen_${FINAL_CHECK_MODE}_${EPISODE_COUNT}ep}
RESULT_DIR=${RESULT_DIR:-results/localqwen_100ep/${RUN_NAME}}
VIDEO_OPTION=${VIDEO_OPTION:-[]}
BASE_PORT=${BASE_PORT:-14181}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}
DATASET_ROOT=${DATASET_ROOT:-data/datasets/objectnav/hm3d/localqwen_${EPISODE_COUNT}ep_fixed/v2/val}
EPISODES_MANIFEST=${EPISODES_MANIFEST:-data/datasets/objectnav/hm3d/localqwen_${EPISODE_COUNT}ep_fixed/episodes_${EPISODE_COUNT}_localqwen_fixed.json}

source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav

cd "$REPO"
mkdir -p "$RESULT_DIR"
ASCENT_DEPS_REPO=${ASCENT_DEPS_REPO:-/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent}
if [ ! -e pretrained_weights ]; then
  ln -s "$ASCENT_DEPS_REPO/pretrained_weights" pretrained_weights
fi
mkdir -p third_party/places365
if [ ! -f third_party/places365/categories_places365.txt ]; then
  cp "$ASCENT_DEPS_REPO/third_party/places365/categories_places365.txt" third_party/places365/categories_places365.txt
fi
export PYTHONPATH="$REPO:$ASCENT_DEPS_REPO:$ASCENT_DEPS_REPO/third_party/D-FINE:$ASCENT_DEPS_REPO/third_party/GroundingDINO:$ASCENT_DEPS_REPO/third_party/MobileSAM:$ASCENT_DEPS_REPO/third_party/recognize-anything:$ASCENT_DEPS_REPO/third_party/habitat-lab/habitat-lab:$ASCENT_DEPS_REPO/third_party/habitat-lab/habitat-baselines:${PYTHONPATH:-}"

if [ "${USE_EXISTING_SPLIT:-false}" != "true" ]; then
  python scripts/make_fixed_100ep_split.py \
    --sample-size "$EPISODE_COUNT" \
    --seed "$SPLIT_SEED" \
    --output-root "$DATASET_ROOT" \
    --manifest "$EPISODES_MANIFEST"
fi

git rev-parse HEAD > "$RESULT_DIR/commit.txt"
git status --short > "$RESULT_DIR/git_status_short.txt"
cp "$EPISODES_MANIFEST" "$RESULT_DIR/episodes_fixed.json"

cat > "$RESULT_DIR/run_command.sh" <<RUNEOF
#!/usr/bin/env bash
set -euo pipefail
source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav
cd "$REPO"
export ASCENT_DEPS_REPO=${ASCENT_DEPS_REPO:-/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent}
if [ ! -e pretrained_weights ]; then
  ln -s "$ASCENT_DEPS_REPO/pretrained_weights" pretrained_weights
fi
mkdir -p third_party/places365
if [ ! -f third_party/places365/categories_places365.txt ]; then
  cp "$ASCENT_DEPS_REPO/third_party/places365/categories_places365.txt" third_party/places365/categories_places365.txt
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
python -u -m ascent.run \\
  --config-name=eval_ascent_hm3d.yaml \\
  "habitat.dataset.data_path='$REPO/$DATASET_ROOT/{split}.json.gz'" \\
  habitat.dataset.scenes_dir=/data/home/sim6g/code/aiaa3201_cv_project/data/scene_datasets \\
  habitat.environment.iterator_options.max_scene_repeat_episodes=-1 \\
  habitat.environment.iterator_options.num_episode_sample=-1 \\
  habitat.environment.iterator_options.shuffle=False \\
  habitat.environment.iterator_options.group_by_scene=False \\
  habitat.task.measurements.multi_floor_map.draw_goal_aabbs=False \\
  habitat_baselines.test_episode_count="$EPISODE_COUNT" \\
  habitat_baselines.video_dir="$RESULT_DIR/videos" \\
  "habitat_baselines.eval.video_option=$VIDEO_OPTION" \\
  habitat_baselines.rl.policy.main_agent.llm.backend=local_qwen \\
  habitat_baselines.rl.policy.main_agent.pointnav_policy_path=/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/vlfm/data/pointnav_weights.pth \\
  habitat_baselines.rl.policy.main_agent.final_check.mode="$FINAL_CHECK_MODE" \\
  habitat_baselines.rl.policy.main_agent.final_check.hybrid_accept_on_qwen_reject_if_blip_high="$HYBRID_ACCEPT_ON_QWEN_REJECT_IF_BLIP_HIGH" \\
  habitat_baselines.rl.policy.main_agent.frontier.stuck_recovery_enabled="$FRONTIER_STUCK_RECOVERY_ENABLED" \\
  habitat_baselines.rl.policy.main_agent.stair.floor_budget_enabled="$STAIR_FLOOR_BUDGET_ENABLED" \\
  habitat_baselines.rl.policy.main_agent.final_check.qwen_approach_recovery_enabled="$QWEN_APPROACH_RECOVERY_ENABLED" \\
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
