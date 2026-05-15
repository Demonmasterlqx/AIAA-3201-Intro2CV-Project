#!/usr/bin/env bash
set -euo pipefail

OPT_REPO=/data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep
BASE_REPO=/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent
ASCENT_DEPS_REPO=/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent
ENV_FILE=/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent_qwenvl_objectpatch/.env
RESULT_ROOT=$OPT_REPO/results/part3_fixed200_final
RUN_NAME=part3_baseline_localqwen_fixed200_v1
RESULT_DIR=$RESULT_ROOT/$RUN_NAME
SESSION=part3_fixed200_localqwen_baseline_v1
DATA_PATH="$OPT_REPO/data/datasets/objectnav/hm3d/part3_200ep_fixed/v1/{split}/{split}.json.gz"
SCENES_DIR="$BASE_REPO/data/scene_datasets"
MANIFEST="$OPT_REPO/data/datasets/objectnav/hm3d/part3_200ep_fixed/episodes_200_part3_fixed.json"
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

cd "$OPT_REPO"
mkdir -p "$RESULT_DIR"
cp "$MANIFEST" "$RESULT_DIR/episodes_fixed.json"
git rev-parse HEAD > "$RESULT_DIR/commit.txt"
git status --short > "$RESULT_DIR/git_status_short.txt"

cat > "$RESULT_DIR/protocol.txt" <<PROTO
Corrected Part3 fixed 200ep Local-Qwen baseline:
- baseline repo/code path: $OPT_REPO
- data_path=$DATA_PATH
- scenes_dir=$SCENES_DIR
- content_scenes=$CONTENT_SCENES
- num_episode_sample=200
- shuffle=True
- group_by_scene=False
- test_episode_count=200
- planner backend=local_qwen
- final_check.mode=blip2
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
cd "$OPT_REPO"
export ASCENT_DEPS_REPO="$ASCENT_DEPS_REPO"
if [ ! -e pretrained_weights ]; then
  ln -s "\$ASCENT_DEPS_REPO/pretrained_weights" pretrained_weights
fi
mkdir -p third_party/places365
if [ ! -f third_party/places365/categories_places365.txt ]; then
  cp "\$ASCENT_DEPS_REPO/third_party/places365/categories_places365.txt" third_party/places365/categories_places365.txt
fi
export PYTHONPATH="$OPT_REPO:$ASCENT_DEPS_REPO:$ASCENT_DEPS_REPO/third_party/D-FINE:$ASCENT_DEPS_REPO/third_party/GroundingDINO:$ASCENT_DEPS_REPO/third_party/MobileSAM:$ASCENT_DEPS_REPO/third_party/recognize-anything:$ASCENT_DEPS_REPO/third_party/habitat-lab/habitat-lab:$ASCENT_DEPS_REPO/third_party/habitat-lab/habitat-baselines:\${PYTHONPATH:-}"
export QWEN2_5_PORT=14181
export BLIP2ITM_PORT=14182
export SAM_PORT=14183
export GROUNDING_DINO_PORT=14184
export RAM_PORT=14185
export DFINE_PORT=14186
export QWEN2_5_VL_PORT=14187
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
  habitat_baselines.video_dir="$RESULT_DIR/videos" \\
  "habitat_baselines.eval.video_option=[]" \\
  habitat_baselines.rl.policy.main_agent.llm.backend=local_qwen \\
  habitat_baselines.rl.policy.main_agent.pointnav_policy_path=/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/vlfm/data/pointnav_weights.pth \\
  habitat_baselines.rl.policy.main_agent.final_check.mode=blip2 \\
  habitat_baselines.rl.policy.main_agent.frontier.stuck_recovery_enabled=false \\
  habitat_baselines.rl.policy.main_agent.stair.floor_budget_enabled=false \\
  habitat_baselines.rl.policy.main_agent.final_check.qwen_approach_recovery_enabled=false
RUNEOF
chmod +x "$RESULT_DIR/run_command.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" \
  "cd '$OPT_REPO' && bash '$RESULT_DIR/run_command.sh' > '$RESULT_DIR/run.log' 2>&1; code=\$?; echo EXIT_CODE:\$code >> '$RESULT_DIR/run.log'; python scripts/parse_100ep_failures.py '$RESULT_DIR' >> '$RESULT_DIR/parse.log' 2>&1 || true; exit \$code"

echo "Started $SESSION"
sleep 3
tmux has-session -t "$SESSION" && echo TMUX_RUNNING || echo TMUX_DONE_OR_MISSING
tail -n 80 "$RESULT_DIR/run.log" || true
