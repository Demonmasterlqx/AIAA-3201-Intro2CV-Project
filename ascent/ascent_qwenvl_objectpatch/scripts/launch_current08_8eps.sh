#!/usr/bin/env bash
set -euo pipefail

source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav

REPO=/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent_qwenvl_objectpatch
SESSION=qwenvl_objectpatch_current08_8
RESULT_DIR=results/baseline_compare_single/qwenvl_objectpatch_current08ada94_8eps_v1

cd "$REPO"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists; refusing to restart." >&2
  exit 2
fi

if [ -e "$RESULT_DIR/run.log" ]; then
  echo "Result log already exists at $RESULT_DIR/run.log; refusing to overwrite." >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
git rev-parse HEAD > "$RESULT_DIR/commit.txt"
git status --short > "$RESULT_DIR/git_status_short.txt"
git show --stat --oneline -5 > "$RESULT_DIR/recent_commits.txt"

cat > "$RESULT_DIR/run_command.sh" <<'RUNEOF'
#!/usr/bin/env bash
set -euo pipefail
source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav
cd /data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent_qwenvl_objectpatch
export QWEN2_5_PORT=14181
export BLIP2ITM_PORT=14182
export SAM_PORT=14183
export GROUNDING_DINO_PORT=14184
export RAM_PORT=14185
export DFINE_PORT=14186
export QWEN2_5_VL_PORT=14187
python -u -m ascent.run \
  --config-name=eval_ascent_hm3d.yaml \
  'habitat.dataset.content_scenes=[4ok3usBNeis,5cdEh9F2hJL,6s7QHgap2fW,DYehNKdT76V,Dd4bFSTQ8gi,Nfvxx8J5NCo,QaLdnwvtxbs,TEEsavR23oF,XB4GS9ShBRE,bxsVRursffK,cvZr5TUy5C5,mL8ThkuaVTM,mv2HUxq3B53,p53SfW6mjZe,q3zU7Yy5E5s,qyAac8rV8Zk,svBbv1Pavdk,wcojb4TFT35,ziup5kvtCCR,zt1RVoi7PcG]' \
  habitat.environment.iterator_options.max_scene_repeat_episodes=-1 \
  habitat.environment.iterator_options.num_episode_sample=8 \
  habitat.environment.iterator_options.shuffle=True \
  habitat.environment.iterator_options.group_by_scene=False \
  habitat_baselines.test_episode_count=8 \
  'habitat_baselines.eval.video_option=[]'
RUNEOF

chmod +x "$RESULT_DIR/run_command.sh"
tmux new-session -d -s "$SESSION" \
  "cd '$REPO' && bash '$RESULT_DIR/run_command.sh' > '$RESULT_DIR/run.log' 2>&1; echo EXIT_CODE:\$? >> '$RESULT_DIR/run.log'"

echo "started $SESSION"
echo "$RESULT_DIR"
sleep 5
tmux has-session -t "$SESSION"
tail -n 80 "$RESULT_DIR/run.log" || true
