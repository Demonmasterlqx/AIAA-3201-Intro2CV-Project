#!/usr/bin/env bash
set -euo pipefail

source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav

BASE=/data/home/sim6g/code/aiaa3201_cv_project/ascent
SINGLE_REPO="$BASE/ascent"
OBJ_REPO="$BASE/ascent_qwenvl_objectpatch_weakcache_rerun"
ASSET_REPO="$BASE/ascent_qwenvl_objectpatch"
SESSION=part3_1000eps_rerun_v1
SINGLE_RESULT="results/baseline_compare_single/20260502_single_1000eps_rerun_v1"
OBJ_RESULT="results/baseline_compare_single/qwenvl_objectpatch_weakcache_1000eps_rerun_v1"
SCENES='habitat.dataset.content_scenes=[4ok3usBNeis,5cdEh9F2hJL,6s7QHgap2fW,DYehNKdT76V,Dd4bFSTQ8gi,Nfvxx8J5NCo,QaLdnwvtxbs,TEEsavR23oF,XB4GS9ShBRE,bxsVRursffK,cvZr5TUy5C5,mL8ThkuaVTM,mv2HUxq3B53,p53SfW6mjZe,q3zU7Yy5E5s,qyAac8rV8Zk,svBbv1Pavdk,wcojb4TFT35,ziup5kvtCCR,zt1RVoi7PcG]'

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists; refusing to start a duplicate." >&2
  exit 2
fi

# The frozen worktree is a clean git checkout, while model weights are local/untracked assets.
cd "$OBJ_REPO"
if [ ! -e pretrained_weights ] && [ -e "$ASSET_REPO/pretrained_weights" ]; then
  ln -s "$ASSET_REPO/pretrained_weights" pretrained_weights
fi

# Refuse to overwrite previous reruns.
for pair in "$SINGLE_REPO:$SINGLE_RESULT" "$OBJ_REPO:$OBJ_RESULT"; do
  repo="${pair%%:*}"
  result="${pair#*:}"
  if [ -e "$repo/$result/run.log" ]; then
    echo "Result already exists: $repo/$result/run.log" >&2
    exit 2
  fi
done

mkdir -p "$SINGLE_REPO/$SINGLE_RESULT" "$OBJ_REPO/$OBJ_RESULT"

git -C "$SINGLE_REPO" rev-parse HEAD > "$SINGLE_REPO/$SINGLE_RESULT/commit.txt"
git -C "$SINGLE_REPO" status --short > "$SINGLE_REPO/$SINGLE_RESULT/git_status_short.txt"
git -C "$SINGLE_REPO" show --stat --oneline -5 > "$SINGLE_REPO/$SINGLE_RESULT/recent_commits.txt"

git -C "$OBJ_REPO" rev-parse HEAD > "$OBJ_REPO/$OBJ_RESULT/commit.txt"
git -C "$OBJ_REPO" status --short > "$OBJ_REPO/$OBJ_RESULT/git_status_short.txt"
git -C "$OBJ_REPO" show --stat --oneline -5 > "$OBJ_REPO/$OBJ_RESULT/recent_commits.txt"

cat > "$SINGLE_REPO/$SINGLE_RESULT/run_command.sh" <<RUNEOF
#!/usr/bin/env bash
set -euo pipefail
source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav
cd "$SINGLE_REPO"
export QWEN2_5_PORT=14181
export BLIP2ITM_PORT=14182
export SAM_PORT=14183
export GROUNDING_DINO_PORT=14184
export RAM_PORT=14185
export DFINE_PORT=14186
export QWEN2_5_VL_PORT=14187
python -u -m ascent.run \\
  --config-name=eval_ascent_hm3d.yaml \\
  '$SCENES' \\
  habitat.environment.iterator_options.max_scene_repeat_episodes=-1 \\
  habitat.environment.iterator_options.num_episode_sample=1000 \\
  habitat.environment.iterator_options.shuffle=True \\
  habitat.environment.iterator_options.group_by_scene=False \\
  habitat_baselines.test_episode_count=1000 \\
  'habitat_baselines.eval.video_option=[]'
RUNEOF

cat > "$OBJ_REPO/$OBJ_RESULT/run_command.sh" <<RUNEOF
#!/usr/bin/env bash
set -euo pipefail
source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav
cd "$OBJ_REPO"
export QWEN2_5_PORT=14181
export BLIP2ITM_PORT=14182
export SAM_PORT=14183
export GROUNDING_DINO_PORT=14184
export RAM_PORT=14185
export DFINE_PORT=14186
export QWEN2_5_VL_PORT=14187
python -u -m ascent.run \\
  --config-name=eval_ascent_hm3d.yaml \\
  '$SCENES' \\
  habitat.environment.iterator_options.max_scene_repeat_episodes=-1 \\
  habitat.environment.iterator_options.num_episode_sample=1000 \\
  habitat.environment.iterator_options.shuffle=True \\
  habitat.environment.iterator_options.group_by_scene=False \\
  habitat_baselines.test_episode_count=1000 \\
  'habitat_baselines.eval.video_option=[]'
RUNEOF
chmod +x "$SINGLE_REPO/$SINGLE_RESULT/run_command.sh" "$OBJ_REPO/$OBJ_RESULT/run_command.sh"

cat > "$BASE/part3_1000eps_queue_v1.sh" <<QUEUEEOF
#!/usr/bin/env bash
set -euo pipefail
source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav

BASE="$BASE"
SINGLE_REPO="$SINGLE_REPO"
OBJ_REPO="$OBJ_REPO"
SINGLE_RESULT="$SINGLE_RESULT"
OBJ_RESULT="$OBJ_RESULT"

printf 'PART3 1000eps queue started at %s\n' "\$(date -Is)" | tee "$BASE/part3_1000eps_queue_v1.log"
printf 'SINGLE_REPO=%s\nOBJ_REPO=%s\n' "\$SINGLE_REPO" "\$OBJ_REPO" | tee -a "$BASE/part3_1000eps_queue_v1.log"

printf '\n[1/2] ASCENT-Single 1000eps started at %s\n' "\$(date -Is)" | tee -a "$BASE/part3_1000eps_queue_v1.log"
cd "\$SINGLE_REPO"
bash "\$SINGLE_RESULT/run_command.sh" > "\$SINGLE_RESULT/run.log" 2>&1
single_code=\$?
echo "EXIT_CODE:\$single_code" >> "\$SINGLE_RESULT/run.log"
printf '[1/2] ASCENT-Single finished with %s at %s\n' "\$single_code" "\$(date -Is)" | tee -a "$BASE/part3_1000eps_queue_v1.log"
if [ "\$single_code" -ne 0 ]; then
  exit "\$single_code"
fi

printf '\n[2/2] Qwen-VL objectpatch weakcache 1000eps started at %s\n' "\$(date -Is)" | tee -a "$BASE/part3_1000eps_queue_v1.log"
cd "\$OBJ_REPO"
bash "\$OBJ_RESULT/run_command.sh" > "\$OBJ_RESULT/run.log" 2>&1
obj_code=\$?
echo "EXIT_CODE:\$obj_code" >> "\$OBJ_RESULT/run.log"
printf '[2/2] Qwen-VL objectpatch weakcache finished with %s at %s\n' "\$obj_code" "\$(date -Is)" | tee -a "$BASE/part3_1000eps_queue_v1.log"
exit "\$obj_code"
QUEUEEOF
chmod +x "$BASE/part3_1000eps_queue_v1.sh"

tmux new-session -d -s "$SESSION" "bash '$BASE/part3_1000eps_queue_v1.sh'"
echo "started $SESSION"
echo "single_log=$SINGLE_REPO/$SINGLE_RESULT/run.log"
echo "objectpatch_log=$OBJ_REPO/$OBJ_RESULT/run.log"
sleep 5
tmux has-session -t "$SESSION"
tail -n 80 "$BASE/part3_1000eps_queue_v1.log" || true
