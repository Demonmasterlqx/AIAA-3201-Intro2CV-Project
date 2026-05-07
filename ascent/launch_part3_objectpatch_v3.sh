#!/usr/bin/env bash
set -u pipefail

source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav

BASE="/data/home/sim6g/code/aiaa3201_cv_project/ascent"
OBJ_REPO="$BASE/ascent_qwenvl_objectpatch_weakcache_rerun"
SRC_RESULT="$OBJ_REPO/results/baseline_compare_single/qwenvl_objectpatch_weakcache_1000eps_rerun_v2"
DST_RESULT="$OBJ_REPO/results/baseline_compare_single/qwenvl_objectpatch_weakcache_1000eps_rerun_v3"
QUEUE_LOG="$BASE/part3_1000eps_objectpatch_v3.log"
SESSION="part3_objectpatch_1000eps_v3"

mkdir -p "$DST_RESULT"
cp "$SRC_RESULT/run_command.sh" "$DST_RESULT/run_command.sh"
chmod +x "$DST_RESULT/run_command.sh"
cd "$OBJ_REPO"
git rev-parse HEAD > "$DST_RESULT/commit.txt"
git status --short > "$DST_RESULT/git_status_short.txt"
git log --oneline -8 > "$DST_RESULT/recent_commits.txt"

tmux kill-session -t "$SESSION" 2>/dev/null || true
cat > "$BASE/part3_objectpatch_1000eps_v3_inner.sh" <<'INNER'
#!/usr/bin/env bash
set -u pipefail
source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav
BASE="/data/home/sim6g/code/aiaa3201_cv_project/ascent"
OBJ_REPO="$BASE/ascent_qwenvl_objectpatch_weakcache_rerun"
DST_RESULT="$OBJ_REPO/results/baseline_compare_single/qwenvl_objectpatch_weakcache_1000eps_rerun_v3"
QUEUE_LOG="$BASE/part3_1000eps_objectpatch_v3.log"
printf 'PART3 objectpatch v3 started at %s\n' "$(date -Is)" | tee "$QUEUE_LOG"
printf 'OBJ_REPO=%s\nDST_RESULT=%s\n' "$OBJ_REPO" "$DST_RESULT" | tee -a "$QUEUE_LOG"
cd "$OBJ_REPO"
bash "$DST_RESULT/run_command.sh" > "$DST_RESULT/run.log" 2>&1
code=$?
echo "EXIT_CODE:$code" >> "$DST_RESULT/run.log"
printf 'PART3 objectpatch v3 finished with %s at %s\n' "$code" "$(date -Is)" | tee -a "$QUEUE_LOG"
exit "$code"
INNER
chmod +x "$BASE/part3_objectpatch_1000eps_v3_inner.sh"

tmux new-session -d -s "$SESSION" "bash '$BASE/part3_objectpatch_1000eps_v3_inner.sh'"
printf 'launched %s\n' "$SESSION"
