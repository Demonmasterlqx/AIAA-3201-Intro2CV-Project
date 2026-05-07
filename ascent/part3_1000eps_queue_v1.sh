#!/usr/bin/env bash
set -euo pipefail
source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav

BASE="/data/home/sim6g/code/aiaa3201_cv_project/ascent"
SINGLE_REPO="/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent"
OBJ_REPO="/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent_qwenvl_objectpatch_weakcache_rerun"
SINGLE_RESULT="results/baseline_compare_single/20260502_single_1000eps_rerun_v1"
OBJ_RESULT="results/baseline_compare_single/qwenvl_objectpatch_weakcache_1000eps_rerun_v1"

printf 'PART3 1000eps queue started at %s\n' "$(date -Is)" | tee "/data/home/sim6g/code/aiaa3201_cv_project/ascent/part3_1000eps_queue_v1.log"
printf 'SINGLE_REPO=%s\nOBJ_REPO=%s\n' "$SINGLE_REPO" "$OBJ_REPO" | tee -a "/data/home/sim6g/code/aiaa3201_cv_project/ascent/part3_1000eps_queue_v1.log"

printf '\n[1/2] ASCENT-Single 1000eps started at %s\n' "$(date -Is)" | tee -a "/data/home/sim6g/code/aiaa3201_cv_project/ascent/part3_1000eps_queue_v1.log"
cd "$SINGLE_REPO"
bash "$SINGLE_RESULT/run_command.sh" > "$SINGLE_RESULT/run.log" 2>&1
single_code=$?
echo "EXIT_CODE:$single_code" >> "$SINGLE_RESULT/run.log"
printf '[1/2] ASCENT-Single finished with %s at %s\n' "$single_code" "$(date -Is)" | tee -a "/data/home/sim6g/code/aiaa3201_cv_project/ascent/part3_1000eps_queue_v1.log"
if [ "$single_code" -ne 0 ]; then
  exit "$single_code"
fi

printf '\n[2/2] Qwen-VL objectpatch weakcache 1000eps started at %s\n' "$(date -Is)" | tee -a "/data/home/sim6g/code/aiaa3201_cv_project/ascent/part3_1000eps_queue_v1.log"
cd "$OBJ_REPO"
bash "$OBJ_RESULT/run_command.sh" > "$OBJ_RESULT/run.log" 2>&1
obj_code=$?
echo "EXIT_CODE:$obj_code" >> "$OBJ_RESULT/run.log"
printf '[2/2] Qwen-VL objectpatch weakcache finished with %s at %s\n' "$obj_code" "$(date -Is)" | tee -a "/data/home/sim6g/code/aiaa3201_cv_project/ascent/part3_1000eps_queue_v1.log"
exit "$obj_code"
