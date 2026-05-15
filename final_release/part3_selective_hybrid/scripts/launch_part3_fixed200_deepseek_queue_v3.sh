#!/usr/bin/env bash
set -euo pipefail

cd /data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep
chmod +x scripts/run_part3_fixed200_deepseek_queue.sh
mkdir -p results/part3_fixed200_final

SESSION=part3_fixed200_deepseek_final_queue_v3
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Existing session $SESSION found; not starting duplicate."
else
  tmux new-session -d -s "$SESSION" \
    "cd /data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep && bash scripts/run_part3_fixed200_deepseek_queue.sh > results/part3_fixed200_final/queue.log 2>&1; code=\$?; echo QUEUE_EXIT_CODE:\$code >> results/part3_fixed200_final/queue.log; exit \$code"
  echo "Started $SESSION"
fi

sleep 3
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo TMUX_RUNNING
else
  echo TMUX_DONE_OR_MISSING
fi

echo "--- queue log ---"
tail -n 80 results/part3_fixed200_final/queue.log 2>/dev/null || true
echo "--- result files ---"
find results/part3_fixed200_final -maxdepth 2 -type f \( -name run_command.sh -o -name protocol.txt -o -name run.log -o -name metrics_summary.json \) -print 2>/dev/null | sort
