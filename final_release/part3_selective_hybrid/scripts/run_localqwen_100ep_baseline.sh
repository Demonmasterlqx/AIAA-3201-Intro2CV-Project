#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME=${RUN_NAME:-baseline_blip2_$(date +%Y%m%d_%H%M%S)}
export FINAL_CHECK_MODE=blip2
export FRONTIER_STUCK_RECOVERY_ENABLED=false
export STAIR_FLOOR_BUDGET_ENABLED=false
export QWEN_APPROACH_RECOVERY_ENABLED=false
export SESSION=${SESSION:-localqwen_100ep_baseline}
exec "$(dirname "$0")/run_localqwen_100ep_common.sh"
