#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME=${RUN_NAME:-variant_selective_hybrid_$(date +%Y%m%d_%H%M%S)}
export FINAL_CHECK_MODE=${FINAL_CHECK_MODE:-selective_hybrid}
export FRONTIER_STUCK_RECOVERY_ENABLED=${FRONTIER_STUCK_RECOVERY_ENABLED:-true}
export STAIR_FLOOR_BUDGET_ENABLED=${STAIR_FLOOR_BUDGET_ENABLED:-true}
export QWEN_APPROACH_RECOVERY_ENABLED=${QWEN_APPROACH_RECOVERY_ENABLED:-true}
export HYBRID_ACCEPT_ON_QWEN_REJECT_IF_BLIP_HIGH=${HYBRID_ACCEPT_ON_QWEN_REJECT_IF_BLIP_HIGH:-true}
export SESSION=${SESSION:-localqwen_100ep_variant}
exec "$(dirname "$0")/run_localqwen_100ep_common.sh"
