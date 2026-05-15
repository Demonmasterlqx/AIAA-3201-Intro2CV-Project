#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT_DIR}/scripts/eval_frontier_policy.sh" "$@"
