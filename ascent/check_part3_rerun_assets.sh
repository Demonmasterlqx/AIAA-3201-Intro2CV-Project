#!/usr/bin/env bash
set -euo pipefail

BASE=/data/home/sim6g/code/aiaa3201_cv_project/ascent
for d in "$BASE/ascent" "$BASE/ascent_qwenvl_objectpatch" "$BASE/ascent_qwenvl_objectpatch_weakcache_rerun"; do
  echo "### $d"
  cd "$d"
  for p in pretrained_weights pretrained_weights/Qwen2.5-VL-7B-Instruct third_party/vlfm/data/pointnav_weights.pth third_party/D-FINE third_party/GroundingDINO third_party/MobileSAM; do
    if [ -e "$p" ]; then
      if [ -L "$p" ]; then kind=symlink; else kind=present; fi
      echo "OK $p $kind"
    else
      echo "MISS $p"
    fi
  done
  echo "commit $(git rev-parse --short HEAD)"
  echo
done
