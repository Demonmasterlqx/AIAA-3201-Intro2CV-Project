#!/usr/bin/env bash
set -euo pipefail
source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav
if [ -f "/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent_qwenvl_objectpatch/.env" ]; then
  set -a
  source "/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent_qwenvl_objectpatch/.env"
  set +a
fi
export DEEPSEEK_API_BASE_URL="${DEEPSEEK_API_BASE_URL:-${DEEPSEEK_BASE_URL:-}}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-${DEEPSEEK_API_BASE_URL:-}}"
export ASCENT_DEPS_REPO="/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent"
cd "/data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep"
if [ ! -e pretrained_weights ]; then
  ln -s "$ASCENT_DEPS_REPO/pretrained_weights" pretrained_weights
fi
mkdir -p third_party/places365
if [ ! -f third_party/places365/categories_places365.txt ]; then
  cp "$ASCENT_DEPS_REPO/third_party/places365/categories_places365.txt" third_party/places365/categories_places365.txt
fi
export PYTHONPATH="/data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/D-FINE:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/GroundingDINO:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/MobileSAM:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/recognize-anything:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/habitat-lab/habitat-lab:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/habitat-lab/habitat-baselines:${PYTHONPATH:-}"
export QWEN2_5_PORT=14181
export BLIP2ITM_PORT=14182
export SAM_PORT=14183
export GROUNDING_DINO_PORT=14184
export RAM_PORT=14185
export DFINE_PORT=14186
export QWEN2_5_VL_PORT=14187
python -u -m ascent.run \
  --config-name=eval_ascent_hm3d.yaml \
  "habitat.dataset.data_path='/data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep/data/datasets/objectnav/hm3d/part3_200ep_fixed/v1/{split}/{split}.json.gz'" \
  "habitat.dataset.scenes_dir=/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/data/scene_datasets" \
  "habitat.dataset.content_scenes=[4ok3usBNeis,5cdEh9F2hJL,6s7QHgap2fW,DYehNKdT76V,Dd4bFSTQ8gi,Nfvxx8J5NCo,QaLdnwvtxbs,TEEsavR23oF,XB4GS9ShBRE,bxsVRursffK,cvZr5TUy5C5,mL8ThkuaVTM,mv2HUxq3B53,p53SfW6mjZe,q3zU7Yy5E5s,qyAac8rV8Zk,svBbv1Pavdk,wcojb4TFT35,ziup5kvtCCR,zt1RVoi7PcG]" \
  habitat.environment.iterator_options.max_scene_repeat_episodes=-1 \
  habitat.environment.iterator_options.num_episode_sample=200 \
  habitat.environment.iterator_options.shuffle=True \
  habitat.environment.iterator_options.group_by_scene=False \
  habitat_baselines.test_episode_count=200 \
  habitat_baselines.video_dir="/data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep/results/part3_fixed200_final/deepseek_selective_hybrid_samepart3_fixed200_v3/videos" \
  "habitat_baselines.eval.video_option=[]" \
  habitat_baselines.rl.policy.main_agent.llm.backend=deepseek_api \
  habitat_baselines.rl.policy.main_agent.llm.model_name=deepseek-chat \
  habitat_baselines.rl.policy.main_agent.pointnav_policy_path=/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/vlfm/data/pointnav_weights.pth \
  habitat_baselines.rl.policy.main_agent.final_check.mode=selective_hybrid_boundary_gate_only \
  habitat_baselines.rl.policy.main_agent.final_check.hybrid_accept_on_qwen_reject_if_blip_high=true \
  habitat_baselines.rl.policy.main_agent.frontier.stuck_recovery_enabled=false \
  habitat_baselines.rl.policy.main_agent.stair.floor_budget_enabled=false \
  habitat_baselines.rl.policy.main_agent.final_check.qwen_approach_recovery_enabled=false
