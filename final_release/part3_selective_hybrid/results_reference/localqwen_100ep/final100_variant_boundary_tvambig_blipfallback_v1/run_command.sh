#!/usr/bin/env bash
set -euo pipefail
source /data/home/sim6g/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav
cd "/data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep"
export ASCENT_DEPS_REPO=/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent
if [ ! -e pretrained_weights ]; then
  ln -s "/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/pretrained_weights" pretrained_weights
fi
mkdir -p third_party/places365
if [ ! -f third_party/places365/categories_places365.txt ]; then
  cp "/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/places365/categories_places365.txt" third_party/places365/categories_places365.txt
fi
export PYTHONPATH="/data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/D-FINE:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/GroundingDINO:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/MobileSAM:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/recognize-anything:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/habitat-lab/habitat-lab:/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/habitat-lab/habitat-baselines:${PYTHONPATH:-}"
export QWEN2_5_PORT=14181
export BLIP2ITM_PORT=14182
export SAM_PORT=14183
export GROUNDING_DINO_PORT=14184
export RAM_PORT=14185
export DFINE_PORT=14186
export QWEN2_5_VL_PORT=14187
EXTRA_OVERRIDES=''
python -u -m ascent.run \
  --config-name=eval_ascent_hm3d.yaml \
  "habitat.dataset.data_path='/data/home/sim6g/code/aiaa3201_cv_project/ascent_opt_localqwen_100ep/data/datasets/objectnav/hm3d/localqwen_100ep_fixed/v2/val/{split}.json.gz'" \
  habitat.dataset.scenes_dir=/data/home/sim6g/code/aiaa3201_cv_project/data/scene_datasets \
  habitat.environment.iterator_options.max_scene_repeat_episodes=-1 \
  habitat.environment.iterator_options.num_episode_sample=-1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat.environment.iterator_options.group_by_scene=False \
  habitat.task.measurements.multi_floor_map.draw_goal_aabbs=False \
  habitat_baselines.test_episode_count="100" \
  habitat_baselines.video_dir="results/localqwen_100ep/final100_variant_boundary_tvambig_blipfallback_v1/videos" \
  "habitat_baselines.eval.video_option=[]" \
  habitat_baselines.rl.policy.main_agent.llm.backend=local_qwen \
  habitat_baselines.rl.policy.main_agent.pointnav_policy_path=/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/third_party/vlfm/data/pointnav_weights.pth \
  habitat_baselines.rl.policy.main_agent.final_check.mode="selective_hybrid_boundary_gate_only" \
  habitat_baselines.rl.policy.main_agent.final_check.hybrid_accept_on_qwen_reject_if_blip_high="true" \
  habitat_baselines.rl.policy.main_agent.frontier.stuck_recovery_enabled="false" \
  habitat_baselines.rl.policy.main_agent.stair.floor_budget_enabled="false" \
  habitat_baselines.rl.policy.main_agent.final_check.qwen_approach_recovery_enabled="false" \
  $EXTRA_OVERRIDES
