# AIAA 3201 Zero-Shot Object Navigation Code Release

This repository contains the code used for the AIAA 3201 project report,
"From Semantic Frontiers to Selective Vision-Language Verification for Zero-Shot Object Navigation."
The code is organized by the three required project parts:

- `part1_zson_plan_a/`: repaired Plan A dense semantic-map ObjectNav branch.
- `part1_vlfm_plan_b/`: VLFM-style Plan B frontier-scoring branch.
- `part2_ascent_baseline/`: ASCENT-style LLM-driven ObjectNav baseline reproduction.
- `part3_selective_hybrid/`: final selective Qwen-VL object-patch verification branch.

Large assets are intentionally not included: HM3D/MP3D scenes, Habitat ObjectNav episode files,
model checkpoints, generated videos, and full run output directories. The submitted Canvas package
contains rendered trajectory videos and selected semantic-map visualizations.

## Expected Assets

The experiments assume a Linux machine with CUDA, Conda, Habitat-Sim/Habitat-Lab compatible with the
original methods, and the following external assets placed locally:

- HM3D ObjectNav episodes: `data/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz`
- HM3D scenes: `data/scene_datasets/hm3d/...`
- MP3D ObjectNav episodes: `data/MatterPort3D/objectnav/mp3d/v1/{split}/{split}.json.gz` or an equivalent path
- MP3D scenes: `data/MatterPort3D/...` or an equivalent path
- ASCENT/VLFM checkpoints: PointNav, BLIP2, MobileSAM, GroundingDINO/D-FINE, RAM++, RedNet, Qwen2.5, and Qwen2.5-VL where required
- API configuration for DeepSeek/OpenAI-compatible backends when running LLM planner variants

Paths in the historical run scripts reflect the remote evaluation machine. For a new machine, set the
environment variables shown below or edit the dataset/checkpoint paths in the shell scripts.

## Part 1A: ZSON Plan A Dense Semantic Map

From `part1_zson_plan_a/`:

```bash
# Optional component tests
python -m unittest tests/test_plan_a_components.py

# HM3D smoke run
PYTHON_BIN=python RUN_NAME=plan_a_hm3d_smoke \
  bash scripts/plan-a-smoke-hm3d.sh PLAN_A.MAX_EPISODES 5 TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 200

# HM3D 200-episode repaired active-goal run used in the report
PYTHON_BIN=python RUN_NAME=plan_a_activegoal_hm3d200_20260515 \
  bash scripts/plan-a-eval-hm3d.sh PLAN_A.MAX_EPISODES 200 TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 500

# MP3D 200-episode repaired active-goal run used in the report
PYTHON_BIN=python RUN_NAME=plan_a_activegoal_mp3d200_20260515 \
  bash scripts/plan-a-eval-mp3d.sh PLAN_A.MAX_EPISODES 200 TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 500
```

The repaired Plan A branch fixes recovery-state reset behavior and active-goal replanning. The reported
subset results use `PLAN_A.USE_SUCCESS_DISTANCE_STOP=True`, so they are reported as repaired subset
evidence rather than a strict non-oracle full benchmark.

## Part 1B: VLFM Plan B Frontier Scoring

From `part1_vlfm_plan_b/`:

```bash
# Install in editable mode after dependencies are available
pip install -e .

# One-episode smoke comparison between dense and frontier policies
SPLIT=val_mini EPISODES=1 VIDEO_OPTION='[]' bash project/run_eval_pair.sh

# Full/large runs can be launched with the project runner or the original scripts.
# Example frontier policy entry point:
bash scripts/eval_frontier_policy.sh
```

The report uses VLFM Plan B as the stronger Part 1 frontier-based baseline.

## Part 2: ASCENT Baseline Reproduction

From `part2_ascent_baseline/`:

```bash
# Check expected data/weight layout
python scripts/check_ascent_data.py --dataset hm3d

# Start model servers as configured for the local machine
bash scripts/launch_vlm_servers_ascent.sh

# Run ASCENT-Single on HM3D. Adjust paths in the Hydra overrides for a new machine.
python -u -m ascent.run \
  --config-name=eval_ascent_hm3d.yaml \
  habitat_baselines.rl.policy.main_agent.llm.backend=deepseek_api \
  habitat_baselines.rl.policy.main_agent.llm.model_name=deepseek-chat \
  habitat_baselines.test_episode_count=1000 \
  habitat.environment.iterator_options.num_episode_sample=1000 \
  "habitat_baselines.eval.video_option=[]"
```

The main Part 2 baseline reported in the paper is the DeepSeek ASCENT-Single 1000-episode run.

## Part 3: Selective Qwen-VL Object-Patch Verification

From `part3_selective_hybrid/`:

```bash
# Check expected data/weight layout
python scripts/check_ascent_data.py --dataset hm3d

# Launch local vision-language model servers before running navigation
bash scripts/launch_vlm_servers_ascent.sh

# Local-Qwen BLIP2 baseline on the fixed 200-episode subset
bash scripts/launch_part3_fixed200_localqwen_baseline.sh

# Final DeepSeek selective-hybrid comparison on the same fixed 200-episode subset
bash scripts/launch_part3_fixed200_deepseek_queue_v3.sh

# Optional 1000-episode scale check for the selective hybrid
bash scripts/run_localqwen_1000ep_samepart3_variant.sh
```

The core method is implemented in the ASCENT policy/final-check code under `part3_selective_hybrid/ascent/`
and the Qwen-VL wrapper under `part3_selective_hybrid/model_api/qwen25_vl_out.py`. Lightweight reference
metrics for the report are stored in `part3_selective_hybrid/results_reference/`.

## Reported Key Results

- Plan A repaired HM3D 200ep: 48.50% SR, 13.28% SPL.
- Plan A repaired MP3D 200ep: 51.00% SR, 13.69% SPL.
- VLFM Plan B HM3D full val: 52.30% SR, 30.34% SPL.
- ASCENT-Single DeepSeek 1000ep baseline: 64.50% SR, 33.36% SPL.
- Fixed-200 Local-Qwen baseline vs DeepSeek selective hybrid: 130/200 to 132/200 successes, 65.00% to 66.00% SR.

## What Is Not Included

This release excludes raw datasets, checkpoints, generated videos, large debug folders, and third-party source trees.
Install third-party packages from their upstream repositories and place data/weights according to each submodule README.
