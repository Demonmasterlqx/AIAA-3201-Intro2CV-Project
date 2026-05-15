# AIAA 3201 Intro2CV Project: Zero-Shot Object Navigation

This repository contains our AIAA 3201 project code for **Visual Navigation in 3D
Environments**. The project studies zero-shot ObjectNav from dense CLIP semantic
mapping, to VLFM-style frontier scoring, to ASCENT-style LLM navigation with a
selective vision-language final verifier.

## Repository Layout

- `zson/`: Part 1 Plan A dense semantic-map branch. This includes the repaired
  recovery state machine and active-goal replanning used for the report.
- `vlfm/`: Part 1 Plan B VLFM-style frontier-scoring branch.
- `ascent/`: Part 2 ASCENT reproduction and Part 3 Qwen-VL / selective-hybrid
  experiment branches. Final fixed-200 result summaries are in
  `ascent/results_reference/`.
- `report/`: CVPR-style LaTeX source and compiled final PDF.

Large assets are intentionally excluded from git: HM3D/MP3D scene datasets,
Habitat episode data, model checkpoints, generated videos, and full debug output
folders.

## Main Results

- Repaired ZSON Plan A, HM3D 200 episodes: 48.50% SR / 13.28% SPL.
- Repaired ZSON Plan A, MP3D 200 episodes: 51.00% SR / 13.69% SPL.
- VLFM Plan B, HM3D full validation: 52.30% SR / 30.34% SPL.
- ASCENT-Single DeepSeek 1000 episodes: 64.50% SR / 33.36% SPL.
- Fixed-200 Local-Qwen baseline vs DeepSeek selective hybrid: 130/200 to 132/200
  successes, with false positives reduced from 36 to 33.

## Running The Main Parts

### Part 1A: ZSON Plan A

```bash
cd zson
python -m unittest tests/test_plan_a_components.py

PYTHON_BIN=python RUN_NAME=plan_a_activegoal_hm3d200 \
  bash scripts/plan-a-eval-hm3d.sh \
  PLAN_A.MAX_EPISODES 200 TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 500

PYTHON_BIN=python RUN_NAME=plan_a_activegoal_mp3d200 \
  bash scripts/plan-a-eval-mp3d.sh \
  PLAN_A.MAX_EPISODES 200 TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 500
```

### Part 1B: VLFM Plan B

```bash
cd vlfm
pip install -e .
SPLIT=val_mini EPISODES=1 VIDEO_OPTION='[]' bash project/run_eval_pair.sh
bash scripts/eval_frontier_policy.sh
```

### Part 2 / Part 3: ASCENT And Selective Hybrid

```bash
cd ascent/ascent
python scripts/check_ascent_data.py --dataset hm3d
bash scripts/launch_vlm_servers_ascent.sh
```

Baseline and object-patch ablation branches are stored under `ascent/`. The final
fixed-200 selective-hybrid launch scripts and compact result artifacts are collected
under:

```text
ascent/scripts_final_selective_hybrid/
ascent/results_reference/
```

## Report

The final report is available at:

```text
report/main.pdf
```

To rebuild:

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## External Assets

The experiments assume a Linux CUDA environment with Habitat-Sim/Habitat-Lab,
HM3D and MP3D ObjectNav episodes/scenes, and the model weights required by ZSON,
VLFM, ASCENT, BLIP2, MobileSAM, D-FINE/GroundingDINO, Qwen2.5, and Qwen2.5-VL.
Paths in historical scripts may need to be adjusted for a new machine.
