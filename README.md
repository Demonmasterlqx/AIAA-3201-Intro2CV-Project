# AIAA 3201 Intro2CV Project: Visual Navigation in 3D Environments

This repository contains the code and report artifacts for the AIAA 3201 project
**From Semantic Frontiers to Selective Vision-Language Verification for Zero-Shot Object Navigation**.

## Where to Start

- `final_release/`: cleaned submission code with runnable entry points and a top-level README.
- `report/`: CVPR-style LaTeX report source and the compiled camera-ready PDF.
- `zson/`, `vlfm/`, `ascent/`: source snapshots kept from the working project history.

For grading and reproduction, start from `final_release/README.md`. It explains the
three-part project structure, required external assets, and the exact commands used
for the reported runs.

## Project Structure

`final_release/` is organized as:

- `part1_zson_plan_a/`: repaired Plan A dense semantic-map ObjectNav branch.
- `part1_vlfm_plan_b/`: VLFM-style Plan B frontier-scoring branch.
- `part2_ascent_baseline/`: ASCENT-style LLM-driven ObjectNav baseline reproduction.
- `part3_selective_hybrid/`: selective Qwen-VL object-patch verification branch.

Large assets are intentionally not stored in git: HM3D/MP3D scenes, Habitat episode
datasets, model checkpoints, generated videos, and full debug output folders.

## Key Results

- Plan A repaired HM3D 200ep: 48.50% SR, 13.28% SPL.
- Plan A repaired MP3D 200ep: 51.00% SR, 13.69% SPL.
- VLFM Plan B HM3D full val: 52.30% SR, 30.34% SPL.
- ASCENT-Single DeepSeek 1000ep baseline: 64.50% SR, 33.36% SPL.
- Fixed-200 Local-Qwen baseline vs DeepSeek selective hybrid: 130/200 to 132/200 successes.

## Report

The final report PDF is:

```text
report/main.pdf
```

The report source can be rebuilt with:

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Canvas Artifacts

The required trajectory package is submitted separately to Canvas as
`navigation_results.zip`. It contains selected top-down, egocentric, and composite
videos plus lightweight summary artifacts. It is not committed to git because it is
a generated media package rather than source code.
