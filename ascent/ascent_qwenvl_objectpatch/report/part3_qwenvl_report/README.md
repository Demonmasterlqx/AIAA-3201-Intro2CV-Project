# Part 3 Qwen-VL Report Package

This folder contains the Part 3 report assets for the ASCENT Qwen-VL object-patch final confirmation experiment.

Files:
- `part3_report.md`: editable narrative section with tables and Mermaid diagrams.
- `part3_section.tex`: CVPR-style LaTeX section for integration into the final report.
- `metrics_summary.csv` and `metrics_summary.json`: frozen metrics parsed from source `run.log` files.
- `failure_taxonomy.csv`, `episode_failures.csv`, `scene_success_rates.csv`: traceability and failure-analysis tables.
- `figures/*.mmd`: Mermaid diagrams for the final confirmation pipeline and full-frame/object-patch comparison.
- `freeze_part3_metrics.py`: reproducible parser used to regenerate the frozen metrics.

Current headline: on the matched 20-episode subset, Qwen-VL object-patch final confirmation improves SR over ASCENT-Single from 80.0% to 85.0%, but SPL is lower (36.62% vs 43.50%).

Current-commit rerun: commit `08ada94` was evaluated on the mandatory matched 8-episode gate in `qwenvl_objectpatch_current08ada94_8eps_v1`. It reached 75.0% SR / 46.88% SPL and introduced one false-positive failure, so it is recorded as a rejected ablation and was not expanded to 20 episodes.
