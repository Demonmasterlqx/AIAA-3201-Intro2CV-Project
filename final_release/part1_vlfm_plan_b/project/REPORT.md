# VLFM Project Report

## Exploration Efficiency Under Different Conditions

The project compares two VLFM exploration conditions on the same split:

- Plan A: dense value matching (`HabitatDenseValueMatchingPolicy`)
- Plan B: frontier-restricted semantic exploration (`HabitatITMPolicyV2`)

| Condition | Episodes | Success Rate | SPL | Soft SPL |
| --- | ---: | ---: | ---: | ---: |
| Plan A Dense | 1 | 0.00% | 0.00% | 7.42% |
| Plan B Frontier | 1 | 100.00% | 73.76% | 72.97% |

A higher SPL indicates more efficient exploration because success is discounted by path inefficiency.

## Failure and Occlusion Analysis

The occlusion limitation of VLFM comes from the policy design itself:

- The semantic value map is projected into a 2D top-down map, so vertical structure is discarded.
- A high VLM score can be triggered by room context, partial target visibility, or nearby co-occurring objects instead of a fully reachable target.
- Under severe occlusion, the projected score can still make a hidden or partially visible region look promising, even when the target is not yet directly observable from that frontier.
- Frontier restriction improves discipline, but it only constrains where the agent explores next. It does not fix incorrect semantic evidence.

Observed failure-cause counts in this run:

Plan A Dense:
- never_saw_target_did_not_travel_stairs_feasible: 1

Plan B Frontier:
- did_not_fail: 1

If false negatives or repeated non-detection failures appear, they are consistent with the expected limitation of relying on 2D projections and simple VLM similarity when the goal is heavily occluded.

## Visual Artifacts

- Per-episode comparison table: `artifacts/episode_comparison.csv`
- Representative dense trajectory video: `artifacts/videos/plan_a_dense_representative.mp4`
- Representative frontier trajectory video: `artifacts/videos/plan_b_frontier_representative.mp4`
- Dense top-down semantic map figure: `artifacts/maps/plan_a_dense_maps.png`
- Frontier top-down semantic map figure: `artifacts/maps/plan_b_frontier_maps.png`

The extracted map figures come from the right half of the saved trajectory frames, where VLFM renders the Habitat top-down map together with the obstacle map and semantic value map.
