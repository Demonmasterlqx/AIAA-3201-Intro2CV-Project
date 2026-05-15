# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any

import numpy as np
from frontier_exploration.measurements import FrontierExplorationMap
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode
from habitat.utils.visualizations import maps


def _safe_draw_target_bbox_mask(self: FrontierExplorationMap, episode: Any) -> None:
    """Fall back to an empty mask when semantic annotations are unavailable.

    Some local HM3D installs only contain the basis meshes and navmeshes required to run
    navigation, but not the semantic scene annotations used for drawing object AABBs.
    The frontier-exploration measurement only needs a bbox mask for post-hoc failure
    analysis, so it is safe to use an empty mask in that case.
    """

    bbox_mask = np.zeros_like(self._top_down_map)
    if not isinstance(episode, ObjectGoalNavEpisode):
        self._static_metrics["target_bboxes_mask"] = bbox_mask
        return

    try:
        sem_scene = self._sim.semantic_annotations()
    except Exception:
        self._static_metrics["target_bboxes_mask"] = bbox_mask
        return

    for goal in episode.goals:
        try:
            object_id = goal.object_id  # type: ignore[attr-defined]
            obj = sem_scene.objects[object_id]
            if int(obj.id.split("_")[-1]) != int(object_id):
                continue

            center = obj.aabb.center
            x_len, _, z_len = obj.aabb.sizes / 2.0
            corners = [
                center + np.array([x, 0, z])
                for x, z in [(-x_len, -z_len), (x_len, z_len)]
                if self._is_on_same_floor(center[1])
            ]
            if not corners:
                continue

            map_corners = [
                maps.to_grid(
                    point[2],
                    point[0],
                    (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                    sim=self._sim,
                )
                for point in corners
            ]
            (y1, x1), (y2, x2) = map_corners
            bbox_mask[y1:y2, x1:x2] = 1
        except Exception:
            continue

    self._static_metrics["target_bboxes_mask"] = bbox_mask


FrontierExplorationMap._draw_target_bbox_mask = _safe_draw_target_bbox_mask
