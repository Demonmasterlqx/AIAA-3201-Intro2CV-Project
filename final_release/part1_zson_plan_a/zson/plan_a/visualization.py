import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio
import numpy as np
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps

from zson.plan_a.semantic_map import build_free_explored_masks

cv2 = try_cv2_import()


def annotate_ego(rgb: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    frame = rgb.copy()
    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return frame


def _fit_to_height(image: np.ndarray, output_height: int) -> np.ndarray:
    old_h, old_w = image.shape[:2]
    output_width = max(1, int(float(output_height) / float(old_h) * float(old_w)))
    return cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_CUBIC)


def render_topdown_frame(
    topdown_info: Dict[str, np.ndarray],
    semantic_map,
    trajectory: Iterable[Tuple[int, int]],
    current_goal_cell: Optional[Tuple[int, int]],
    output_height: int,
) -> np.ndarray:
    map_data = topdown_info["map"].copy()
    fog_mask = topdown_info["fog_of_war_mask"]
    canvas = maps.colorize_topdown_map(map_data, fog_mask)

    free_mask, explored_mask, _ = build_free_explored_masks(topdown_info)
    planning_map = semantic_map.planning_map()
    valid = (semantic_map.hit_count > 0) & free_mask
    if np.any(valid):
        overlay_values = np.zeros_like(planning_map, dtype=np.float32)
        clipped = np.clip((planning_map + 1.0) / 2.0, 0.0, 1.0)
        overlay_values[valid] = clipped[valid]
        heatmap = (overlay_values * 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        blend_mask = valid & explored_mask
        if np.any(blend_mask):
            blended = (
                canvas[blend_mask].astype(np.float32) * 0.45
                + heatmap[blend_mask].astype(np.float32) * 0.55
            )
            canvas[blend_mask] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    valid_trajectory = [cell for cell in trajectory if cell[0] >= 0 and cell[1] >= 0]
    if valid_trajectory:
        points = np.array([[cell[1], cell[0]] for cell in valid_trajectory], dtype=np.int32)
        if points.shape[0] > 1:
            cv2.polylines(
                canvas,
                [points.reshape(-1, 1, 2)],
                isClosed=False,
                color=(255, 255, 0),
                thickness=2,
            )
        start = tuple(points[0])
        end = tuple(points[-1])
        cv2.circle(canvas, start, 6, (0, 0, 255), -1)
        cv2.circle(canvas, end, 6, (255, 0, 0), -1)

    if current_goal_cell is not None:
        cv2.circle(
            canvas,
            (int(current_goal_cell[1]), int(current_goal_cell[0])),
            8,
            (0, 255, 255),
            2,
        )

    canvas = maps.draw_agent(
        image=canvas,
        agent_center_coord=topdown_info["agent_map_coord"],
        agent_rotation=topdown_info["agent_angle"],
        agent_radius_px=min(canvas.shape[:2]) // 32,
    )
    if canvas.shape[0] > canvas.shape[1]:
        canvas = np.rot90(canvas, 1)
    return _fit_to_height(canvas, output_height)


def compose_frame(ego_frame: np.ndarray, topdown_frame: np.ndarray) -> np.ndarray:
    if ego_frame.shape[0] != topdown_frame.shape[0]:
        topdown_frame = _fit_to_height(topdown_frame, ego_frame.shape[0])
    return np.concatenate([ego_frame, topdown_frame], axis=1)


def write_video(frames: List[np.ndarray], output_path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, quality=5)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
