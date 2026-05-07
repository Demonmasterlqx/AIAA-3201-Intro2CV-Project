# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from frontier_exploration.utils.general_utils import xyz_to_habitat
from habitat.utils.common import flatten_dict
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.maps import MAP_TARGET_POINT_INDICATOR
from habitat.utils.visualizations.utils import overlay_text_to_image
from habitat_baselines.common.tensor_dict import TensorDict

from vlfm.utils.geometry_utils import transform_points
from vlfm.utils.img_utils import (
    reorient_rescale_map,
    resize_image,
    resize_images,
    rotate_image,
)
from vlfm.utils.visualization import add_text_to_image, pad_images


class HabitatVis:
    def __init__(self) -> None:
        self.rgb: List[np.ndarray] = []
        self.depth: List[np.ndarray] = []
        self.maps: List[np.ndarray] = []
        self.vis_maps: List[List[np.ndarray]] = []
        self.texts: List[List[str]] = []
        self.using_vis_maps = False
        self.using_annotated_rgb = False
        self.using_annotated_depth = False

    def reset(self) -> None:
        self.rgb = []
        self.depth = []
        self.maps = []
        self.vis_maps = []
        self.texts = []
        self.using_annotated_rgb = False
        self.using_annotated_depth = False

    def collect_data(
        self,
        observations: TensorDict,
        infos: List[Dict[str, Any]],
        policy_info: List[Dict[str, Any]],
    ) -> None:
        assert len(infos) == 1, "Only support one environment for now"

        if "annotated_depth" in policy_info[0]:
            depth = policy_info[0]["annotated_depth"]
            self.using_annotated_depth = True
        else:
            depth = (observations["depth"][0].cpu().numpy() * 255.0).astype(np.uint8)
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        depth = overlay_frame(depth, infos[0])
        self.depth.append(depth)

        if "annotated_rgb" in policy_info[0]:
            rgb = policy_info[0]["annotated_rgb"]
            self.using_annotated_rgb = True
        else:
            rgb = observations["rgb"][0].cpu().numpy()
        self.rgb.append(rgb)

        # Visualize target point cloud on the map
        color_point_cloud_on_map(infos, policy_info)

        map = maps.colorize_draw_agent_and_fit_to_height(infos[0]["top_down_map"], self.depth[0].shape[0])
        self.maps.append(map)
        vis_map_imgs = [
            self._reorient_rescale_habitat_map(infos, policy_info[0][vkey])
            for vkey in ["obstacle_map", "value_map"]
            if vkey in policy_info[0]
        ]
        if vis_map_imgs:
            self.using_vis_maps = True
            self.vis_maps.append(vis_map_imgs)
        text = [
            policy_info[0][text_key]
            for text_key in policy_info[0].get("render_below_images", [])
            if text_key in policy_info[0]
        ]
        self.texts.append(text)

    def flush_frame_sets(self, failure_cause: str) -> Dict[str, List[np.ndarray]]:
        """Flush all frames and return the supported video views."""
        # Because the annotated frames are actually one step delayed, pop the first one
        # and add a placeholder frame to the end (gets removed anyway)
        if self.using_annotated_rgb:
            self.rgb.append(self.rgb.pop(0))
        if self.using_annotated_depth:
            self.depth.append(self.depth.pop(0))
        if self.using_vis_maps:  # Cost maps are also one step delayed
            self.vis_maps.append(self.vis_maps.pop(0))

        frame_sets = {
            "composite": [],
            "egocentric": [],
            "topdown": [],
        }
        num_frames = len(self.depth) - 1  # last frame is from next episode, remove it
        for i in range(num_frames):
            composite = self._create_frame(
                self.depth[i],
                self.rgb[i],
                self.maps[i],
                self.vis_maps[i],
                self.texts[i],
            )
            egocentric = self._create_egocentric_frame(self.rgb[i], self.texts[i])
            topdown = self._create_topdown_frame(self.maps[i], self.vis_maps[i], self.texts[i])
            for key, frame in (
                ("composite", composite),
                ("egocentric", egocentric),
                ("topdown", topdown),
            ):
                frame = add_text_to_image(frame, f"step: {i}", top=True)
                frame = add_text_to_image(frame, "Failure cause: " + failure_cause, top=True)
                frame_sets[key].append(frame)

        for key, frames in frame_sets.items():
            if len(frames) == 0:
                continue
            frames = pad_images(frames, pad_from_top=True)
            target_width = 480 * 2 if key == "composite" else 640
            frame_sets[key] = [resize_image(frame, target_width) for frame in frames]

        self.reset()

        return frame_sets

    def flush_frames(self, failure_cause: str) -> List[np.ndarray]:
        """Backward-compatible alias returning only composite frames."""
        frame_sets = self.flush_frame_sets(failure_cause)
        frames = frame_sets["composite"]
        self.reset()

        return frames

    @staticmethod
    def _reorient_rescale_habitat_map(infos: List[Dict[str, Any]], vis_map: np.ndarray) -> np.ndarray:
        # Rotate the cost map to match the agent's orientation at the start
        # of the episode
        start_yaw = infos[0]["start_yaw"]
        if start_yaw != 0.0:
            vis_map = rotate_image(vis_map, start_yaw, border_value=(255, 255, 255))

        # Rotate the image 90 degrees if the corresponding map is taller than it is wide
        habitat_map = infos[0]["top_down_map"]["map"]
        if habitat_map.shape[0] > habitat_map.shape[1]:
            vis_map = np.rot90(vis_map, 1)

        vis_map = reorient_rescale_map(vis_map)

        return vis_map

    @staticmethod
    def _create_frame(
        depth: np.ndarray,
        rgb: np.ndarray,
        map: np.ndarray,
        vis_map_imgs: List[np.ndarray],
        text: List[str],
    ) -> np.ndarray:
        """Create a frame using all the given images.

        First, the depth and rgb images are stacked vertically. Then, all the maps are
        combined as a separate images. Then these two images should be stitched together
        horizontally (depth-rgb on the left, maps on the right).

        The combined map image contains two rows of images and at least one column.
        First, the 'map' argument is at the top left, then the first element of the
        'vis_map_imgs' argument is at the bottom left. If there are more than one
        element in 'vis_map_imgs', then the second element is at the top right, the
        third element is at the bottom right, and so on.

        Args:
            depth: The depth image (H, W, 3).
            rgb: The rgb image (H, W, 3).
            map: The map image, a 3-channel rgb image, but can have different shape from
                depth and rgb.
            vis_map_imgs: A list of other map images. Each are 3-channel rgb images, but
                can have different sizes.
            text: A list of strings to be rendered above the images.
        Returns:
            np.ndarray: The combined frame image.
        """
        # Stack depth and rgb images vertically
        depth_rgb = np.vstack((depth, rgb))

        # Prepare the list of images to be combined
        map_imgs = [map] + vis_map_imgs
        if len(map_imgs) % 2 == 1:
            # If there are odd number of images, add a placeholder image
            map_imgs.append(np.ones_like(map_imgs[-1]) * 255)

        even_index_imgs = map_imgs[::2]
        odd_index_imgs = map_imgs[1::2]
        top_row = np.hstack(resize_images(even_index_imgs, match_dimension="height"))
        bottom_row = np.hstack(resize_images(odd_index_imgs, match_dimension="height"))

        frame = np.vstack(resize_images([top_row, bottom_row], match_dimension="width"))
        depth_rgb, frame = resize_images([depth_rgb, frame], match_dimension="height")
        frame = np.hstack((depth_rgb, frame))

        # Add text to the top of the frame
        for t in text[::-1]:
            frame = add_text_to_image(frame, t, top=True)

        return frame

    @staticmethod
    def _create_egocentric_frame(rgb: np.ndarray, text: List[str]) -> np.ndarray:
        frame = rgb.copy()
        for value in text[::-1]:
            frame = add_text_to_image(frame, value, top=True)
        return frame

    @staticmethod
    def _create_topdown_frame(map: np.ndarray, vis_map_imgs: List[np.ndarray], text: List[str]) -> np.ndarray:
        if vis_map_imgs:
            topdown = HabitatVis._stack_map_images([map] + vis_map_imgs)
        else:
            topdown = map.copy()
        for value in text[::-1]:
            topdown = add_text_to_image(topdown, value, top=True)
        return topdown

    @staticmethod
    def _stack_map_images(map_imgs: List[np.ndarray]) -> np.ndarray:
        if len(map_imgs) % 2 == 1:
            map_imgs.append(np.ones_like(map_imgs[-1]) * 255)
        even_index_imgs = map_imgs[::2]
        odd_index_imgs = map_imgs[1::2]
        top_row = np.hstack(resize_images(even_index_imgs, match_dimension="height"))
        bottom_row = np.hstack(resize_images(odd_index_imgs, match_dimension="height"))
        return np.vstack(resize_images([top_row, bottom_row], match_dimension="width"))


def sim_xy_to_grid_xy(
    upper_bound: Tuple[int, int],
    lower_bound: Tuple[int, int],
    grid_resolution: Tuple[int, int],
    sim_xy: np.ndarray,
    remove_duplicates: bool = True,
) -> np.ndarray:
    """Converts simulation coordinates to grid coordinates.

    Args:
        upper_bound (Tuple[int, int]): The upper bound of the grid.
        lower_bound (Tuple[int, int]): The lower bound of the grid.
        grid_resolution (Tuple[int, int]): The resolution of the grid.
        sim_xy (np.ndarray): A numpy array of 2D simulation coordinates.
        remove_duplicates (bool): Whether to remove duplicate grid coordinates.

    Returns:
        np.ndarray: A numpy array of 2D grid coordinates.
    """
    grid_size = np.array(
        [
            abs(upper_bound[1] - lower_bound[1]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        ]
    )
    grid_xy = ((sim_xy - lower_bound[::-1]) / grid_size).astype(int)

    if remove_duplicates:
        grid_xy = np.unique(grid_xy, axis=0)

    return grid_xy


def grid_xy_to_sim_xy(
    upper_bound: Tuple[int, int],
    lower_bound: Tuple[int, int],
    grid_resolution: Tuple[int, int],
    grid_xy: np.ndarray,
) -> np.ndarray:
    grid_xy = np.asarray(grid_xy, dtype=np.float32)
    grid_size = np.array(
        [
            abs(upper_bound[1] - lower_bound[1]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        ],
        dtype=np.float32,
    )
    return grid_xy * grid_size + lower_bound[::-1]


def color_point_cloud_on_map(infos: List[Dict[str, Any]], policy_info: List[Dict[str, Any]]) -> None:
    if len(policy_info[0]["target_point_cloud"]) == 0:
        return

    upper_bound = infos[0]["top_down_map"]["upper_bound"]
    lower_bound = infos[0]["top_down_map"]["lower_bound"]
    grid_resolution = infos[0]["top_down_map"]["grid_resolution"]
    tf_episodic_to_global = infos[0]["top_down_map"]["tf_episodic_to_global"]

    cloud_episodic_frame = policy_info[0]["target_point_cloud"][:, :3]
    cloud_global_frame_xyz = transform_points(tf_episodic_to_global, cloud_episodic_frame)
    cloud_global_frame_habitat = xyz_to_habitat(cloud_global_frame_xyz)
    cloud_global_frame_habitat_xy = cloud_global_frame_habitat[:, [2, 0]]

    grid_xy = sim_xy_to_grid_xy(
        upper_bound,
        lower_bound,
        grid_resolution,
        cloud_global_frame_habitat_xy,
        remove_duplicates=True,
    )

    new_map = infos[0]["top_down_map"]["map"].copy()
    valid_mask = (
        (grid_xy[:, 0] >= 0)
        & (grid_xy[:, 0] < new_map.shape[0])
        & (grid_xy[:, 1] >= 0)
        & (grid_xy[:, 1] < new_map.shape[1])
    )
    grid_xy = grid_xy[valid_mask]
    if len(grid_xy) == 0:
        return
    new_map[grid_xy[:, 0], grid_xy[:, 1]] = MAP_TARGET_POINT_INDICATOR

    infos[0]["top_down_map"]["map"] = new_map


def overlay_frame(frame: np.ndarray, info: Dict[str, Any], additional: Optional[List[str]] = None) -> np.ndarray:
    """
    Renders text from the `info` dictionary to the `frame` image.
    """

    lines = []
    flattened_info = flatten_dict(info)
    for k, v in flattened_info.items():
        if isinstance(v, str):
            lines.append(f"{k}: {v}")
        else:
            try:
                lines.append(f"{k}: {v:.2f}")
            except TypeError:
                pass
    if additional is not None:
        lines.extend(additional)

    frame = overlay_text_to_image(frame, lines, font_size=0.25)

    return frame
