from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import quaternion
import torch
import torchvision.transforms.functional as TF
from habitat.utils.visualizations import maps


@dataclass
class SemanticUpdate:
    center_depth: Optional[float]
    center_similarity: float
    projected_cells: int
    max_projected_similarity: float


def compute_resize_center_crop_geometry(
    raw_height: int,
    raw_width: int,
    output_size: int,
) -> Dict[str, float]:
    scale = float(output_size) / float(min(raw_height, raw_width))
    resized_height = int(round(raw_height * scale))
    resized_width = int(round(raw_width * scale))
    crop_y = max((resized_height - output_size) / 2.0, 0.0)
    crop_x = max((resized_width - output_size) / 2.0, 0.0)
    return {
        "scale": scale,
        "resized_height": float(resized_height),
        "resized_width": float(resized_width),
        "crop_y": crop_y,
        "crop_x": crop_x,
    }


def make_intrinsics(
    raw_height: int,
    raw_width: int,
    hfov_deg: float,
    output_size: int,
) -> Tuple[float, float, float, float]:
    geom = compute_resize_center_crop_geometry(raw_height, raw_width, output_size)
    hfov = np.deg2rad(hfov_deg)
    fx = (raw_width / 2.0) / np.tan(hfov / 2.0)
    fy = fx
    cx = raw_width / 2.0
    cy = raw_height / 2.0
    fx *= geom["scale"]
    fy *= geom["scale"]
    cx = cx * geom["scale"] - geom["crop_x"]
    cy = cy * geom["scale"] - geom["crop_y"]
    return float(fx), float(fy), float(cx), float(cy)


def resize_center_crop_depth(depth: np.ndarray, output_size: int) -> np.ndarray:
    tensor = torch.from_numpy(depth[None, None].astype(np.float32))
    tensor = TF.resize(
        tensor,
        output_size,
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    tensor = TF.center_crop(tensor, output_size=[output_size, output_size])
    return tensor[0, 0].numpy()


def build_free_explored_masks(topdown_info: Dict[str, np.ndarray]):
    map_data = topdown_info["map"]
    fog_mask = topdown_info["fog_of_war_mask"].astype(bool)
    free_mask = np.logical_not(
        np.isin(map_data, [maps.MAP_INVALID_POINT, maps.MAP_BORDER_INDICATOR])
    )
    explored_mask = free_mask & fog_mask
    obstacle_mask = np.logical_not(free_mask)
    return free_mask, explored_mask, obstacle_mask


def gaussian_smooth(values: np.ndarray, valid_mask: np.ndarray, kernel_size: int):
    if kernel_size <= 1:
        return np.where(valid_mask, values, -1.0)
    kernel = (kernel_size, kernel_size)
    values = values.astype(np.float32)
    valid = valid_mask.astype(np.float32)
    smoothed_values = cv2.GaussianBlur(values * valid, kernel, sigmaX=0)
    smoothed_valid = cv2.GaussianBlur(valid, kernel, sigmaX=0)
    output = np.full_like(values, -1.0, dtype=np.float32)
    keep = smoothed_valid > 1e-6
    output[keep] = smoothed_values[keep] / smoothed_valid[keep]
    output[~valid_mask] = -1.0
    return output


def frontier_mask(free_mask: np.ndarray, explored_mask: np.ndarray) -> np.ndarray:
    unknown_free = free_mask & np.logical_not(explored_mask)
    explored_free = free_mask & explored_mask
    neighbors = np.zeros_like(unknown_free, dtype=bool)
    neighbors[1:, :] |= explored_free[:-1, :]
    neighbors[:-1, :] |= explored_free[1:, :]
    neighbors[:, 1:] |= explored_free[:, :-1]
    neighbors[:, :-1] |= explored_free[:, 1:]
    return unknown_free & neighbors


class SemanticMap:
    def __init__(
        self,
        map_shape: Tuple[int, int],
        patch_grid: int,
        clip_input_size: int,
        min_depth_m: float,
        max_depth_m: float,
        hfov_deg: float,
        projection_stride: int,
    ):
        self.map_shape = tuple(map_shape)
        self.patch_grid = int(patch_grid)
        self.clip_input_size = int(clip_input_size)
        self.patch_size = self.clip_input_size // self.patch_grid
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.hfov_deg = float(hfov_deg)
        self.projection_stride = int(projection_stride)
        self.max_similarity = np.full(self.map_shape, -1.0, dtype=np.float32)
        self.mean_similarity = np.zeros(self.map_shape, dtype=np.float32)
        self.hit_count = np.zeros(self.map_shape, dtype=np.int32)

    def planning_map(self) -> np.ndarray:
        planning = np.full(self.map_shape, -1.0, dtype=np.float32)
        valid = self.hit_count > 0
        planning[valid] = self.mean_similarity[valid]
        return planning

    def update(
        self,
        rgb_shape: Tuple[int, int],
        depth_observation: np.ndarray,
        patch_features: np.ndarray,
        text_embedding: np.ndarray,
        sensor_state,
        topdown_info: Dict[str, np.ndarray],
        pathfinder,
    ) -> SemanticUpdate:
        depth_map = depth_observation.squeeze(-1).astype(np.float32)
        if depth_map.ndim != 2:
            raise ValueError("Depth observation must be HxW or HxWx1")
        depth_map = resize_center_crop_depth(depth_map, self.clip_input_size)
        fx, fy, cx, cy = make_intrinsics(
            raw_height=int(rgb_shape[0]),
            raw_width=int(rgb_shape[1]),
            hfov_deg=self.hfov_deg,
            output_size=self.clip_input_size,
        )

        patch_similarities = np.tensordot(
            patch_features.astype(np.float32),
            text_embedding.astype(np.float32),
            axes=([-1], [0]),
        )
        dense_similarity = cv2.resize(
            patch_similarities.astype(np.float32),
            (self.clip_input_size, self.clip_input_size),
            interpolation=cv2.INTER_CUBIC,
        )

        rotation = quaternion.as_rotation_matrix(sensor_state.rotation)
        position = np.asarray(sensor_state.position, dtype=np.float32)
        free_mask, _, _ = build_free_explored_masks(topdown_info)

        projected_cells = 0
        max_projected_similarity = -1.0
        stride = max(1, self.projection_stride)
        for y0 in range(0, depth_map.shape[0], stride):
            for x0 in range(0, depth_map.shape[1], stride):
                y1 = min(y0 + stride, depth_map.shape[0])
                x1 = min(x0 + stride, depth_map.shape[1])
                patch_depth = depth_map[y0:y1, x0:x1]
                valid_depth = patch_depth[
                    (patch_depth >= self.min_depth_m)
                    & (patch_depth <= self.max_depth_m)
                ]
                if valid_depth.size == 0:
                    continue

                depth_value = float(np.median(valid_depth))
                pixel_x = (x0 + x1) / 2.0
                pixel_y = (y0 + y1) / 2.0
                camera_point = np.array(
                    [
                        (pixel_x - cx) / fx * depth_value,
                        -(pixel_y - cy) / fy * depth_value,
                        -depth_value,
                    ],
                    dtype=np.float32,
                )
                world_point = position + rotation.dot(camera_point)
                grid_x, grid_y = maps.to_grid(
                    world_point[2],
                    world_point[0],
                    self.map_shape,
                    pathfinder=pathfinder,
                )
                if (
                    grid_x < 0
                    or grid_x >= self.map_shape[0]
                    or grid_y < 0
                    or grid_y >= self.map_shape[1]
                ):
                    continue
                if not free_mask[grid_x, grid_y]:
                    continue

                similarity = float(np.mean(dense_similarity[y0:y1, x0:x1]))
                projected_cells += 1
                max_projected_similarity = max(max_projected_similarity, similarity)
                self.max_similarity[grid_x, grid_y] = max(
                    self.max_similarity[grid_x, grid_y], similarity
                )
                count = int(self.hit_count[grid_x, grid_y])
                if count == 0:
                    self.mean_similarity[grid_x, grid_y] = similarity
                else:
                    mean = self.mean_similarity[grid_x, grid_y]
                    self.mean_similarity[grid_x, grid_y] = (
                        mean * count + similarity
                    ) / float(count + 1)
                self.hit_count[grid_x, grid_y] = count + 1

        center_row = self.patch_grid // 2
        center_col = self.patch_grid // 2
        cy0 = center_row * self.patch_size
        cy1 = min((center_row + 1) * self.patch_size, depth_map.shape[0])
        cx0 = center_col * self.patch_size
        cx1 = min((center_col + 1) * self.patch_size, depth_map.shape[1])
        center_patch = depth_map[cy0:cy1, cx0:cx1]
        center_valid = center_patch[
            (center_patch >= self.min_depth_m) & (center_patch <= self.max_depth_m)
        ]
        center_depth = None
        if center_valid.size > 0:
            center_depth = float(np.median(center_valid))

        return SemanticUpdate(
            center_depth=center_depth,
            center_similarity=float(patch_similarities[center_row, center_col]),
            projected_cells=projected_cells,
            max_projected_similarity=max_projected_similarity,
        )
