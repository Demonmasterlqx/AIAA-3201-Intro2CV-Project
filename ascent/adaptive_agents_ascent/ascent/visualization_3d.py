from __future__ import annotations

from io import BytesIO
from typing import Optional

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


MAIN_COLOR = "#12B5CB"
SUB_COLOR = "#FF8A3D"
TARGET_COLOR = "#38A169"
STAIR_COLOR = "#7C5CE6"
OTHER_COLOR = "#A0AEC0"


def render_pointcloud_panel(
    *,
    main_traj: np.ndarray,
    sub_traj: np.ndarray,
    target_cloud: np.ndarray,
    stair_cloud: np.ndarray,
    other_cloud: np.ndarray,
    main_current: np.ndarray,
    sub_current: np.ndarray,
    width: int = 960,
    height: int = 720,
    max_points_per_cloud: int = 1800,
) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.view_init(elev=35, azim=-60)
    ax.grid(False)

    traj_main_xyz = _traj_to_xyz(main_traj)
    traj_sub_xyz = _traj_to_xyz(sub_traj)
    target_xyz = _cloud_to_plot_xyz(target_cloud)
    stair_xyz = _cloud_to_plot_xyz(stair_cloud)
    other_xyz = _cloud_to_plot_xyz(other_cloud)
    main_current_xyz = _single_to_xyz(main_current)
    sub_current_xyz = _single_to_xyz(sub_current)

    plotted = False
    if traj_main_xyz.size > 0:
        ax.plot(
            traj_main_xyz[:, 0],
            traj_main_xyz[:, 1],
            traj_main_xyz[:, 2],
            color=MAIN_COLOR,
            linewidth=3.0,
            label="Main path",
        )
        plotted = True
    if traj_sub_xyz.size > 0:
        ax.plot(
            traj_sub_xyz[:, 0],
            traj_sub_xyz[:, 1],
            traj_sub_xyz[:, 2],
            color=SUB_COLOR,
            linewidth=2.4,
            linestyle="--",
            label="Sub path",
        )
        plotted = True

    other_xyz = _downsample(other_xyz, max_points_per_cloud)
    target_xyz = _downsample(target_xyz, max_points_per_cloud)
    stair_xyz = _downsample(stair_xyz, max_points_per_cloud // 2)

    if other_xyz.size > 0:
        ax.scatter(
            other_xyz[:, 0],
            other_xyz[:, 1],
            other_xyz[:, 2],
            s=3,
            c=OTHER_COLOR,
            alpha=0.18,
            depthshade=False,
            label="Observed cloud",
        )
        plotted = True
    if stair_xyz.size > 0:
        ax.scatter(
            stair_xyz[:, 0],
            stair_xyz[:, 1],
            stair_xyz[:, 2],
            s=16,
            c=STAIR_COLOR,
            alpha=0.9,
            depthshade=False,
            label="Stair landmarks",
        )
        plotted = True
    if target_xyz.size > 0:
        ax.scatter(
            target_xyz[:, 0],
            target_xyz[:, 1],
            target_xyz[:, 2],
            s=10,
            c=TARGET_COLOR,
            alpha=0.8,
            depthshade=False,
            label="Target cloud",
        )
        plotted = True

    if main_current_xyz is not None:
        ax.scatter(
            [main_current_xyz[0]],
            [main_current_xyz[1]],
            [main_current_xyz[2]],
            s=160,
            c=MAIN_COLOR,
            edgecolors="black",
            linewidths=1.0,
            depthshade=False,
            label="Main current",
        )
        plotted = True
    if sub_current_xyz is not None:
        ax.scatter(
            [sub_current_xyz[0]],
            [sub_current_xyz[1]],
            [sub_current_xyz[2]],
            s=150,
            c=SUB_COLOR,
            edgecolors="black",
            linewidths=1.0,
            depthshade=False,
            label="Sub current",
        )
        plotted = True

    if plotted:
        limits = _compute_axis_limits(
            [
                traj_main_xyz,
                traj_sub_xyz,
                target_xyz,
                stair_xyz,
                other_xyz,
                _optional_to_array(main_current_xyz),
                _optional_to_array(sub_current_xyz),
            ]
        )
        _apply_equal_limits(ax, limits)
    else:
        ax.text2D(0.18, 0.52, "3D map pending\nNo point cloud yet", transform=ax.transAxes, fontsize=18)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y / floor")
    ax.set_title("Point-Cloud Map", fontsize=16, pad=10)
    if plotted:
        ax.legend(loc="upper left", fontsize=8, frameon=True)

    fig.tight_layout(pad=0.6)
    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image = cv2.imdecode(np.frombuffer(buf.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return np.full((height, width, 3), 255, dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _traj_to_xyz(points: np.ndarray) -> np.ndarray:
    if points is None:
        return np.empty((0, 3), dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if points.ndim == 1:
        points = points[None, :]
    if points.shape[1] >= 3:
        return np.stack((points[:, 0], points[:, 2], points[:, 1]), axis=1)
    if points.shape[1] == 2:
        return np.stack((points[:, 0], points[:, 1], np.zeros(points.shape[0], dtype=np.float32)), axis=1)
    return np.empty((0, 3), dtype=np.float32)


def _cloud_to_plot_xyz(points: np.ndarray) -> np.ndarray:
    if points is None:
        return np.empty((0, 3), dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if points.ndim == 1:
        points = points[None, :]
    if points.shape[1] < 3:
        return _traj_to_xyz(points)
    return np.stack((points[:, 0], points[:, 2], points[:, 1]), axis=1)


def _single_to_xyz(point: np.ndarray) -> Optional[np.ndarray]:
    point = np.asarray(point, dtype=np.float32)
    if point.size == 0:
        return None
    if point.ndim > 1:
        point = point.reshape(-1)
    if point.shape[0] >= 3:
        return np.array([point[0], point[2], point[1]], dtype=np.float32)
    if point.shape[0] == 2:
        return np.array([point[0], point[1], 0.0], dtype=np.float32)
    return None


def _optional_to_array(point: Optional[np.ndarray]) -> np.ndarray:
    if point is None:
        return np.empty((0, 3), dtype=np.float32)
    return point[None, :]


def _downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.size == 0 or len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step][:max_points]


def _compute_axis_limits(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    valid = [arr for arr in arrays if arr.size > 0]
    stacked = np.concatenate(valid, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    margin = np.maximum((maxs - mins) * 0.08, 0.4)
    return mins - margin, maxs + margin


def _apply_equal_limits(ax, limits: tuple[np.ndarray, np.ndarray]) -> None:
    mins, maxs = limits
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
