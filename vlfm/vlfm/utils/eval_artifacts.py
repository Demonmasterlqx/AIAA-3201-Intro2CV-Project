from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import cv2
import numpy as np


DEFAULT_VIDEO_VIEWS = ("composite", "egocentric", "topdown")
VIDEO_FILENAMES = {
    "composite": "composite.mp4",
    "egocentric": "egocentric.mp4",
    "topdown": "topdown.mp4",
}


def scene_id_to_short(scene_id_full: str) -> str:
    scene_name = Path(scene_id_full).stem
    return scene_name.removesuffix(".basis")


def episode_dir_name(scene_short: str, episode_id: str) -> str:
    return f"{scene_short}__ep_{episode_id}"


def get_episode_dir(episodes_root: Path, scene_short: str, episode_id: str) -> Path:
    return episodes_root / episode_dir_name(scene_short, episode_id)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Mapping) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2) + "\n")


def normalize_video_views(video_views: Iterable[str]) -> List[str]:
    normalized = []
    for view in video_views:
        stripped = view.strip().lower()
        if not stripped:
            continue
        if stripped not in VIDEO_FILENAMES:
            raise ValueError(f"Unknown video view '{view}'. Expected one of {sorted(VIDEO_FILENAMES)}.")
        if stripped not in normalized:
            normalized.append(stripped)
    return normalized


def parse_video_views(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(DEFAULT_VIDEO_VIEWS)
    return normalize_video_views(value.split(","))


def save_episode_videos(
    episode_dir: Path,
    frame_sets: Mapping[str, List[np.ndarray]],
    fps: int,
    requested_views: Iterable[str],
    run_root: Path,
) -> Dict[str, str]:
    saved_paths: Dict[str, str] = {}
    episode_dir = ensure_dir(episode_dir)
    for view in normalize_video_views(requested_views):
        frames = frame_sets.get(view, [])
        if not frames:
            continue
        video_path = episode_dir / VIDEO_FILENAMES[view]
        write_video(video_path, frames, fps=fps)
        saved_paths[view] = os.path.relpath(video_path, run_root)
    return saved_paths


def write_video(path: Path, frames: List[np.ndarray], fps: int) -> None:
    if len(frames) == 0:
        return

    normalized_frames = [_normalize_frame(frame) for frame in frames]
    height, width = normalized_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")

    try:
        for frame in normalized_frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    height, width = frame.shape[:2]
    even_height = height - (height % 2)
    even_width = width - (width % 2)
    if even_height != height or even_width != width:
        frame = frame[:even_height, :even_width]
    return frame
