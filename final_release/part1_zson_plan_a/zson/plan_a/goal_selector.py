from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from zson.plan_a.semantic_map import frontier_mask


@dataclass
class GoalCandidate:
    cell: Tuple[int, int]
    score: float
    strategy: str


def select_semantic_candidates(
    planning_map: np.ndarray,
    hit_count: np.ndarray,
    free_mask: np.ndarray,
    explored_mask: np.ndarray,
    suppression_mask: np.ndarray,
    threshold: float,
    top_k: int,
) -> List[GoalCandidate]:
    valid = (
        (hit_count > 0)
        & free_mask
        & explored_mask
        & (suppression_mask <= 0)
        & (planning_map >= threshold)
    )
    if not np.any(valid):
        return []

    flat_scores = planning_map.reshape(-1)
    flat_valid = valid.reshape(-1)
    valid_indices = np.flatnonzero(flat_valid)
    top_k = min(int(top_k), int(valid_indices.size))
    selected = valid_indices[np.argpartition(flat_scores[valid_indices], -top_k)[-top_k:]]
    selected = selected[np.argsort(flat_scores[selected])[::-1]]
    return [
        GoalCandidate(
            cell=tuple(np.unravel_index(int(idx), planning_map.shape)),
            score=float(flat_scores[idx]),
            strategy="semantic",
        )
        for idx in selected
    ]


def select_frontier_candidates(
    free_mask: np.ndarray,
    explored_mask: np.ndarray,
    suppression_mask: np.ndarray,
    agent_cell: Tuple[int, int],
    limit: int,
    unknown_radius: int,
    distance_penalty: float,
) -> List[GoalCandidate]:
    frontier = frontier_mask(free_mask, explored_mask) & (suppression_mask <= 0)
    coords = np.argwhere(frontier)
    if coords.size == 0:
        return []

    unknown_free = free_mask & np.logical_not(explored_mask)
    kernel = 2 * int(unknown_radius) + 1
    local_unknown = np.zeros_like(unknown_free, dtype=np.float32)
    if kernel > 1:
        local_unknown = (
            np.pad(unknown_free.astype(np.float32), int(unknown_radius), mode="constant")
        )
        integral = np.cumsum(np.cumsum(local_unknown, axis=0), axis=1)
        h, w = unknown_free.shape
        for r in range(h):
            r0 = r
            r1 = r + kernel
            for c in range(w):
                c0 = c
                c1 = c + kernel
                total = integral[r1 - 1, c1 - 1]
                if r0 > 0:
                    total -= integral[r0 - 1, c1 - 1]
                if c0 > 0:
                    total -= integral[r1 - 1, c0 - 1]
                if r0 > 0 and c0 > 0:
                    total += integral[r0 - 1, c0 - 1]
                local_unknown[r, c] = total

    distances = np.linalg.norm(coords - np.asarray(agent_cell)[None], axis=1)
    frontier_gain = local_unknown[coords[:, 0], coords[:, 1]]
    scores = frontier_gain - float(distance_penalty) * distances
    order = np.argsort(scores)[::-1][:limit]
    return [
        GoalCandidate(
            cell=(int(coords[idx, 0]), int(coords[idx, 1])),
            score=float(scores[idx]),
            strategy="frontier",
        )
        for idx in order
    ]
