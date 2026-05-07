from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

from ascent.as_ascent_structures import FrontierRecord
from ascent.as_ascent_structures import FrontierStatus
from ascent.as_ascent_structures import UtilityBreakdown


@dataclass
class FrontierCluster:
    cluster_id: str
    floor_id: int
    members: List[str]
    centroid: Tuple[float, float]
    extent: float


def cluster_frontiers(
    frontiers: Iterable[FrontierRecord], separation_threshold: float = 2.0
) -> List[FrontierCluster]:
    frontier_list = list(frontiers)
    clusters: List[FrontierCluster] = []
    for frontier in frontier_list:
        placed = False
        for cluster in clusters:
            if cluster.floor_id != frontier.floor_id:
                continue
            if _l2(cluster.centroid, frontier.centroid) <= separation_threshold:
                cluster.members.append(frontier.frontier_id)
                cluster.centroid = _mean_centroid([frontier.centroid] + [
                    frontier_lookup.centroid
                    for frontier_lookup in frontier_list
                    if frontier_lookup.frontier_id in cluster.members[:-1]
                ])
                cluster.extent = max(cluster.extent, _l2(cluster.centroid, frontier.centroid))
                placed = True
                break
        if not placed:
            clusters.append(
                FrontierCluster(
                    cluster_id=f"cluster_{len(clusters):03d}",
                    floor_id=frontier.floor_id,
                    members=[frontier.frontier_id],
                    centroid=frontier.centroid,
                    extent=max(frontier.cluster_extent, 0.1),
                )
            )
    return clusters


def should_trigger_fine_grained(
    frontiers: Sequence[FrontierRecord],
    main_position: Sequence[float],
    distance_threshold: float,
    min_frontiers: int,
    min_cluster_count: int,
    top2_gap_threshold: float,
) -> Tuple[bool, List[FrontierCluster]]:
    viable = [
        frontier
        for frontier in frontiers
        if frontier.status in (FrontierStatus.FREE, FrontierStatus.RESERVED, FrontierStatus.EXPLORING)
    ]
    if len(viable) < min_frontiers:
        return False, []
    min_distance = min(_l2(frontier.centroid, main_position) for frontier in viable)
    clusters = cluster_frontiers(viable)
    if min_distance <= distance_threshold or len(clusters) < min_cluster_count:
        return False, clusters
    sorted_utilities = sorted(
        [frontier.utility.total_utility for frontier in viable], reverse=True
    )
    if len(sorted_utilities) >= 2 and (
        sorted_utilities[0] - sorted_utilities[1]
    ) > top2_gap_threshold:
        return False, clusters
    return True, clusters


def frontier_overlap_penalty(
    candidate: Sequence[float], assigned: Optional[Sequence[float]]
) -> float:
    if assigned is None:
        return 0.0
    distance = _l2(candidate, assigned)
    if distance >= 3.0:
        return 0.0
    return float(max(0.0, 1.0 - distance / 3.0))


def compute_frontier_utility(
    frontier: FrontierRecord,
    floor_prior: float,
    area_prior: float,
    overlap_penalty: float,
    blacklist_penalty: float,
    stale_penalty: float,
    weights,
    for_subagent: bool,
) -> UtilityBreakdown:
    path_cost = frontier.path_cost_from_sub if for_subagent else frontier.path_cost_from_main
    breakdown = UtilityBreakdown(
        semantic_score=frontier.semantic_score if weights.use_semantic else 0.0,
        geometry_gain=frontier.geometry_gain if weights.use_geometry else 0.0,
        floor_prior=floor_prior if weights.use_floor_prior else 0.0,
        area_prior=area_prior if weights.use_area_prior else 0.0,
        path_cost=path_cost,
        overlap_penalty=overlap_penalty if weights.use_overlap_penalty else 0.0,
        blacklist_penalty=blacklist_penalty if weights.use_blacklist_penalty else 0.0,
        stale_penalty=stale_penalty if weights.use_stale_penalty else 0.0,
    )
    breakdown.total_utility = (
        weights.alpha * breakdown.semantic_score
        + weights.beta * breakdown.geometry_gain
        + weights.gamma * breakdown.floor_prior
        + weights.eta * breakdown.area_prior
        - weights.lambda_path * breakdown.path_cost
        - weights.mu_overlap * breakdown.overlap_penalty
        - weights.nu_blacklist * breakdown.blacklist_penalty
        - weights.xi_stale * breakdown.stale_penalty
    )
    return breakdown


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2]), ord=2))


def _mean_centroid(points: Sequence[Sequence[float]]) -> Tuple[float, float]:
    arr = np.array(points, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())
