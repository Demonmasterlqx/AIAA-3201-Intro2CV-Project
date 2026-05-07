from __future__ import annotations

from collections import defaultdict
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np

from ascent.g_ascent_structures import AgentState
from ascent.g_ascent_structures import FloorNode
from ascent.g_ascent_structures import RegionGraphSnapshot
from ascent.g_ascent_structures import RegionNode


class RegionGraphBuilder:
    def __init__(self, cluster_radius: float = 2.5) -> None:
        self._cluster_radius = cluster_radius

    def build(
        self,
        *,
        agent_states: Iterable[AgentState],
        frontier_sets: Dict[str, List[Tuple[Tuple[float, float], float]]],
        explored_ratios: Dict[int, float],
        floor_priors: Dict[int, float],
        step: int,
    ) -> RegionGraphSnapshot:
        snapshot = RegionGraphSnapshot(step=step)
        per_floor_points: Dict[int, List[Tuple[str, Tuple[float, float], float]]] = defaultdict(list)
        for agent_state in agent_states:
            floor = snapshot.floors.setdefault(
                agent_state.floor_id,
                FloorNode(
                    floor_id=agent_state.floor_id,
                    explored_ratio=explored_ratios.get(agent_state.floor_id, 0.0),
                    goal_prior=floor_priors.get(agent_state.floor_id, 0.0),
                ),
            )
            for point, score in frontier_sets.get(agent_state.agent_id, []):
                per_floor_points[agent_state.floor_id].append((agent_state.agent_id, point, score))
            if "stairs" not in floor.stairs and floor.explored_ratio < 0.25:
                floor.stairs.append("unknown")

        for floor_id, points in per_floor_points.items():
            clusters = self._cluster_points(points)
            for idx, cluster in enumerate(clusters):
                centroid = tuple(np.mean([point for _, point, _ in cluster], axis=0).tolist())
                region_id = f"floor_{floor_id}_region_{idx:02d}"
                region_score = float(np.mean([score for _, _, score in cluster])) if cluster else 0.0
                region = RegionNode(
                    region_id=region_id,
                    floor_id=floor_id,
                    centroid=(float(centroid[0]), float(centroid[1])),
                    semantic_score=float(np.tanh(region_score)),
                    unknown_gain=float(len(cluster)),
                    reachable=True,
                    owner=None,
                    frontier_points=[(float(point[0]), float(point[1])) for _, point, _ in cluster],
                    source_agents=sorted({agent_id for agent_id, _, _ in cluster}),
                )
                snapshot.regions[region_id] = region
                snapshot.floors[floor_id].frontier_clusters.append(region_id)
                snapshot.floors[floor_id].region_ids.append(region_id)
        return snapshot

    def _cluster_points(
        self,
        points: List[Tuple[str, Tuple[float, float], float]],
    ) -> List[List[Tuple[str, Tuple[float, float], float]]]:
        clusters: List[List[Tuple[str, Tuple[float, float], float]]] = []
        for agent_id, point, score in points:
            placed = False
            for cluster in clusters:
                centroid = np.mean([cluster_point for _, cluster_point, _ in cluster], axis=0)
                if float(np.linalg.norm(np.asarray(point) - centroid)) <= self._cluster_radius:
                    cluster.append((agent_id, point, score))
                    placed = True
                    break
            if not placed:
                clusters.append([(agent_id, point, score)])
        return clusters
