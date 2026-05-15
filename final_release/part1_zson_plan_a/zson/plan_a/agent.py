from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat_sim.errors import GreedyFollowerError

from zson.plan_a.goal_selector import (
    GoalCandidate,
    select_frontier_candidates,
    select_semantic_candidates,
)
from zson.plan_a.semantic_map import SemanticMap, SemanticUpdate, build_free_explored_masks, gaussian_smooth


@dataclass
class StepDecision:
    action: int
    action_name: str
    reason: str
    strategy: str
    center_similarity: float
    center_depth: Optional[float]
    map_max_similarity: float
    current_goal_score: Optional[float]
    current_goal_cell: Optional[Tuple[int, int]]
    projected_cells: int
    force_frontier_steps: int = 0
    recovery_queue_len: int = 0
    stuck: bool = False
    rotation_stuck: bool = False


class PlanAAgent:
    ACTION_NAMES = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

    def __init__(self, sim, config, clip_encoder, text_cache):
        self.sim = sim
        self.config = config
        self.plan_cfg = config.PLAN_A
        self.clip_encoder = clip_encoder
        self.text_cache = text_cache
        self.follower = ShortestPathFollower(
            sim,
            goal_radius=self.plan_cfg.LOCAL_GOAL_RADIUS,
            return_one_hot=False,
            stop_on_error=False,
        )
        self.goal_embedding = None
        self.goal_category = None
        self.semantic_map = None
        self.suppression_mask = None
        self.subgoal_cell = None
        self.subgoal_world = None
        self.subgoal_score = None
        self.subgoal_strategy = "none"
        self.last_replan_step = -1
        self.step_id = 0
        self.position_history = deque(maxlen=self.plan_cfg.STUCK_WINDOW)
        self.trajectory = []
        self.max_similarity_seen = -1.0
        self.failed_goal_count = 0
        self.stuck_events = 0
        self.semantic_selections = 0
        self.frontier_selections = 0
        self.stop_called = False
        self.last_info = None
        self.force_frontier_steps = 0
        self.recovery_actions = deque()
        self.recent_actions = deque(maxlen=self.plan_cfg.ROTATION_STUCK_WINDOW)
        self.recovery_cooldown_steps = 0

    def reset(self, episode, info: Dict[str, object]) -> None:
        self.goal_category = episode.object_category
        self.goal_embedding = self.text_cache.get_embedding(self.goal_category)
        topdown_info = info["top_down_map"]
        if topdown_info is None:
            map_resolution = int(self.config.TASK_CONFIG.TASK.TOP_DOWN_MAP.MAP_RESOLUTION)
            map_shape = (map_resolution, map_resolution)
        else:
            map_shape = topdown_info["map"].shape
        hfov = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        self.semantic_map = SemanticMap(
            map_shape=map_shape,
            patch_grid=self.plan_cfg.PATCH_GRID,
            clip_input_size=self.plan_cfg.CLIP_INPUT_SIZE,
            min_depth_m=self.plan_cfg.MIN_DEPTH_M,
            max_depth_m=self.plan_cfg.MAX_DEPTH_M,
            hfov_deg=hfov,
            projection_stride=self.plan_cfg.PROJECTION_STRIDE,
        )
        self.suppression_mask = np.zeros(map_shape, dtype=np.int32)
        self.subgoal_cell = None
        self.subgoal_world = None
        self.subgoal_score = None
        self.subgoal_strategy = "none"
        self.last_replan_step = -1
        self.step_id = 0
        self.position_history.clear()
        self.trajectory = []
        self.max_similarity_seen = -1.0
        self.failed_goal_count = 0
        self.stuck_events = 0
        self.semantic_selections = 0
        self.frontier_selections = 0
        self.stop_called = False
        self.last_info = info
        self.force_frontier_steps = int(self.plan_cfg.FRONTIER_BOOTSTRAP_STEPS)
        self.recovery_actions.clear()
        self.recent_actions.clear()
        self.recovery_cooldown_steps = 0

    def _sensor_state(self):
        agent_state = self.sim.get_agent_state()
        sensor_states = getattr(agent_state, "sensor_states", {})
        sensor_uuid = self.plan_cfg.DEPTH_SENSOR
        return sensor_states.get(sensor_uuid, agent_state)

    def _ensure_map_shape(self, info: Dict[str, object]) -> None:
        topdown_info = info["top_down_map"]
        if topdown_info is None:
            return
        map_shape = topdown_info["map"].shape
        if tuple(self.semantic_map.map_shape) == tuple(map_shape):
            return
        hfov = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        self.semantic_map = SemanticMap(
            map_shape=map_shape,
            patch_grid=self.plan_cfg.PATCH_GRID,
            clip_input_size=self.plan_cfg.CLIP_INPUT_SIZE,
            min_depth_m=self.plan_cfg.MIN_DEPTH_M,
            max_depth_m=self.plan_cfg.MAX_DEPTH_M,
            hfov_deg=hfov,
            projection_stride=self.plan_cfg.PROJECTION_STRIDE,
        )
        self.suppression_mask = np.zeros(map_shape, dtype=np.int32)

    def _decrement_suppression(self):
        mask = self.suppression_mask > 0
        self.suppression_mask[mask] -= 1

    def _agent_cell(self, info: Dict[str, object]) -> Tuple[int, int]:
        coord = info["top_down_map"]["agent_map_coord"]
        return int(coord[0]), int(coord[1])

    def _agent_position(self, info: Dict[str, object]) -> np.ndarray:
        return np.asarray(info["agent_position"], dtype=np.float32)

    def _current_map_max(self) -> float:
        assert self.semantic_map is not None
        valid = self.semantic_map.hit_count > 0
        if not np.any(valid):
            return -1.0
        return float(np.max(self.semantic_map.mean_similarity[valid]))

    def _goal_reached(self, position: np.ndarray) -> bool:
        if self.subgoal_world is None:
            return False
        delta = position[[0, 2]] - self.subgoal_world[[0, 2]]
        return float(np.linalg.norm(delta)) <= float(self.plan_cfg.LOCAL_GOAL_RADIUS)

    def _is_stuck(self) -> bool:
        if len(self.position_history) < self.position_history.maxlen:
            return False
        start = self.position_history[0]
        end = self.position_history[-1]
        displacement = float(np.linalg.norm(end[[0, 2]] - start[[0, 2]]))
        return displacement < float(self.plan_cfg.STUCK_DISTANCE_M)

    def _is_rotation_stuck(self) -> bool:
        if len(self.recent_actions) < self.recent_actions.maxlen:
            return False
        turn_count = sum(action in (2, 3) for action in self.recent_actions)
        forward_count = sum(action == 1 for action in self.recent_actions)
        return (
            turn_count >= int(self.plan_cfg.ROTATION_STUCK_TURNS)
            and forward_count <= 1
        )

    def _start_recovery(self) -> None:
        if len(self.recovery_actions) > 0:
            return
        self.recovery_actions = deque(int(a) for a in self.plan_cfg.RECOVERY_ACTIONS)
        self.force_frontier_steps = max(
            self.force_frontier_steps,
            int(self.plan_cfg.FRONTIER_RECOVERY_STEPS),
        )
        self.recovery_cooldown_steps = max(
            self.recovery_cooldown_steps,
            int(self.plan_cfg.RECOVERY_COOLDOWN_STEPS),
        )
        self.recent_actions.clear()

    def _suppress_current_goal(self):
        if self.subgoal_cell is None:
            return
        radius = int(self.plan_cfg.SUPPRESSION_RADIUS)
        row, col = self.subgoal_cell
        r0 = max(row - radius, 0)
        r1 = min(row + radius + 1, self.suppression_mask.shape[0])
        c0 = max(col - radius, 0)
        c1 = min(col + radius + 1, self.suppression_mask.shape[1])
        self.suppression_mask[r0:r1, c0:c1] = int(self.plan_cfg.FAILED_GOAL_COOLDOWN)
        self.failed_goal_count += 1

    def _frontier_candidates(self, info: Dict[str, object]):
        free_mask, explored_mask, _ = build_free_explored_masks(info["top_down_map"])
        return select_frontier_candidates(
            free_mask=free_mask,
            explored_mask=explored_mask,
            suppression_mask=self.suppression_mask,
            agent_cell=self._agent_cell(info),
            limit=int(self.plan_cfg.FRONTIER_CANDIDATE_LIMIT),
            unknown_radius=int(self.plan_cfg.FRONTIER_UNKNOWN_RADIUS),
            distance_penalty=float(self.plan_cfg.FRONTIER_DISTANCE_PENALTY),
        )

    def _semantic_candidates(self, info: Dict[str, object]):
        free_mask, explored_mask, _ = build_free_explored_masks(info["top_down_map"])
        planning_map = gaussian_smooth(
            self.semantic_map.planning_map(),
            self.semantic_map.hit_count > 0,
            int(self.plan_cfg.SMOOTHING_KERNEL),
        )
        semantic_candidates = select_semantic_candidates(
            planning_map=planning_map,
            hit_count=self.semantic_map.hit_count,
            free_mask=free_mask,
            explored_mask=explored_mask,
            suppression_mask=self.suppression_mask,
            threshold=float(self.plan_cfg.SEMANTIC_THRESHOLD),
            top_k=int(self.plan_cfg.TOP_K_SEMANTIC_CANDIDATES),
        )
        return semantic_candidates, planning_map

    def _cell_to_nav_point(
        self,
        cell: Tuple[int, int],
        current_y: float,
    ) -> Optional[np.ndarray]:
        assert self.semantic_map is not None
        world_z, world_x = maps.from_grid(
            cell[0],
            cell[1],
            self.semantic_map.map_shape,
            pathfinder=self.sim.pathfinder,
        )
        candidate = np.array([world_x, current_y, world_z], dtype=np.float32)
        snapped = self.sim.pathfinder.snap_point(candidate)
        if snapped is None:
            return None
        snapped = np.asarray(snapped, dtype=np.float32)
        if np.any(~np.isfinite(snapped)):
            return None
        if not self.sim.is_navigable(snapped.tolist()):
            return None
        return snapped

    def _is_reachable(self, agent_position: np.ndarray, nav_point: np.ndarray) -> bool:
        distance = float(self.sim.geodesic_distance(agent_position, nav_point))
        return np.isfinite(distance) and distance < 1e8

    def _geodesic_distance(self, agent_position: np.ndarray, nav_point: np.ndarray) -> float:
        return float(self.sim.geodesic_distance(agent_position, nav_point))

    def _within_success_distance(self, info: Dict[str, object]) -> bool:
        distance = info.get("distance_to_goal")
        if distance is None:
            return False
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            return False
        success_distance = float(self.config.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE)
        return np.isfinite(distance) and distance <= success_distance

    def _choose_candidate(self, info: Dict[str, object]) -> Optional[GoalCandidate]:
        assert self.semantic_map is not None
        if info["top_down_map"] is None:
            return None
        semantic_candidates, _ = self._semantic_candidates(info)
        frontier_candidates = self._frontier_candidates(info)

        prefer_frontier = False
        if self.force_frontier_steps > 0:
            prefer_frontier = True
        elif len(semantic_candidates) == 0:
            prefer_frontier = True

        if not prefer_frontier and len(semantic_candidates) > 0:
            return semantic_candidates[0]
        if len(frontier_candidates) > 0:
            return frontier_candidates[0]
        if len(semantic_candidates) > 0:
            return semantic_candidates[0]
        return None

    def _replan(self, info: Dict[str, object]) -> None:
        candidate = self._choose_candidate(info)
        if candidate is None:
            self.subgoal_cell = None
            self.subgoal_world = None
            self.subgoal_score = None
            self.subgoal_strategy = "none"
            return

        agent_position = self._agent_position(info)
        candidate_groups = []
        if candidate.strategy == "frontier":
            candidate_groups.append(self._frontier_candidates(info))
            semantic_candidates, _ = self._semantic_candidates(info)
            candidate_groups.append(semantic_candidates)
        else:
            semantic_candidates, _ = self._semantic_candidates(info)
            candidate_groups.append(semantic_candidates)
            candidate_groups.append(self._frontier_candidates(info))

        for candidates in candidate_groups:
            for item in candidates:
                nav_point = self._cell_to_nav_point(item.cell, float(agent_position[1]))
                if nav_point is None or not self._is_reachable(agent_position, nav_point):
                    continue
                geodesic_distance = self._geodesic_distance(agent_position, nav_point)
                if geodesic_distance < float(self.plan_cfg.MIN_SUBGOAL_DISTANCE_M):
                    continue

                self.subgoal_cell = item.cell
                self.subgoal_world = nav_point
                self.subgoal_score = item.score
                self.subgoal_strategy = item.strategy
                if item.strategy == "semantic":
                    self.semantic_selections += 1
                elif item.strategy == "frontier":
                    self.frontier_selections += 1
                self.last_replan_step = self.step_id
                return

        self.subgoal_cell = None
        self.subgoal_world = None
        self.subgoal_score = None
        self.subgoal_strategy = "none"

    def observe(self, observation: Dict[str, np.ndarray], info: Dict[str, object]):
        assert self.semantic_map is not None
        self.last_info = info
        self._decrement_suppression()
        agent_position = self._agent_position(info)
        self.position_history.append(agent_position)
        if info["top_down_map"] is not None:
            self.trajectory.append(self._agent_cell(info))
        else:
            self.trajectory.append((-1, -1))

        if info["top_down_map"] is None:
            return SemanticUpdate(
                center_depth=None,
                center_similarity=-1.0,
                projected_cells=0,
                max_projected_similarity=-1.0,
            )

        self._ensure_map_shape(info)

        rgb = observation[self.plan_cfg.EGO_SENSOR]
        depth = observation[self.plan_cfg.DEPTH_SENSOR]
        image_features = self.clip_encoder.encode_image_patches(rgb)
        update = self.semantic_map.update(
            rgb_shape=rgb.shape[:2],
            depth_observation=depth,
            patch_features=image_features.patch_features,
            text_embedding=self.goal_embedding,
            sensor_state=self._sensor_state(),
            topdown_info=info["top_down_map"],
            pathfinder=self.sim.pathfinder,
        )
        self.max_similarity_seen = max(
            self.max_similarity_seen, update.max_projected_similarity
        )
        return update

    def act(self, observation: Dict[str, np.ndarray], info: Dict[str, object]) -> StepDecision:
        update = self.observe(observation, info)
        self.step_id += 1
        if self.force_frontier_steps > 0:
            self.force_frontier_steps -= 1
        if self.recovery_cooldown_steps > 0:
            self.recovery_cooldown_steps -= 1
        center_depth = update.center_depth
        center_similarity = update.center_similarity
        map_max_similarity = self._current_map_max()
        action = 3
        reason = "frontier-spin"
        strategy = self.subgoal_strategy
        position = self._agent_position(info)
        if info["top_down_map"] is None:
            return StepDecision(
                action=3,
                action_name=self.ACTION_NAMES[3],
                reason="bootstrap-topdown",
                strategy="none",
                center_similarity=center_similarity,
                center_depth=center_depth,
                map_max_similarity=map_max_similarity,
                current_goal_score=self.subgoal_score,
                current_goal_cell=self.subgoal_cell,
                projected_cells=update.projected_cells,
            )
        if self.plan_cfg.USE_SUCCESS_DISTANCE_STOP and self._within_success_distance(info):
            self.stop_called = True
            return StepDecision(
                action=0,
                action_name=self.ACTION_NAMES[0],
                reason="stop-success-distance",
                strategy="success-distance",
                center_similarity=center_similarity,
                center_depth=center_depth,
                map_max_similarity=map_max_similarity,
                current_goal_score=self.subgoal_score,
                current_goal_cell=self.subgoal_cell,
                projected_cells=update.projected_cells,
            )
        stuck = self._is_stuck()
        rotation_stuck = self._is_rotation_stuck()
        should_start_recovery = (
            (stuck or rotation_stuck)
            and self.recovery_cooldown_steps <= 0
            and len(self.recovery_actions) == 0
        )
        if should_start_recovery:
            self.stuck_events += 1
            self._suppress_current_goal()
            self.subgoal_world = None
            self.subgoal_cell = None
            self.subgoal_score = None
            self.subgoal_strategy = "none"
            self._start_recovery()

        in_recovery = len(self.recovery_actions) > 0
        interval_due = (self.step_id - self.last_replan_step) >= int(
            self.plan_cfg.REPLAN_INTERVAL
        )
        keep_active_goal = (
            self.subgoal_world is not None
            and not self._goal_reached(position)
        )
        needs_replan = (
            not in_recovery
            and (
                self.subgoal_world is None
                or self._goal_reached(position)
                or (interval_due and not keep_active_goal)
            )
        )
        if needs_replan:
            if self._goal_reached(position):
                self.subgoal_world = None
                self.subgoal_cell = None
                self.subgoal_score = None
                self.subgoal_strategy = "none"
            self._replan(info)

        strategy = self.subgoal_strategy
        if len(self.recovery_actions) > 0:
            action = int(self.recovery_actions.popleft())
            reason = "recovery-sequence"
            strategy = "frontier"
        elif self.subgoal_world is not None:
            try:
                follower_action = int(
                    self.follower.get_next_action(self.subgoal_world.tolist())
                )
                if follower_action == 0:
                    should_stop = (
                        self.subgoal_strategy == "semantic"
                        and center_depth is not None
                        and center_depth <= float(self.plan_cfg.STOP_DEPTH_M)
                        and center_similarity
                        >= float(self.plan_cfg.STOP_SIMILARITY_THRESHOLD)
                    )
                    if should_stop:
                        self.stop_called = True
                        action = 0
                        reason = "stop-heuristic-subgoal"
                    else:
                        # Reaching a local subgoal triggers replanning instead of terminating the episode.
                        self.subgoal_world = None
                        self.subgoal_cell = None
                        self.subgoal_score = None
                        self.subgoal_strategy = "none"
                        action = 3
                        reason = "subgoal-reached-replan"
                        strategy = "none"
                        if center_similarity < float(self.plan_cfg.STOP_SIMILARITY_THRESHOLD):
                            self.force_frontier_steps = max(
                                self.force_frontier_steps,
                                int(self.plan_cfg.FRONTIER_RECOVERY_STEPS // 2),
                            )
                else:
                    action = follower_action
                    reason = "follow-subgoal"
            except GreedyFollowerError:
                self._suppress_current_goal()
                self.subgoal_world = None
                self.subgoal_cell = None
                self.subgoal_score = None
                self.subgoal_strategy = "none"
                action = 3
                reason = "follower-error"
                self.force_frontier_steps = max(
                    self.force_frontier_steps,
                    int(self.plan_cfg.FRONTIER_RECOVERY_STEPS),
                )
        else:
            action = 3
            reason = "no-subgoal"
            self.force_frontier_steps = max(
                self.force_frontier_steps,
                int(self.plan_cfg.FRONTIER_RECOVERY_STEPS // 2),
            )

        self.recent_actions.append(int(action))

        return StepDecision(
            action=action,
            action_name=self.ACTION_NAMES[action],
            reason=reason,
            strategy=strategy,
            center_similarity=center_similarity,
            center_depth=center_depth,
            map_max_similarity=map_max_similarity,
            current_goal_score=self.subgoal_score,
            current_goal_cell=self.subgoal_cell,
            projected_cells=update.projected_cells,
            force_frontier_steps=self.force_frontier_steps,
            recovery_queue_len=len(self.recovery_actions),
            stuck=stuck,
            rotation_stuck=rotation_stuck,
        )
