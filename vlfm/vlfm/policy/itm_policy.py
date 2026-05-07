# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

try:
    from part3_multi_agent.shared_memory import (
        SharedEpisodeMemory,
        compute_claim_cell_xy,
        upsert_region_in_state,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    try:
        from part3_multi_agent.shared_memory import (
            SharedEpisodeMemory,
            compute_claim_cell_xy,
            upsert_region_in_state,
        )
    except ModuleNotFoundError:
        SharedEpisodeMemory = None  # type: ignore[assignment]
        compute_claim_cell_xy = None  # type: ignore[assignment]
        upsert_region_in_state = None  # type: ignore[assignment]

PROMPT_SEPARATOR = "|"


class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = AcyclicEnforcer()

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            self._termination_hint = "frontier_collapse"
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        return self._select_best_point(sorted_pts, sorted_values)

    def _select_best_point(
        self,
        sorted_pts: np.ndarray,
        sorted_values: List[float],
    ) -> Tuple[np.ndarray, float]:
        if len(sorted_pts) == 0:
            raise ValueError("Expected at least one candidate point.")

        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(sorted_pts)),
                key=lambda i: np.linalg.norm(sorted_pts[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info

    def _update_value_map(self) -> None:
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        cosines = [
            [
                self._itm.cosine(
                    rgb,
                    p.replace("target_object", self._target_object.replace("|", "/")),
                )
                for p in self._text_prompt.split(PROMPT_SEPARATOR)
            ]
            for rgb in all_rgb
        ]
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values


class DenseValueMatchingPolicy(BaseITMPolicy):
    def __init__(
        self,
        dense_point_radius: float,
        dense_min_value: float,
        dense_min_distance: float,
        dense_max_candidates: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._dense_point_radius = dense_point_radius
        self._dense_min_value = dense_min_value
        self._dense_min_distance = dense_min_distance
        self._dense_max_candidates = dense_max_candidates

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        del observations
        sorted_pts, sorted_values = self._get_sorted_value_points()
        if len(sorted_pts) == 0:
            print("No dense value goals found during exploration, stopping.")
            return self._stop_action

        best_point, best_value = self._select_best_point(sorted_pts, sorted_values)
        os.environ["DEBUG_INFO"] = f"Dense goal value: {best_value*100:.2f}%"
        print(f"Dense goal value: {best_value*100:.2f}%")
        return self._pointnav(best_point, stop=False)

    def _get_sorted_value_points(self) -> Tuple[np.ndarray, List[float]]:
        reduce_fn = self._vis_reduce_fn if self._value_map._value_channels > 1 else None
        candidate_points, candidate_values = self._value_map.get_value_peaks(
            radius=self._dense_point_radius,
            reduce_fn=reduce_fn,
            min_value=self._dense_min_value,
            max_points=self._dense_max_candidates,
        )
        if len(candidate_points) == 0:
            return candidate_points, candidate_values

        robot_xy = self._observations_cache["robot_xy"]
        distances = np.linalg.norm(candidate_points - robot_xy, axis=1)
        far_enough = distances >= self._dense_min_distance
        if np.any(far_enough):
            candidate_points = candidate_points[far_enough]
            candidate_values = [candidate_values[idx] for idx, keep in enumerate(far_enough) if keep]
            distances = distances[far_enough]

        order = sorted(
            range(len(candidate_values)),
            key=lambda idx: (candidate_values[idx], distances[idx]),
            reverse=True,
        )
        sorted_points = np.array([candidate_points[idx] for idx in order])
        sorted_values = [candidate_values[idx] for idx in order]
        return sorted_points, sorted_values


class AdaptiveHybridITMPolicy(DenseValueMatchingPolicy):
    """Switch between semantic dense matching and geometry-driven frontier search.

    This is a lightweight Part 2 baseline that stays close to the existing VLFM
    codepath. The policy uses the dense value map as a short-term semantic memory.
    When the recent dense scores are strong and stable, it follows the dense
    semantic peak. Otherwise, it falls back to frontier exploration using a simple
    geometry score that prefers nearby unexplored entrances.
    """

    def __init__(
        self,
        adaptive_semantic_threshold: float,
        adaptive_score_ema_alpha: float,
        adaptive_score_history_length: int,
        adaptive_min_confident_fraction: float,
        adaptive_max_score_std: float,
        adaptive_frontier_min_distance: float,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._adaptive_semantic_threshold = adaptive_semantic_threshold
        self._adaptive_score_ema_alpha = adaptive_score_ema_alpha
        self._adaptive_score_history_length = adaptive_score_history_length
        self._adaptive_min_confident_fraction = adaptive_min_confident_fraction
        self._adaptive_max_score_std = adaptive_max_score_std
        self._adaptive_frontier_min_distance = adaptive_frontier_min_distance
        self._adaptive_mode = "geometry"
        self._adaptive_score_history: List[float] = []
        self._adaptive_score_ema = 0.0

    def _reset(self) -> None:
        super()._reset()
        self._adaptive_mode = "geometry"
        self._adaptive_score_history = []
        self._adaptive_score_ema = 0.0

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        dense_points, dense_values = self._get_sorted_value_points()
        best_dense_score = dense_values[0] if dense_values else 0.0
        self._update_semantic_memory(best_dense_score)

        if self._should_use_semantic_mode(best_dense_score, len(dense_points) > 0):
            self._adaptive_mode = "semantic"
            best_point, best_value = self._select_best_point(dense_points, dense_values)
            os.environ["DEBUG_INFO"] = (
                f"Mode: semantic | Dense score: {best_value*100:.2f}% | "
                f"EMA: {self._adaptive_score_ema*100:.2f}%"
            )
            print(
                f"Adaptive mode: semantic | Dense score: {best_value*100:.2f}% | "
                f"EMA: {self._adaptive_score_ema*100:.2f}%"
            )
            return self._pointnav(best_point, stop=False)

        frontier_points, frontier_values = self._get_sorted_geometry_frontiers()
        if len(frontier_points) > 0:
            self._adaptive_mode = "geometry"
            best_point, best_value = self._select_best_point(frontier_points, frontier_values)
            os.environ["DEBUG_INFO"] = (
                f"Mode: geometry | Dense score: {best_dense_score*100:.2f}% | "
                f"EMA: {self._adaptive_score_ema*100:.2f}% | "
                f"Frontier score: {best_value:.3f}"
            )
            print(
                f"Adaptive mode: geometry | Dense score: {best_dense_score*100:.2f}% | "
                f"EMA: {self._adaptive_score_ema*100:.2f}% | "
                f"Frontier score: {best_value:.3f}"
            )
            return self._pointnav(best_point, stop=False)

        if len(dense_points) > 0:
            self._adaptive_mode = "semantic_fallback"
            best_point, best_value = self._select_best_point(dense_points, dense_values)
            os.environ["DEBUG_INFO"] = (
                f"Mode: semantic_fallback | Dense score: {best_value*100:.2f}% | "
                f"EMA: {self._adaptive_score_ema*100:.2f}%"
            )
            print(
                f"Adaptive mode: semantic_fallback | Dense score: {best_value*100:.2f}% | "
                f"EMA: {self._adaptive_score_ema*100:.2f}%"
            )
            return self._pointnav(best_point, stop=False)

        print("No dense peaks or frontiers found during adaptive exploration, stopping.")
        self._termination_hint = "no_candidate_goals"
        return self._stop_action

    def _update_semantic_memory(self, current_score: float) -> None:
        if not self._adaptive_score_history:
            self._adaptive_score_ema = current_score
        else:
            alpha = self._adaptive_score_ema_alpha
            self._adaptive_score_ema = alpha * current_score + (1 - alpha) * self._adaptive_score_ema
        self._adaptive_score_history.append(current_score)
        if len(self._adaptive_score_history) > self._adaptive_score_history_length:
            self._adaptive_score_history = self._adaptive_score_history[-self._adaptive_score_history_length :]

    def _should_use_semantic_mode(self, current_score: float, has_dense_candidates: bool) -> bool:
        if not has_dense_candidates:
            return False

        history = self._adaptive_score_history[-self._adaptive_score_history_length :]
        if len(history) == 0:
            return False

        if len(history) == 1:
            return (
                current_score >= self._adaptive_semantic_threshold
                and self._adaptive_score_ema >= self._adaptive_semantic_threshold
            )

        confident_fraction = sum(v >= self._adaptive_semantic_threshold for v in history) / len(history)
        score_std = float(np.std(history))
        return (
            self._adaptive_score_ema >= self._adaptive_semantic_threshold
            and confident_fraction >= self._adaptive_min_confident_fraction
            and score_std <= self._adaptive_max_score_std
        )

    def _get_sorted_geometry_frontiers(self) -> Tuple[np.ndarray, List[float]]:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            return np.array([]), []

        robot_xy = self._observations_cache["robot_xy"]
        distances = np.linalg.norm(frontiers - robot_xy, axis=1)
        far_enough = distances >= self._adaptive_frontier_min_distance
        if np.any(far_enough):
            frontiers = frontiers[far_enough]
            distances = distances[far_enough]

        if len(frontiers) == 0:
            return np.array([]), []

        scores = [1.0 / max(distance, 1e-6) for distance in distances]
        order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        sorted_frontiers = np.array([frontiers[idx] for idx in order])
        sorted_scores = [scores[idx] for idx in order]
        return sorted_frontiers, sorted_scores

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)
        policy_info["adaptive_mode_name"] = self._adaptive_mode
        policy_info["adaptive_score_ema_value"] = self._adaptive_score_ema

        if not self._visualize:
            return policy_info

        policy_info["render_below_images"].extend(["adaptive_mode", "adaptive_score_ema"])
        policy_info["adaptive_mode"] = f"adaptive_mode: {self._adaptive_mode}"
        policy_info["adaptive_score_ema"] = f"adaptive_score_ema: {self._adaptive_score_ema:.2f}"
        return policy_info


class CollaborativeAdaptiveHybridITMPolicy(AdaptiveHybridITMPolicy):
    """Peer-aware collaborative wrapper around the Part 2 adaptive hybrid policy."""

    _collab_peer_stale_timeout_sec: float = 30.0
    _collab_sector_bonus_scale: float = 0.03
    _collab_claim_penalty: float = 1e6
    _collab_assignment_bonus_scale: float = 0.15
    _collab_peer_assignment_penalty: float = 0.3

    def __init__(
        self,
        collab_agent_id: str,
        collab_shared_memory_path: str,
        collab_claim_cell_size_m: float,
        collab_claim_radius_m: float,
        collab_candidate_score_margin: float,
        collab_peer_success_stop: bool,
        collab_initial_sector_bias: float,
        collab_shared_update_every_n_steps: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if SharedEpisodeMemory is None or compute_claim_cell_xy is None or upsert_region_in_state is None:
            raise ImportError(
                "CollaborativeAdaptiveHybridITMPolicy requires part3_multi_agent on PYTHONPATH."
            )
        if collab_agent_id not in {"agent_0", "agent_1"}:
            raise ValueError(f"Unsupported collab_agent_id '{collab_agent_id}'.")
        if not collab_shared_memory_path:
            raise ValueError("collab_shared_memory_path must be provided for collaborative evaluation.")

        self._collab_agent_id = collab_agent_id
        self._collab_peer_agent_id = "agent_1" if collab_agent_id == "agent_0" else "agent_0"
        self._collab_shared_memory_path = collab_shared_memory_path
        self._collab_claim_cell_size_m = collab_claim_cell_size_m
        self._collab_claim_radius_m = collab_claim_radius_m
        self._collab_candidate_score_margin = collab_candidate_score_margin
        self._collab_peer_success_stop = collab_peer_success_stop
        self._collab_initial_sector_bias = collab_initial_sector_bias
        self._collab_shared_update_every_n_steps = max(int(collab_shared_update_every_n_steps), 1)
        self._collab_memory: Optional["SharedEpisodeMemory"] = None
        self._collab_last_best_candidate: Optional[np.ndarray] = None
        self._collab_last_best_candidate_score = 0.0
        self._collab_last_selected_goal: Optional[np.ndarray] = None
        self._collab_claimed_cell_xy: Optional[List[float]] = None
        self._collab_suppressed_candidates = 0
        self._collab_total_suppressed_candidates = 0
        self._collab_conflict_replans = 0
        self._collab_peer_success_abort = False
        self._collab_assigned_region_id: Optional[str] = None
        self._collab_assigned_cell_xy: Optional[List[float]] = None
        self._collab_intent_text: str = ""
        self._collab_assignment_source: str = ""

    def _reset(self) -> None:
        super()._reset()
        self._collab_last_best_candidate = None
        self._collab_last_best_candidate_score = 0.0
        self._collab_last_selected_goal = None
        self._collab_claimed_cell_xy = None
        self._collab_suppressed_candidates = 0
        self._collab_total_suppressed_candidates = 0
        self._collab_conflict_replans = 0
        self._collab_peer_success_abort = False
        self._collab_assigned_region_id = None
        self._collab_assigned_cell_xy = None
        self._collab_intent_text = ""
        self._collab_assignment_source = ""
        self._publish_shared_state(force=True)

    def _collab_episode_key(self) -> str:
        default_key = f"collab::{self._collab_agent_id}"
        return os.environ.get("COLLAB_EPISODE_KEY", default_key)

    def _shared_memory(self) -> "SharedEpisodeMemory":
        if self._collab_memory is None:
            self._collab_memory = SharedEpisodeMemory(
                self._collab_shared_memory_path,
                self._collab_episode_key(),
            )
        return self._collab_memory

    def _read_peer_state(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        shared_state = self._shared_memory().read()
        peer_state = dict(shared_state.get("agents", {}).get(self._collab_peer_agent_id, {}))
        return shared_state, peer_state

    def _peer_state_is_stale(self, peer_state: Dict[str, Any]) -> bool:
        last_update_time = float(peer_state.get("last_update_time") or 0.0)
        if last_update_time <= 0:
            return False
        return (time.time() - last_update_time) > self._collab_peer_stale_timeout_sec

    def _should_abort_for_peer_success(self, shared_state: Dict[str, Any]) -> bool:
        return bool(
            self._collab_peer_success_stop
            and shared_state.get("success_flag")
            and shared_state.get("winner_agent_id")
            and shared_state.get("winner_agent_id") != self._collab_agent_id
        )

    def _normalise_xy(self, point_xy: Optional[Union[np.ndarray, List[float], Tuple[float, float]]]) -> Optional[List[float]]:
        if point_xy is None:
            return None
        arr = np.asarray(point_xy, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            return None
        return [float(arr[0]), float(arr[1])]

    def _current_claimed_cell(self) -> Optional[List[float]]:
        if self._collab_last_selected_goal is not None:
            claimed = compute_claim_cell_xy(self._collab_last_selected_goal, self._collab_claim_cell_size_m)
        elif self._collab_last_best_candidate is not None:
            claimed = compute_claim_cell_xy(self._collab_last_best_candidate, self._collab_claim_cell_size_m)
        else:
            claimed = None
        return self._normalise_xy(claimed)

    def _publish_shared_state(
        self,
        force: bool = False,
        finished: Optional[bool] = None,
        success: Optional[bool] = None,
    ) -> None:
        if not force and (self._num_steps % self._collab_shared_update_every_n_steps) != 0:
            return

        robot_xy = self._normalise_xy(self._observations_cache.get("robot_xy"))
        selected_goal_xy = self._normalise_xy(self._collab_last_selected_goal)
        best_candidate_xy = self._normalise_xy(self._collab_last_best_candidate)
        claimed_cell_xy = self._current_claimed_cell()
        self._collab_claimed_cell_xy = claimed_cell_xy
        target_detected = bool(self._target_object and self._object_map.has_object(self._target_object))
        step_idx = int(self._num_steps)
        finished_value = bool(finished) if finished is not None else bool(self._called_stop or self._collab_peer_success_abort)
        success_value = bool(success) if success is not None else False

        def updater(state: Dict[str, Any]) -> None:
            agent_state = state.setdefault("agents", {}).setdefault(self._collab_agent_id, {})
            assignment = (
                state.setdefault("coordination", {})
                .setdefault("assignments", {})
                .setdefault(self._collab_agent_id, {})
            )
            self._collab_assigned_region_id = assignment.get("assigned_region_id")
            self._collab_assigned_cell_xy = self._normalise_xy(assignment.get("assigned_cell_xy"))
            self._collab_intent_text = str(assignment.get("intent", "")).strip()
            self._collab_assignment_source = str(assignment.get("source", "")).strip()
            agent_state.update(
                {
                    "step_idx": step_idx,
                    "robot_xy": robot_xy,
                    "selected_goal_xy": selected_goal_xy,
                    "best_candidate_xy": best_candidate_xy,
                    "best_candidate_score": float(self._collab_last_best_candidate_score),
                    "claimed_cell_xy": claimed_cell_xy,
                    "target_detected": target_detected,
                    "finished": finished_value,
                    "success": success_value,
                    "target_object": self._target_object,
                    "intent_text": self._collab_intent_text,
                    "assigned_region_id": self._collab_assigned_region_id,
                    "assigned_cell_xy": self._collab_assigned_cell_xy,
                    "assignment_source": self._collab_assignment_source,
                    "last_update_time": time.time(),
                }
            )
            upsert_region_in_state(
                state,
                agent_id=self._collab_agent_id,
                target_object=self._target_object,
                point_xy=best_candidate_xy or selected_goal_xy,
                claimed_cell_xy=claimed_cell_xy,
                score=float(self._collab_last_best_candidate_score),
                selected_goal_xy=selected_goal_xy,
                target_detected=target_detected,
                cell_size_m=self._collab_claim_cell_size_m,
            )

        self._shared_memory().update(updater)

    def _coordination_assignment(self, shared_state: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        assignment = (
            shared_state.get("coordination", {})
            .get("assignments", {})
            .get(agent_id, {})
        )
        return dict(assignment)

    def _sector_bonus(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0 or self._collab_initial_sector_bias == 0.0:
            return np.zeros(len(points), dtype=np.float32)
        robot_xy = self._observations_cache["robot_xy"]
        heading = float(self._observations_cache["robot_heading"])
        rel = points - robot_xy
        angles = np.arctan2(rel[:, 1], rel[:, 0]) - heading
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        preferred_sign = -1.0 if self._collab_initial_sector_bias < 0 else 1.0
        same_side = np.sign(angles) == preferred_sign
        magnitude = np.maximum(0.0, 1.0 - (np.abs(angles) / np.pi))
        return np.where(same_side, self._collab_sector_bonus_scale * magnitude, 0.0)

    def _peer_reference_point(self, peer_state: Dict[str, Any]) -> Optional[np.ndarray]:
        for key in ("selected_goal_xy", "best_candidate_xy", "claimed_cell_xy"):
            value = peer_state.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                return arr[:2]
        return None

    def _apply_assignment_guidance(
        self,
        points: np.ndarray,
        values: List[float],
        own_assignment: Dict[str, Any],
        peer_assignment: Dict[str, Any],
    ) -> Tuple[np.ndarray, List[float], int]:
        if len(points) == 0:
            return points, values, 0

        adjusted_values = np.asarray(values, dtype=np.float32).copy()
        interventions = np.zeros(len(points), dtype=bool)

        own_xy = self._normalise_xy(own_assignment.get("assigned_cell_xy"))
        if own_xy is not None:
            own_xy_arr = np.asarray(own_xy, dtype=np.float32)
            own_dist = np.linalg.norm(points - own_xy_arr, axis=1)
            adjusted_values += self._collab_assignment_bonus_scale * np.exp(
                -own_dist / max(self._collab_claim_radius_m, 1e-3)
            )
            interventions |= own_dist <= max(self._collab_claim_radius_m * 1.25, 1.0)

        peer_xy = self._normalise_xy(peer_assignment.get("assigned_cell_xy"))
        if peer_xy is not None:
            peer_xy_arr = np.asarray(peer_xy, dtype=np.float32)
            peer_dist = np.linalg.norm(points - peer_xy_arr, axis=1)
            peer_hits = peer_dist <= max(self._collab_claim_radius_m, 1.0)
            adjusted_values[peer_hits] -= self._collab_peer_assignment_penalty
            interventions |= peer_hits

        order = np.argsort(adjusted_values)[::-1]
        sorted_points = np.asarray(points[order])
        sorted_values = [float(adjusted_values[idx]) for idx in order]
        return sorted_points, sorted_values, int(interventions.sum())

    def _apply_peer_suppression(
        self,
        points: np.ndarray,
        values: List[float],
        peer_state: Dict[str, Any],
        compare_scores: bool,
    ) -> Tuple[np.ndarray, List[float], int]:
        if len(points) == 0:
            return points, values, 0

        adjusted_values = np.asarray(values, dtype=np.float32).copy()
        original_values = adjusted_values.copy()
        suppressed = np.zeros(len(points), dtype=bool)
        sector_bonus = self._sector_bonus(points)

        if not self._peer_state_is_stale(peer_state):
            peer_claim = peer_state.get("claimed_cell_xy")
            if peer_claim is not None:
                peer_claim_xy = np.asarray(peer_claim, dtype=np.float32).reshape(-1)[:2]
                claim_hits = np.linalg.norm(points - peer_claim_xy, axis=1) <= max(
                    self._collab_claim_radius_m,
                    self._collab_claim_cell_size_m * 0.75,
                )
                adjusted_values[claim_hits] -= self._collab_claim_penalty
                suppressed |= claim_hits

            peer_point = self._peer_reference_point(peer_state)
            peer_score = float(peer_state.get("best_candidate_score") or 0.0)
            if peer_point is not None:
                overlap_hits = np.linalg.norm(points - peer_point, axis=1) <= self._collab_claim_radius_m
                if compare_scores:
                    lower_score_hits = overlap_hits & (adjusted_values <= (peer_score + self._collab_candidate_score_margin))
                    adjusted_values[lower_score_hits] -= self._collab_claim_penalty
                    suppressed |= lower_score_hits
                else:
                    adjusted_values[overlap_hits] -= self._collab_claim_penalty
                    suppressed |= overlap_hits

        adjusted_values += sector_bonus
        if np.all(suppressed):
            adjusted_values = original_values + sector_bonus
            suppressed = np.zeros(len(points), dtype=bool)

        order = np.argsort(adjusted_values)[::-1]
        sorted_points = np.asarray(points[order])
        sorted_values = [float(adjusted_values[idx]) for idx in order]
        return sorted_points, sorted_values, int(suppressed.sum())

    def _select_and_record(
        self,
        sorted_pts: np.ndarray,
        sorted_values: List[float],
        best_candidate_xy: Optional[np.ndarray],
        best_candidate_score: float,
    ) -> Tuple[np.ndarray, float]:
        selected_point, selected_value = self._select_best_point(sorted_pts, sorted_values)
        self._collab_last_selected_goal = np.asarray(selected_point, dtype=np.float32)
        self._collab_last_best_candidate = (
            np.asarray(best_candidate_xy, dtype=np.float32) if best_candidate_xy is not None else None
        )
        self._collab_last_best_candidate_score = float(best_candidate_score)
        self._collab_claimed_cell_xy = self._current_claimed_cell()
        return selected_point, selected_value

    def _abort_for_peer_success(self) -> Tensor:
        self._collab_peer_success_abort = True
        self._termination_hint = "peer_success_abort"
        self._publish_shared_state(force=True, finished=True, success=False)
        return self._stop_action

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        del observations
        self._collab_suppressed_candidates = 0

        shared_state, peer_state = self._read_peer_state()
        own_assignment = self._coordination_assignment(shared_state, self._collab_agent_id)
        peer_assignment = self._coordination_assignment(shared_state, self._collab_peer_agent_id)
        self._collab_assigned_region_id = own_assignment.get("assigned_region_id")
        self._collab_assigned_cell_xy = self._normalise_xy(own_assignment.get("assigned_cell_xy"))
        self._collab_intent_text = str(own_assignment.get("intent", "")).strip()
        self._collab_assignment_source = str(own_assignment.get("source", "")).strip()
        if self._should_abort_for_peer_success(shared_state):
            print(f"{self._collab_agent_id}: peer success detected, stopping early.")
            return self._abort_for_peer_success()

        dense_points, dense_values = self._get_sorted_value_points()
        dense_points, dense_values, dense_suppressed = self._apply_peer_suppression(
            dense_points,
            dense_values,
            peer_state,
            compare_scores=True,
        )
        dense_points, dense_values, dense_assignment_interventions = self._apply_assignment_guidance(
            dense_points,
            dense_values,
            own_assignment,
            peer_assignment,
        )
        self._collab_suppressed_candidates += dense_suppressed
        self._collab_suppressed_candidates += dense_assignment_interventions
        self._collab_total_suppressed_candidates += dense_suppressed
        self._collab_total_suppressed_candidates += dense_assignment_interventions

        best_dense_point = dense_points[0] if len(dense_points) > 0 else None
        best_dense_score = dense_values[0] if dense_values else 0.0
        self._update_semantic_memory(best_dense_score)

        if self._should_use_semantic_mode(best_dense_score, len(dense_points) > 0):
            self._adaptive_mode = "semantic"
            best_point, best_value = self._select_and_record(
                dense_points,
                dense_values,
                best_candidate_xy=best_dense_point,
                best_candidate_score=best_dense_score,
            )
            os.environ["DEBUG_INFO"] = (
                f"Mode: semantic | Dense score: {best_value*100:.2f}% | "
                f"EMA: {self._adaptive_score_ema*100:.2f}% | "
                f"Suppressed: {self._collab_suppressed_candidates}"
            )
            print(
                f"Adaptive mode: semantic | Dense score: {best_value*100:.2f}% | "
                f"EMA: {self._adaptive_score_ema*100:.2f}% | "
                f"Suppressed: {self._collab_suppressed_candidates}"
            )
            return self._pointnav(best_point, stop=False)

        frontier_points, frontier_values = self._get_sorted_geometry_frontiers()
        frontier_points, frontier_values, frontier_suppressed = self._apply_peer_suppression(
            frontier_points,
            frontier_values,
            peer_state,
            compare_scores=False,
        )
        frontier_points, frontier_values, frontier_assignment_interventions = self._apply_assignment_guidance(
            frontier_points,
            frontier_values,
            own_assignment,
            peer_assignment,
        )
        self._collab_suppressed_candidates += frontier_suppressed
        self._collab_suppressed_candidates += frontier_assignment_interventions
        self._collab_total_suppressed_candidates += frontier_suppressed
        self._collab_total_suppressed_candidates += frontier_assignment_interventions

        if len(frontier_points) > 0:
            self._adaptive_mode = "geometry"
            best_point, best_value = self._select_and_record(
                frontier_points,
                frontier_values,
                best_candidate_xy=best_dense_point,
                best_candidate_score=best_dense_score,
            )
            os.environ["DEBUG_INFO"] = (
                f"Mode: geometry | Dense score: {best_dense_score*100:.2f}% | "
                f"EMA: {self._adaptive_score_ema*100:.2f}% | "
                f"Frontier score: {best_value:.3f} | "
                f"Suppressed: {self._collab_suppressed_candidates}"
            )
            print(
                f"Adaptive mode: geometry | Dense score: {best_dense_score*100:.2f}% | "
                f"EMA: {self._adaptive_score_ema*100:.2f}% | "
                f"Frontier score: {best_value:.3f} | "
                f"Suppressed: {self._collab_suppressed_candidates}"
            )
            return self._pointnav(best_point, stop=False)

        if len(dense_points) > 0:
            self._adaptive_mode = "semantic_fallback"
            best_point, best_value = self._select_and_record(
                dense_points,
                dense_values,
                best_candidate_xy=best_dense_point,
                best_candidate_score=best_dense_score,
            )
            os.environ["DEBUG_INFO"] = (
                f"Mode: semantic_fallback | Dense score: {best_value*100:.2f}% | "
                f"EMA: {self._adaptive_score_ema*100:.2f}% | "
                f"Suppressed: {self._collab_suppressed_candidates}"
            )
            print(
                f"Adaptive mode: semantic_fallback | Dense score: {best_value*100:.2f}% | "
                f"EMA: {self._adaptive_score_ema*100:.2f}% | "
                f"Suppressed: {self._collab_suppressed_candidates}"
            )
            return self._pointnav(best_point, stop=False)

        print("No dense peaks or frontiers found during collaborative exploration, stopping.")
        self._termination_hint = "no_candidate_goals"
        self._publish_shared_state(force=True, finished=True, success=False)
        return self._stop_action

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if self._collab_suppressed_candidates > 0:
            self._collab_conflict_replans += 1

        self._publish_shared_state(force=False)
        policy_info.update(
            {
                "collab_agent_id": self._collab_agent_id,
                "collab_claimed_cell_xy": self._collab_claimed_cell_xy,
                "collab_suppressed_candidates": float(self._collab_suppressed_candidates),
                "collab_total_suppressed_candidates": float(self._collab_total_suppressed_candidates),
                "collab_conflict_replans": float(self._collab_conflict_replans),
                "collab_peer_success_abort": float(self._collab_peer_success_abort),
                "collab_assigned_region_id": self._collab_assigned_region_id,
                "collab_assigned_cell_xy": self._collab_assigned_cell_xy,
                "collab_intent_text": self._collab_intent_text,
                "collab_assignment_source": self._collab_assignment_source,
            }
        )

        if not self._visualize:
            self._collab_suppressed_candidates = 0
            return policy_info

        policy_info["render_below_images"].extend(
            [
                "collab_agent",
                "collab_suppressed",
                "collab_claim_cell",
                "collab_assignment",
            ]
        )
        policy_info["collab_agent"] = f"collab_agent: {self._collab_agent_id}"
        policy_info["collab_suppressed"] = (
            f"collab_suppressed: {self._collab_suppressed_candidates} "
            f"(total {self._collab_total_suppressed_candidates})"
        )
        policy_info["collab_claim_cell"] = f"collab_claim_cell: {self._collab_claimed_cell_xy}"
        policy_info["collab_assignment"] = (
            f"collab_assignment: {self._collab_assigned_region_id} "
            f"| {self._collab_assignment_source} | {self._collab_intent_text}"
        )
        self._collab_suppressed_candidates = 0
        return policy_info


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]
