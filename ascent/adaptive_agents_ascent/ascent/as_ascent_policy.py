from __future__ import annotations

import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from torch import Tensor
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.policy.habitat_policies import VLFMPolicyConfig
from vlfm.utils.geometry_utils import rho_theta

from ascent.as_ascent_config import AscentMultiAgentPolicyConfig
from ascent.as_ascent_coordinator import compute_frontier_utility
from ascent.as_ascent_coordinator import frontier_overlap_penalty
from ascent.as_ascent_coordinator import should_trigger_fine_grained
from ascent.as_ascent_logging import ASEpisodeLogger
from ascent.as_ascent_shared_memory import SharedMemoryManager
from ascent.as_ascent_structures import ConfirmationState
from ascent.as_ascent_structures import CoordinatorState
from ascent.as_ascent_structures import FrontierRecord
from ascent.as_ascent_structures import FrontierStatus
from ascent.as_ascent_structures import IntentMode
from ascent.as_ascent_structures import IntentRecord
from ascent.as_ascent_structures import SubagentState
from ascent.ascent_policy import Ascent_Policy
from ascent.pointnav_policy import WrappedPointNavResNetPolicy
from ascent.utils import build_non_coco_caption
from ascent.utils import get_action_tensor
from ascent.utils import infer_dataset_type_from_path
from ascent.utils import resolve_object_goal_names
from constants import LOOK_DOWN
from constants import LOOK_UP
from constants import MOVE_FORWARD
from constants import STOP
from constants import TURN_LEFT
from constants import TURN_RIGHT


NOOP_ACTION_ID = 6


@baseline_registry.register_policy
class AscentMultiAgentPolicy(Ascent_Policy):
    @classmethod
    def from_config(
        cls,
        config,
        *args_unused: Any,
        **kwargs_unused: Any,
    ) -> "AscentMultiAgentPolicy":
        rl_policy_config = config.habitat_baselines.rl.policy
        policy_config: AscentMultiAgentPolicyConfig = rl_policy_config.main_agent
        kwargs = {k: policy_config[k] for k in VLFMPolicyConfig.kwaarg_names}
        sim_sensors_cfg = config.habitat.simulator.agents.agent_0.sim_sensors
        kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]
        kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
        kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
        kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
        kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width
        kwargs["image_height"] = sim_sensors_cfg.rgb_sensor.height
        kwargs["visualize"] = len(config.habitat_baselines.eval.video_option) > 0
        kwargs["action_space"] = args_unused[-1]
        kwargs["dataset_type"] = infer_dataset_type_from_path(config.habitat.dataset.data_path)
        kwargs["full_config"] = config
        kwargs["num_envs"] = config.habitat_baselines.num_environments
        kwargs["as_ascent_config"] = policy_config.as_ascent
        kwargs["llm_config"] = policy_config.as_ascent.llm
        return cls(**kwargs)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._as_ascent_cfg = kwargs.pop("as_ascent_config")
        super().__init__(*args, **kwargs)
        self._joint_action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([6.0, 6.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._sub_pointnav_policy: List[WrappedPointNavResNetPolicy] = [
            WrappedPointNavResNetPolicy(
                kwargs["pointnav_policy_path"],
                original_config=kwargs["full_config"],
                agent_name="agent_1",
                device=self.device,
            )
            for _ in range(self._num_envs)
        ]
        self._sub_last_goal: List[np.ndarray] = [np.zeros(2) for _ in range(self._num_envs)]
        self._sub_called_stop: List[bool] = [False] * self._num_envs
        self._sub_history_action: List[List[int]] = [[] for _ in range(self._num_envs)]
        self._sub_pitch_angle: List[int] = [0] * self._num_envs
        self._sub_observations_cache: List[dict] = [{} for _ in range(self._num_envs)]
        self._sub_active: List[bool] = [
            self._as_ascent_cfg.coordination_mode == "naive_shared"
            for _ in range(self._num_envs)
        ]
        self._sub_frontier: List[np.ndarray] = [np.array([]) for _ in range(self._num_envs)]
        self._shared_memory: List[SharedMemoryManager] = [
            SharedMemoryManager(
                reservation_ttl=self._as_ascent_cfg.spawn.reservation_ttl,
                candidate_merge_radius=self._as_ascent_cfg.evidence.candidate_merge_radius,
            )
            for _ in range(self._num_envs)
        ]
        self._coordinator_state: List[CoordinatorState] = [
            CoordinatorState.SINGLE_AGENT_NORMAL for _ in range(self._num_envs)
        ]
        self._episode_loggers: List[Optional[ASEpisodeLogger]] = [None] * self._num_envs
        self._spawn_count: List[int] = [0] * self._num_envs
        self._reclaim_count: List[int] = [0] * self._num_envs
        self._cross_floor_transition_count: List[int] = [0] * self._num_envs
        self._cross_floor_success_count: List[int] = [0] * self._num_envs
        self._first_detection_step: List[int] = [-1] * self._num_envs
        self._verification_step: List[int] = [-1] * self._num_envs
        self._false_positive_abort_count: List[int] = [0] * self._num_envs
        self._overlap_ratio: List[float] = [0.0] * self._num_envs
        self._last_floor_index: List[int] = [0] * self._num_envs
        self._verification_source: List[str] = ["" for _ in range(self._num_envs)]
        self._pending_spawn: List[bool] = [False] * self._num_envs
        self._pending_reclaim: List[bool] = [False] * self._num_envs
        self._sub_last_progress_step: List[int] = [0] * self._num_envs
        self._sub_commit_until_step: List[int] = [-1] * self._num_envs
        self._sub_fail_count: List[int] = [0] * self._num_envs
        self._sub_last_target_distance: List[float] = [float("inf")] * self._num_envs
        self._sub_state: List[SubagentState] = [SubagentState.DORMANT] * self._num_envs
        self._last_policy_info: List[Dict[str, Any]] = [{} for _ in range(self._num_envs)]
        self._last_episode_logger_dir: List[str] = [""] * self._num_envs
        self._sub_traj_points: List[List[np.ndarray]] = [[] for _ in range(self._num_envs)]
        self._event_history: List[List[str]] = [[] for _ in range(self._num_envs)]
        self._last_spawn_reason: List[str] = ["not_evaluated"] * self._num_envs
        self._video_layout_version: str = "as_ascent_v2"
        self._last_main_frontier_id: List[str] = [""] * self._num_envs
        self._pointcloud_panel_cache: List[np.ndarray] = [np.array([]) for _ in range(self._num_envs)]
        self._pointcloud_panel_step: List[int] = [-9999] * self._num_envs
        self._pointcloud_refresh_interval: int = 5
        self._last_assignment_rejection_signature: List[str] = [""] * self._num_envs

    @property
    def should_load_agent_state(self) -> bool:
        return False

    @property
    def policy_action_space(self):
        return self._joint_action_space

    def eval(self) -> None:
        return None

    def train(self) -> None:
        return None

    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> PolicyActionData:
        del deterministic
        current_episodes_info = kwargs.get("current_episodes_info")
        main_obs = observations.to_tree()
        agent_0_obs = _extract_prefixed_obs(main_obs, "agent_0_")
        agent_1_obs = _extract_prefixed_obs(main_obs, "agent_1_")

        object_ids = agent_0_obs[ObjectGoalSensor.cls_uuid]
        goal_names = resolve_object_goal_names(
            self._dataset_type,
            object_ids,
            current_episodes_info=current_episodes_info,
        )
        agent_0_obs[ObjectGoalSensor.cls_uuid] = goal_names
        agent_1_obs[ObjectGoalSensor.cls_uuid] = goal_names
        non_coco_caption = build_non_coco_caption(self._dataset_type, goal_names)
        self._non_coco_caption = non_coco_caption or ""

        self._pre_step(agent_0_obs, masks)
        for env in range(self._num_envs):
            self._maybe_reset_multi_agent(env, masks, current_episodes_info)
            self._cache_secondary_observations(agent_1_obs, env)

        img_height, img_width = agent_0_obs["rgb"].shape[1:3]
        self._map_controller._update_object_map_with_stair_and_person(
            img_height,
            img_width,
            self._observations_cache,
            self._non_coco_caption,
            self._num_steps,
            self._try_to_navigate,
        )
        self.red_semantic_pred_list = []
        self.seg_map_color_list = []
        for env in range(self._num_envs):
            rgb = agent_0_obs["rgb"][env : env + 1].float()
            depth = agent_0_obs["depth"][env : env + 1].float()
            with torch.no_grad():
                red_semantic_pred = self.red_sem_pred(rgb, depth).squeeze().cpu().numpy().astype(np.uint8)
            self.red_semantic_pred_list.append(red_semantic_pred)
            self.seg_map_color_list.append(np.zeros((*red_semantic_pred.shape, 3), dtype=np.uint8))

        self._map_controller._update_obstacle_map(self._observations_cache, self.red_semantic_pred_list, self._pitch_angle)
        self._map_controller._update_value_map(self._observations_cache)
        self._map_controller._update_distance_on_object_map(self._observations_cache)

        joint_actions: List[Tensor] = []
        for env in range(self._num_envs):
            for stair_event in self._map_controller.pop_stair_events(env):
                stair_payload = dict(stair_event)
                event_type = str(stair_payload.pop("event_type", "stair_event"))
                if event_type == "stair_success":
                    from_floor = int(stair_payload.get("from_floor", self._last_floor_index[env]))
                    to_floor = int(stair_payload.get("to_floor", self._map_controller._cur_floor_index[env]))
                    self._cross_floor_transition_count[env] += 1
                    self._cross_floor_success_count[env] += 1
                    self._last_floor_index[env] = to_floor
                    self._log_event(
                        env,
                        "cross_floor_transition",
                        {
                            "from_floor": from_floor,
                            "to_floor": to_floor,
                            "success_rate": self._cross_floor_success_count[env]
                            / max(1, self._cross_floor_transition_count[env]),
                        },
                    )
                elif event_type == "stair_disabled":
                    self._last_floor_index[env] = int(self._map_controller._cur_floor_index[env])
                self._log_event(env, event_type, stair_payload)
            self._ingest_shared_memory(env)
            self._update_target_evidence(env, "main", self._observations_cache[env], self._map_controller.target_detection_list[env])
            if self._sub_active[env] or self._pending_spawn[env]:
                sub_target_detections = self._detect_sub_target(env)
                self._update_target_evidence(env, "sub", self._sub_observations_cache[env], sub_target_detections)

            spawn_signal, reclaim_signal = self._update_subagent_state(env)
            main_action = self._compute_main_action(agent_0_obs, env, masks)
            sub_action = self._compute_sub_action(env, masks)
            joint_actions.append(
                torch.tensor(
                    [[float(main_action), float(sub_action), spawn_signal, reclaim_signal]],
                    dtype=torch.float32,
                    device=masks.device,
                )
            )
            self._last_policy_info[env] = self._build_policy_info(env)
            self._observations_cache[env] = {}
            self._sub_observations_cache[env] = {}
            self._did_reset[env] = False

        action_tensor = torch.cat(joint_actions, dim=0)
        return PolicyActionData(
            actions=action_tensor,
            take_actions=action_tensor,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=self._last_policy_info,
        )

    def get_extra(self, action_data: PolicyActionData, infos, dones) -> List[Dict[str, float]]:
        del infos
        del dones
        return action_data.policy_info or []

    def _maybe_reset_multi_agent(self, env: int, masks: Tensor, current_episodes_info) -> None:
        if masks[env][0] != 0:
            return
        # `_pre_step` already reset the single-agent ASCENT state and set the target object.
        # Do not call `_reset` again here, otherwise the target object is cleared.
        self._sub_pointnav_policy[env].reset()
        self._sub_last_goal[env] = np.zeros(2)
        self._sub_called_stop[env] = False
        self._sub_history_action[env].clear()
        self._sub_pitch_angle[env] = 0
        self._sub_active[env] = self._as_ascent_cfg.coordination_mode == "naive_shared"
        self._shared_memory[env].reset()
        self._coordinator_state[env] = CoordinatorState.SINGLE_AGENT_NORMAL
        self._spawn_count[env] = 0
        self._reclaim_count[env] = 0
        self._cross_floor_transition_count[env] = 0
        self._cross_floor_success_count[env] = 0
        self._first_detection_step[env] = -1
        self._verification_step[env] = -1
        self._false_positive_abort_count[env] = 0
        self._overlap_ratio[env] = 0.0
        self._last_floor_index[env] = 0
        self._verification_source[env] = ""
        self._pending_spawn[env] = False
        self._pending_reclaim[env] = False
        self._sub_last_progress_step[env] = 0
        self._sub_commit_until_step[env] = -1
        self._sub_fail_count[env] = 0
        self._sub_last_target_distance[env] = float("inf")
        self._sub_state[env] = SubagentState.DORMANT
        self._sub_traj_points[env] = []
        self._event_history[env] = []
        self._last_spawn_reason[env] = "reset"
        self._last_main_frontier_id[env] = ""
        self._pointcloud_panel_cache[env] = np.array([])
        self._pointcloud_panel_step[env] = -9999
        self._last_assignment_rejection_signature[env] = ""
        self._episode_loggers[env] = self._build_logger(env, current_episodes_info)
        self._last_episode_logger_dir[env] = str(self._episode_loggers[env].run_dir)
        self._log_event(env, "llm_backend", self.llm_planner.get_logging_info())

    def _build_logger(self, env: int, current_episodes_info) -> ASEpisodeLogger:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        scene_id = "unknown_scene"
        episode_id = "unknown_episode"
        goal_name = self._map_controller._target_object[env].split("|")[0] if self._map_controller._target_object[env] else "unknown_goal"
        if current_episodes_info is not None and env < len(current_episodes_info):
            episode = current_episodes_info[env]
            scene_id = str(episode.scene_id).split("/")[-1].split(".")[0]
            episode_id = str(episode.episode_id)
            run_id = f"{run_id}_{episode_id}"
        run_dir_template = self._as_ascent_cfg.logging.log_dir_template
        replacements = {
            "{run_id}": run_id,
            "{scene_id}": _slugify(scene_id),
            "{episode_id}": _slugify(episode_id),
            "{goal_name}": _slugify(goal_name),
            "__RUN_ID__": run_id,
            "__SCENE_ID__": _slugify(scene_id),
            "__EPISODE_ID__": _slugify(episode_id),
            "__GOAL_NAME__": _slugify(goal_name),
        }
        for src, dst in replacements.items():
            run_dir_template = run_dir_template.replace(src, dst)
        run_dir = run_dir_template
        return ASEpisodeLogger(run_dir)

    def _detect_sub_target(self, env: int) -> Any:
        if not self._sub_observations_cache[env]:
            return None
        cache = self._sub_observations_cache[env]
        rgb = cache["rgb"]
        height, width = rgb.shape[:2]
        target_detections, _, _ = self._map_controller._get_object_detections_with_stair_and_person(
            rgb,
            self._non_coco_caption,
            env,
        )
        if getattr(target_detections, "num_detections", 0) <= 0:
            return None
        depth = cache["depth"]
        for idx in range(len(target_detections.logits)):
            target_bbox_denorm = target_detections.boxes[idx] * np.array([width, height, width, height])
            target_mask = self._map_controller._mobile_sam.segment_bbox(rgb, target_bbox_denorm.tolist())
            target_mask = (target_mask > 0).astype(np.uint8)
            self._map_controller._object_map[env].update_map(
                self._map_controller._target_object[env],
                depth,
                target_mask,
                cache["tf_camera_to_episodic"],
                cache["min_depth"],
                cache["max_depth"],
                cache["fx"],
                cache["fy"],
            )
        self._map_controller._object_map[env].update_explored(
            cache["tf_camera_to_episodic"],
            cache["max_depth"],
            cache["camera_fov"],
        )
        return target_detections

    def _cache_secondary_observations(self, observations: Dict[str, Tensor], env: int) -> None:
        rgb = observations["rgb"][env].cpu().numpy()
        depth = observations["depth"][env].cpu().numpy()
        x, y = observations["gps"][env].cpu().numpy()
        camera_yaw = observations["compass"][env].cpu().item()
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]
        camera_pitch = np.radians(-self._sub_pitch_angle[env])
        tf_camera_to_episodic = self._build_tf(camera_position, camera_yaw, camera_pitch)
        self._sub_observations_cache[env] = {
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "tf_camera_to_episodic": tf_camera_to_episodic,
            "rgb": rgb,
            "depth": depth.reshape(depth.shape[:2]),
            "min_depth": self._min_depth,
            "max_depth": self._max_depth,
            "fx": self._fx,
            "fy": self._fy,
            "camera_fov": self._camera_fov,
            "habitat_start_yaw": observations["heading"][env].item(),
            "nav_rgb": torch.unsqueeze(observations["rgb"][env], dim=0),
            "nav_depth": torch.unsqueeze(observations["depth"][env], dim=0),
        }
        if self._sub_active[env]:
            self._sub_traj_points[env].append(robot_xy.copy())

    def _build_tf(self, camera_position: np.ndarray, camera_yaw: float, camera_pitch: float):
        from ascent.utils import xyz_yaw_pitch_roll_to_tf_matrix

        return xyz_yaw_pitch_roll_to_tf_matrix(camera_position, camera_yaw, camera_pitch, 0.0)

    def _ingest_shared_memory(self, env: int) -> None:
        current_floor = int(self._map_controller._cur_floor_index[env])
        floor_prior = 0.0
        if hasattr(self.llm_planner, "get_floor_probabilities"):
            try:
                probs = self.llm_planner.get_floor_probabilities(
                    self.floor_probabilities_df,
                    self._map_controller._target_object[env].split("|")[0],
                    max(1, self._map_controller.floor_num[env]),
                )
                floor_prior = float(probs.get(current_floor + 1, 0.0)) / 100.0
            except Exception:
                floor_prior = 0.0

        self._shared_memory[env].update_floor(
            current_floor,
            visited_ratio=float(self._map_controller._obstacle_map[env].explored_area.mean()),
            frontier_count=int(len(self._map_controller._obstacle_map[env].frontiers)),
            stairs_up=bool(self._map_controller._obstacle_map[env]._has_up_stair),
            stairs_down=bool(self._map_controller._obstacle_map[env]._has_down_stair),
            floor_prior_for_goal=floor_prior,
            last_updated_step=int(self._num_steps[env]),
            agent_presence_state={
                "main": f"floor_{current_floor}",
                "sub": f"floor_{current_floor}" if self._sub_active[env] else "dormant",
            },
        )

        frontiers = np.array(self._map_controller._obstacle_map[env].frontiers)
        frontier_records = self._build_frontier_records(env, frontiers, current_floor, floor_prior)
        self._shared_memory[env].release_expired_reservations(int(self._num_steps[env]))
        self._shared_memory[env].refresh_frontiers(current_floor, frontier_records, int(self._num_steps[env]))
        self._sync_main_frontier_assignment(env, current_floor)

    def _sync_main_frontier_assignment(self, env: int, current_floor: int) -> None:
        snapshot = self._shared_memory[env].snapshot
        for frontier in snapshot.frontiers.values():
            if frontier.assigned_agent == "main":
                frontier.assigned_agent = None

        if self.cur_frontier[env].size != 2:
            self._last_main_frontier_id[env] = ""
            return

        frontier_xy = self.cur_frontier[env]
        best_frontier = None
        best_distance = float("inf")
        for frontier in snapshot.frontiers.values():
            if frontier.floor_id != current_floor:
                continue
            distance = float(np.linalg.norm(np.array(frontier.centroid, dtype=np.float32) - frontier_xy))
            if distance < best_distance:
                best_frontier = frontier
                best_distance = distance

        if best_frontier is None or best_distance > 0.75:
            self._last_main_frontier_id[env] = ""
            return

        best_frontier.assigned_agent = "main"
        if self._last_main_frontier_id[env] != best_frontier.frontier_id:
            self._log_event(
                env,
                "frontier_assignment",
                {
                    "agent": "main",
                    "frontier_id": best_frontier.frontier_id,
                    "utility": best_frontier.utility.total_utility,
                },
            )
        self._last_main_frontier_id[env] = best_frontier.frontier_id

    def _build_frontier_records(
        self, env: int, frontiers: np.ndarray, current_floor: int, floor_prior: float
    ) -> List[FrontierRecord]:
        if frontiers.size == 0 or np.array_equal(frontiers, np.zeros((1, 2))):
            return []
        sorted_pts, sorted_values = self._map_controller._value_map[env].sort_waypoints(frontiers, 0.5)
        value_lookup = {tuple(point.tolist()): float(value) for point, value in zip(sorted_pts, sorted_values)}
        records: List[FrontierRecord] = []
        main_xy = self._observations_cache[env]["robot_xy"]
        sub_xy = self._sub_observations_cache[env].get("robot_xy") if self._sub_observations_cache[env] else None
        assigned_main = None
        if self.cur_frontier[env].size == 2:
            assigned_main = self.cur_frontier[env]
        frontier_id_counts: Dict[str, int] = {}
        for idx, frontier in enumerate(frontiers):
            centroid = (float(frontier[0]), float(frontier[1]))
            frontier_id = self._make_frontier_id(env, current_floor, centroid)
            if frontier_id in frontier_id_counts:
                frontier_id_counts[frontier_id] += 1
                frontier_id = f"{frontier_id}_{frontier_id_counts[frontier_id]}"
            else:
                frontier_id_counts[frontier_id] = 0
            record = FrontierRecord(
                frontier_id=frontier_id,
                floor_id=current_floor,
                centroid=centroid,
                cluster_extent=0.5,
                semantic_score=value_lookup.get(tuple(frontier.tolist()), 0.0),
                geometry_gain=float(np.linalg.norm(frontier - main_xy)),
                path_cost_from_main=float(np.linalg.norm(frontier - main_xy)),
                path_cost_from_sub=float(np.linalg.norm(frontier - sub_xy)) if sub_xy is not None else float(np.linalg.norm(frontier - main_xy)),
            )
            breakdown = compute_frontier_utility(
                record,
                floor_prior=floor_prior,
                area_prior=0.0,
                overlap_penalty=frontier_overlap_penalty(record.centroid, assigned_main),
                blacklist_penalty=1.0 if self._shared_memory[env].is_blacklisted(record.frontier_id) else 0.0,
                stale_penalty=0.0,
                weights=self._as_ascent_cfg.utility,
                for_subagent=False,
            )
            record.utility = breakdown
            records.append(record)
        return records

    def _make_frontier_id(self, env: int, current_floor: int, centroid: Tuple[float, float]) -> str:
        qx = int(round(centroid[0] * 4.0))
        qy = int(round(centroid[1] * 4.0))
        return f"f_{env}_{current_floor}_{qx}_{qy}"

    def _update_target_evidence(
        self,
        env: int,
        agent_id: str,
        observation_cache: Dict[str, Any],
        target_detections: Any,
    ) -> None:
        if target_detections is None:
            return
        if getattr(target_detections, "num_detections", 0) <= 0:
            return
        if self._first_detection_step[env] < 0:
            self._first_detection_step[env] = int(self._num_steps[env])
        logits = target_detections.logits
        if hasattr(logits, "detach"):
            logits_score = float(logits.max().detach().cpu().item()) if logits.numel() > 0 else 0.5
        else:
            logits_score = float(np.max(logits)) if len(logits) > 0 else 0.5
        score = logits_score
        score = max(score, float(self._map_controller._blip_cosine[env]))
        pose = observation_cache["tf_camera_to_episodic"][:3, 3]
        previous_state = None
        previous_candidate_id = None
        for candidate_record in self._shared_memory[env].snapshot.target_evidence.values():
            if candidate_record.floor_id != int(self._map_controller._cur_floor_index[env]):
                continue
            dx = float(candidate_record.pose_estimate[0] - pose[0])
            dz = float(candidate_record.pose_estimate[2] - pose[2])
            if (dx * dx + dz * dz) ** 0.5 <= self._as_ascent_cfg.evidence.candidate_merge_radius:
                previous_state = candidate_record.confirmation_state.value
                previous_candidate_id = candidate_record.candidate_id
                break
        candidate = self._shared_memory[env].upsert_candidate(
            floor_id=int(self._map_controller._cur_floor_index[env]),
            pose_estimate=pose,
            agent_id=agent_id,
            score=score,
            step=int(self._num_steps[env]),
            viewpoint_cluster=f"{agent_id}_{round(float(pose[0]), 1)}_{round(float(pose[2]), 1)}",
            likely_threshold=self._as_ascent_cfg.evidence.likely_threshold,
            verified_threshold=self._as_ascent_cfg.evidence.verified_threshold,
        )
        if previous_state != candidate.confirmation_state.value:
            self._log_event(
                env,
                "verification_state_change",
                {
                    "candidate_id": candidate.candidate_id,
                    "previous": previous_state or "none",
                    "current": candidate.confirmation_state.value,
                    "agent": agent_id,
                    "fused_score": candidate.fused_score,
                },
            )
        if candidate.confirmation_state == ConfirmationState.LIKELY:
            self._coordinator_state[env] = CoordinatorState.VERIFICATION_PRIORITY
        if (
            candidate.confirmation_state == ConfirmationState.VERIFIED
            and self._passes_stop_guard(candidate)
        ):
            if self._verification_step[env] < 0:
                self._verification_step[env] = int(self._num_steps[env])
                self._verification_source[env] = agent_id
                self._log_event(
                    env,
                    "verification_state_change",
                    {
                        "candidate_id": candidate.candidate_id,
                        "previous": previous_state or candidate.confirmation_state.value,
                        "current": "verified_stop_guard_passed",
                        "agent": agent_id,
                        "fused_score": candidate.fused_score,
                    },
                )
            self._map_controller._double_check_goal[env] = True
        elif candidate.confirmation_state == ConfirmationState.VERIFIED:
            candidate.confirmation_state = ConfirmationState.LIKELY
            candidate.rejection_reason = "stop_guard_failed"
            self._false_positive_abort_count[env] += 1
            self._coordinator_state[env] = CoordinatorState.VERIFICATION_PRIORITY
            self._log_event(
                env,
                "verification_state_change",
                {
                    "candidate_id": candidate.candidate_id if previous_candidate_id is None else previous_candidate_id,
                    "previous": ConfirmationState.VERIFIED.value,
                    "current": ConfirmationState.LIKELY.value,
                    "agent": agent_id,
                    "reason": "stop_guard_failed",
                },
            )

    def _passes_stop_guard(self, candidate) -> bool:
        if candidate.supporting_frames < self._as_ascent_cfg.evidence.stability_window:
            return False
        if len(candidate.supporting_agents) == 0:
            return False
        if len(candidate.viewpoint_clusters) < self._as_ascent_cfg.evidence.min_viewpoint_clusters:
            return False
        return True

    def _update_subagent_state(self, env: int) -> Tuple[float, float]:
        frontiers = list(self._shared_memory[env].snapshot.frontiers.values())
        if self._map_controller._cur_floor_index[env] != self._last_floor_index[env]:
            self._last_floor_index[env] = int(self._map_controller._cur_floor_index[env])

        if self._as_ascent_cfg.coordination_mode == "single_agent":
            self._sub_active[env] = False
            self._coordinator_state[env] = CoordinatorState.SINGLE_AGENT_NORMAL
            self._last_spawn_reason[env] = "single_agent_mode"
            return 0.0, 0.0

        if self._as_ascent_cfg.coordination_mode == "naive_shared":
            if self._num_steps[env] == 0:
                self._sub_active[env] = True
                self._sub_state[env] = SubagentState.SPAWNED
                self._spawn_count[env] += 1
                if self._sub_observations_cache[env]:
                    self._sub_traj_points[env].append(self._sub_observations_cache[env]["robot_xy"].copy())
                self._log_event(env, "spawn", {"step": self._num_steps[env], "mode": "naive_shared"})
                self._last_spawn_reason[env] = "naive_shared"
                return 1.0, 0.0
            self._coordinator_state[env] = CoordinatorState.DUAL_AGENT_ACTIVE
            self._last_spawn_reason[env] = "naive_shared_active"
            return 0.0, 0.0

        triggered, clusters = should_trigger_fine_grained(
            frontiers,
            self._observations_cache[env]["robot_xy"],
            self._as_ascent_cfg.trigger.distance_threshold,
            self._as_ascent_cfg.trigger.min_frontiers,
            self._as_ascent_cfg.trigger.min_cluster_count,
            self._as_ascent_cfg.trigger.top2_gap_threshold,
        )
        self._coordinator_state[env] = (
            CoordinatorState.FINE_GRAINED_TRIGGERED if triggered else CoordinatorState.SINGLE_AGENT_NORMAL
        )
        main_candidate = self._select_main_frontier_candidate(env)
        sub_eval = self._evaluate_sub_frontiers(
            env,
            main_candidate=main_candidate,
            include_current_assignment=False,
        )
        best_candidate = self._shared_memory[env].best_candidate()
        can_use_verifier = (
            self._as_ascent_cfg.spawn.allow_verifier_without_search
            and best_candidate is not None
            and best_candidate.confirmation_state == ConfirmationState.LIKELY
        )
        if (
            self._as_ascent_cfg.spawn.allow_verifier_priority_spawn
            and not self._sub_active[env]
            and best_candidate is not None
            and best_candidate.confirmation_state in (
                ConfirmationState.LIKELY,
                ConfirmationState.VERIFIED,
            )
            and (self.max_episode_steps - self._num_steps[env]) >= self._as_ascent_cfg.spawn.min_remaining_steps
            and (
                self._verification_step[env] < 0
                or len(best_candidate.supporting_agents)
                < self._as_ascent_cfg.evidence.preferred_supporting_agents
            )
        ):
            self._sub_active[env] = True
            self._sub_state[env] = SubagentState.VERIFYING
            self._sub_frontier[env] = np.array(best_candidate.pose_estimate[:2], dtype=np.float32)
            self._sub_commit_until_step[env] = (
                self._num_steps[env] + self._as_ascent_cfg.commit.commit_window_steps
            )
            self._sub_last_progress_step[env] = int(self._num_steps[env])
            self._sub_last_target_distance[env] = float("inf")
            self._spawn_count[env] += 1
            self._coordinator_state[env] = CoordinatorState.VERIFICATION_PRIORITY
            self._log_event(
                env,
                "spawn",
                {
                    "step": self._num_steps[env],
                    "reason": "verifier_priority_spawn",
                    "candidate_id": best_candidate.candidate_id,
                    "candidate_score": best_candidate.fused_score,
                },
            )
            self._last_spawn_reason[env] = "verifier_priority_spawn"
            return 1.0, 0.0
        if (
            (triggered or self._should_opportunistically_spawn(env, frontiers, main_candidate, sub_eval))
            and not self._sub_active[env]
            and (self.max_episode_steps - self._num_steps[env]) >= self._as_ascent_cfg.spawn.min_remaining_steps
            and self._verification_step[env] < 0
        ):
            spawn_allowed, spawn_reason = self._should_spawn_subagent(
                env,
                triggered=triggered,
                clusters=clusters,
                main_candidate=main_candidate,
                sub_eval=sub_eval,
            )
            if not spawn_allowed and not triggered and self._should_opportunistically_spawn(env, frontiers, main_candidate, sub_eval):
                spawn_allowed = True
                spawn_reason = "opportunistic_spawn"
            if spawn_allowed:
                self._sub_active[env] = True
                self._sub_state[env] = SubagentState.SPAWNED
                self._sub_commit_until_step[env] = (
                    self._num_steps[env] + self._as_ascent_cfg.commit.commit_window_steps
                )
                self._sub_last_progress_step[env] = int(self._num_steps[env])
                self._sub_last_target_distance[env] = float("inf")
                self._spawn_count[env] += 1
                self._coordinator_state[env] = CoordinatorState.DUAL_AGENT_ACTIVE
                self._log_event(
                    env,
                    "spawn",
                    {
                        "step": self._num_steps[env],
                        "reason": spawn_reason,
                        "main_frontier_id": main_candidate.frontier_id if main_candidate is not None else "",
                        "sub_frontier_id": sub_eval["best"].frontier_id if sub_eval["best"] is not None else "",
                    },
                )
                self._last_spawn_reason[env] = spawn_reason
                return 1.0, 0.0
            self._coordinator_state[env] = CoordinatorState.SPAWN_EVALUATION
            self._last_spawn_reason[env] = spawn_reason
            self._record_assignment_rejections(
                env,
                sub_eval["rejection_counts"],
                context="spawn",
                reason=spawn_reason,
            )
            if can_use_verifier:
                self._coordinator_state[env] = CoordinatorState.VERIFICATION_PRIORITY
                self._last_spawn_reason[env] = f"{spawn_reason}_verifier_only"
            return 0.0, 0.0

        if self._sub_active[env]:
            if best_candidate is not None and best_candidate.confirmation_state == ConfirmationState.VERIFIED:
                if len(best_candidate.supporting_agents) < self._as_ascent_cfg.evidence.preferred_supporting_agents:
                    self._coordinator_state[env] = CoordinatorState.VERIFICATION_PRIORITY
                    self._last_spawn_reason[env] = "await_cross_agent_verification"
                    return 0.0, 0.0
                return self._reclaim_subagent(env, "verified_target")
            if best_candidate is not None and best_candidate.confirmation_state == ConfirmationState.LIKELY:
                self._coordinator_state[env] = CoordinatorState.VERIFICATION_PRIORITY
                self._last_spawn_reason[env] = "verifier_priority"
                return 0.0, 0.0

            if (
                self._sub_frontier[env].size == 2
                and (sub_eval["current_valid"] or self._sub_can_keep_coordinate_target(env, main_candidate))
                and self._sub_should_hold_current_frontier(env)
            ):
                self._coordinator_state[env] = CoordinatorState.DUAL_AGENT_ACTIVE
                self._last_spawn_reason[env] = "hold_current_frontier"
                return 0.0, 0.0

            current_overlap = sub_eval["best_overlap"]
            current_reason = None
            if sub_eval["best"] is None:
                current_reason = "no_disjoint_frontier"
            elif current_overlap > self._as_ascent_cfg.spawn.max_overlap:
                current_reason = "overlap_too_high"
            elif (
                self._sub_frontier[env].size == 2
                and self._num_steps[env] <= self._sub_commit_until_step[env]
                and sub_eval["current_valid"]
            ):
                self._coordinator_state[env] = CoordinatorState.DUAL_AGENT_ACTIVE
                self._last_spawn_reason[env] = "commit_window_active"
                return 0.0, 0.0

            if current_reason is not None:
                self._record_assignment_rejections(
                    env,
                    sub_eval["rejection_counts"],
                    context="active_subagent",
                    reason=current_reason,
                )
                if can_use_verifier:
                    self._sub_state[env] = SubagentState.VERIFYING
                    self._coordinator_state[env] = CoordinatorState.VERIFICATION_PRIORITY
                    self._last_spawn_reason[env] = f"{current_reason}_verifier_only"
                    return 0.0, 0.0
                return self._reclaim_subagent(env, current_reason)
            self._coordinator_state[env] = CoordinatorState.DUAL_AGENT_ACTIVE
            self._last_spawn_reason[env] = "subagent_active"
        elif not triggered:
            self._last_spawn_reason[env] = "fine_grained_not_triggered"
        elif self._verification_step[env] >= 0:
            self._coordinator_state[env] = CoordinatorState.VERIFICATION_PRIORITY
            self._last_spawn_reason[env] = "verified_target_exists"
        else:
            self._coordinator_state[env] = CoordinatorState.SPAWN_EVALUATION
            self._last_spawn_reason[env] = "budget_or_state_blocked"
        return 0.0, 0.0

    def _compute_main_action(self, observations: Dict[str, Tensor], env: int, masks: Tensor) -> int:
        robot_xy = self._observations_cache[env]["robot_xy"]
        goal = self._get_target_object_location(robot_xy, env)
        if goal is None:
            if not self._map_controller._done_initializing[env]:
                action = self._initialize(env, masks)
                action_value = int(action.detach().cpu().numpy()[0].item())
                self._num_steps[env] += 1
                self._map_controller._obstacle_map[env]._floor_num_steps += 1
                return action_value
            action = self._explore(observations, env, masks)
        else:
            self._try_to_navigate[env] = True
            action = self._navigate(observations, goal[:2], stop=True, env=env, ori_masks=masks)
        action_value = int(action.detach().cpu().numpy()[0].item())
        self._num_steps[env] += 1
        self._map_controller._obstacle_map[env]._floor_num_steps += 1
        if action_value == 0 and self._verification_step[env] < 0:
            action_value = MOVE_FORWARD
        return action_value

    def _compute_sub_action(self, env: int, masks: Tensor) -> int:
        if not self._sub_active[env]:
            return NOOP_ACTION_ID
        if self._as_ascent_cfg.coordination_mode == "naive_shared":
            return self._compute_naive_sub_action(env, masks)
        best_candidate = self._shared_memory[env].best_candidate()
        if best_candidate is not None and best_candidate.confirmation_state in (
            ConfirmationState.LIKELY,
            ConfirmationState.VERIFIED,
        ):
            self._sub_state[env] = SubagentState.VERIFYING
            self._sub_frontier[env] = np.array(best_candidate.pose_estimate[:2], dtype=np.float32)
            self._sub_last_progress_step[env] = int(self._num_steps[env])
            self._sub_last_target_distance[env] = float("inf")
        else:
            current_frontier = self._current_sub_frontier_record(env)
            if (
                (current_frontier is not None or self._sub_can_keep_coordinate_target(env, self._select_main_frontier_candidate(env)))
                and self._sub_should_hold_current_frontier(env)
            ):
                frontier = current_frontier or FrontierRecord(
                    frontier_id="sub_coordinate_hold",
                    floor_id=int(self._map_controller._cur_floor_index[env]),
                    centroid=(float(self._sub_frontier[env][0]), float(self._sub_frontier[env][1])),
                    cluster_extent=0.5,
                    semantic_score=0.0,
                    geometry_gain=0.0,
                    path_cost_from_main=0.0,
                    path_cost_from_sub=0.0,
                )
            else:
                frontier = self._choose_sub_frontier(env)
                if frontier is None:
                    self._sub_state[env] = SubagentState.DORMANT
                    return NOOP_ACTION_ID
                self._sub_frontier[env] = np.array(frontier.centroid, dtype=np.float32)
                self._sub_last_progress_step[env] = int(self._num_steps[env])
                self._sub_last_target_distance[env] = float("inf")
            self._overlap_ratio[env] = frontier_overlap_penalty(
                frontier.centroid,
                self.cur_frontier[env] if self.cur_frontier[env].size == 2 else None,
            )
            self._sub_state[env] = (
                SubagentState.COMMITTED
                if self._num_steps[env] <= self._sub_commit_until_step[env]
                else SubagentState.EXPLORING
            )
        self._update_subagent_progress(env)
        action = self._sub_pointnav(env, self._sub_frontier[env], masks)
        action_value = int(action.detach().cpu().numpy()[0].item())
        self._sub_history_action[env].append(action_value)
        return action_value

    def _compute_naive_sub_action(self, env: int, masks: Tensor) -> int:
        frontier = self._choose_naive_sub_frontier(env)
        if frontier is None:
            self._sub_state[env] = SubagentState.DORMANT
            return NOOP_ACTION_ID
        self._sub_frontier[env] = np.array(frontier.centroid, dtype=np.float32)
        self._sub_state[env] = SubagentState.EXPLORING
        action = self._sub_pointnav(env, self._sub_frontier[env], masks)
        action_value = int(action.detach().cpu().numpy()[0].item())
        self._sub_history_action[env].append(action_value)
        return action_value

    def _choose_naive_sub_frontier(self, env: int) -> Optional[FrontierRecord]:
        candidates = [
            frontier
            for frontier in self._shared_memory[env].snapshot.frontiers.values()
            if frontier.status in (FrontierStatus.FREE, FrontierStatus.RESERVED, FrontierStatus.EXPLORING)
            and frontier.status not in (FrontierStatus.STALE, FrontierStatus.UNREACHABLE)
            and not self._shared_memory[env].is_blacklisted(frontier.frontier_id)
        ]
        if not candidates:
            return None
        best_frontier = None
        best_score = -1e9
        for frontier in candidates:
            breakdown = compute_frontier_utility(
                frontier,
                floor_prior=self._shared_memory[env].snapshot.floors[frontier.floor_id].floor_prior_for_goal,
                area_prior=0.0,
                overlap_penalty=0.0,
                blacklist_penalty=0.0,
                stale_penalty=0.0,
                weights=self._as_ascent_cfg.utility,
                for_subagent=True,
            )
            if breakdown.total_utility > best_score:
                best_score = breakdown.total_utility
                best_frontier = frontier
        if best_frontier is not None:
            self._log_event(
                env,
                "frontier_assignment",
                {
                    "agent": "sub_naive",
                    "frontier_id": best_frontier.frontier_id,
                    "utility": best_score,
                },
            )
        return best_frontier

    def _current_sub_frontier_record(self, env: int) -> Optional[FrontierRecord]:
        if self._sub_frontier[env].size != 2:
            return None
        snapshot = self._shared_memory[env].snapshot
        best_match = None
        best_distance = float("inf")
        for frontier in snapshot.frontiers.values():
            distance = float(
                np.linalg.norm(np.array(frontier.centroid, dtype=np.float32) - self._sub_frontier[env])
            )
            if distance < best_distance:
                best_distance = distance
                best_match = frontier
        if best_match is None or best_distance > 0.75:
            return None
        return best_match

    def _sub_should_hold_current_frontier(self, env: int) -> bool:
        if self._sub_frontier[env].size != 2:
            return False
        if self._num_steps[env] <= self._sub_commit_until_step[env]:
            return True
        return (
            int(self._num_steps[env]) - int(self._sub_last_progress_step[env])
            <= self._as_ascent_cfg.commit.progress_patience
        )

    def _sub_can_keep_coordinate_target(
        self,
        env: int,
        main_candidate: Optional[FrontierRecord],
    ) -> bool:
        if self._sub_frontier[env].size != 2:
            return False
        main_assignment = main_candidate.centroid if main_candidate is not None else None
        overlap = frontier_overlap_penalty(
            self._sub_frontier[env],
            main_assignment,
        )
        if overlap > self._as_ascent_cfg.spawn.max_overlap:
            return False
        return True

    def _update_subagent_progress(self, env: int) -> None:
        if self._sub_frontier[env].size != 2 or not self._sub_observations_cache[env]:
            return
        robot_xy = self._sub_observations_cache[env]["robot_xy"]
        current_distance = float(np.linalg.norm(robot_xy - self._sub_frontier[env]))
        if self._sub_last_target_distance[env] == float("inf"):
            self._sub_last_target_distance[env] = current_distance
            self._sub_last_progress_step[env] = int(self._num_steps[env])
            return
        if current_distance < self._sub_last_target_distance[env] - 0.10:
            self._sub_last_progress_step[env] = int(self._num_steps[env])
        self._sub_last_target_distance[env] = current_distance

    def _choose_sub_frontier(self, env: int) -> Optional[FrontierRecord]:
        main_candidate = self._select_main_frontier_candidate(env)
        eval_result = self._evaluate_sub_frontiers(
            env,
            main_candidate=main_candidate,
            include_current_assignment=True,
        )
        best_frontier = eval_result["best"]
        if best_frontier is not None:
            self._shared_memory[env].reserve_frontier(
                best_frontier.frontier_id,
                "sub",
                self._num_steps[env],
            )
            self._log_event(
                env,
                "frontier_assignment",
                {
                    "agent": "sub",
                    "frontier_id": best_frontier.frontier_id,
                    "utility": best_frontier.utility.total_utility,
                },
            )
            self._last_assignment_rejection_signature[env] = ""
            return best_frontier
        self._record_assignment_rejections(
            env,
            eval_result["rejection_counts"],
            context="sub_assignment",
            reason="no_disjoint_frontier",
        )
        return best_frontier

    def _select_main_frontier_candidate(self, env: int) -> Optional[FrontierRecord]:
        candidates = [
            frontier
            for frontier in self._shared_memory[env].snapshot.frontiers.values()
            if frontier.status in (FrontierStatus.FREE, FrontierStatus.RESERVED, FrontierStatus.EXPLORING)
            and frontier.status not in (FrontierStatus.STALE, FrontierStatus.UNREACHABLE)
            and not self._shared_memory[env].is_blacklisted(frontier.frontier_id)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.utility.total_utility)

    def _evaluate_sub_frontiers(
        self,
        env: int,
        *,
        main_candidate: Optional[FrontierRecord],
        include_current_assignment: bool,
    ) -> Dict[str, Any]:
        rejection_counts: Dict[str, int] = {}
        main_assignment = main_candidate.centroid if main_candidate is not None else None
        current_frontier_id = None
        for frontier in self._shared_memory[env].snapshot.frontiers.values():
            if frontier.assigned_agent == "sub":
                current_frontier_id = frontier.frontier_id
                break
        best_frontier: Optional[FrontierRecord] = None
        best_score = -1e9
        best_overlap = 0.0
        current_valid = False
        for frontier in self._shared_memory[env].snapshot.frontiers.values():
            allow_current = include_current_assignment and frontier.frontier_id == current_frontier_id
            allowed, reason, overlap, breakdown = self._assess_sub_frontier_candidate(
                env,
                frontier,
                main_candidate=main_candidate,
                allow_current_assignment=allow_current,
            )
            if breakdown is not None:
                self._shared_memory[env].set_frontier_utility(frontier.frontier_id, breakdown)
            if not allowed:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            if allow_current:
                current_valid = True
            if breakdown is not None and breakdown.total_utility > best_score:
                best_score = breakdown.total_utility
                best_frontier = frontier
                best_overlap = overlap
        return {
            "best": best_frontier,
            "best_overlap": best_overlap,
            "rejection_counts": rejection_counts,
            "current_valid": current_valid,
            "main_candidate": main_candidate,
        }

    def _assess_sub_frontier_candidate(
        self,
        env: int,
        frontier: FrontierRecord,
        *,
        main_candidate: Optional[FrontierRecord],
        allow_current_assignment: bool,
    ) -> Tuple[bool, str, float, Optional[Any]]:
        if frontier.status not in (FrontierStatus.FREE, FrontierStatus.RESERVED, FrontierStatus.EXPLORING):
            return False, "status_blocked", 0.0, None
        if frontier.status in (FrontierStatus.STALE, FrontierStatus.UNREACHABLE):
            return False, "status_blocked", 0.0, None
        if self._shared_memory[env].is_blacklisted(frontier.frontier_id):
            return False, "blacklisted", 0.0, None
        if frontier.assigned_agent == "main":
            return False, "assigned_to_main", 0.0, None
        if main_candidate is not None and frontier.frontier_id == main_candidate.frontier_id:
            return False, "same_frontier", 0.0, None
        if frontier.reserved_by not in (None, "sub") and not allow_current_assignment:
            return False, "reserved_elsewhere", 0.0, None

        overlap = frontier_overlap_penalty(
            frontier.centroid,
            main_candidate.centroid if main_candidate is not None else None,
        )
        if (
            self._as_ascent_cfg.spawn.require_disjoint_frontier
            and overlap > self._as_ascent_cfg.spawn.max_overlap
        ):
            return False, "overlap_too_high", overlap, None

        breakdown = compute_frontier_utility(
            frontier,
            floor_prior=self._shared_memory[env].snapshot.floors[frontier.floor_id].floor_prior_for_goal,
            area_prior=0.0,
            overlap_penalty=overlap,
            blacklist_penalty=1.0 if self._shared_memory[env].is_blacklisted(frontier.frontier_id) else 0.0,
            stale_penalty=1.0 if frontier.status == FrontierStatus.STALE else 0.0,
            weights=self._as_ascent_cfg.utility,
            for_subagent=True,
        )
        return True, "", overlap, breakdown

    def _should_spawn_subagent(
        self,
        env: int,
        *,
        triggered: bool,
        clusters: List[Any],
        main_candidate: Optional[FrontierRecord],
        sub_eval: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if not triggered:
            return False, "fine_grained_not_triggered"
        if len(clusters) < 2:
            return False, "no_cluster_separation"
        if main_candidate is None:
            return False, "no_main_frontier"
        best_sub = sub_eval["best"]
        if best_sub is None:
            return False, "no_disjoint_frontier"
        sub_utility = best_sub.utility.total_utility
        main_utility = main_candidate.utility.total_utility
        if sub_utility < (main_utility - self._as_ascent_cfg.spawn.spawn_margin):
            return False, "spawn_margin_not_met"
        max_allowed_path = max(1e-3, main_candidate.path_cost_from_main) * (
            1.0 + self._as_ascent_cfg.spawn.max_spawn_overhead_ratio
        )
        if best_sub.path_cost_from_sub > max_allowed_path:
            return False, "budget_blocked"
        return True, "triggered_spawn"

    def _should_opportunistically_spawn(
        self,
        env: int,
        frontiers: List[FrontierRecord],
        main_candidate: Optional[FrontierRecord],
        sub_eval: Dict[str, Any],
    ) -> bool:
        if self._sub_active[env]:
            return False
        if len(frontiers) < 2:
            return False
        if main_candidate is None or sub_eval["best"] is None:
            return False
        best_sub = sub_eval["best"]
        return (
            best_sub.utility.total_utility > 0.0
            and sub_eval["best_overlap"] <= self._as_ascent_cfg.spawn.max_overlap
        )

    def _release_sub_assignments(self, env: int, status: FrontierStatus) -> None:
        for frontier in self._shared_memory[env].snapshot.frontiers.values():
            if frontier.assigned_agent == "sub" or frontier.reserved_by == "sub":
                self._shared_memory[env].release_frontier(frontier.frontier_id, status)
        self._sub_frontier[env] = np.array([])

    def _reclaim_subagent(self, env: int, reason: str) -> Tuple[float, float]:
        self._release_sub_assignments(env, FrontierStatus.FREE)
        self._sub_active[env] = False
        self._sub_state[env] = SubagentState.RECLAIMED
        self._sub_last_target_distance[env] = float("inf")
        self._reclaim_count[env] += 1
        self._coordinator_state[env] = CoordinatorState.RECLAIM_SUBAGENT
        self._log_event(env, "reclaim", {"reason": reason})
        self._last_spawn_reason[env] = f"reclaimed_{reason}"
        return 0.0, 1.0

    def _record_assignment_rejections(
        self,
        env: int,
        rejection_counts: Dict[str, int],
        *,
        context: str,
        reason: str,
    ) -> None:
        if not self._as_ascent_cfg.logging.record_assignment_rejections:
            return
        if not rejection_counts and not reason:
            return
        signature = f"{context}|{reason}|{sorted(rejection_counts.items())}"
        if signature == self._last_assignment_rejection_signature[env]:
            return
        self._last_assignment_rejection_signature[env] = signature
        self._log_event(
            env,
            "assignment_rejected",
            {
                "context": context,
                "reason": reason,
                "rejection_counts": rejection_counts,
            },
        )

    def _action_name(self, action_id: Optional[int]) -> str:
        action_map = {
            STOP: "stop",
            MOVE_FORWARD: "forward",
            TURN_LEFT: "turn_left",
            TURN_RIGHT: "turn_right",
            LOOK_UP: "look_up",
            LOOK_DOWN: "look_down",
            NOOP_ACTION_ID: "noop",
        }
        if action_id is None:
            return "-"
        return action_map.get(int(action_id), str(action_id))

    def _sub_pointnav(self, env: int, goal: np.ndarray, masks: Tensor) -> Tensor:
        del masks
        cache = self._sub_observations_cache[env]
        pointnav_policy = self._sub_pointnav_policy[env]
        pointnav_device = pointnav_policy.device
        reset_mask = torch.tensor([self._num_steps[env] != 0], dtype=torch.bool, device=pointnav_device)
        if not np.array_equal(goal, self._sub_last_goal[env]):
            if np.linalg.norm(goal - self._sub_last_goal[env]) > 0.1:
                pointnav_policy.reset()
                reset_mask = torch.zeros_like(reset_mask)
            self._sub_last_goal[env] = goal
        rho, theta = rho_theta(cache["robot_xy"], cache["robot_heading"], goal)
        obs_pointnav = {
            "depth": image_resize(
                cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": torch.tensor(
                [[rho, theta]], device=pointnav_device, dtype=torch.float32
            ),
        }
        return pointnav_policy.act(obs_pointnav, reset_mask, deterministic=True)

    def _build_policy_info(self, env: int) -> Dict[str, Any]:
        info = self._get_policy_info(self._map_controller.target_detection_list[env], env)
        verified_delta = -1
        if self._first_detection_step[env] >= 0 and self._verification_step[env] >= 0:
            verified_delta = self._verification_step[env] - self._first_detection_step[env]
        self._overlap_ratio[env] = frontier_overlap_penalty(
            self._sub_frontier[env] if self._sub_frontier[env].size == 2 else None,
            self.cur_frontier[env] if self.cur_frontier[env].size == 2 else None,
        ) if self._sub_frontier[env].size == 2 and self.cur_frontier[env].size == 2 else 0.0
        snapshot = self._shared_memory[env].snapshot
        current_floor = int(self._map_controller._cur_floor_index[env])
        best_candidate = self._shared_memory[env].best_candidate()
        current_sub_floor = current_floor
        sub_assignment = next(
            (frontier for frontier in snapshot.frontiers.values() if frontier.assigned_agent == "sub"),
            None,
        )
        if sub_assignment is not None:
            current_sub_floor = sub_assignment.floor_id
        cooperative_map = self._build_cooperative_topdown_map(env)
        coordination_heatmap = self._build_coordination_heatmap(env)
        pointcloud_3d_map = self._build_pointcloud_panel(env)
        info.update(
            {
                "spawn_count": float(self._spawn_count[env]),
                "reclaim_count": float(self._reclaim_count[env]),
                "first_detection_step": float(self._first_detection_step[env]),
                "verification_step": float(self._verification_step[env]),
                "overlap_ratio": float(self._overlap_ratio[env]),
                "cross_floor_transition_count": float(self._cross_floor_transition_count[env]),
                "cross_floor_success_rate": float(
                    self._cross_floor_success_count[env] / max(1, self._cross_floor_transition_count[env])
                ),
                "suspect_to_verified_time": float(verified_delta),
                "false_positive_abort_count": float(self._false_positive_abort_count[env]),
                "subagent_state": str(self._sub_state[env]),
                "coordinator_state": str(self._coordinator_state[env]),
                "verification_source": self._verification_source[env],
                "episode_log_dir": self._last_episode_logger_dir[env],
                "video_layout_version": self._video_layout_version,
                "main_agent_label": "Main Agent",
                "subagent_label": "Subagent",
                "main_status": self._format_agent_status(env, main_agent=True),
                "sub_status": self._format_agent_status(env, main_agent=False),
                "event_timeline": self._event_history[env][-6:],
                "spawn_reason": self._last_spawn_reason[env],
                "current_floor_main": current_floor,
                "current_floor_sub": current_sub_floor,
                "main_action": self._action_name(self.history_action[env][-1] if self.history_action[env] else None),
                "sub_action": self._action_name(self._sub_history_action[env][-1] if self._sub_history_action[env] else None),
                "verification_state": best_candidate.confirmation_state.value if best_candidate is not None else "none",
                "verification_score": float(best_candidate.fused_score) if best_candidate is not None else 0.0,
                "active_candidate_id": best_candidate.candidate_id if best_candidate is not None else "",
                "main_traj_xy": [pt.tolist() for pt in self._map_controller._obstacle_map[env]._camera_positions],
                "sub_traj_xy": [pt.tolist() for pt in self._sub_traj_points[env]],
                "main_heading_rad": float(self._observations_cache[env].get("robot_heading", 0.0)),
                "sub_heading_rad": float(self._sub_observations_cache[env].get("robot_heading", 0.0)) if self._sub_observations_cache[env] else 0.0,
                "main_position_xy": self._observations_cache[env].get("robot_xy", np.zeros(2)).tolist(),
                "sub_position_xy": self._sub_observations_cache[env].get("robot_xy", np.zeros(2)).tolist() if self._sub_observations_cache[env] else [],
                "frontier_registry": self._serialize_frontiers_for_vis(env),
                "candidate_states": self._serialize_candidates_for_vis(env),
                "cooperative_topdown_map": cooperative_map,
                "coordination_heatmap": coordination_heatmap,
                "pointcloud_3d_map": pointcloud_3d_map,
            }
        )
        return info

    def _log_event(self, env: int, event_type: str, payload: Dict[str, Any]) -> None:
        logger = self._episode_loggers[env]
        if logger is None:
            return
        logger.log_event(event_type, payload)
        rendered = f"[{int(self._num_steps[env]):03d}] {event_type}: {payload}"
        self._event_history[env].append(rendered)
        self._event_history[env] = self._event_history[env][-20:]

    def _format_agent_status(self, env: int, *, main_agent: bool) -> str:
        if main_agent:
            current_frontier = np.array2string(self.cur_frontier[env], precision=2) if self.cur_frontier[env].size else "-"
            last_action = self._action_name(self.history_action[env][-1] if self.history_action[env] else None)
            return (
                f"M | mode={self._coordinator_state[env]} | floor={self._map_controller._cur_floor_index[env]} "
                f"| frontier={current_frontier} | act={last_action} | verify={self._verification_source[env] or '-'}"
            )
        target_frontier = np.array2string(self._sub_frontier[env], precision=2) if self._sub_frontier[env].size else "-"
        last_action = self._action_name(self._sub_history_action[env][-1] if self._sub_history_action[env] else None)
        return (
            f"S | state={self._sub_state[env]} | floor={self._map_controller._cur_floor_index[env]} "
            f"| frontier={target_frontier} | act={last_action} | reason={self._last_spawn_reason[env]} | commit={self._sub_commit_until_step[env]}"
        )

    def _serialize_frontiers_for_vis(self, env: int) -> List[Dict[str, Any]]:
        frontiers = []
        for frontier in self._shared_memory[env].snapshot.frontiers.values():
            frontiers.append(
                {
                    "frontier_id": frontier.frontier_id,
                    "floor_id": frontier.floor_id,
                    "centroid": list(frontier.centroid),
                    "status": str(frontier.status),
                    "assigned_agent": frontier.assigned_agent or "",
                    "reserved_by": frontier.reserved_by or "",
                    "utility": frontier.utility.to_dict(),
                }
            )
        return frontiers

    def _serialize_candidates_for_vis(self, env: int) -> List[Dict[str, Any]]:
        candidates = []
        for candidate in self._shared_memory[env].snapshot.target_evidence.values():
            candidates.append(candidate.to_dict())
        return candidates

    def _build_cooperative_topdown_map(self, env: int) -> np.ndarray:
        base = cv2.cvtColor(self._map_controller._obstacle_map[env].visualize(), cv2.COLOR_BGR2RGB)
        drawer = self._map_controller._obstacle_map[env]._traj_vis
        main_color = (18, 181, 203)
        sub_color = (255, 138, 61)
        self._draw_path(base, self._map_controller._obstacle_map[env]._camera_positions, drawer, main_color, dashed=False)
        self._draw_path(base, self._sub_traj_points[env], drawer, sub_color, dashed=True)
        if len(self._map_controller._obstacle_map[env]._camera_positions) > 0:
            self._draw_agent_marker(
                base,
                self._map_controller._obstacle_map[env]._camera_positions[-1],
                float(self._observations_cache[env].get("robot_heading", 0.0)),
                drawer,
                main_color,
                "M",
                filled=True,
            )
        if len(self._sub_traj_points[env]) > 0:
            self._draw_agent_marker(
                base,
                self._sub_traj_points[env][-1],
                float(self._sub_observations_cache[env].get("robot_heading", 0.0)) if self._sub_observations_cache[env] else 0.0,
                drawer,
                sub_color,
                "S",
                filled=False,
            )
        for frontier in self._shared_memory[env].snapshot.frontiers.values():
            px = drawer._metric_to_pixel(np.array(frontier.centroid, dtype=np.float32))
            status = str(frontier.status)
            if frontier.assigned_agent == "sub":
                cv2.circle(base, tuple(px[::-1]), 10, sub_color, 2)
            elif frontier.assigned_agent == "main":
                cv2.circle(base, tuple(px[::-1]), 10, main_color, -1)
            elif status in ("stale", "unreachable"):
                cv2.drawMarker(base, tuple(px[::-1]), (120, 80, 80), cv2.MARKER_TILTED_CROSS, 12, 2)
            elif frontier.reserved_by:
                cv2.rectangle(base, (px[1] - 8, px[0] - 8), (px[1] + 8, px[0] + 8), (160, 60, 160), 2)
        focused = self._focus_map_on_interest(
            base,
            drawer,
            self._collect_interest_points(env, include_frontiers=True),
            margin=84,
            min_crop_size=220,
        )
        legend_items = [
            ("M current / path", main_color),
            ("S current / path", sub_color),
            ("frontier", (66, 84, 180)),
            ("reserved", (160, 60, 160)),
            ("stale", (120, 80, 80)),
        ]
        return self._append_side_legend(
            focused,
            [item[0] for item in legend_items],
            main_color,
            sub_color,
        )

    def _build_coordination_heatmap(self, env: int) -> np.ndarray:
        obstacle_vis = cv2.cvtColor(self._map_controller._obstacle_map[env].visualize(), cv2.COLOR_BGR2RGB)
        base = cv2.addWeighted(obstacle_vis, 0.18, np.full_like(obstacle_vis, 255), 0.82, 0.0)
        drawer = self._map_controller._obstacle_map[env]._traj_vis
        frontiers = list(self._shared_memory[env].snapshot.frontiers.values())
        values = [frontier.utility.total_utility for frontier in frontiers] or [0.0]
        min_v, max_v = min(values), max(values)
        spread = max(max_v - min_v, 1e-6)
        for frontier in frontiers:
            norm = (frontier.utility.total_utility - min_v) / spread
            color = self._heat_color(norm)
            px = drawer._metric_to_pixel(np.array(frontier.centroid, dtype=np.float32))
            cv2.circle(base, tuple(px[::-1]), 14, color, -1)
            cv2.circle(base, tuple(px[::-1]), 14, (255, 255, 255), 1)
            if frontier.status == FrontierStatus.STALE:
                cv2.drawMarker(base, tuple(px[::-1]), (120, 80, 80), cv2.MARKER_TILTED_CROSS, 14, 2)
            if frontier.assigned_agent == "main":
                cv2.circle(base, tuple(px[::-1]), 18, (18, 181, 203), 2)
            if frontier.assigned_agent == "sub":
                cv2.circle(base, tuple(px[::-1]), 18, (255, 138, 61), 2)
        focused = self._focus_map_on_interest(
            base,
            drawer,
            self._collect_interest_points(env, include_frontiers=True),
            margin=110,
            min_crop_size=240,
        )
        utility_lines = self._build_utility_lines(env)
        return self._append_text_panel(focused, utility_lines[:10], panel_width=244)

    def _build_pointcloud_panel(self, env: int) -> np.ndarray:
        from ascent.visualization_3d import render_pointcloud_panel

        if (
            self._pointcloud_panel_cache[env].size > 0
            and int(self._num_steps[env]) - self._pointcloud_panel_step[env] < self._pointcloud_refresh_interval
        ):
            return self._pointcloud_panel_cache[env]

        target_cloud = np.array([])
        if self._map_controller._object_map[env].has_object(self._map_controller._target_object[env]):
            target_cloud = self._map_controller._object_map[env].get_target_cloud(self._map_controller._target_object[env])
        other_clouds = []
        for name, cloud in self._map_controller._object_map[env].clouds.items():
            if name == self._map_controller._target_object[env]:
                continue
            if len(cloud) > 0:
                other_clouds.append(cloud[:, :3])
        other_cloud = np.concatenate(other_clouds, axis=0) if other_clouds else np.array([])
        stair_points = self._build_stair_landmarks(env)
        panel = render_pointcloud_panel(
            main_traj=np.array(self._map_controller._obstacle_map[env]._camera_positions),
            sub_traj=np.array(self._sub_traj_points[env]),
            target_cloud=target_cloud[:, :3] if len(target_cloud) > 0 else np.array([]),
            stair_cloud=stair_points,
            other_cloud=other_cloud,
            main_current=np.array(self._map_controller._obstacle_map[env]._camera_positions[-1]) if self._map_controller._obstacle_map[env]._camera_positions else np.array([]),
            sub_current=np.array(self._sub_traj_points[env][-1]) if self._sub_traj_points[env] else np.array([]),
        )
        self._pointcloud_panel_cache[env] = panel
        self._pointcloud_panel_step[env] = int(self._num_steps[env])
        return panel

    def _build_stair_landmarks(self, env: int) -> np.ndarray:
        stair_points = []
        for stair_frontiers in [self._map_controller._obstacle_map[env]._up_stair_frontiers, self._map_controller._obstacle_map[env]._down_stair_frontiers]:
            if isinstance(stair_frontiers, np.ndarray) and stair_frontiers.size > 0:
                for point in np.atleast_2d(stair_frontiers):
                    stair_points.append([float(point[0]), float(point[1]), 0.0])
        return np.array(stair_points, dtype=np.float32) if stair_points else np.array([])

    def _build_utility_lines(self, env: int) -> List[str]:
        snapshot = self._shared_memory[env].snapshot
        best_candidate = self._shared_memory[env].best_candidate()
        lines = [
            f"M floor={self._map_controller._cur_floor_index[env]}",
            f"S state={self._sub_state[env]}",
            f"Reason: {self._last_spawn_reason[env]}",
            f"Cross-floor: {self._cross_floor_transition_count[env]} / {self._cross_floor_success_count[env]}",
        ]
        main_frontier = self.cur_frontier[env].tolist() if self.cur_frontier[env].size else []
        sub_frontier = self._sub_frontier[env].tolist() if self._sub_frontier[env].size else []
        lines.append(f"M frontier: {main_frontier or '-'}")
        lines.append(f"S frontier: {sub_frontier or '-'}")
        if best_candidate is not None:
            lines.append(
                f"Verify: {best_candidate.confirmation_state.value} "
                f"({best_candidate.candidate_id}, score={best_candidate.fused_score:.2f})"
            )
        if self._sub_state[env] == SubagentState.DORMANT:
            lines.append(f"No-spawn: {self._last_spawn_reason[env]}")
        ranked_frontiers = sorted(
            list(snapshot.frontiers.values()),
            key=lambda item: item.utility.total_utility,
            reverse=True,
        )
        for frontier in ranked_frontiers[:2]:
            util = frontier.utility
            lines.append(f"{frontier.frontier_id} U={util.total_utility:.2f}")
            lines.append(
                f"sem={util.semantic_score:.2f} geo={util.geometry_gain:.2f} "
                f"floor={util.floor_prior:.2f}"
            )
            lines.append(
                f"path={util.path_cost:.2f} ov={util.overlap_penalty:.2f} "
                f"stale={util.stale_penalty:.2f}"
            )
        return lines

    def _collect_interest_points(self, env: int, *, include_frontiers: bool) -> List[np.ndarray]:
        points: List[np.ndarray] = []
        for traj_point in self._map_controller._obstacle_map[env]._camera_positions[-20:]:
            points.append(np.array(traj_point[:2], dtype=np.float32))
        for traj_point in self._sub_traj_points[env][-20:]:
            points.append(np.array(traj_point[:2], dtype=np.float32))
        if self.cur_frontier[env].size == 2:
            points.append(self.cur_frontier[env].astype(np.float32))
        if self._sub_frontier[env].size == 2:
            points.append(self._sub_frontier[env].astype(np.float32))
        if include_frontiers:
            ranked_frontiers = sorted(
                list(self._shared_memory[env].snapshot.frontiers.values()),
                key=lambda item: item.utility.total_utility,
                reverse=True,
            )
            for frontier in ranked_frontiers[:4]:
                points.append(np.array(frontier.centroid, dtype=np.float32))
        return points

    def _focus_map_on_interest(
        self,
        image: np.ndarray,
        drawer: Any,
        metric_points: List[np.ndarray],
        *,
        margin: int,
        min_crop_size: int,
    ) -> np.ndarray:
        if image.size == 0 or not metric_points:
            return image
        pixel_points: List[np.ndarray] = []
        for point in metric_points:
            if point.size < 2:
                continue
            try:
                pixel_points.append(drawer._metric_to_pixel(np.array(point[:2], dtype=np.float32)))
            except Exception:
                continue
        if not pixel_points:
            return image

        pixels = np.stack(pixel_points, axis=0)
        row_min = int(np.min(pixels[:, 0])) - margin
        row_max = int(np.max(pixels[:, 0])) + margin
        col_min = int(np.min(pixels[:, 1])) - margin
        col_max = int(np.max(pixels[:, 1])) + margin

        crop_h = max(row_max - row_min, min_crop_size)
        crop_w = max(col_max - col_min, min_crop_size)
        crop_size = max(crop_h, crop_w)
        center_row = (row_min + row_max) // 2
        center_col = (col_min + col_max) // 2

        row0 = center_row - crop_size // 2
        col0 = center_col - crop_size // 2
        row1 = row0 + crop_size
        col1 = col0 + crop_size

        if row0 < 0:
            row1 -= row0
            row0 = 0
        if col0 < 0:
            col1 -= col0
            col0 = 0
        if row1 > image.shape[0]:
            row0 = max(0, row0 - (row1 - image.shape[0]))
            row1 = image.shape[0]
        if col1 > image.shape[1]:
            col0 = max(0, col0 - (col1 - image.shape[1]))
            col1 = image.shape[1]

        cropped = image[row0:row1, col0:col1]
        return cropped if cropped.size > 0 else image

    def _overlay_corner_legend(
        self,
        image: np.ndarray,
        entries: List[Tuple[str, Tuple[int, int, int]]],
        *,
        anchor: str,
    ) -> None:
        panel_w = 172
        panel_h = 18 + 20 * len(entries)
        if anchor == "top_right":
            x0 = max(8, image.shape[1] - panel_w - 8)
            y0 = 8
        else:
            x0 = 8
            y0 = 8
        overlay = image.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.88, image, 0.12, 0, image)
        cv2.rectangle(image, (x0, y0), (x0 + panel_w, y0 + panel_h), (210, 210, 210), 1)
        y = y0 + 18
        for label, color in entries:
            cv2.circle(image, (x0 + 12, y - 5), 4, color, -1)
            cv2.putText(
                image,
                label,
                (x0 + 24, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (35, 35, 35),
                1,
                cv2.LINE_AA,
            )
            y += 20

    def _overlay_info_box(
        self,
        image: np.ndarray,
        lines: List[str],
        *,
        anchor: str,
        width: int,
    ) -> None:
        line_count = max(1, min(len(lines), 10))
        box_h = 16 + 18 * line_count
        if anchor == "top_left":
            x0 = 8
            y0 = 8
        elif anchor == "bottom_left":
            x0 = 8
            y0 = max(8, image.shape[0] - box_h - 8)
        else:
            x0 = max(8, image.shape[1] - width - 8)
            y0 = 8
        overlay = image.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + box_h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.90, image, 0.10, 0, image)
        cv2.rectangle(image, (x0, y0), (x0 + width, y0 + box_h), (210, 210, 210), 1)
        y = y0 + 16
        for raw in lines[:10]:
            for line in [raw[i : i + 30] for i in range(0, len(raw), 30)][:2]:
                cv2.putText(
                    image,
                    line,
                    (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.41,
                    (30, 30, 30),
                    1,
                    cv2.LINE_AA,
                )
                y += 17
                if y > y0 + box_h - 4:
                    return

    def _draw_path(self, canvas: np.ndarray, points: List[np.ndarray], drawer: Any, color: Tuple[int, int, int], dashed: bool) -> None:
        if len(points) < 2:
            return
        for idx in range(len(points) - 1):
            if dashed and idx % 2 == 1:
                continue
            p0 = drawer._metric_to_pixel(points[idx])
            p1 = drawer._metric_to_pixel(points[idx + 1])
            cv2.line(canvas, tuple(p0[::-1]), tuple(p1[::-1]), color, 3 if not dashed else 2)

    def _draw_agent_marker(self, canvas: np.ndarray, point: np.ndarray, heading: float, drawer: Any, color: Tuple[int, int, int], label: str, filled: bool) -> None:
        px = drawer._metric_to_pixel(point)
        cv2.circle(canvas, tuple(px[::-1]), 9, color, -1 if filled else 2)
        end = (
            int(px[0] - 12 * np.cos(heading)),
            int(px[1] - 12 * np.sin(heading)),
        )
        cv2.line(canvas, tuple(px[::-1]), tuple(end[::-1]), (0, 0, 0), 2)
        cv2.putText(canvas, label, (px[1] + 8, px[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def _append_side_legend(self, image: np.ndarray, legend_lines: List[str], main_color: Tuple[int, int, int], sub_color: Tuple[int, int, int]) -> np.ndarray:
        panel = np.full((image.shape[0], 240, 3), 255, dtype=np.uint8)
        y = 40
        colors = [main_color, sub_color, (0, 0, 180), (160, 60, 160), (120, 80, 80)]
        for line, color in zip(legend_lines, colors):
            cv2.circle(panel, (20, y - 8), 7, color, -1)
            cv2.putText(panel, line, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            y += 28
        return np.hstack((image, panel))

    def _append_text_panel(self, image: np.ndarray, lines: List[str], panel_width: int = 320) -> np.ndarray:
        panel = np.full((image.shape[0], panel_width, 3), 255, dtype=np.uint8)
        y = 28
        for raw in lines[:16]:
            wrapped = [raw[i : i + 34] for i in range(0, len(raw), 34)]
            for line in wrapped:
                cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)
                y += 24
                if y >= panel.shape[0] - 10:
                    break
            if y >= panel.shape[0] - 10:
                break
        return np.hstack((image, panel))

    def _heat_color(self, norm_value: float) -> Tuple[int, int, int]:
        color = cv2.applyColorMap(np.array([[int(norm_value * 255)]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0, 0]
        return int(color[2]), int(color[1]), int(color[0])


def _extract_prefixed_obs(observations: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    for key, value in observations.items():
        if key.startswith(prefix):
            extracted[key[len(prefix) :]] = value
    return extracted


def _slugify(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )
