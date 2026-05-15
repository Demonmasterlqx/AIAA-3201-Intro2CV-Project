from vlfm.policy.itm_policy import ITMPolicyV2
from vlfm.policy.habitat_policies import HabitatMixin
from habitat_baselines.common.baseline_registry import baseline_registry
from typing import Dict, Tuple, Any, Union, List, Optional
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from habitat_baselines.common.tensor_dict import TensorDict
from depth_camera_filtering import filter_depth
import numpy as np
import torch
import cv2
import os
from constants import MPCAT40_RGB_COLORS
from torch import Tensor
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from vlfm.utils.geometry_utils import rho_theta
from vlfm.obs_transformers.utils import image_resize
from ascent.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.vlm.detections import ObjectDetections
from vlfm.policy.habitat_policies import VLFMPolicyConfig
from constants import (
    STAIR_CLASS_ID,
    STOP,
    MOVE_FORWARD,
    TURN_LEFT,
    TURN_RIGHT,
    LOOK_UP,
    LOOK_DOWN,
)
from ascent.llm_planner import Ascent_LLM_Planner
from ascent.map_controller import Map_Controller
from ascent.utils import (
    build_non_coco_caption,
    xyz_yaw_pitch_roll_to_tf_matrix,
    check_stairs_in_upper_50_percent,
    infer_dataset_type_from_path,
    load_place365_categories,
    load_floor_probabilities_by_dataset,
    extract_room_categories,
    get_action_tensor,
    load_rednet_model,
    resolve_object_goal_names,
)
from omegaconf import DictConfig

@baseline_registry.register_policy
class Ascent_Policy(HabitatMixin, ITMPolicyV2):

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> "Ascent_Policy":
        """
        只需要处理一下 policy_config 的访问方式
        """
        # 获取 policy 配置 - 如果是字典就取 main_agent，否则直接用
        rl_policy_config = config.habitat_baselines.rl.policy
        
        if "main_agent" in rl_policy_config:
            policy_config = rl_policy_config.main_agent
        else:
            policy_config = rl_policy_config
        
        # 提取参数（和 HabitatMixin.from_config 一样的逻辑）
        kwargs = {k: policy_config[k] for k in VLFMPolicyConfig.kwaarg_names}
        
        # 添加 Habitat 相关参数
        sim_sensors_cfg = config.habitat.simulator.agents.main_agent.sim_sensors
        kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]
        kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
        kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
        kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
        kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width
        kwargs["image_height"] = sim_sensors_cfg.rgb_sensor.height
        kwargs["visualize"] = len(config.habitat_baselines.eval.video_option) > 0
        # For Habitat 3.0
        kwargs["action_space"]= args_unused[-1]
        # 数据集类型
        kwargs["dataset_type"] = infer_dataset_type_from_path(config.habitat.dataset.data_path)
        
        # 添加你需要的额外参数
        kwargs["full_config"] = config
        kwargs["num_envs"] = config.habitat_baselines.num_environments
        
        return cls(**kwargs)
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        # super().__init__(*args, **kwargs)
        # 1. 结构化参数获取
        config = kwargs.get('full_config')
        
        # 从 kwargs 或 config 获取策略相关参数
        self.max_episode_steps = config.habitat.environment.max_episode_steps if config else None
        self.nearby_distance = config.habitat_baselines.rl.policy.main_agent.nearby_distance if config else None
        self.topk = config.habitat_baselines.rl.policy.main_agent.topk if config else None

        self._action_space = kwargs["action_space"]
        self._policy_info = {}
        self._pointnav_stop_radius = kwargs["pointnav_stop_radius"]
        self._visualize = kwargs["visualize"]

        # 3. 批量初始化列表和地图相关参数
        self._num_envs = kwargs['num_envs']
        self._depth_image_shape = tuple(kwargs["depth_image_shape"]) # (224, 224)
        self._num_steps: List[int] = [0] * self._num_envs
        self._did_reset: List[bool] = [False] * self._num_envs
        self._last_goal: List[np.ndarray] = [np.zeros(2) for _ in range(self._num_envs)]
        self._called_stop: List[bool] = [False] * self._num_envs
                
        self._pointnav_policy: List[WrappedPointNavResNetPolicy] = [
            WrappedPointNavResNetPolicy(kwargs["pointnav_policy_path"], original_config=config) for _ in range(self._num_envs)
        ]
        
        # BaseITMPolicy 相关属性
        self._target_object_color = (0, 255, 0)
        self._selected__frontier_color = (0, 255, 255)
        self._frontier_color = (0, 0, 255)
        self._circle_marker_thickness = 2
        self._circle_marker_radius = 5
        self._acyclic_enforcer: List[AcyclicEnforcer] = [AcyclicEnforcer() for _ in range(self._num_envs)]

        # HabitatMixin 相关属性
        self._camera_height = kwargs["camera_height"]
        self._min_depth = kwargs["min_depth"]
        self._max_depth = kwargs["max_depth"]
        camera_fov_rad = np.deg2rad(kwargs["camera_fov"])
        self._camera_fov = camera_fov_rad
        self._image_width = kwargs["image_width"]
        self._image_height = kwargs["image_height"]
        self._fx = self._fy = self._image_width / (2 * np.tan(camera_fov_rad / 2))
        self._cx, self._cy = self._image_width // 2, (config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height // 2) if config else (self._image_height // 2)
        self._dataset_type = kwargs["dataset_type"]
        self._observations_cache: List[dict] = [{} for _ in range(self._num_envs)]

        # 4. torch.device 的处理
        if torch.cuda.is_available():
            cuda_id = config.habitat_baselines.torch_gpu_id if config and hasattr(config.habitat_baselines, 'torch_gpu_id') else 0
            self.device = torch.device(f"cuda:{cuda_id}")
        else:
            self.device = torch.device("cpu")
        
        self._pitch_angle_offset = config.habitat.task.actions.look_down.tilt_angle if config and hasattr(config.habitat.task.actions.look_down, 'tilt_angle') else self.DEFAULT_PITCH_ANGLE_OFFSET
        self._stop_action = get_action_tensor(STOP, device=self.device) # 假设 get_action_tensor 已定义

        # 5. 模型加载封装
        self.red_sem_pred = load_rednet_model(model_path="pretrained_weights/rednet_semmap_mp3d_40.pth", device=self.device)
        
        # 楼梯和楼层管理相关
        main_agent_cfg = config.habitat_baselines.rl.policy.main_agent if config else None
        final_check_cfg = main_agent_cfg.final_check if main_agent_cfg is not None else None

        self._map_controller = Map_Controller(device=self.device, num_envs = kwargs['num_envs'], text_prompt=kwargs["text_prompt"],
                                              object_map_erosion_size=kwargs["object_map_erosion_size"],
                                              min_obstacle_height = kwargs["min_obstacle_height"],
                                              max_obstacle_height = kwargs["max_obstacle_height"],
                                              obstacle_map_area_threshold = kwargs["obstacle_map_area_threshold"],
                                              agent_radius = kwargs["agent_radius"],
                                              hole_area_thresh = kwargs["hole_area_thresh"],
                                              use_max_confidence = kwargs["use_max_confidence"],
                                              coco_threshold = kwargs["coco_threshold"],
                                              non_coco_threshold = kwargs["non_coco_threshold"],
                                              final_check_config=final_check_cfg)
        self._final_check_cfg = final_check_cfg
        self._frontier_cfg = self._safe_cfg_get(main_agent_cfg, "frontier", None)
        self._stair_cfg = self._safe_cfg_get(main_agent_cfg, "stair", None)
        
        self._pitch_angle: List[int] = [0] * self._num_envs

        self.all_detection_list: List[Any] = [None] * self._num_envs
        self.target_might_detected: List[bool] = [False] * self._num_envs
        self._last_frontier_distance: List[float] = [0.0] * self._num_envs
        self.red_semantic_pred_list: List[List] = [[] for _ in range(self._num_envs)]
        self.seg_map_color_list: List[List] = [[] for _ in range(self._num_envs)]
        self.history_action: List[List] = [[] for _ in range(self._num_envs)] 
        self._try_to_navigate: List[bool] = [False] * self._num_envs
        self._try_to_navigate_step: List[int] = [0] * self._num_envs
        self.min_distance_xy: List[float] = [np.inf] * self._num_envs
        self.cur_frontier: List[np.ndarray] = [np.array([]) for _ in range(self._num_envs)]
        self._explore_frontier_repeat_count: List[int] = [0] * self._num_envs
        self._last_explore_frontier: List[np.ndarray] = [np.array([]) for _ in range(self._num_envs)]
        self._last_explore_frontier_robot_xy: List[np.ndarray] = [np.array([]) for _ in range(self._num_envs)]
        self._disabled_frontier_recovery_count: List[int] = [0] * self._num_envs
        self._no_target_frontier_retry_count: List[int] = [0] * self._num_envs
        self._late_stair_waypoint_key: List[Optional[Tuple[str, float, float]]] = [None] * self._num_envs
        self._late_stair_waypoint_steps: List[int] = [0] * self._num_envs
        self._late_stair_waypoint_best_distance: List[float] = [np.inf] * self._num_envs
        self._qwen_approach_goal_cache: List[Dict[str, Any]] = [{} for _ in range(self._num_envs)]
        self._qwen_close_stall_steps: List[int] = [0] * self._num_envs
        self._qwen_close_stall_best_distance: List[float] = [np.inf] * self._num_envs
        
        
        # LLM Planner
        self.floor_probabilities_df = load_floor_probabilities_by_dataset(self._dataset_type)
        self.llm_planner = Ascent_LLM_Planner(
            num_envs=self._num_envs, 
            nearby_distance=self.nearby_distance, 
            topk=self.topk, 
            target_object_list=self._map_controller._target_object, ##
            floor_probabilities_df=self.floor_probabilities_df,
            llm_config=config.habitat_baselines.rl.policy.main_agent.llm if config else None,
        )

    @staticmethod
    def _safe_cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        try:
            return getattr(cfg, key)
        except Exception:
            return default

    def _cfg_bool(self, cfg: Any, key: str, default: bool = False) -> bool:
        value = self._safe_cfg_get(cfg, key, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _frontier_stuck_recovery_enabled(self) -> bool:
        return self._cfg_bool(self._frontier_cfg, "stuck_recovery_enabled", False)

    def _floor_budget_enabled(self) -> bool:
        return self._cfg_bool(self._stair_cfg, "floor_budget_enabled", False)

    def _qwen_approach_recovery_enabled(self) -> bool:
        return self._cfg_bool(self._final_check_cfg, "qwen_approach_recovery_enabled", False)
    def _reset(self, env: int) -> None:
        ## 核心策略状态重置
        self._pointnav_policy[env].reset()
        self._last_goal[env] = np.zeros(2)
        self._num_steps[env] = 0
        self._called_stop[env] = False
        self._did_reset[env] = True
        self._acyclic_enforcer[env] = AcyclicEnforcer() # 重新创建 Enforcer

        ## 地图和楼层管理重置
        self._map_controller.reset(env)

        ## 导航和探索状态重置
        self._try_to_navigate_step[env] = 0
        self._try_to_navigate[env] = False


        ## 爬楼梯相关状态重置
        # 防止之前episode爬楼梯异常退出
        self._pitch_angle[env] = 0

        ## 目标检测和探索相关状态重置
        # 防止识别正确之后造成误识别
        self.target_might_detected[env] = False
        self._last_frontier_distance[env] = 0
        self.all_detection_list[env] = None

        self.min_distance_xy[env] = np.inf
        self.cur_frontier[env] = np.array([])
        self._explore_frontier_repeat_count[env] = 0
        self._last_explore_frontier[env] = np.array([])
        self._last_explore_frontier_robot_xy[env] = np.array([])
        self._disabled_frontier_recovery_count[env] = 0
        self._no_target_frontier_retry_count[env] = 0
        self._late_stair_waypoint_key[env] = None
        self._late_stair_waypoint_steps[env] = 0
        self._late_stair_waypoint_best_distance[env] = np.inf
        self._qwen_approach_goal_cache[env] = {}
        self._qwen_close_stall_steps[env] = 0
        self._qwen_close_stall_best_distance[env] = np.inf

        ## 辅助缓存和历史记录重置
        self.history_action[env].clear()
                                         
    def _cache_observations(self: Union["HabitatMixin", BaseObjectNavPolicy], observations: TensorDict, env: int) -> None:
        """Caches the rgb, depth, and camera transform from the observations.

        Args:
           observations (TensorDict): The observations from the current timestep.
        """
        if len(self._observations_cache[env]) > 0:
            return
        rgb = observations["rgb"][env].cpu().numpy() ## modify this to fit on multiple environments
        depth = observations["depth"][env].cpu().numpy()
        x, y = observations["gps"][env].cpu().numpy()
        camera_yaw = observations["compass"][env].cpu().item()
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]
        camera_pitch = np.radians(-self._pitch_angle[env]) # 应该是弧度制 -
        camera_roll = 0
        tf_camera_to_episodic = xyz_yaw_pitch_roll_to_tf_matrix(camera_position, camera_yaw, camera_pitch, camera_roll)

        self._observations_cache[env] = {
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "tf_camera_to_episodic": tf_camera_to_episodic,
            "rgb": rgb,
            "depth": depth,
            "min_depth": self._min_depth,
            "max_depth": self._max_depth,
            "fx": self._fx,
            "fy": self._fy,
            "camera_fov": self._camera_fov,
            "habitat_start_yaw": observations["heading"][env].item(),
            
        }

        self._observations_cache[env]["nav_rgb"]=torch.unsqueeze(observations["rgb"][env], dim=0)
        self._observations_cache[env]["nav_depth"]=torch.unsqueeze(observations["depth"][env], dim=0)

        # if "third_rgb" in observations:
        #     self._observations_cache[env]["third_rgb"]=observations["third_rgb"][env].cpu().numpy()

    def _get_policy_info(self, detections: ObjectDetections,  env: int = 0) -> Dict[str, Any]: # seg_map_color:np.ndarray,
        """Get policy info for logging, especially, we add rednet to add seg_map"""
        # 获取目标点云信息
        if self._map_controller._object_map[env].has_object(self._map_controller._target_object[env]):
            target_point_cloud = self._map_controller._object_map[env].get_target_cloud(self._map_controller._target_object[env])
        else:
            target_point_cloud = np.array([])

        # 初始化 policy_info
        policy_info = {
            "target_object": self._map_controller._target_object[env].split("|")[0],
            "gps": str(self._observations_cache[env]["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache[env]["robot_heading"]),
            "target_detected": self._map_controller._object_map[env].has_object(self._map_controller._target_object[env]),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal[env],
            "stop_called": self._called_stop[env],
            "render_below_images": ["target_object"],
            "seg_map": self.seg_map_color_list[env], # seg_map_color,
            "num_steps": self._num_steps[env],
            "final_check_backend": self._map_controller._final_check_backend[env],
            "final_check_mode": self._map_controller._final_check_mode[env],
            "final_check_confidence": self._map_controller._final_check_confidence[env],
            "final_check_decision": self._map_controller._final_check_decision[env],
            "final_check_trigger_step": self._map_controller._final_check_trigger_step[env],
            "final_check_raw_response": self._map_controller._final_check_raw_response[env],
            "final_check_parse_error": self._map_controller._final_check_parse_error[env],
            "final_check_call_count": float(self._map_controller._final_check_call_count[env]),
            "final_check_success_count": float(self._map_controller._final_check_success_count[env]),
            "final_check_parse_error_count": float(self._map_controller._final_check_parse_error_count[env]),
            "final_check_patch_count": float(self._map_controller._final_check_patch_count[env]),
            "final_check_patch_index": float(self._map_controller._final_check_patch_index[env]),
            "final_check_patch_bbox": self._map_controller._final_check_patch_bbox[env],
            "final_check_patch_iou_cache_hit": self._map_controller._final_check_patch_iou_cache_hit[env],
            # "floor_num_steps": self._map_controller._obstacle_map[env]._floor_num_steps,
        }

        # 若不需要可视化,直接返回
        if not self._visualize:
            return policy_info

        # 处理注释深度图和 RGB 图
        annotated_depth = self._observations_cache[env]["depth"] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # 定义所有需要处理的掩膜和对应的标注帧
        annotated_rgb = self._observations_cache[env]["rgb"]
        masks_info = [
            (self._map_controller._object_masks[env], (255, 0, 0)), # object: 红色
            (self._map_controller._person_masks[env], (255, 105, 180)), # person: 粉色
            (self._map_controller._stair_masks[env], (0, 0, 255)) # stair: 蓝色
        ]

        for mask, color in masks_info:
            if mask.sum() == 0:
                continue
            
            # 统一转换为uint8类型（即使原类型已经是uint8也不影响）
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_TREE, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # 绘制到RGB和深度图
            annotated_rgb = cv2.drawContours(annotated_rgb, contours, -1, color, 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, color, 2)

        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        # # 添加第三视角 RGB
        # if "third_rgb" in self._observations_cache[env]:
        #     policy_info["third_rgb"] = self._observations_cache[env]["third_rgb"]

        # 绘制 frontiers
        policy_info["obstacle_map"] = cv2.cvtColor(self._map_controller._obstacle_map[env].visualize(), cv2.COLOR_BGR2RGB)
        policy_info["vlm_input"] = self.llm_planner.frontier_rgb_list[env]
        policy_info["vlm_response"] = self.llm_planner.vlm_response[env]
        policy_info["object_map"] = cv2.cvtColor(self._map_controller._object_map[env].visualize(), cv2.COLOR_BGR2RGB) 
        markers = []
        frontiers = self._observations_cache[env]["frontier_sensor"]
        for frontier in frontiers:
            markers.append((frontier[:2], {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }))

        if not np.array_equal(self._last_goal[env], np.zeros(2)):
            goal_color = (self._selected__frontier_color
                        if any(np.array_equal(self._last_goal[env], frontier) for frontier in frontiers)
                        else self._target_object_color)
            markers.append((self._last_goal[env], {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": goal_color,
            }))

        policy_info["value_map"] = cv2.cvtColor(
            self._map_controller._value_map[env].visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        policy_info["start_yaw"] = self._observations_cache[env]["habitat_start_yaw"]
        
        return policy_info
    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        self._policy_info = []
        self._num_envs = masks.shape[0]
        for env in range(self._num_envs):
            try:
                if not self._did_reset[env] and masks[env][0] == 0:
                    self._reset(env)
                    self._map_controller._target_object[env] = observations["objectgoal"][env]
            except IndexError as e:
                print(f"Caught an IndexError: {e}")
                print(f"self._did_reset: {self._did_reset}")
                print(f"masks: {masks}")
                raise StopIteration
            try:
                self._cache_observations(observations, env)
            except IndexError as e:
                print(e)
                print("Reached edge of map, stopping.")
                raise StopIteration
            self._policy_info.append({})

    def _get_target_object_location(self, position: np.ndarray, env: int = 0) -> Union[None, np.ndarray]:
        if self._map_controller._object_map[env].has_object(self._map_controller._target_object[env]):
            object_goal = self._map_controller._object_map[env].get_best_object(
                self._map_controller._target_object[env],
                position,
            )
            return self._get_qwen_approach_goal(object_goal, position, env)
        else:
            return None

    def _get_qwen_approach_goal(self, object_goal: np.ndarray, robot_xy: np.ndarray, env: int) -> np.ndarray:
        if not self._qwen_approach_recovery_enabled():
            self._qwen_approach_goal_cache[env] = {}
            return object_goal
        if self._map_controller._double_check_goal[env] is not True:
            self._qwen_approach_goal_cache[env] = {}
            return object_goal
        if not self._is_qwen_verified_goal(env):
            self._qwen_approach_goal_cache[env] = {}
            return object_goal

        robot_xy = np.asarray(robot_xy, dtype=np.float64)[:2]
        object_goal = np.asarray(object_goal, dtype=np.float64)[:2]
        vector_to_robot = robot_xy - object_goal
        distance = float(np.linalg.norm(vector_to_robot))
        if not np.isfinite(distance) or distance <= 0.05:
            return object_goal

        stand_off = float(getattr(self._final_check_cfg, "qwen_approach_standoff", 0.55))
        if distance <= stand_off + 0.1:
            return object_goal
        direction_to_robot = vector_to_robot / distance
        desired_goal = object_goal + direction_to_robot * stand_off
        cached_goal = self._lookup_qwen_approach_goal_cache(object_goal, desired_goal, distance, env)
        if cached_goal is not None:
            return cached_goal
        navigable_goal = self._nearest_navigable_goal(desired_goal, robot_xy, env)
        if navigable_goal is None:
            return object_goal
        if np.linalg.norm(navigable_goal - object_goal) < 0.2:
            return object_goal
        self._qwen_approach_goal_cache[env] = {
            "object_goal": object_goal.copy(),
            "approach_goal": navigable_goal.copy(),
            "desired_goal": desired_goal.copy(),
            "robot_distance": distance,
        }
        print(
            "Qwen-VL approach goal selected "
            f"(object_goal={object_goal.tolist()}, approach_goal={navigable_goal.tolist()}, "
            f"robot_distance={distance:.2f})."
        )
        return navigable_goal

    def _lookup_qwen_approach_goal_cache(
        self,
        object_goal: np.ndarray,
        desired_goal: np.ndarray,
        robot_distance: float,
        env: int,
    ) -> Optional[np.ndarray]:
        cache = self._qwen_approach_goal_cache[env]
        if not cache:
            return None
        cached_object = np.asarray(cache.get("object_goal", []), dtype=np.float64)
        cached_goal = np.asarray(cache.get("approach_goal", []), dtype=np.float64)
        cached_desired = np.asarray(cache.get("desired_goal", []), dtype=np.float64)
        if cached_object.shape != (2,) or cached_goal.shape != (2,) or cached_desired.shape != (2,):
            return None
        if np.linalg.norm(cached_object - object_goal) > 0.25:
            return None
        if abs(float(cache.get("robot_distance", np.inf)) - robot_distance) > 0.45:
            return None
        if np.linalg.norm(cached_desired - desired_goal) > 0.35:
            return None
        return cached_goal.copy()

    def _is_qwen_verified_goal(self, env: int) -> bool:
        return (
            self._qwen_approach_recovery_enabled()
            and
            self._map_controller._double_check_goal[env] is True
            and self._map_controller._final_check_mode[env] in {"qwenvl_objectpatch", "selective_hybrid"}
        )

    def _nearest_navigable_goal(self, desired_goal: np.ndarray, robot_xy: np.ndarray, env: int) -> Optional[np.ndarray]:
        obstacle_map = self._map_controller._obstacle_map[env]
        candidates = [desired_goal]
        for radius_m in (0.25, 0.5, 0.75):
            for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
                offset = np.array([np.cos(angle), np.sin(angle)]) * radius_m
                candidates.append(desired_goal + offset)

        best_goal = None
        best_score = np.inf
        for candidate in candidates:
            px = obstacle_map._xy_to_px(np.atleast_2d(candidate))[0]
            x, y = int(px[0]), int(px[1])
            if not (0 <= x < obstacle_map._navigable_map.shape[1] and 0 <= y < obstacle_map._navigable_map.shape[0]):
                continue
            if not obstacle_map._navigable_map[y, x]:
                continue
            robot_px = obstacle_map._xy_to_px(np.atleast_2d(robot_xy))[0]
            if obstacle_map.check_path_collision(robot_px, px):
                continue
            score = float(np.linalg.norm(candidate - desired_goal)) + 0.1 * float(np.linalg.norm(candidate - robot_xy))
            if score < best_score:
                best_score = score
                best_goal = candidate
        return best_goal

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # 提取 object_ids，假设 observations[ObjectGoalSensor.cls_uuid] 包含多个值
        object_ids = observations[ObjectGoalSensor.cls_uuid]

        # 转换 observations 为字典格式
        obs_dict = observations.to_tree()

        current_episodes_info = kwargs.get("current_episodes_info")
        goal_names = resolve_object_goal_names(
            self._dataset_type,
            object_ids,
            current_episodes_info=current_episodes_info,
        )
        obs_dict[ObjectGoalSensor.cls_uuid] = goal_names

        non_coco_caption = build_non_coco_caption(self._dataset_type, goal_names)
        if non_coco_caption:
            self._non_coco_caption = non_coco_caption

        self._pre_step(obs_dict, masks)
        img_height, img_width = observations["rgb"].shape[1:3]
        self._map_controller._update_object_map_with_stair_and_person(img_height, img_width, self._observations_cache,
                                                                       self._non_coco_caption, self._num_steps, self._try_to_navigate)

        self.red_semantic_pred_list = []
        self.seg_map_color_list = []

        # 语义分割预测
        for env in range(self._num_envs):
            rgb = observations["rgb"][env:env+1].float() # 优化：直接切片
            depth = observations["depth"][env:env+1].float()

            with torch.no_grad():
                red_semantic_pred = self.red_sem_pred(rgb, depth).squeeze().cpu().numpy().astype(np.uint8)
            
            self.red_semantic_pred_list.append(red_semantic_pred)
            
            color_map = np.array(MPCAT40_RGB_COLORS, dtype=np.uint8)
            seg_map_color = color_map[red_semantic_pred]
            self.seg_map_color_list.append(seg_map_color)

        self._map_controller._update_obstacle_map(self._observations_cache, self.red_semantic_pred_list, self._pitch_angle) # observations
        self._map_controller._update_value_map(self._observations_cache)
        self._map_controller._update_distance_on_object_map(self._observations_cache)
        
        pointnav_action_env_list = []

        for env in range(self._num_envs):
            robot_xy = self._observations_cache[env]["robot_xy"]
            goal = self._get_target_object_location(robot_xy, env)
            robot_px = self._map_controller._obstacle_map[env]._xy_to_px(np.atleast_2d(robot_xy))
            x, y = int(robot_px[0, 0]), int(robot_px[0, 1]) # 确保索引是整数

            mode = "unknown" # 明确初始化 mode 变量

            # 楼梯状态判断与动作逻辑
            if not self._map_controller._climb_stair_over[env]:
                if self._map_controller._reach_stair[env]:
                    if self._pitch_angle[env] == 0 and self._map_controller._climb_stair_flag[env] == 2:
                        self._pitch_angle[env] -= self._pitch_angle_offset
                        mode = "look_down"
                        pointnav_action = get_action_tensor(LOOK_DOWN, device=masks.device)
                    elif self._map_controller._climb_stair_flag[env] == 2 and self._pitch_angle[env] >= -30 and not self._map_controller._reach_stair_centroid[env]:
                        self._pitch_angle[env] -= self._pitch_angle_offset
                        mode = "look_down_twice"
                        pointnav_action = get_action_tensor(LOOK_DOWN, device=masks.device)
                    else:
                        if self._map_controller._obstacle_map[env]._climb_stair_paused_step < 30:
                            mode = "climb_stair"
                            pointnav_action = self._climb_stair(observations, env, masks)
                        else:
                            current_floor = self._map_controller._cur_floor_index[env]
                            # 楼层切换逻辑 - 上楼
                            if self._map_controller._climb_stair_flag[env] == 1:
                                next_floor = current_floor + 1
                                if next_floor < len(self._map_controller._obstacle_map_list[env]) and \
                                not self._map_controller._obstacle_map_list[env][next_floor]._done_initializing:
                                    
                                    # 保存当前楼层的上楼梯信息到新楼层的下楼梯属性
                                    self._map_controller._obstacle_map_list[env][next_floor]._down_stair_map = \
                                        self._map_controller._obstacle_map[env]._up_stair_map.copy()
                                    self._map_controller._obstacle_map_list[env][next_floor]._down_stair_start = \
                                        self._map_controller._obstacle_map[env]._up_stair_start.copy()
                                    self._map_controller._obstacle_map_list[env][next_floor]._down_stair_end = \
                                        self._map_controller._obstacle_map[env]._up_stair_end.copy()
                                    self._map_controller._obstacle_map_list[env][next_floor]._down_stair_frontiers = \
                                        self._map_controller._obstacle_map[env]._up_stair_frontiers.copy()
                                    self._map_controller._obstacle_map_list[env][next_floor]._has_down_stair = True
                                    
                                    print(f"Saved upstairs info to floor {next_floor} as downstairs")
                            
                            # 楼层切换逻辑 - 下楼
                            elif self._map_controller._climb_stair_flag[env] == 2:
                                prev_floor = current_floor - 1
                                if prev_floor >= 0 and \
                                not self._map_controller._obstacle_map_list[env][prev_floor]._done_initializing:
                                    
                                    # 保存当前楼层的下楼梯信息到新楼层的上楼梯属性
                                    self._map_controller._obstacle_map_list[env][prev_floor]._up_stair_map = \
                                        self._map_controller._obstacle_map[env]._down_stair_map.copy()
                                    self._map_controller._obstacle_map_list[env][prev_floor]._up_stair_start = \
                                        self._map_controller._obstacle_map[env]._down_stair_start.copy()
                                    self._map_controller._obstacle_map_list[env][prev_floor]._up_stair_end = \
                                        self._map_controller._obstacle_map[env]._down_stair_end.copy()
                                    self._map_controller._obstacle_map_list[env][prev_floor]._up_stair_frontiers = \
                                        self._map_controller._obstacle_map[env]._down_stair_frontiers.copy()
                                    self._map_controller._obstacle_map_list[env][prev_floor]._has_up_stair = True
                                    
                                    print(f"Saved downstairs info to floor {prev_floor} as upstairs")
                            
                            # 继续原有的初始化逻辑
                            mode = "climb_stair_initialize"
                            if self._pitch_angle[env] > 0:
                                self._pitch_angle[env] -= self._pitch_angle_offset
                                pointnav_action = get_action_tensor(LOOK_DOWN, device=masks.device)
                            elif self._pitch_angle[env] < 0:
                                self._pitch_angle[env] += self._pitch_angle_offset
                                pointnav_action = get_action_tensor(LOOK_UP, device=masks.device)
                            else:
                                self._map_controller._obstacle_map[env]._done_initializing = False # for initial floor and new floor
                                self._map_controller._initialize_step[env] = 0
                                pointnav_action = self._initialize(env, masks)
                            self._map_controller._update_stair_state(env) # 重置楼梯相关状态
                else: # 未达到楼梯，但在楼梯附近
                    # 检查是否不知不觉到了楼梯
                    if self._map_controller._climb_stair_over[env] and self._map_controller._obstacle_map[env]._down_stair_map[y,x] == 1 and len(self._map_controller._obstacle_map[env]._down_stair_frontiers) > 0 and not self._map_controller._obstacle_map_list[env][self._map_controller._cur_floor_index[env] - 1]._explored_up_stair:
                        self._map_controller._reach_stair[env] = True
                        self._map_controller._get_close_to_stair_step[env] = 0
                        self._map_controller._climb_stair_over[env] = False
                        self._map_controller._climb_stair_flag[env] = 2
                        self._map_controller._obstacle_map[env]._down_stair_start = robot_px[0].copy()
                        mode = "down_stair_detected"
                    elif self._map_controller._climb_stair_over[env] and self._map_controller._obstacle_map[env]._up_stair_map[y,x] == 1 and len(self._map_controller._obstacle_map[env]._up_stair_frontiers) > 0 and not self._map_controller._obstacle_map_list[env][self._map_controller._cur_floor_index[env] + 1]._explored_down_stair:
                        self._map_controller._reach_stair[env] = True
                        self._map_controller._get_close_to_stair_step[env] = 0
                        self._map_controller._climb_stair_over[env] = False
                        self._map_controller._climb_stair_flag[env] = 1
                        self._map_controller._obstacle_map[env]._up_stair_start = robot_px[0].copy()
                        mode = "up_stair_detected"
                    
                    if self._map_controller._obstacle_map[env]._look_for_downstair_flag:
                        mode = "look_for_downstair"
                        pointnav_action = self._look_for_downstair(observations, env, masks)
                    elif self._map_controller._climb_stair_flag[env] == 1 and self._pitch_angle[env] == 0 and np.sum(self._map_controller._obstacle_map[env]._up_stair_map) > 0:
                        min_dis_to_upstair = np.min(np.abs(np.argwhere(self._map_controller._obstacle_map[env]._up_stair_map) - robot_px[0]).sum(axis=1))
                        print(f"min_dis_to_upstair: {min_dis_to_upstair}")
                        if min_dis_to_upstair <= 2.0 * self._map_controller._obstacle_map[env].pixels_per_meter and check_stairs_in_upper_50_percent(self.red_semantic_pred_list[env] == STAIR_CLASS_ID):
                            self._pitch_angle[env] += self._pitch_angle_offset
                            mode = "look_up"
                            pointnav_action = get_action_tensor(LOOK_UP, device=masks.device)
                        else:
                            mode = "get_close_to_stair"
                            pointnav_action = self._get_close_to_stair(observations, env, masks)
                    elif self._map_controller._climb_stair_flag[env] == 2 and self._pitch_angle[env] == 0 and np.sum(self._map_controller._obstacle_map[env]._down_stair_map) > 0:
                        min_dis_to_downstair = np.min(np.abs(np.argwhere(self._map_controller._obstacle_map[env]._down_stair_map) - robot_px[0]).sum(axis=1))
                        print(f"min_dis_to_downstair: {min_dis_to_downstair}")
                        if min_dis_to_downstair <= 2.0 * self._map_controller._obstacle_map[env].pixels_per_meter:
                            self._pitch_angle[env] -= self._pitch_angle_offset
                            mode = "look_down"
                            pointnav_action = get_action_tensor(LOOK_DOWN, device=masks.device)
                        else:
                            mode = "get_close_to_stair"
                            pointnav_action = self._get_close_to_stair(observations, env, masks)
                    else:
                        mode = "get_close_to_stair"
                        pointnav_action = self._get_close_to_stair(observations, env, masks)
            else: # 非楼梯爬行状态下的通用导航和探索逻辑
                if self._pitch_angle[env] > 0:
                    mode = "look_down_back"
                    self._pitch_angle[env] -= self._pitch_angle_offset
                    pointnav_action = get_action_tensor(LOOK_DOWN, device=masks.device)
                elif self._pitch_angle[env] < 0 and not self._map_controller._obstacle_map[env]._look_for_downstair_flag:
                    mode = "look_up_back"
                    self._pitch_angle[env] += self._pitch_angle_offset
                    pointnav_action = get_action_tensor(LOOK_UP, device=masks.device)
                elif not self._map_controller._done_initializing[env]:
                    self._map_controller._obstacle_map[env]._done_initializing = True
                    mode = "initialize"
                    pointnav_action = self._initialize(env, masks)
                elif goal is None:
                    if self._map_controller._obstacle_map[env]._look_for_downstair_flag:
                        mode = "look_for_downstair"
                        pointnav_action = self._look_for_downstair(observations, env, masks)
                    else:
                        mode = "explore"
                        pointnav_action = self._explore(observations, env, masks)
                else:
                    mode = "navigate"
                    self._try_to_navigate[env] = True
                    pointnav_action = self._navigate(observations, goal[:2], stop=True, env=env, ori_masks=masks)

            if pointnav_action is None:
                action_numpy = 0
                pointnav_action = torch.tensor([[action_numpy]], dtype=torch.int64, device=masks.device)
            
            action_numpy = pointnav_action.detach().cpu().numpy()[0]
            if isinstance(action_numpy, np.ndarray) and action_numpy.size == 1: # 确保action_numpy是标量
                action_numpy = action_numpy.item() # 使用.item()获取标量值

            # Assuming 'action_numpy' is the initial action from the policy or some other source
            # Initialize pointnav_action with the original action_numpy
            pointnav_action = torch.tensor([[action_numpy]], dtype=torch.int64, device=masks.device)

            # Check history and potentially override the action
            if len(self.history_action[env]) >= 30: # Use >= to correctly handle the window size
                # Store the first action before popping to see what it was, if needed for debugging
                # old_action_in_history = self.history_action[env][0] 
                self.history_action[env].pop(0) # Remove the oldest action to maintain window size

                if all(action in [2, 3] for action in self.history_action[env]):
                    action_numpy = 1 # Force forward
                    pointnav_action = torch.tensor([[action_numpy]], dtype=torch.int64, device=masks.device)
                    print("Continuous turns to force forward.")
                elif all(action in [1] for action in self.history_action[env]):
                    action_numpy = 3 # Force turn right
                    pointnav_action = torch.tensor([[action_numpy]], dtype=torch.int64, device=masks.device)
                    print("Continuous forward to force turn right.")

            if self.max_episode_steps is not None and self._num_steps[env] == self.max_episode_steps - 1:
                if self._should_suppress_budget_stop(env, mode):
                    self._clear_qwen_verified_goal(env)
                    if action_numpy == STOP:
                        action_numpy = MOVE_FORWARD
                    pointnav_action = torch.tensor([[action_numpy]], dtype=torch.int64, device=masks.device)
                    print(
                        "Suppress force stop for distant Qwen-VL objectpatch target "
                        f"(distance={self._map_controller.cur_dis_to_goal[env]:.2f})."
                    )
                else:
                    action_numpy = STOP
                    pointnav_action = torch.tensor([[action_numpy]], dtype=torch.int64, device=masks.device)
                    print("Force stop.")

            pointnav_action_env_list.append(pointnav_action)

            # AFTER all potential overrides, append the FINAL action that will be executed
            self.history_action[env].append(action_numpy)
            
            print(f"Env: {env} | Step: {self._num_steps[env]} | Floor_step: {self._map_controller._obstacle_map[env]._floor_num_steps} | Mode: {mode} | Stair_flag: {self._map_controller._climb_stair_flag[env]} | Action: {action_numpy}")
            if not self._map_controller._climb_stair_over[env]:
                print(f"Reach_stair_centroid: {self._map_controller._reach_stair_centroid[env]}")
            
            self._num_steps[env] += 1
            self._map_controller._obstacle_map[env]._floor_num_steps += 1
            self._policy_info[env].update(self._get_policy_info(self.all_detection_list[env], env))

            self._observations_cache[env] = {}
            self._did_reset[env] = False

        pointnav_action_tensor = torch.cat(pointnav_action_env_list, dim=0)

        return PolicyActionData(
            actions=pointnav_action_tensor,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=self._policy_info,
        )

    def _should_suppress_budget_stop(self, env: int, mode: str) -> bool:
        """Avoid turning late, distant Qwen-VL confirmations into explicit false-positive stops."""
        if not bool(getattr(self._final_check_cfg, "suppress_distant_budget_stop", False)):
            return False
        if mode != "navigate":
            return False
        if self._called_stop[env]:
            return False
        if not self._is_qwen_verified_goal(env):
            return False
        distance_to_goal = float(self._map_controller.cur_dis_to_goal[env])
        if not np.isfinite(distance_to_goal):
            return True
        final_confidence = float(self._map_controller._final_check_confidence[env])
        safe_stop_radius = (
            float(getattr(self._final_check_cfg, "qwen_stalled_stop_radius", 0.82))
            if final_confidence >= float(getattr(self._final_check_cfg, "strong_confidence_threshold", 0.99))
            else float(getattr(self._final_check_cfg, "qwen_close_stop_radius", 0.60))
        )
        return distance_to_goal > safe_stop_radius

    def _clear_qwen_verified_goal(self, env: int) -> None:
        """Drop a late Qwen target lock while preserving the rest of the exploration state."""
        target_object = self._map_controller._target_object[env]
        self._map_controller._object_map[env].clear_verified_cloud(target_object)
        self._map_controller._object_map[env].clouds = {}
        self._try_to_navigate[env] = False
        self._try_to_navigate_step[env] = 0
        self._map_controller._double_check_goal[env] = False
        self._map_controller._final_check_decision[env] = False
        self._qwen_close_stall_steps[env] = 0
        self._qwen_close_stall_best_distance[env] = np.inf
        self._map_controller._object_map[env]._disabled_object_map[
            self._map_controller._object_map[env]._map == 1
        ] = 1
        self._map_controller._object_map[env]._map.fill(0)

    def _initialize(self, env: int, masks: Tensor) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        # self._map_controller._done_initializing[env] = not self._num_steps[env] < 11  # type: ignore
        if self._map_controller._initialize_step[env] > 11: # 11, 12 step is for the first step in this floor
            self._map_controller._done_initializing[env] = True
            self._map_controller._obstacle_map[env]._tight_search_thresh = False 
        else:
            self._map_controller._initialize_step[env] += 1 
        return get_action_tensor(TURN_LEFT, device=masks.device)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"], env: int, masks: Tensor) -> Tensor:
        """
        根据当前观测和环境状态执行探索行为。
        """
        initial_frontiers = self._observations_cache[env]["frontier_sensor"]
        obstacle_map = self._map_controller._obstacle_map[env]
        if self._frontier_stuck_recovery_enabled():
            frontiers = [
                f for f in initial_frontiers
                if not obstacle_map.is_frontier_disabled(f)
            ]
        else:
            frontiers = initial_frontiers

        if (self._frontier_stuck_recovery_enabled()
            and not self._has_valid_frontiers(frontiers)
            and self._has_valid_frontiers(initial_frontiers)
            and len(obstacle_map._disabled_frontiers) > 0
            and self._disabled_frontier_recovery_count[env] < 2):
            recovered_count = len(obstacle_map._disabled_frontiers)
            obstacle_map._disabled_frontiers.clear()
            obstacle_map._disabled_frontiers_px = np.array([], dtype=np.float64).reshape(0, 2)
            self._disabled_frontier_recovery_count[env] += 1
            self._explore_frontier_repeat_count[env] = 0
            self._last_explore_frontier[env] = np.array([])
            self._last_explore_frontier_robot_xy[env] = np.array([])
            frontiers = list(initial_frontiers)
            print(f"Environment {env}: Re-enabled {recovered_count} disabled frontier clusters to avoid premature no-frontier stop (recovery {self._disabled_frontier_recovery_count[env]}/2).")

        # Current floor has no valid frontier, including the Habitat [[0, 0]] sentinel.
        if not self._has_valid_frontiers(frontiers):
            # 如果还没有初始化过，并且在该楼层步数很短，并且有没探索过的高层或者低层并且没有找到对应的楼梯，如果在楼梯间且未探索完（防止卡在楼梯间），尝试重置并初始化.
            if not self._map_controller._obstacle_map[env]._reinitialize_flag and \
               self._map_controller._obstacle_map[env]._floor_num_steps < 50 and \
               ((self._map_controller._obstacle_map[env]._explored_up_stair == False and self._map_controller._obstacle_map[env]._up_stair_frontiers.size == 0) or \
                (self._map_controller._obstacle_map[env]._explored_down_stair == False and self._map_controller._obstacle_map[env]._down_stair_frontiers.size == 0)):
                return self._handle_stairwell_reinitialization(env, masks)

            # 标记当前楼层已探索
            self._map_controller._obstacle_map[env]._this_floor_explored = True

            # 尝试导航到未探索的楼层 (优先上楼，其次下楼)
            # 2025.02.24: 改成优先探索没爬过的楼梯
            action = None
            if not self._map_controller._obstacle_map[env]._explored_up_stair:
                action = self._navigate_stair_if_unexplored_floor(observations, env, 'up')
            
            if action is None and not self._map_controller._obstacle_map[env]._explored_down_stair:
                action = self._navigate_stair_if_unexplored_floor(observations, env, 'down')

            if action is not None:
                return action
            else:
                retry_action = self._try_no_target_frontier_retry(env, masks)
                if retry_action is not None:
                    return retry_action
                print(f"Environment {env}: In all floors, no unexplored stairs or frontiers found, stopping.")
                return self._stop_action.to(masks.device)

        # 场景二：当前楼层有有效 Frontier，使用 LLM 规划器选择最佳 Frontier
        else:
            late_stair_action = self._try_late_stair_escape(observations, env)
            if late_stair_action is not None:
                return late_stair_action
            late_stair_waypoint_action = self._try_late_stair_waypoint_approach(observations, env, masks)
            if late_stair_waypoint_action is not None:
                return late_stair_waypoint_action

            best_frontier, best_value = self.llm_planner._get_best_frontier_with_llm(
                self._observations_cache, self._map_controller._obstacle_map, self._map_controller._value_map, self._map_controller._object_map,
                self._map_controller._obstacle_map_list, self._map_controller._value_map_list, self._map_controller._object_map_list,
                frontiers, env, topk=self.topk, use_multi_floor=True,
                cur_floor_index=self._map_controller._cur_floor_index, num_steps=self._num_steps,
                last_frontier_distance=self._last_frontier_distance,
                frontier_stick_step=self._map_controller._frontier_stick_step,
            )

            # LLM 判断上楼或下楼的保底机制
            if best_value == -100: # LLM 判断上楼
                action = self._navigate_stair_if_unexplored_floor(observations, env, 'up')
                if action: return action
                print(f"Environment {env}: Can't go upstairs or have already fully explored upstairs, exploring current floor instead.")
            elif best_value == -200: # LLM 判断下楼
                action = self._navigate_stair_if_unexplored_floor(observations, env, 'down')
                if action: return action
                print(f"Environment {env}: Can't go downstairs or have already fully explored downstairs, exploring current floor instead.")

            if not self._has_valid_frontiers([best_frontier]):
                print(f"Environment {env}: Planner returned no valid frontier; stopping exploration on this floor.")
                self._map_controller._obstacle_map[env]._this_floor_explored = True
                retry_action = self._try_no_target_frontier_retry(env, masks)
                if retry_action is not None:
                    return retry_action
                return self._stop_action.to(masks.device)
            
            # 执行点导航到最佳 Frontier
            robot_xy = self._observations_cache[env]["robot_xy"]
            if self._frontier_stuck_recovery_enabled() and self._should_disable_stalled_explore_frontier(env, best_frontier, robot_xy):
                return self._explore(observations, env, masks)
            self.cur_frontier[env] = best_frontier
            pointnav_action = self._pointnav(observations, self.cur_frontier[env], stop=False, env=env, stop_radius=self._pointnav_stop_radius)
            
            # 如果点导航动作是停止（0），则强制前进（1），以避免卡死
            if pointnav_action.item() == 0:
                print(f"Environment {env}: PointNav suggested stopping, forcing move forward.")
                pointnav_action.fill_(1)
            return pointnav_action

    def _has_valid_frontiers(self, frontiers: Any) -> bool:
        """Return False for empty frontier sensors and the Habitat [[0, 0]] sentinel."""
        if frontiers is None:
            return False
        try:
            frontier_array = np.asarray(frontiers, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if frontier_array.size == 0:
            return False
        frontier_array = np.atleast_2d(frontier_array)
        if frontier_array.shape[-1] < 2:
            return False
        return bool(np.any(np.linalg.norm(frontier_array[:, :2], axis=1) > 1e-6))

    def _try_no_target_frontier_retry(self, env: int, masks: Tensor) -> Optional[Tensor]:
        """Do a bounded recovery scan before stopping an unseen-target episode."""
        if not self._frontier_stuck_recovery_enabled():
            return None
        if self.max_episode_steps is None:
            return None
        remaining_steps = self.max_episode_steps - self._num_steps[env]
        if remaining_steps <= 80:
            return None
        if self._map_controller._object_map[env].has_object(self._map_controller._target_object[env]):
            return None
        if self._no_target_frontier_retry_count[env] >= 2:
            return None

        obstacle_map = self._map_controller._obstacle_map[env]
        if len(obstacle_map._disabled_frontiers) == 0:
            return None

        recovered_count = len(obstacle_map._disabled_frontiers)
        obstacle_map._disabled_frontiers.clear()
        obstacle_map._disabled_frontiers_px = np.array([], dtype=np.float64).reshape(0, 2)
        obstacle_map._this_floor_explored = False
        obstacle_map._tight_search_thresh = True
        self._map_controller._done_initializing[env] = False
        self._map_controller._initialize_step[env] = 0
        self._explore_frontier_repeat_count[env] = 0
        self._last_explore_frontier[env] = np.array([])
        self._last_explore_frontier_robot_xy[env] = np.array([])
        self._no_target_frontier_retry_count[env] += 1
        print(
            f"Environment {env}: Retry unseen-target exploration instead of stopping; "
            f"cleared {recovered_count} disabled frontier clusters "
            f"(retry {self._no_target_frontier_retry_count[env]}/2, remaining_steps={remaining_steps})."
        )
        return self._initialize(env, masks)

    def _try_late_stair_escape(self, observations: Union[Dict[str, Tensor], "TensorDict"], env: int) -> Optional[Tensor]:
        """Use remaining budget on unseen-target episodes to try an unvisited floor before timing out."""
        if not self._floor_budget_enabled():
            return None
        if self.max_episode_steps is None:
            return None
        remaining_steps = self.max_episode_steps - self._num_steps[env]
        if remaining_steps > 170:
            return None
        if not self._map_controller._climb_stair_over[env]:
            return None
        if self._map_controller._obstacle_map[env]._floor_num_steps < 220:
            return None
        if self._map_controller._object_map[env].has_object(self._map_controller._target_object[env]):
            return None

        for direction in ("up", "down"):
            explored_attr = f"_explored_{direction}_stair"
            if getattr(self._map_controller._obstacle_map[env], explored_attr):
                continue
            if not self._is_stair_close_enough_for_late_escape(env, direction):
                continue
            action = self._navigate_stair_if_unexplored_floor(observations, env, direction)
            if action is not None:
                print(f"Environment {env}: Late no-target stair escape via {direction}stairs with {remaining_steps} steps remaining.")
                return action
        return None

    def _try_late_stair_waypoint_approach(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        env: int,
        masks: Tensor,
    ) -> Optional[Tensor]:
        """Approach a known stair map from afar, but keep climbing gated by the close-distance check."""
        if not self._floor_budget_enabled():
            return None
        if self.max_episode_steps is None:
            return None
        remaining_steps = self.max_episode_steps - self._num_steps[env]
        if remaining_steps > 170:
            return None
        if not self._map_controller._climb_stair_over[env]:
            return None
        if self._map_controller._obstacle_map[env]._floor_num_steps < 220:
            return None
        if self._map_controller._object_map[env].has_object(self._map_controller._target_object[env]):
            return None

        obstacle_map = self._map_controller._obstacle_map[env]
        robot_xy = np.asarray(self._observations_cache[env]["robot_xy"], dtype=np.float64)
        robot_px = obstacle_map._xy_to_px(np.atleast_2d(robot_xy))[0]
        robot_yx = np.array([robot_px[1], robot_px[0]])
        close_distance_px = 4.0 * obstacle_map.pixels_per_meter
        # Start walking toward known stairs earlier when the target has never been
        # seen; the actual floor transition is still gated by the close stair check.
        max_approach_distance_px = 14.0 * obstacle_map.pixels_per_meter

        best_choice = None
        for direction in ("up", "down"):
            explored_attr = f"_explored_{direction}_stair"
            if getattr(obstacle_map, explored_attr):
                continue
            stair_map = obstacle_map._up_stair_map if direction == "up" else obstacle_map._down_stair_map
            if stair_map is None or np.sum(stair_map) == 0:
                continue
            stair_points_yx = np.argwhere(stair_map == 1)
            if stair_points_yx.size == 0:
                continue
            distances_px = np.abs(stair_points_yx - robot_yx).sum(axis=1)
            min_idx = int(np.argmin(distances_px))
            min_distance_px = float(distances_px[min_idx])
            if min_distance_px <= close_distance_px or min_distance_px > max_approach_distance_px:
                if min_distance_px > max_approach_distance_px:
                    print(
                        f"Environment {env}: Skip late {direction}stairs waypoint because stair is too far "
                        f"({min_distance_px:.1f}px > {max_approach_distance_px:.1f}px)."
                    )
                continue

            stair_yx = stair_points_yx[min_idx]
            stair_px = np.array([[stair_yx[1], stair_yx[0]]], dtype=np.float64)
            stair_xy = obstacle_map._px_to_xy(stair_px)[0]
            robot_to_stair = float(np.linalg.norm(stair_xy - robot_xy))
            choice = (min_distance_px, direction, stair_xy, robot_to_stair)
            if best_choice is None or choice[0] < best_choice[0]:
                best_choice = choice

        if best_choice is None:
            self._late_stair_waypoint_key[env] = None
            self._late_stair_waypoint_steps[env] = 0
            self._late_stair_waypoint_best_distance[env] = np.inf
            return None

        min_distance_px, direction, stair_xy, robot_to_stair = best_choice
        waypoint_key = (
            direction,
            round(float(stair_xy[0]), 2),
            round(float(stair_xy[1]), 2),
        )
        if self._late_stair_waypoint_key[env] != waypoint_key:
            self._late_stair_waypoint_key[env] = waypoint_key
            self._late_stair_waypoint_steps[env] = 0
            self._late_stair_waypoint_best_distance[env] = robot_to_stair
        else:
            self._late_stair_waypoint_steps[env] += 1
            if robot_to_stair + 0.25 < self._late_stair_waypoint_best_distance[env]:
                self._late_stair_waypoint_best_distance[env] = robot_to_stair
                self._late_stair_waypoint_steps[env] = 0

        if self._late_stair_waypoint_steps[env] >= 12:
            print(
                f"Environment {env}: Abort late {direction}stairs waypoint after "
                f"{self._late_stair_waypoint_steps[env]} low-progress steps "
                f"(best_distance={self._late_stair_waypoint_best_distance[env]:.2f}m, current={robot_to_stair:.2f}m)."
            )
            return None

        self.cur_frontier[env] = stair_xy
        print(
            f"Environment {env}: Late no-target stair waypoint approach toward {direction}stairs "
            f"with {remaining_steps} steps remaining "
            f"(stair_distance={min_distance_px:.1f}px/{robot_to_stair:.2f}m, "
            f"steps={self._late_stair_waypoint_steps[env]}, target={stair_xy})."
        )
        pointnav_action = self._pointnav(
            observations,
            self.cur_frontier[env],
            stop=False,
            env=env,
            stop_radius=self._pointnav_stop_radius,
        )
        if pointnav_action.item() == 0:
            print(f"Environment {env}: Stair-waypoint PointNav suggested stopping, forcing move forward.")
            pointnav_action.fill_(1)
        return pointnav_action

    def _is_stair_close_enough_for_late_escape(self, env: int, direction: str) -> bool:
        """Avoid spending the last budget on a far or low-quality stair approach."""
        obstacle_map = self._map_controller._obstacle_map[env]
        stair_map = obstacle_map._up_stair_map if direction == "up" else obstacle_map._down_stair_map
        if stair_map is None or np.sum(stair_map) == 0:
            return False
        robot_xy = self._observations_cache[env]["robot_xy"]
        robot_px = obstacle_map._xy_to_px(np.atleast_2d(robot_xy))[0]
        robot_yx = np.array([robot_px[1], robot_px[0]])
        stair_points = np.argwhere(stair_map == 1)
        if stair_points.size == 0:
            return False
        min_stair_distance_px = float(np.min(np.abs(stair_points - robot_yx).sum(axis=1)))
        max_stair_distance_px = 4.0 * obstacle_map.pixels_per_meter
        if min_stair_distance_px > max_stair_distance_px:
            print(
                f"Environment {env}: Skip late {direction}stairs escape because stair is too far "
                f"({min_stair_distance_px:.1f}px > {max_stair_distance_px:.1f}px)."
            )
            return False
        return True

    def _should_disable_stalled_explore_frontier(self, env: int, best_frontier: np.ndarray, robot_xy: np.ndarray) -> bool:
        """Disable a repeatedly selected exploration frontier when the robot makes little progress."""
        if not self._frontier_stuck_recovery_enabled():
            return False
        if best_frontier is None or len(best_frontier) == 0:
            self._explore_frontier_repeat_count[env] = 0
            return False

        best_frontier = np.asarray(best_frontier)
        last_frontier = self._last_explore_frontier[env]
        last_robot_xy = self._last_explore_frontier_robot_xy[env]

        same_frontier = len(last_frontier) > 0 and np.linalg.norm(best_frontier - last_frontier) < 0.4
        if not same_frontier:
            self._last_explore_frontier[env] = best_frontier.copy()
            self._last_explore_frontier_robot_xy[env] = np.asarray(robot_xy).copy()
            self._explore_frontier_repeat_count[env] = 1
            return False

        robot_progress = 0.0
        if len(last_robot_xy) > 0:
            robot_progress = float(np.linalg.norm(np.asarray(robot_xy) - last_robot_xy))

        if robot_progress > 0.35:
            self._last_explore_frontier_robot_xy[env] = np.asarray(robot_xy).copy()
            self._explore_frontier_repeat_count[env] = 1
            return False

        self._explore_frontier_repeat_count[env] += 1
        frontier_distance = float(np.linalg.norm(best_frontier - robot_xy))
        stick_step = self._map_controller._frontier_stick_step[env]
        remaining_steps = self.max_episode_steps - self._num_steps[env] if self.max_episode_steps is not None else None
        target_seen = self._map_controller._object_map[env].has_object(self._map_controller._target_object[env])
        late_never_saw = (
            remaining_steps is not None
            and remaining_steps <= 120
            and self._map_controller._obstacle_map[env]._floor_num_steps >= 260
            and not target_seen
        )
        should_disable = (
            (same_frontier and stick_step >= 18)
            or (
                self._explore_frontier_repeat_count[env] >= 18
                and robot_progress < 0.2
                and (frontier_distance < 3.0 or stick_step >= 10)
            )
        )
        if late_never_saw:
            should_disable = should_disable or (
                (same_frontier and stick_step >= 6 and robot_progress < 0.25)
                or (self._explore_frontier_repeat_count[env] >= 8 and robot_progress < 0.2)
                or (frontier_distance < 2.0 and stick_step >= 4)
            )

        if not should_disable:
            return False

        disabled_frontier = tuple(best_frontier)
        obstacle_map = self._map_controller._obstacle_map[env]
        obstacle_map._disabled_frontiers.add(disabled_frontier)
        print(
            f"Environment {env}: Disabled stalled exploration frontier {disabled_frontier} "
            f"within radius {obstacle_map.disabled_frontier_radius:.2f}m after {self._explore_frontier_repeat_count[env]} repeated selections "
            f"(distance={frontier_distance:.2f}, robot_progress={robot_progress:.2f}, stick_step={stick_step}, late_never_saw={late_never_saw})."
        )
        self._explore_frontier_repeat_count[env] = 0
        self._last_explore_frontier[env] = np.array([])
        self._last_explore_frontier_robot_xy[env] = np.array([])
        return True

    def _handle_stairwell_reinitialization(self, env: int, masks: Tensor) -> Tensor:
        """
        辅助函数：处理楼梯间场景的地图重置和初始化。
        在没有 Frontier 且处于楼梯间状态时调用。
        """
        # 重置当前环境的对象地图和价值地图
        self._map_controller._object_map[env].reset()
        self._map_controller._value_map[env].reset()

        # 临时存储现有楼梯信息
        stair_data = {}
        for stair_type in ["up", "down"]:
            has_stair = getattr(self._map_controller._obstacle_map[env], f"_has_{stair_type}_stair")
            if has_stair:
                stair_data[stair_type] = {
                    "map": getattr(self._map_controller._obstacle_map[env], f"_{stair_type}_stair_map").copy(),
                    "start": getattr(self._map_controller._obstacle_map[env], f"_{stair_type}_stair_start").copy(),
                    "end": getattr(self._map_controller._obstacle_map[env], f"_{stair_type}_stair_end").copy(),
                    "frontiers": getattr(self._map_controller._obstacle_map[env], f"_{stair_type}_stair_frontiers").copy(),
                    "explored": getattr(self._map_controller._obstacle_map[env], f"_explored_{stair_type}_stair"),
                }
        
        # 重置障碍物地图
        self._map_controller._obstacle_map[env].reset()

        # 恢复之前存储的楼梯信息
        for stair_type, data in stair_data.items():
            setattr(self._map_controller._obstacle_map[env], f"_has_{stair_type}_stair", True)
            setattr(self._map_controller._obstacle_map[env], f"_{stair_type}_stair_map", data["map"])
            setattr(self._map_controller._obstacle_map[env], f"_{stair_type}_stair_start", data["start"])
            setattr(self._map_controller._obstacle_map[env], f"_{stair_type}_stair_end", data["end"])
            setattr(self._map_controller._obstacle_map[env], f"_{stair_type}_stair_frontiers", data["frontiers"])
            setattr(self._map_controller._obstacle_map[env], f"_explored_{stair_type}_stair", data["explored"])

        self._map_controller._obstacle_map[env]._reinitialize_flag = True # 标记已重初始化

        # 重置与导航状态相关的其他变量
        self._map_controller._obstacle_map[env]._tight_search_thresh = True
        self._map_controller._climb_stair_over[env] = True
        self._map_controller._reach_stair[env] = False
        self._map_controller._reach_stair_centroid[env] = False
        self._map_controller._stair_dilate_flag[env] = False
        self._pitch_angle[env] = 0
        self._map_controller._done_initializing[env] = False
        self._map_controller._initialize_step[env] = 0
        
        # 执行初始化动作
        return self._initialize(env, masks)

    def _navigate_stair_if_unexplored_floor(self, observations: Union[Dict[str, Tensor], "TensorDict"], env: int, direction: str) -> Optional[Tensor]:
        """
        辅助函数：检查是否存在未探索的楼层，并通过楼梯导航。
        Args:
            direction (str): 'up' 或 'down'，表示向上或向下探索楼梯。
        Returns:
            Optional[Tensor]: 如果找到未探索的楼层并成功规划到楼梯，则返回点导航动作；否则返回 None。
        """
        temp_flag = False
        
        # 根据方向选择对应的楼梯属性和爬楼标志值
        has_stair_attr = f"_has_{direction}_stair"
        stair_frontiers_attr = f"_{direction}_stair_frontiers"
        climb_flag_value = 1 if direction == 'up' else 2

        if getattr(self._map_controller._obstacle_map[env], has_stair_attr):
            # 根据方向确定楼层遍历范围
            if direction == 'up':
                floor_range = range(self._map_controller._cur_floor_index[env] + 1, len(self._map_controller._object_map_list[env]))
            else: # 'down'
                floor_range = range(self._map_controller._cur_floor_index[env] - 1, -1, -1)
            
            # 检查目标方向是否有未探索的楼层
            for ith_floor in floor_range:
                if not self._map_controller._obstacle_map_list[env][ith_floor]._this_floor_explored:
                    temp_flag = True
                    break

            if temp_flag:
                self._map_controller._climb_stair_over[env] = False
                self._map_controller._climb_stair_flag[env] = climb_flag_value
                self._map_controller._stair_frontier[env] = getattr(self._map_controller._obstacle_map[env], stair_frontiers_attr)
                print(f"Environment {env}: Navigating {direction}stairs to unexplored floor.")
                # 假设 _stair_frontier[env] 包含了多个前沿，取第一个作为目标
                return self._pointnav(observations, self._map_controller._stair_frontier[env][0], stop=False, env=env)
        
        return None # 未找到符合条件的楼梯或未探索楼层
    
    def _look_for_downstair(self, observations: Union[Dict[str, Tensor], "TensorDict"], env: int, masks: Tensor) -> Tensor:
        # 如果已经有centroid就不用了
        if self._pitch_angle[env] >= 0:
            self._pitch_angle[env] -= self._pitch_angle_offset
            pointnav_action = get_action_tensor(LOOK_DOWN, device=masks.device)
        else:
            robot_xy = self._observations_cache[env]["robot_xy"]
            robot_xy_2d = np.atleast_2d(robot_xy) 
            dis_to_potential_stair = np.linalg.norm(self._map_controller._obstacle_map[env]._potential_stair_centroid - robot_xy_2d)
            if dis_to_potential_stair > 0.2:
                pointnav_action = self._pointnav(observations,self._map_controller._obstacle_map[env]._potential_stair_centroid[0], stop=False, env=env, stop_radius=self._pointnav_stop_radius) # 探索的时候可以远一点停？
                if pointnav_action.item() == 0:
                    print("Might false recognize down stairs, change to other mode.")
                    self._map_controller._obstacle_map[env]._disabled_frontiers.add(tuple(self._map_controller._obstacle_map[env]._potential_stair_centroid[0]))
                    print(f"Frontier {self._map_controller._obstacle_map[env]._potential_stair_centroid[0]} is disabled due to no movement.")
                    # 需验证，一般来说，如果真有向下的楼梯，并不会执行到这里
                    self._map_controller._obstacle_map[env]._disabled_stair_map[self._map_controller._obstacle_map[env]._down_stair_map == 1] = 1
                    self._map_controller._obstacle_map[env]._down_stair_map.fill(0)
                    self._map_controller._obstacle_map[env]._has_down_stair = False
                    self._pitch_angle[env] += self._pitch_angle_offset
                    self._map_controller._obstacle_map[env]._look_for_downstair_flag = False
                    pointnav_action = get_action_tensor(LOOK_UP, device=masks.device)

            else:
                print("Might false recognize down stairs, change to other mode.")
                self._map_controller._obstacle_map[env]._disabled_frontiers.add(tuple(self._map_controller._obstacle_map[env]._potential_stair_centroid[0]))
                print(f"Frontier {self._map_controller._obstacle_map[env]._potential_stair_centroid[0]} is disabled due to no movement.")
                # 需验证，一般来说，如果真有向下的楼梯，并不会执行到这里
                self._map_controller._obstacle_map[env]._disabled_stair_map[self._map_controller._obstacle_map[env]._down_stair_map == 1] = 1
                self._map_controller._obstacle_map[env]._down_stair_map.fill(0)
                self._map_controller._obstacle_map[env]._has_down_stair = False
                self._pitch_angle[env] += self._pitch_angle_offset
                self._map_controller._obstacle_map[env]._look_for_downstair_flag = False
                pointnav_action = get_action_tensor(LOOK_UP, device=masks.device) 

        return pointnav_action
        
    def _pointnav(self, observations: "TensorDict", goal: np.ndarray, stop: bool = False, env: int = 0, ori_masks: Tensor = None, stop_radius: float = 0.9) -> Tensor: #, 
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        masks = torch.tensor([self._num_steps[env] != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal[env]):
            if np.linalg.norm(goal - self._last_goal[env]) > 0.1:
                self._pointnav_policy[env].reset()
                masks = torch.zeros_like(masks)
            self._last_goal[env] = goal
        robot_xy = self._observations_cache[env]["robot_xy"]
        heading = self._observations_cache[env]["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache[env]["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info[env]["rho_theta"] = np.array([rho, theta])
        if rho < stop_radius: # self._pointnav_stop_radius
            if stop:
                    self._called_stop[env] = True
                    return self._stop_action.to(ori_masks.device)
        action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)
        return action

    def _navigate(self, observations: "TensorDict", goal: np.ndarray, stop: bool = False, env: int = 0, ori_masks: Tensor = None, stop_radius: float = 0.9) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        self._try_to_navigate_step[env] += 1
        masks = torch.tensor([self._num_steps[env] != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal[env]):
            if np.linalg.norm(goal - self._last_goal[env]) > 0.1:
                self._pointnav_policy[env].reset()
                masks = torch.zeros_like(masks)
            self._last_goal[env] = goal
        robot_xy = self._observations_cache[env]["robot_xy"]
        heading = self._observations_cache[env]["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache[env]["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info[env]["rho_theta"] = np.array([rho, theta])
        distance_to_goal = float(self._map_controller.cur_dis_to_goal[env])
        print(f"Distance to goal: {self._map_controller.cur_dis_to_goal[env]}")
        close_stop_radius = 0.6
        stalled_stop_radius = 1.0
        qwen_verified_goal = self._is_qwen_verified_goal(env)
        if qwen_verified_goal:
            close_stop_radius = float(getattr(self._final_check_cfg, "qwen_close_stop_radius", 0.60))
            final_confidence = float(self._map_controller._final_check_confidence[env])
            strong_threshold = float(getattr(self._final_check_cfg, "strong_confidence_threshold", 0.99))
            stalled_stop_radius = (
                max(close_stop_radius, float(getattr(self._final_check_cfg, "qwen_stalled_stop_radius", 0.82)) - 0.15)
                if final_confidence < strong_threshold
                else float(getattr(self._final_check_cfg, "qwen_stalled_stop_radius", 0.82))
            )
        if qwen_verified_goal:
            self._update_qwen_close_stall(env, distance_to_goal)
            if self._should_stop_qwen_close_stall(env, distance_to_goal):
                self._called_stop[env] = True
                print(
                    "Qwen-VL close-stall stop "
                    f"(distance={distance_to_goal:.3f}, "
                    f"best={self._qwen_close_stall_best_distance[env]:.3f}, "
                    f"stall_steps={self._qwen_close_stall_steps[env]})."
                )
                return self._stop_action.to(ori_masks.device)
        else:
            self._qwen_close_stall_steps[env] = 0
            self._qwen_close_stall_best_distance[env] = np.inf
        if distance_to_goal < 1.0: # close to the goal, but might be some noise, so get close as possible 
            if distance_to_goal <= close_stop_radius or (distance_to_goal <= stalled_stop_radius and np.abs(distance_to_goal - self.min_distance_xy[env]) < 0.1): # close enough or cannot move forward more #  or self._num_steps[env] == (500 - 1)
                if self._map_controller._double_check_goal[env] == True: # self._try_to_navigate_step[env] < 5 or 
                    self._called_stop[env] = True
                    # self._map_controller._obstacle_map[env].visualize_and_save_frontiers() ## for debug
                    return self._stop_action.to(ori_masks.device)
                else:
                    print("Might false positive, change to look for the true goal.")
                    self._map_controller._object_map[env].clear_verified_cloud(self._map_controller._target_object[env])
                    self._map_controller._object_map[env].clouds = {}
                    self._try_to_navigate[env] = False
                    self._try_to_navigate_step[env] = 0
                    self._map_controller._object_map[env]._disabled_object_map[self._map_controller._object_map[env]._map == 1] = 1
                    self._map_controller._object_map[env]._map.fill(0)
                    action = self._explore(observations, env, ori_masks) # 果断换成探索
                    return action
            else:
                self.min_distance_xy[env] = distance_to_goal
                return get_action_tensor(MOVE_FORWARD, device=ori_masks.device) # force to move forward
        else:        
            action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)
        if self._try_to_navigate_step[env] >= 100:
            print("Might false positive, change to look for the true goal.")
            self._map_controller._object_map[env].clear_verified_cloud(self._map_controller._target_object[env])
            self._map_controller._object_map[env].clouds = {}
            self._try_to_navigate[env] = False
            self._try_to_navigate_step[env] = 0
            self._map_controller._object_map[env]._disabled_object_map[self._map_controller._object_map[env]._map == 1] = 1
            self._map_controller._object_map[env]._map.fill(0)
            action = self._explore(observations, env, ori_masks) # 果断换成探索
            return action
        return action

    def _update_qwen_close_stall(self, env: int, distance_to_goal: float) -> None:
        """Track local progress after Qwen-VL has verified an objectpatch target."""
        if not np.isfinite(distance_to_goal) or distance_to_goal >= 1.05:
            self._qwen_close_stall_steps[env] = 0
            self._qwen_close_stall_best_distance[env] = np.inf
            return

        best_distance = self._qwen_close_stall_best_distance[env]
        if not np.isfinite(best_distance) or distance_to_goal < best_distance - 0.04:
            self._qwen_close_stall_best_distance[env] = distance_to_goal
            self._qwen_close_stall_steps[env] = 0
            return

        self._qwen_close_stall_steps[env] += 1

    def _should_stop_qwen_close_stall(self, env: int, distance_to_goal: float) -> bool:
        if self._called_stop[env]:
            return False
        if not np.isfinite(distance_to_goal):
            return False
        final_confidence = float(self._map_controller._final_check_confidence[env])
        if final_confidence < 0.99:
            return False
        if self._try_to_navigate_step[env] < 8:
            return False
        if distance_to_goal > 0.95:
            return False
        return self._qwen_close_stall_steps[env] >= int(getattr(self._final_check_cfg, "qwen_stall_steps", 8))

    def _get_close_to_stair(self, observations: "TensorDict", env: int, ori_masks: Tensor) -> Tensor:
        """
        处理导航到楼梯前沿的逻辑，包括卡顿检测和楼梯禁用。
        """
        # 参数校验和初始化
        if self._map_controller._climb_stair_flag[env] not in [1, 2]:
            print(f"Warning: _climb_stair_flag[{env}] is not 1 or 2. Skipping _get_close_to_stair.")
            return self._explore(observations, env, ori_masks) # 兜底，避免非预期状态

        masks = torch.tensor([self._num_steps[env] != 0], dtype=torch.bool, device="cuda")
        robot_xy = self._observations_cache[env]["robot_xy"]

        # 根据爬楼梯标志设置目标楼梯前沿
        target_stair_frontier = self._map_controller._obstacle_map[env]._up_stair_frontiers if self._map_controller._climb_stair_flag[env] == 1 else self._map_controller._obstacle_map[env]._down_stair_frontiers
        
        # 确保目标楼梯前沿存在，否则返回探索行为
        if target_stair_frontier.size == 0:
            print(f"Error: Stair frontier for climb_stair_flag {self._map_controller._climb_stair_flag[env]} is empty. Returning to explore.")
            return self._explore(observations, env, ori_masks)
        
        # 目标楼梯点统一取第一个，如果逻辑允许有多个，需要更复杂的选择策略
        target_stair_point = target_stair_frontier[0]

        # --- 楼梯前沿卡顿检测逻辑重构 ---
        if np.array_equal(self.llm_planner._last_frontier[env], target_stair_point):
            current_distance = np.linalg.norm(target_stair_point - robot_xy)

            if self._map_controller._frontier_stick_step[env] == 0:
                self._last_frontier_distance[env] = current_distance
                self._map_controller._frontier_stick_step[env] += 1
                self._map_controller._get_close_to_stair_step[env] += 1
            else:
                # 检查距离变化是否超过阈值（0.3米）
                if np.abs(self._last_frontier_distance[env] - current_distance) > 0.3:
                    self._map_controller._frontier_stick_step[env] = 0
                    self._last_frontier_distance[env] = current_distance
                else:
                    self._map_controller._frontier_stick_step[env] += 1
                    self._map_controller._get_close_to_stair_step[env] += 1

                    # 达到卡顿阈值，禁用楼梯前沿
                    if self._map_controller._frontier_stick_step[env] >= 30 or self._map_controller._get_close_to_stair_step[env] >= 60:
                        self._map_controller._disable_stair_and_reset_state(env, target_stair_point)
                        return self._explore(observations, env, ori_masks) # 禁用后切换到探索
        else:
            # 如果选中了不同的前沿，重置卡顿计数
            self._map_controller._frontier_stick_step[env] = 0
            self._last_frontier_distance[env] = 0
            self._map_controller._get_close_to_stair_step[env] = 0
        
        # 更新 LLM Planner 的 last_frontier，使其知道当前正在导航到楼梯
        self.llm_planner._last_frontier[env] = target_stair_point

        # --- 使用点导航模型算动作 ---
        heading = self._observations_cache[env]["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, target_stair_point)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)

        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache[env]["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info[env]["rho_theta"] = np.array([rho, theta])
        action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)

        # 如果点导航模型输出停止动作 (0)，则禁用楼梯并切换到探索
        if action.item() == 0:
            print(f"Pointnav policy stopped. Disabling stair frontier {target_stair_point}.")
            self._map_controller._disable_stair_and_reset_state(env, target_stair_point)
            return self._explore(observations, env, ori_masks) # 切换到探索

        return action
    
    def _climb_stair(self, observations: "TensorDict", env: int, ori_masks: Tensor) -> Tensor:
        """
        处理爬楼梯（上楼或下楼）的逻辑，包括视角调整和目标点导航。
        """
        masks = torch.tensor([self._num_steps[env] != 0], dtype=torch.bool, device="cuda")
        robot_xy = self._observations_cache[env]["robot_xy"]
        heading = self._observations_cache[env]["robot_heading"]

        # 根据爬楼梯标志设置目标楼梯前沿
        target_stair_frontier = self._map_controller._obstacle_map[env]._up_stair_frontiers if self._map_controller._climb_stair_flag[env] == 1 else self._map_controller._obstacle_map[env]._down_stair_frontiers
        
        if target_stair_frontier.size == 0:
            print(f"Error: Stair frontier for climb_stair_flag {self._map_controller._climb_stair_flag[env]} is empty. Returning to explore.")
            return self._explore(observations, env, ori_masks)

        current_distance = np.linalg.norm(target_stair_frontier[0] - robot_xy)
        print(f"Climb Stair - Distance Change: {np.abs(self._last_frontier_distance[env] - current_distance):.2f}m, Climb Stair Paused Step: {self._map_controller._obstacle_map[env]._climb_stair_paused_step}")

        # 检测是否卡顿在楼梯上
        if np.abs(self._last_frontier_distance[env] - current_distance) > 0.2:
            self._map_controller._obstacle_map[env]._climb_stair_paused_step = 0
            self._last_frontier_distance[env] = current_distance
        else:
            self._map_controller._obstacle_map[env]._climb_stair_paused_step += 1
        
        if self._map_controller._obstacle_map[env]._climb_stair_paused_step > 15:
            # 如果长时间卡顿，可能楼梯已经走完，或者遇到了障碍
            self._map_controller._obstacle_map[env]._disable_end = True # 标记楼梯终点可能不可达

        # 阶段1: 接近楼梯质心 (如果尚未到达)
        if not self._map_controller._reach_stair_centroid[env]:
            stair_centroid_point = target_stair_frontier[0] # 楼梯的质心点
            rho, theta = rho_theta(robot_xy, heading, stair_centroid_point)
            rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
            
            obs_pointnav = {
                "depth": image_resize(self._observations_cache[env]["nav_depth"], self._depth_image_shape, channels_last=True, interpolation_mode="area"),
                "pointgoal_with_gps_compass": rho_theta_tensor,
            }
            self._policy_info[env]["rho_theta"] = np.array([rho, theta])
            action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)
            
            if action.item() == 0:
                self._map_controller._reach_stair_centroid[env] = True
                print("Agent is near stair centroid, switching to move forward.")
                action[0] = 1 # 强制向前移动
            return action

        # 阶段2: 视角调整 (如果需要)
        # 仅在下楼梯且俯仰角过高时调整
        if self._map_controller._climb_stair_flag[env] == 2 and self._pitch_angle[env] < -30: 
            self._pitch_angle[env] += self._pitch_angle_offset
            print("Adjusting pitch angle for downstair (looking up a little).")
            return get_action_tensor(LOOK_UP, device=masks.device)
        
        # 阶段3: 沿楼梯方向前进 (胡萝卜策略)
        else:
            distance = 0.8 # 目标点距离

            depth_map = self._observations_cache[env]["nav_depth"].squeeze(0).cpu().numpy()
            if depth_map.size == 0: # 避免空深度图
                print("Warning: Depth map is empty. Cannot determine target point.")
                return get_action_tensor(1, device=masks.device) # 默认向前

            max_value = np.max(depth_map)
            max_indices = np.argwhere(depth_map == max_value)
            
            if max_indices.size == 0: # 避免没有最大值点
                print("Warning: No max depth value found. Cannot determine target point.")
                return get_action_tensor(1, device=masks.device) # 默认向前

            center_point = np.mean(max_indices, axis=0).astype(int)
            v, u = center_point[0], center_point[1]

            normalized_u = np.clip((u - self._cx) / self._cx, -1, 1)
            angle_offset = normalized_u * (self._camera_fov / 2)
            target_heading = heading - angle_offset # 尝试减去角度偏移
            target_heading = target_heading % (2 * np.pi)

            x_target = robot_xy[0] + distance * np.cos(target_heading)
            y_target = robot_xy[1] + distance * np.sin(target_heading)
            current_target_point_xy = np.array([x_target, y_target])
            current_target_point_px = self._map_controller._obstacle_map[env]._xy_to_px(np.atleast_2d(current_target_point_xy))

            this_stair_end_px = self._map_controller._obstacle_map[env]._up_stair_end if self._map_controller._climb_stair_flag[env] == 1 else self._map_controller._obstacle_map[env]._down_stair_end

            # 第一次或接近楼梯终点时重置胡萝卜目标点
            if (len(self._map_controller._last_carrot_xy[env]) == 0 or this_stair_end_px.size == 0 or 
                np.linalg.norm(this_stair_end_px - self._map_controller._obstacle_map[env]._xy_to_px(np.atleast_2d(robot_xy))[0]) <= 0.5 * self._map_controller._obstacle_map[env].pixels_per_meter or 
                self._map_controller._obstacle_map[env]._disable_end):
                
                self._map_controller._carrot_goal_xy[env] = current_target_point_xy
                self._map_controller._obstacle_map[env]._carrot_goal_px = current_target_point_px
                self._map_controller._last_carrot_xy[env] = current_target_point_xy
                self._map_controller._last_carrot_px[env] = current_target_point_px
            else:
                # 比较当前胡萝卜目标点与上次胡萝卜目标点到楼梯终点的L1距离
                l1_distance_current = np.abs(this_stair_end_px[0] - current_target_point_px[0][0]) + np.abs(this_stair_end_px[1] - current_target_point_px[0][1])
                l1_distance_last = np.abs(this_stair_end_px[0] - self._map_controller._last_carrot_px[env][0][0]) + np.abs(this_stair_end_px[1] - self._map_controller._last_carrot_px[env][0][1])
                
                if l1_distance_last > l1_distance_current: # 如果新的胡萝卜点离终点更近，则更新
                    self._map_controller._carrot_goal_xy[env] = current_target_point_xy
                    self._map_controller._obstacle_map[env]._carrot_goal_px = current_target_point_px
                    self._map_controller._last_carrot_xy[env] = current_target_point_xy
                    self._map_controller._last_carrot_px[env] = current_target_point_px
                # 否则，保持上一个胡萝卜目标点不变，即 self._map_controller._carrot_goal_xy[env] 和 _carrot_goal_px 已经是 _last_carrot 的值

            rho, theta = rho_theta(robot_xy, heading, self._map_controller._carrot_goal_xy[env])
            rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)

            obs_pointnav = {
                "depth": image_resize(self._observations_cache[env]["nav_depth"], self._depth_image_shape, channels_last=True, interpolation_mode="area"),
                "pointgoal_with_gps_compass": rho_theta_tensor,
            }
            self._policy_info[env]["rho_theta"] = np.array([rho, theta])
            action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)
            
            if action.item() == 0:
                print("Agent might stop, forcing move forward.")
                action[0] = 1 # 强制向前移动
            return action
