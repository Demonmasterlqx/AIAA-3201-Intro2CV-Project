from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from frontier_exploration.utils.general_utils import xyz_to_habitat
from habitat.utils.common import flatten_dict
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.maps import MAP_TARGET_POINT_INDICATOR
from habitat.utils.visualizations.utils import overlay_text_to_image
from habitat_baselines.common.tensor_dict import TensorDict

from vlfm.utils.geometry_utils import transform_points
from vlfm.utils.img_utils import (
    reorient_rescale_map,
    resize_image,
    resize_images,
    rotate_image,
)
from vlfm.utils.visualization import add_text_to_image, pad_images
import textwrap  # 用于分割长字符串

class HabitatVis:
    def __init__(self, num_envs: int) -> None:
        self.rgb: List[List[np.ndarray]] = [[] for _ in range(num_envs)]
        self.depth: List[List[np.ndarray]] = [[] for _ in range(num_envs)]
        self.maps: List[List[np.ndarray]] = [[] for _ in range(num_envs)]
        self.vis_maps: List[List[List[np.ndarray]]] = [[] for _ in range(num_envs)]
        self.texts: List[List[List[str]]] = [[] for _ in range(num_envs)]
        self.using_vis_maps = [False for _ in range(num_envs)]
        self.using_annotated_rgb = [False for _ in range(num_envs)]
        self.using_annotated_depth = [False for _ in range(num_envs)]
        
        # with 3rd view
        self.using_third_rgb = [False for _ in range(num_envs)]
        self.third_rgb: List[List[np.ndarray]] = [[] for _ in range(num_envs)]

        # with seg map
        self.using_seg_map = [False for _ in range(num_envs)]
        self.seg_map: List[List[np.ndarray]] = [[] for _ in range(num_envs)]

        # with vlm input
        self.using_vlm_input = [False for _ in range(num_envs)]
        self.vlm_input: List[List[np.ndarray]] = [[] for _ in range(num_envs)]
        self.vlm_response: List[List[str]] = [[] for _ in range(num_envs)]
        self.as_payloads: List[List[Dict[str, Any]]] = [[] for _ in range(num_envs)]

    def reset(self, env: int) -> None:
        self.rgb[env] = []
        self.depth[env]  = []
        self.maps[env]  = []
        self.vis_maps[env]  = []
        self.texts[env]  = []
        self.using_annotated_rgb[env] = False
        self.using_annotated_depth[env] = False

        # with 3rd view
        self.using_third_rgb[env] = False
        self.third_rgb[env] = []

        # with seg map
        self.using_seg_map[env] = False
        self.seg_map[env] = []

        # with vlm input
        self.using_vlm_input[env] = False
        self.vlm_input[env] = []
        self.vlm_response[env] = []
        self.as_payloads[env] = []

    def _collect_as_ascent_payload(self, policy_frame: Dict[str, Any], env: int) -> None:
        if "video_layout_version" not in policy_frame:
            return
        payload: Dict[str, Any] = {
            "video_layout_version": policy_frame.get("video_layout_version", ""),
            "scene_id": str(policy_frame.get("scene_id", "")),
            "episode_id": str(policy_frame.get("episode_id", "")),
            "goal": str(policy_frame.get("target_object", "")),
            "llm_backend": str(policy_frame.get("llm_backend", "")),
            "llm_model": str(policy_frame.get("llm_model", "")),
            "main_status": str(policy_frame.get("main_status", "")),
            "sub_status": str(policy_frame.get("sub_status", "")),
            "main_label": str(policy_frame.get("main_agent_label", "Main Agent")),
            "sub_label": str(policy_frame.get("subagent_label", "Subagent")),
            "event_timeline": list(policy_frame.get("event_timeline", [])),
            "spawn_reason": str(policy_frame.get("spawn_reason", "")),
            "current_floor_main": int(policy_frame.get("current_floor_main", 0)),
            "current_floor_sub": int(policy_frame.get("current_floor_sub", 0)),
            "cross_floor_transition_count": int(policy_frame.get("cross_floor_transition_count", 0)),
            "cross_floor_success_rate": float(policy_frame.get("cross_floor_success_rate", 0.0)),
            "spawn_count": int(policy_frame.get("spawn_count", 0)),
            "reclaim_count": int(policy_frame.get("reclaim_count", 0)),
            "subagent_state": str(policy_frame.get("subagent_state", "")),
            "coordinator_state": str(policy_frame.get("coordinator_state", "")),
            "verification_state": str(policy_frame.get("verification_state", "none")),
            "verification_score": float(policy_frame.get("verification_score", 0.0)),
            "verification_source": str(policy_frame.get("verification_source", "")),
            "main_action": str(policy_frame.get("main_action", "-")),
            "sub_action": str(policy_frame.get("sub_action", "-")),
            "active_candidate_id": str(policy_frame.get("active_candidate_id", "")),
            "vlm_response": str(policy_frame.get("vlm_response", "")),
        }
        for image_key in ("cooperative_topdown_map", "coordination_heatmap", "pointcloud_3d_map"):
            image = policy_frame.get(image_key)
            payload[image_key] = image.copy() if isinstance(image, np.ndarray) else None
        self.as_payloads[env].append(payload)
    def collect_data(
        self,
        observations: TensorDict,
        infos: List[Dict[str, Any]],
        policy_info: List[Dict[str, Any]],
    ) -> None:
        assert len(infos) == 1, "Only support one environment for now"

        if "annotated_depth" in policy_info[0]:
            depth = policy_info[0]["annotated_depth"]
            self.using_annotated_depth = True
        else:
            depth = (observations["depth"][0].cpu().numpy() * 255.0).astype(np.uint8)
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        depth = overlay_frame(depth, infos[0])
        self.depth.append(depth)

        if "annotated_rgb" in policy_info[0]:
            rgb = policy_info[0]["annotated_rgb"]
            self.using_annotated_rgb = True
        else:
            rgb = observations["rgb"][0].cpu().numpy()
        self.rgb.append(rgb)

        # Visualize target point cloud on the map
        color_point_cloud_on_map(infos, policy_info)

        map = maps.colorize_draw_agent_and_fit_to_height(infos[0]["top_down_map"], self.depth[0].shape[0])
        self.maps.append(map)
        vis_map_imgs = [
            self._reorient_rescale_habitat_map(infos, policy_info[0][vkey])
            for vkey in ["obstacle_map", "value_map"]
            if vkey in policy_info[0]
        ]
        if vis_map_imgs:
            self.using_vis_maps = True
            self.vis_maps.append(vis_map_imgs)
        text = [
            policy_info[0][text_key]
            for text_key in policy_info[0].get("render_below_images", [])
            if text_key in policy_info[0]
        ]
        self.texts.append(text)

    def collect_data_with_third_view(
        self,
        observations: TensorDict,
        infos: List[Dict[str, Any]],
        policy_info: List[Dict[str, Any]],
        env: int = 0,
    ) -> None:
        # assert len(infos) == 1, "Only support one environment for now"

        if "annotated_depth" in policy_info[env]:
            depth = policy_info[env]["annotated_depth"]
            self.using_annotated_depth[env] = True
        else:
            depth = (observations["depth"][env].cpu().numpy() * 255.0).astype(np.uint8)
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        depth = overlay_frame(depth, infos[env])
        self.depth[env].append(depth)

        if "annotated_rgb" in policy_info[env]:
            rgb = policy_info[env]["annotated_rgb"]
            self.using_annotated_rgb[env] = True
        else:
            rgb = observations["rgb"][env].cpu().numpy()
        self.rgb[env].append(rgb)

        # Visualize target point cloud on the map
        color_point_cloud_on_map(infos, policy_info, env)

        map = maps.colorize_draw_agent_and_fit_to_height(infos[env]["top_down_map"], self.depth[env][0].shape[0])
        self.maps[env].append(map)
        vis_map_imgs = [
            self._reorient_rescale_habitat_map(infos[env], policy_info[env][vkey])
            for vkey in ["obstacle_map", "value_map"]
            if vkey in policy_info[env]
        ]
        if vis_map_imgs:
            self.using_vis_maps[env] = True
            self.vis_maps[env].append(vis_map_imgs)
        text = [
            policy_info[env][text_key]
            for text_key in policy_info[env].get("render_below_images", [])
            if text_key in policy_info[env]
        ]
        self.texts[env].append(text)

        if "third_rgb" in policy_info[env]:
            third_rgb = policy_info[env]["third_rgb"]
            self.using_third_rgb[env] = True
            self.third_rgb[env].append(third_rgb)
        # else:
        #     third_rgb = observations["third_rgb"][0].cpu().numpy()
        # self.third_rgb.append(third_rgb)

    def collect_data_with_third_view_and_seg_map(
        self,
        observations: TensorDict,
        infos: List[Dict[str, Any]],
        policy_info: List[Dict[str, Any]],
        env: int = 0,
    ) -> None:
        # 初始化 step 计数器
        # if not hasattr(self, "step_count"):
        #     self.step_count = 0  # 添加 step_count 属性

        # 获取深度信息
        if len(policy_info) > 0 and "annotated_depth" in policy_info[env]:
            depth = policy_info[env]["annotated_depth"]
            self.using_annotated_depth[env] = True
        else:
            depth = (observations["depth"][env].cpu().numpy() * 255.0).astype(np.uint8)
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        depth = overlay_frame(depth, infos[env])
        self.depth[env].append(depth)

        # 获取 RGB 信息
        if policy_info is not None and "annotated_rgb" in policy_info[env]:
            rgb = policy_info[env]["annotated_rgb"]
            self.using_annotated_rgb[env] = True
        else:
            rgb = observations["rgb"][env].cpu().numpy()
        self.rgb[env].append(rgb)

        # 可视化目标点云在地图上的信息
        color_point_cloud_on_map(infos, policy_info, env)

        # 获取地图信息
        map = maps.colorize_draw_agent_and_fit_to_height(
            infos[env]["top_down_map"], self.depth[env][0].shape[0]
        )
        self.maps[env].append(map)

        if policy_info is not None:
            vis_map_imgs = [
                self._reorient_rescale_habitat_map(infos[env], policy_info[env][vkey])
                for vkey in ["obstacle_map", "value_map", "scene_map"] # scene_map
                if vkey in policy_info[env]
            ]
            if vis_map_imgs:
                self.using_vis_maps[env] = True
                self.vis_maps[env].append(vis_map_imgs)

            # 获取文本信息
            text = [
                policy_info[env][text_key]
                for text_key in policy_info[env].get("render_below_images", [])
                if text_key in policy_info[env]
            ]
            self.texts[env].append(text)

            # 获取第三视角 RGB 信息
            if "third_rgb" in policy_info[env]:
                third_rgb = policy_info[env]["third_rgb"]
                self.using_third_rgb[env] = True
                self.third_rgb[env].append(third_rgb)

            # 获取分割图信息
            if "seg_map" in policy_info[env]:
                seg_map = policy_info[env]["seg_map"]
                self.using_seg_map[env] = True
                self.seg_map[env].append(seg_map)
            self._collect_as_ascent_payload(policy_info[env], env)


    def collect_data_with_third_view_and_seg_map_vlm_input(
        self,
        observations: TensorDict,
        infos: List[Dict[str, Any]],
        policy_info: List[Dict[str, Any]],
        env: int = 0,
    ) -> None:

        # 获取深度信息
        if "annotated_depth" in policy_info[env]:
            depth = policy_info[env]["annotated_depth"]
            self.using_annotated_depth[env] = True
        else:
            depth = (observations["depth"][env].cpu().numpy() * 255.0).astype(np.uint8)
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        depth = overlay_frame(depth, infos[env])
        self.depth[env].append(depth)

        # 获取 RGB 信息
        if "annotated_rgb" in policy_info[env]:
            rgb = policy_info[env]["annotated_rgb"]
            self.using_annotated_rgb[env] = True
        else:
            rgb = observations["rgb"][env].cpu().numpy()
        self.rgb[env].append(rgb)

        # 可视化目标点云在地图上的信息
        color_point_cloud_on_map(infos, policy_info, env)

        # 获取地图信息
        map = maps.colorize_draw_agent_and_fit_to_height(
            infos[env]["top_down_map"], self.depth[env][0].shape[0]
        )
        self.maps[env].append(map)
        vis_map_imgs = [
            self._reorient_rescale_habitat_map(infos[env], policy_info[env][vkey])
            # policy_info[env][vkey]
            for vkey in ["obstacle_map", "value_map", "object_map"] # 画出目标距离
            if vkey in policy_info[env]
        ]
        if vis_map_imgs:
            self.using_vis_maps[env] = True
            self.vis_maps[env].append(vis_map_imgs)

        # 获取文本信息
        text = [
            policy_info[env][text_key]
            for text_key in policy_info[env].get("render_below_images", [])
            if text_key in policy_info[env]
        ]
        self.texts[env].append(text)

        # 获取第三视角 RGB 信息
        if "third_rgb" in policy_info[env]:
            third_rgb = policy_info[env]["third_rgb"]
            self.using_third_rgb[env] = True
            self.third_rgb[env].append(third_rgb)

        # 获取分割图信息
        if "seg_map" in policy_info[env]:
            seg_map = policy_info[env]["seg_map"]
            self.using_seg_map[env] = True
            self.seg_map[env].append(seg_map)
        
        if "vlm_input" in policy_info[env]:
            vlm_input = policy_info[env]["vlm_input"]
            vlm_response = policy_info[env]["vlm_response"]
            self.using_vlm_input[env] = True
            self.vlm_input[env].append(vlm_input)
            self.vlm_response[env].append(vlm_response)
        self._collect_as_ascent_payload(policy_info[env], env)


    def flush_frames(self, failure_cause: str, env: int) -> List[np.ndarray]:
        """Flush all frames and return them"""
        # Because the annotated frames are actually one step delayed, pop the first one
        # and add a placeholder frame to the end (gets removed anyway)
        if self.using_annotated_rgb[env] is not None:
            self.rgb[env].append(self.rgb[env].pop(0))
        if self.using_annotated_depth[env] is not None:
            self.depth[env].append(self.depth[env].pop(0))
        if self.using_vis_maps[env]:  # Cost maps are also one step delayed
            self.vis_maps[env].append(self.vis_maps[env].pop(0))

        frames = []
        num_frames = len(self.depth[env]) - 1  # last frame is from next episode, remove it
        for i in range(num_frames):
            if self.vis_maps[0] != []:
                frame = self._create_frame(
                    self.depth[env][i],
                    self.rgb[env][i],
                    self.maps[env][i],
                    self.vis_maps[env][i],
                    self.texts[env][i],
                )
            else:
                frame = self._create_frame(
                    self.depth[env][i],
                    self.rgb[env][i],
                    self.maps[env][i],
                    None,
                    None,
                )
            failure_cause_text = "Failure cause: " + failure_cause
            frame = add_text_to_image(frame, failure_cause_text, top=True)
            frames.append(frame)

        if len(frames) > 0:
            frames = pad_images(frames, pad_from_top=True)

        frames = [resize_image(f, 480 * 2) for f in frames]

        self.reset(env)

        return frames

    def flush_frames_with_rednet(self, failure_cause: str, env: int) -> List[np.ndarray]:
        """Flush all frames and return them"""
        # Because the annotated frames are actually one step delayed, pop the first one
        # and add a placeholder frame to the end (gets removed anyway)
        if self.using_annotated_rgb[env] is not None:
            self.rgb[env].append(self.rgb[env].pop(0))
        if self.using_annotated_depth[env] is not None:
            self.depth[env].append(self.depth[env].pop(0))
        if self.using_vis_maps[env]:  # Cost maps are also one step delayed
            self.vis_maps[env].append(self.vis_maps[env].pop(0))
        if self.using_third_rgb[env]:  # Add third_rgb if enabled
            self.third_rgb[env].append(self.third_rgb[env].pop(0))
        if self.using_seg_map[env]:  # Add seg_map if enabled
            self.seg_map[env].append(self.seg_map[env].pop(0))
        if self.as_payloads[env]:
            self.as_payloads[env].append(self.as_payloads[env].pop(0))
        frames = []
        lengths = [len(self.depth[env]), len(self.rgb[env])]
        if self.using_seg_map[env]:
            lengths.append(len(self.seg_map[env]))
        if self.as_payloads[env]:
            lengths.append(len(self.as_payloads[env]))
        num_frames = min(lengths) - 1
        for i in range(num_frames):
            if self.as_payloads[env]:
                seg_map = self.seg_map[env][i] if self.using_seg_map[env] and len(self.seg_map[env]) > i else None
                frame = self._create_frame_with_as_ascent_layout(
                    depth=self.depth[env][i],
                    rgb=self.rgb[env][i],
                    seg_map=seg_map,
                    payload=self.as_payloads[env][i],
                    failure_cause=failure_cause,
                )
            elif self.vis_maps[0] != []:
                # Dynamically decide parameters to pass to _create_frame
                frame_args = {
                    "depth": self.depth[env][i],
                    "rgb": self.rgb[env][i],
                    "map": self.maps[env][i],
                    "vis_map_imgs": self.vis_maps[env][i],
                    "text": self.texts[env][i],
                    "third_rgb": self.third_rgb[env][i], 
                    "seg_map": self.seg_map[env][i],
                }
            else:
                frame_args = {
                    "depth": self.depth[env][i],
                    "rgb": self.rgb[env][i],
                    "map": self.maps[env][i],
                    "vis_map_imgs": None,
                    "text": None,
                    "third_rgb": self.third_rgb[env][i], 
                    "seg_map": self.seg_map[env][i],
                }

            # Call _create_frame with dynamic arguments
            frame = self._create_frame_with_custom_layout(**frame_args) # _create_frame

            failure_cause_text = "Failure cause: " + failure_cause
            if not self.as_payloads[env]:
                frame = add_text_to_image(frame, failure_cause_text, top=True)
            frames.append(frame)

        if len(frames) > 0:
            frames = pad_images(frames, pad_from_top=True)

        frames = [resize_image(f, 480 * 2) for f in frames]

        self.reset(env)

        return frames

    def flush_frames_with_rednet_vlm_input(self, failure_cause: str, env: int) -> List[np.ndarray]:
        """Flush all frames and return them"""
        # Because the annotated frames are actually one step delayed, pop the first one
        # and add a placeholder frame to the end (gets removed anyway)
        if self.using_annotated_rgb[env] is not None:
            self.rgb[env].append(self.rgb[env].pop(0))
        if self.using_annotated_depth[env] is not None:
            self.depth[env].append(self.depth[env].pop(0))
        if self.using_vis_maps[env]:  # Cost maps are also one step delayed
            self.vis_maps[env].append(self.vis_maps[env].pop(0))
        if self.using_third_rgb[env]:  # Add third_rgb if enabled
            self.third_rgb[env].append(self.third_rgb[env].pop(0))
        if self.using_seg_map[env]:  # Add seg_map if enabled
            self.seg_map[env].append(self.seg_map[env].pop(0))
        if self.using_vlm_input[env]:  # Add vlm_input if enabled
            self.vlm_input[env].append(self.vlm_input[env].pop(0))
        if self.as_payloads[env]:
            self.as_payloads[env].append(self.as_payloads[env].pop(0))
        frames = []
        lengths = [len(self.depth[env]), len(self.rgb[env])]
        if self.using_seg_map[env]:
            lengths.append(len(self.seg_map[env]))
        if self.as_payloads[env]:
            lengths.append(len(self.as_payloads[env]))
        num_frames = min(lengths) - 1
        for i in range(num_frames):
            if self.as_payloads[env]:
                seg_map = self.seg_map[env][i] if self.using_seg_map[env] and len(self.seg_map[env]) > i else None
                frame = self._create_frame_with_as_ascent_layout(
                    depth=self.depth[env][i],
                    rgb=self.rgb[env][i],
                    seg_map=seg_map,
                    payload=self.as_payloads[env][i],
                    failure_cause=failure_cause,
                )
            else:
                # Dynamically decide parameters to pass to _create_frame
                frame_args = {
                    "depth": self.depth[env][i],
                    "rgb": self.rgb[env][i],
                    "map": self.maps[env][i],
                    "vis_map_imgs": self.vis_maps[env][i],
                    "text": self.texts[env][i],
                    "seg_map": self.seg_map[env][i],
                }
                if len(self.third_rgb[env]) > 0:
                    frame_args["third_rgb"] = self.third_rgb[env][i], 
                if len(self.vlm_input[env]) > 0:
                    frame_args["vlm_input"] = self.vlm_input[env][i]
                if len(self.vlm_response[env]) > 0:
                    frame_args["vlm_response"] = self.vlm_response[env][i]
                # Call _create_frame with dynamic arguments
                frame = self._create_frame_with_custom_layout(**frame_args) # _create_frame

            failure_cause_text = "Failure cause: " + failure_cause
            if not self.as_payloads[env]:
                frame = add_text_to_image(frame, failure_cause_text, top=True)
            frames.append(frame)

        if len(frames) > 0:
            frames = pad_images(frames, pad_from_top=True)

        frames = [resize_image(f, 480 * 2) for f in frames]

        self.reset(env)

        return frames

    @staticmethod
    def _reorient_rescale_habitat_map(infos: List[Dict[str, Any]], vis_map: np.ndarray) -> np.ndarray:
        # Rotate the cost map to match the agent's orientation at the start
        # of the episode
        # start_yaw = infos["start_yaw"]
        # if start_yaw != 0.0:
        #     vis_map = rotate_image(vis_map, start_yaw, border_value=(255, 255, 255))

        # # Rotate the image 90 degrees if the corresponding map is taller than it is wide
        # habitat_map = infos["top_down_map"]["map"]
        # if habitat_map.shape[0] > habitat_map.shape[1]:
        #     vis_map = np.rot90(vis_map, 1)

        vis_map = reorient_rescale_map(vis_map)

        return vis_map

    @staticmethod
    def _create_frame(
        depth: np.ndarray,
        rgb: np.ndarray,
        map: np.ndarray,
        vis_map_imgs: List[np.ndarray],
        text: List[str],
    ) -> np.ndarray:
        """Create a frame using all the given images.

        First, the depth and rgb images are stacked vertically. Then, all the maps are
        combined as a separate images. Then these two images should be stitched together
        horizontally (depth-rgb on the left, maps on the right).

        The combined map image contains two rows of images and at least one column.
        First, the 'map' argument is at the top left, then the first element of the
        'vis_map_imgs' argument is at the bottom left. If there are more than one
        element in 'vis_map_imgs', then the second element is at the top right, the
        third element is at the bottom right, and so on.

        Args:
            depth: The depth image (H, W, 3).
            rgb: The rgb image (H, W, 3).
            map: The map image, a 3-channel rgb image, but can have different shape from
                depth and rgb.
            vis_map_imgs: A list of other map images. Each are 3-channel rgb images, but
                can have different sizes.
            text: A list of strings to be rendered above the images.
        Returns:
            np.ndarray: The combined frame image.
        """
        # Stack depth and rgb images vertically
        depth_rgb = np.vstack((depth, rgb))

        # Prepare the list of images to be combined
        if vis_map_imgs is not None:
            map_imgs = [map] + vis_map_imgs
        else:
            map_imgs = [map]
        if len(map_imgs) % 2 == 1:
            # If there are odd number of images, add a placeholder image
            map_imgs.append(np.ones_like(map_imgs[-1]) * 255)

        even_index_imgs = map_imgs[::2]
        odd_index_imgs = map_imgs[1::2]
        top_row = np.hstack(resize_images(even_index_imgs, match_dimension="height"))
        bottom_row = np.hstack(resize_images(odd_index_imgs, match_dimension="height"))

        frame = np.vstack(resize_images([top_row, bottom_row], match_dimension="width"))
        depth_rgb, frame = resize_images([depth_rgb, frame], match_dimension="height")
        frame = np.hstack((depth_rgb, frame))

        # Add text to the top of the frame
        if text is not None:
            for t in text[::-1]:
                frame = add_text_to_image(frame, t, top=True)

        return frame

    @staticmethod
    def _create_frame_with_as_ascent_layout(
        depth: np.ndarray,
        rgb: np.ndarray,
        seg_map: Optional[np.ndarray],
        payload: Dict[str, Any],
        failure_cause: str,
    ) -> np.ndarray:
        content_h = 310
        content_w = 430
        panel_a = HabitatVis._panel_with_title(
            HabitatVis._draw_agent_badges(
                HabitatVis._fit_panel(rgb, content_h, content_w),
                payload,
            ),
            "Main RGB",
        )
        panel_b = HabitatVis._panel_with_title(
            HabitatVis._create_depth_seg_panel(depth, seg_map, content_h, content_w),
            "Depth + Seg",
        )
        panel_c = HabitatVis._panel_with_title(
            HabitatVis._annotate_map_panel(
                payload.get("cooperative_topdown_map"),
                payload,
                content_h,
                content_w,
            ),
            "Cooperative Map",
        )
        panel_d = HabitatVis._panel_with_title(
            HabitatVis._fit_or_placeholder(
                payload.get("coordination_heatmap"),
                content_h,
                content_w,
                "Coordination heatmap pending",
                crop_whitespace=True,
            ),
            "Coordination Heatmap",
        )
        panel_e = HabitatVis._panel_with_title(
            HabitatVis._fit_or_placeholder(
                payload.get("pointcloud_3d_map"),
                content_h,
                content_w,
                "3D map pending",
                crop_whitespace=True,
            ),
            "3D Point-Cloud Map",
        )
        panel_f = HabitatVis._panel_with_title(
            HabitatVis._build_event_panel(payload, failure_cause, content_h, content_w),
            "Events + LLM + Verification",
        )

        row_top = np.hstack((panel_a, panel_b, panel_c))
        row_bottom = np.hstack((panel_d, panel_e, panel_f))
        grid = np.vstack((row_top, row_bottom))
        title_bar = HabitatVis._build_title_bar(payload, failure_cause, grid.shape[1])
        return np.vstack((title_bar, grid))

    @staticmethod
    def _fit_or_placeholder(
        image: Optional[np.ndarray],
        target_h: int,
        target_w: int,
        placeholder_text: str,
        crop_whitespace: bool = False,
    ) -> np.ndarray:
        if isinstance(image, np.ndarray) and image.size > 0:
            if crop_whitespace:
                image = HabitatVis._crop_to_content(image)
            return HabitatVis._fit_panel(image, target_h, target_w)
        return HabitatVis._render_text_panel([placeholder_text], target_h, target_w)

    @staticmethod
    def _fit_panel(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if image is None or image.size == 0:
            return np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image.astype(np.uint8)
        src_h, src_w = image.shape[:2]
        scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
        resized = cv2.resize(
            image,
            (max(1, int(src_w * scale)), max(1, int(src_h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        y0 = (target_h - resized.shape[0]) // 2
        x0 = (target_w - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        return canvas

    @staticmethod
    def _crop_to_content(image: np.ndarray, threshold: int = 245, padding: int = 18) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        if image.ndim == 2:
            mask = image < threshold
        else:
            mask = np.any(image < threshold, axis=2)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return image
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        y0 = max(0, int(y0) - padding)
        x0 = max(0, int(x0) - padding)
        y1 = min(image.shape[0], int(y1) + padding + 1)
        x1 = min(image.shape[1], int(x1) + padding + 1)
        cropped = image[y0:y1, x0:x1]
        if cropped.size == 0:
            return image
        return cropped

    @staticmethod
    def _panel_with_title(image: np.ndarray, title: str) -> np.ndarray:
        header_h = 34
        header = np.full((header_h, image.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(header, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (32, 32, 32), 2, cv2.LINE_AA)
        cv2.line(header, (0, header_h - 1), (image.shape[1], header_h - 1), (220, 220, 220), 1)
        return np.vstack((header, image))

    @staticmethod
    def _draw_agent_badges(image: np.ndarray, payload: Dict[str, Any]) -> np.ndarray:
        canvas = image.copy()
        HabitatVis._draw_status_badge(
            canvas,
            label="M",
            color=(18, 181, 203),
            lines=HabitatVis._wrap_status_lines(
                [
                    payload.get("main_label", "Main Agent"),
                    payload.get("main_status", ""),
                    f"action={payload.get('main_action', '-')}",
                ]
            ),
            origin=(14, 14),
        )
        HabitatVis._draw_status_badge(
            canvas,
            label="S",
            color=(255, 138, 61),
            lines=HabitatVis._wrap_status_lines(
                [
                    payload.get("sub_label", "Subagent"),
                    payload.get("sub_status", ""),
                    f"action={payload.get('sub_action', '-')}",
                ]
            ),
            origin=(14, 122),
        )

        verify_state = str(payload.get("verification_state", "none"))
        verify_color = HabitatVis._verification_color(verify_state)
        chip_text = f"verify={verify_state}"
        if verify_state != "none":
            chip_text += f" ({payload.get('verification_score', 0.0):.2f})"
        text_size, _ = cv2.getTextSize(chip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        x0 = max(10, canvas.shape[1] - text_size[0] - 30)
        cv2.rectangle(canvas, (x0, 14), (canvas.shape[1] - 14, 46), verify_color, -1)
        cv2.putText(
            canvas,
            chip_text,
            (x0 + 10, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    @staticmethod
    def _draw_status_badge(
        image: np.ndarray,
        *,
        label: str,
        color: Tuple[int, int, int],
        lines: List[str],
        origin: Tuple[int, int],
    ) -> None:
        x0, y0 = origin
        width = min(280, image.shape[1] - x0 - 12)
        height = 92
        overlay = image.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (255, 255, 255), -1)
        cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), color, 2)
        cv2.rectangle(overlay, (x0, y0), (x0 + 34, y0 + height), color, -1)
        cv2.addWeighted(overlay, 0.86, image, 0.14, 0, image)
        cv2.putText(
            image,
            label,
            (x0 + 10, y0 + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.92,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y = y0 + 24
        for idx, line in enumerate(lines[:4]):
            cv2.putText(
                image,
                line,
                (x0 + 44, y + idx * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (28, 28, 28),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _wrap_status_lines(lines: List[str], width: int = 34) -> List[str]:
        wrapped: List[str] = []
        for line in lines:
            if not line:
                continue
            for chunk in textwrap.wrap(line, width=width)[:2]:
                wrapped.append(chunk)
        return wrapped[:4]

    @staticmethod
    def _create_depth_seg_panel(
        depth: np.ndarray,
        seg_map: Optional[np.ndarray],
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        depth_panel = HabitatVis._fit_panel(depth, target_h, target_w)
        if seg_map is not None and seg_map.size > 0:
            seg_panel = HabitatVis._fit_panel(seg_map, target_h, target_w)
            mixed = cv2.addWeighted(depth_panel, 0.64, seg_panel, 0.36, 0)
        else:
            mixed = depth_panel
        legend_items = [
            ("target", (255, 0, 0)),
            ("person", (255, 105, 180)),
            ("stair", (0, 0, 255)),
        ]
        x = 12
        for label, color in legend_items:
            cv2.rectangle(mixed, (x, target_h - 32), (x + 16, target_h - 16), color, -1)
            cv2.putText(
                mixed,
                label,
                (x + 22, target_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (36, 36, 36),
                1,
                cv2.LINE_AA,
            )
            x += 108
        return mixed

    @staticmethod
    def _annotate_map_panel(
        image: Optional[np.ndarray],
        payload: Dict[str, Any],
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        panel = HabitatVis._fit_or_placeholder(
            image,
            target_h,
            target_w,
            "Cooperative map pending",
            crop_whitespace=True,
        )
        footer_h = 58
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, target_h - footer_h), (target_w, target_h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.9, panel, 0.1, 0, panel)
        lines = [
            f"floors M/S: {payload.get('current_floor_main', 0)} / {payload.get('current_floor_sub', 0)}",
            (
                "cross-floor: "
                f"{payload.get('cross_floor_transition_count', 0)} "
                f"(succ={payload.get('cross_floor_success_rate', 0.0):.2f})"
            ),
        ]
        for idx, line in enumerate(lines):
            cv2.putText(
                panel,
                line,
                (12, target_h - 28 + idx * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (26, 26, 26),
                1,
                cv2.LINE_AA,
            )
        return panel

    @staticmethod
    def _build_event_panel(
        payload: Dict[str, Any],
        failure_cause: str,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        lines = [
            f"Coordinator: {payload.get('coordinator_state', '-')}",
            f"Subagent: {payload.get('subagent_state', '-')}",
            f"Spawn/Reclaim: {payload.get('spawn_count', 0)} / {payload.get('reclaim_count', 0)}",
            f"Spawn reason: {payload.get('spawn_reason', '-')}",
            (
                f"Verification: {payload.get('verification_state', 'none')} "
                f"score={payload.get('verification_score', 0.0):.2f}"
            ),
            f"Source/Candidate: {payload.get('verification_source', '-') or '-'} / {payload.get('active_candidate_id', '-') or '-'}",
            f"Backend: {payload.get('llm_backend', '-')}",
            f"Model: {payload.get('llm_model', '-')}",
            f"Failure cause: {failure_cause}",
            "Recent events:",
        ]
        events = payload.get("event_timeline", []) or ["no recent event"]
        for event in events[-5:]:
            lines.append(f"- {event}")
        vlm_response = str(payload.get("vlm_response", "")).strip()
        if vlm_response:
            lines.append("LLM summary:")
            lines.extend([f"> {chunk}" for chunk in textwrap.wrap(vlm_response, width=42)[:5]])
        return HabitatVis._render_text_panel(lines, target_h, target_w, payload.get("verification_state", "none"))

    @staticmethod
    def _render_text_panel(
        lines: List[str],
        target_h: int,
        target_w: int,
        verification_state: str = "none",
    ) -> np.ndarray:
        panel = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        header_color = HabitatVis._verification_color(verification_state)
        cv2.rectangle(panel, (0, 0), (target_w, 28), header_color, -1)
        header_text = f"verification={verification_state}"
        cv2.putText(panel, header_text, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)
        y = 48
        for raw in lines:
            wrapped = textwrap.wrap(str(raw), width=52) or [""]
            for line in wrapped:
                cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (30, 30, 30), 1, cv2.LINE_AA)
                y += 18
                if y >= target_h - 10:
                    return panel
        return panel

    @staticmethod
    def _verification_color(state: str) -> Tuple[int, int, int]:
        color_map = {
            "suspect": (214, 184, 26),
            "likely": (232, 104, 52),
            "verified": (46, 160, 67),
            "rejected": (145, 92, 92),
            "none": (90, 90, 90),
        }
        return color_map.get(str(state), (90, 90, 90))

    @staticmethod
    def _build_title_bar(payload: Dict[str, Any], failure_cause: str, width: int) -> np.ndarray:
        bar = np.full((78, width, 3), 248, dtype=np.uint8)
        title = (
            f"scene={payload.get('scene_id', '-') or '-'} | "
            f"episode={payload.get('episode_id', '-') or '-'} | "
            f"goal={payload.get('goal', '-') or '-'}"
        )
        subtitle = (
            f"backend={payload.get('llm_backend', '-') or '-'} / "
            f"model={payload.get('llm_model', '-') or '-'} | "
            f"result={failure_cause}"
        )
        cv2.putText(bar, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (24, 24, 24), 2, cv2.LINE_AA)
        cv2.putText(bar, subtitle, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (54, 54, 54), 1, cv2.LINE_AA)
        cv2.line(bar, (0, 77), (width, 77), (220, 220, 220), 1)
        return bar

    @staticmethod
    def _create_frame_with_custom_layout(
        depth: np.ndarray,
        rgb: np.ndarray,
        map: np.ndarray,
        vis_map_imgs: List[np.ndarray],
        text: List[str],
        seg_map: np.ndarray,
        third_rgb: np.ndarray = None,
        vlm_input: List[np.ndarray] = None,  # 新增参数
        vlm_response: List[str] = None, 
        max_image_num: int = 3,  # 
    ) -> np.ndarray:
        """
        Create a custom layout frame with the given images:
        - Leftmost column: RGB (top), Depth (bottom).
        - Second column: Segmentation Map (top), Third RGB (bottom).
        - Third column: Map and Visual Map Images or Visual Map Images only.
        - Fourth column with a single row `map`, and padded with a black placeholder.
        - If vlm_input is provided and vis_map_imgs has 2 images, vlm_input is drawn below the fourth column.

        Args:
            depth: Depth image (H, W, 3).
            rgb: RGB image (H, W, 3).
            map: Main map image (may differ in size).
            vis_map_imgs: List of additional map images (H, W, 3 each).
            text: List of strings to overlay above the frame.
            third_rgb: Third RGB view image (H, W, 3).
            seg_map: Segmentation map (H, W, 3).
            vlm_input: List of up to three images to be drawn below the fourth column (optional).

        Returns:
            np.ndarray: Combined frame with the desired layout.
        """
        # 基准高度为 rgb 图像的高度
        base_height = rgb.shape[0]

        # 定义 vlm_input 的固定宽度和高度
        vlm_width = 300  # 每张 vlm_input 图像的固定宽度
        vlm_height = 300  # 每张 vlm_input 图像的固定高度
        vlm_gap = 20  # 图像之间的间隙

        # 定义一个辅助函数，用于调整图像高度
        def resize_to_height(image, target_height):
            h, w = image.shape[:2]
            if h == target_height:
                return image
            scale_factor = target_height / h
            new_width = int(w * scale_factor)
            resized_image = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
            return resized_image

        # 调整所有图像的高度到 base_height
        depth_resized = resize_to_height(depth, base_height)
        rgb_resized = resize_to_height(rgb, base_height)
        seg_map_resized = resize_to_height(seg_map, base_height)

        # 调整 vis_map_imgs 的高度
        vis_map_imgs_resized = [resize_to_height(img, base_height) for img in vis_map_imgs] if vis_map_imgs is not None else []

        # 调整 map 的高度
        map_resized = resize_to_height(map, base_height)

        # 第一列: RGB (上) 和 Depth (下)
        first_column = np.vstack((rgb_resized, depth_resized))

        # 第二列: Segmentation Map (上) 和 object_map (下) # Third RGB (下) 
        if third_rgb is not None:  
            third_rgb_resized = resize_to_height(third_rgb, base_height)
            second_column = np.vstack((seg_map_resized, third_rgb_resized))
        elif len(vis_map_imgs_resized) == 3:
            # 多个 vis_map_imgs: vis_map_imgs[0] (上), vis_map_imgs[1] (下)
            obj_map_img_bottom = vis_map_imgs_resized[2]
            # 确保所有 vis_map_imgs_resized 的宽度一致
            top_width = seg_map_resized.shape[1]
            bottom_width = obj_map_img_bottom.shape[1]
            if top_width != bottom_width:
                # 调整底部 vis_map_img 的宽度以匹配顶部
                obj_map_img_bottom = cv2.resize(obj_map_img_bottom, (top_width, base_height), interpolation=cv2.INTER_AREA)
            second_column = np.vstack((seg_map_resized, obj_map_img_bottom))
        else:
            second_column = seg_map_resized
            # 目标高度为第一列的高度
            target_height = first_column.shape[0]
            current_height = second_column.shape[0]

            if current_height < target_height:
                # 计算需要填充的高度
                padding_height = target_height - current_height
                # 创建黑色填充
                black_placeholder = np.zeros((padding_height, second_column.shape[1], second_column.shape[2]), dtype=second_column.dtype)
                second_column = np.vstack((second_column, black_placeholder))
            elif current_height > target_height:
                # 如果当前高度超过目标高度，进行缩放
                scale_factor = target_height / current_height
                new_width = int(second_column.shape[1] * scale_factor)
                second_column = cv2.resize(second_column, (new_width, target_height), interpolation=cv2.INTER_AREA)

        # 第三列和第四列处理
        third_column = None
        fourth_column = None

        if len(vis_map_imgs_resized) == 1:
            # 单个 vis_map_img: Map (上), vis_map_img (下)
            vis_map_img_resized = vis_map_imgs_resized[0]
            # 确保 map_resized 和 vis_map_img_resized 有相同的宽度
            map_width = map_resized.shape[1]
            vis_map_width = vis_map_img_resized.shape[1]
            if map_width != vis_map_width:
                # 调整 vis_map_img_resized 的宽度以匹配 map_resized
                vis_map_img_resized = cv2.resize(vis_map_img_resized, (map_width, base_height), interpolation=cv2.INTER_AREA)
            third_column = np.vstack((map_resized, vis_map_img_resized))
        elif len(vis_map_imgs_resized) == 2:
            # 多个 vis_map_imgs: vis_map_imgs[0] (上), vis_map_imgs[1] (下)
            vis_map_img_top = vis_map_imgs_resized[0]
            vis_map_img_bottom = vis_map_imgs_resized[1]
            # 确保所有 vis_map_imgs_resized 的宽度一致
            top_width = vis_map_img_top.shape[1]
            bottom_width = vis_map_img_bottom.shape[1]
            if top_width != bottom_width:
                # 调整底部 vis_map_img 的宽度以匹配顶部
                vis_map_img_bottom = cv2.resize(vis_map_img_bottom, (top_width, base_height), interpolation=cv2.INTER_AREA)
            third_column = np.vstack((vis_map_img_top, vis_map_img_bottom))
            fourth_column = map_resized

            # 如果有 vlm_input，将其绘制在第四列的下方
            if vlm_input is not None and len(vlm_input) > 0 and len(vlm_input) <= max_image_num:
                # 调整 vlm_input 图像的大小到固定宽度和高度
                vlm_input_resized = [cv2.resize(img, (vlm_width, vlm_height)) for img in vlm_input]

                # 计算 vlm_canvas 的总宽度
                total_vlm_width = len(vlm_input_resized) * vlm_width + (len(vlm_input_resized) - 1) * vlm_gap

                # 如果 map_resized 的宽度小于 total_vlm_width，则在右边填充黑色
                if map_resized.shape[1] < total_vlm_width:
                    padding_width = total_vlm_width - map_resized.shape[1]
                    white_padding = np.full((map_resized.shape[0], padding_width, 3), 255, dtype=np.uint8)
                    map_resized = np.hstack((map_resized, white_padding))
                    fourth_column = map_resized

                # 创建一个黑色背景的画布，用于放置 vlm_input 图像
                vlm_canvas = np.full((vlm_height, total_vlm_width, 3), 255, dtype=np.uint8)

                # 将 vlm_input 图像水平拼接
                for i, img in enumerate(vlm_input_resized):
                    x_start = i * (vlm_width + vlm_gap)
                    x_end = x_start + vlm_width
                    vlm_canvas[:, x_start:x_end] = img

                # 在 vlm_canvas 下方添加 padding_height 的空间
                padding_height = base_height - vlm_height
                # black_padding = np.zeros((padding_height, total_vlm_width, 3), dtype=np.uint8)
                white_padding = np.full((padding_height, total_vlm_width, 3), 255, dtype=np.uint8)

                # 将 vlm_canvas 和 black_padding 拼接
                vlm_section = np.vstack((vlm_canvas, white_padding))

                # 确保 vlm_section 的宽度与 fourth_column 的宽度一致
                if fourth_column.shape[1] > vlm_section.shape[1]:
                    # 如果 fourth_column 的宽度更大，则在 vlm_section 的右边填充黑色
                    padding_width = fourth_column.shape[1] - vlm_section.shape[1]
                    white_padding = np.full((vlm_section.shape[0], padding_width, 3), 255, dtype=np.uint8)
                    vlm_section = np.hstack((vlm_section, white_padding))
                elif fourth_column.shape[1] < vlm_section.shape[1]:
                    # 如果 vlm_section 的宽度更大，则在 fourth_column 的右边填充白色，适配地图底色
                    padding_width = vlm_section.shape[1] - fourth_column.shape[1]
                    # 创建一个白色的填充数组
                    white_padding = np.full((fourth_column.shape[0], padding_width, 3), 255, dtype=np.uint8)
                    # black_padding = np.zeros((fourth_column.shape[0], padding_width, 3), dtype=np.uint8)
                    fourth_column = np.hstack((fourth_column, white_padding))

                # 将 vlm_section 添加到 fourth_column 的下方
                fourth_column = np.vstack((fourth_column, vlm_section))

                # 将 vlm_response 分割成多行文本
                max_text_width = fourth_column.shape[1] - 20  # 最大文本宽度（留出左右边距）
                wrapped_text = textwrap.wrap(vlm_response, width=int(max_text_width / 10))  # 按字符数分割

                # 在 black_padding 区域添加文本
                text_height = 30  # 每行文本的高度
                for i, t in enumerate(wrapped_text[:4]):  # 只取前四行文本
                    text_y = fourth_column.shape[0] - padding_height + (i + 1) * text_height  # 从 vlm_section 的顶部开始插入文本
                    cv2.putText(
                        fourth_column,
                        t,
                        (10, text_y),  # 文本插入的位置 (x, y)
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,  # 字体大小
                        (0, 0, 0),  # 白色字体
                        2,
                        cv2.LINE_AA,
                    )
        elif len(vis_map_imgs_resized) == 3:
            # 和 2 的情况一样，多的那个替掉了第三人称视角
            # 多个 vis_map_imgs: vis_map_imgs[0] (上), vis_map_imgs[1] (下)
            vis_map_img_top = vis_map_imgs_resized[0]
            vis_map_img_bottom = vis_map_imgs_resized[1]
            # 确保所有 vis_map_imgs_resized 的宽度一致
            top_width = vis_map_img_top.shape[1]
            bottom_width = vis_map_img_bottom.shape[1]
            if top_width != bottom_width:
                # 调整底部 vis_map_img 的宽度以匹配顶部
                vis_map_img_bottom = cv2.resize(vis_map_img_bottom, (top_width, base_height), interpolation=cv2.INTER_AREA)
            third_column = np.vstack((vis_map_img_top, vis_map_img_bottom))
            fourth_column = map_resized

            # 如果有 vlm_input，将其绘制在第四列的下方
            if vlm_input is not None and len(vlm_input) > 0 and len(vlm_input) <= max_image_num:
                # 调整 vlm_input 图像的大小到固定宽度和高度
                vlm_input_resized = [cv2.resize(img, (vlm_width, vlm_height)) for img in vlm_input]

                # 计算 vlm_canvas 的总宽度
                total_vlm_width = len(vlm_input_resized) * vlm_width + (len(vlm_input_resized) - 1) * vlm_gap

                # 如果 map_resized 的宽度小于 total_vlm_width，则在右边填充黑色
                if map_resized.shape[1] < total_vlm_width:
                    padding_width = total_vlm_width - map_resized.shape[1]
                    white_padding = np.full((map_resized.shape[0], padding_width, 3), 255, dtype=np.uint8)
                    map_resized = np.hstack((map_resized, white_padding))
                    fourth_column = map_resized

                # 创建一个黑色背景的画布，用于放置 vlm_input 图像
                vlm_canvas = np.full((vlm_height, total_vlm_width, 3), 255, dtype=np.uint8)

                # 将 vlm_input 图像水平拼接
                for i, img in enumerate(vlm_input_resized):
                    x_start = i * (vlm_width + vlm_gap)
                    x_end = x_start + vlm_width
                    vlm_canvas[:, x_start:x_end] = img

                # 在 vlm_canvas 下方添加 padding_height 的空间
                padding_height = base_height - vlm_height
                # black_padding = np.zeros((padding_height, total_vlm_width, 3), dtype=np.uint8)
                white_padding = np.full((padding_height, total_vlm_width, 3), 255, dtype=np.uint8)

                # 将 vlm_canvas 和 black_padding 拼接
                vlm_section = np.vstack((vlm_canvas, white_padding))

                # 确保 vlm_section 的宽度与 fourth_column 的宽度一致
                if fourth_column.shape[1] > vlm_section.shape[1]:
                    # 如果 fourth_column 的宽度更大，则在 vlm_section 的右边填充黑色
                    padding_width = fourth_column.shape[1] - vlm_section.shape[1]
                    white_padding = np.full((vlm_section.shape[0], padding_width, 3), 255, dtype=np.uint8)
                    vlm_section = np.hstack((vlm_section, white_padding))
                elif fourth_column.shape[1] < vlm_section.shape[1]:
                    # 如果 vlm_section 的宽度更大，则在 fourth_column 的右边填充白色，适配地图底色
                    padding_width = vlm_section.shape[1] - fourth_column.shape[1]
                    # 创建一个白色的填充数组
                    white_padding = np.full((fourth_column.shape[0], padding_width, 3), 255, dtype=np.uint8)
                    # black_padding = np.zeros((fourth_column.shape[0], padding_width, 3), dtype=np.uint8)
                    fourth_column = np.hstack((fourth_column, white_padding))

                # 将 vlm_section 添加到 fourth_column 的下方
                fourth_column = np.vstack((fourth_column, vlm_section))

                # 将 vlm_response 分割成多行文本
                max_text_width = fourth_column.shape[1] - 20  # 最大文本宽度（留出左右边距）
                wrapped_text = textwrap.wrap(vlm_response, width=int(max_text_width / 10))  # 按字符数分割

                # 在 black_padding 区域添加文本
                text_height = 30  # 每行文本的高度
                for i, t in enumerate(wrapped_text[:4]):  # 只取前四行文本
                    text_y = fourth_column.shape[0] - padding_height + (i + 1) * text_height  # 从 vlm_section 的顶部开始插入文本
                    cv2.putText(
                        fourth_column,
                        t,
                        (10, text_y),  # 文本插入的位置 (x, y)
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,  # 字体大小
                        (0, 0, 0),  # 白色字体
                        2,
                        cv2.LINE_AA,
                    )
        else:
            # 没有 vis_map_imgs: 只有 map
            third_column = map_resized

        # 处理第四列（如果存在）
        if fourth_column is not None:
            # 目标高度为第一列的高度
            target_height = first_column.shape[0]
            current_height = fourth_column.shape[0]

            if current_height < target_height:
                # 计算需要填充的高度
                padding_height = target_height - current_height
                # 创建黑色填充
                black_placeholder = np.zeros((padding_height, fourth_column.shape[1], fourth_column.shape[2]), dtype=fourth_column.dtype)
                fourth_column = np.vstack((fourth_column, black_placeholder))
            elif current_height > target_height:
                # 如果当前高度超过目标高度，进行缩放
                scale_factor = target_height / current_height
                new_width = int(fourth_column.shape[1] * scale_factor)
                fourth_column = cv2.resize(fourth_column, (new_width, target_height), interpolation=cv2.INTER_AREA)

        # 组合所有列
        if fourth_column is not None:
            main_frame = np.hstack((first_column, second_column, third_column, fourth_column))
        else:
            main_frame = np.hstack((first_column, second_column, third_column))

        # 添加文本注释到图像顶部
        if text is not None:
            for t in text[::-1]:
                main_frame = add_text_to_image(main_frame, t, top=True)

        return main_frame

    @staticmethod
    def _create_frame_with_third_view(
        depth: np.ndarray,
        rgb: np.ndarray,
        map: np.ndarray,
        vis_map_imgs: List[np.ndarray],
        text: List[str],
        third_rgb: np.ndarray,
    ) -> np.ndarray:
        """Create a frame using all the given images.

        First, the depth and rgb images are stacked vertically. Then, all the maps are
        combined as a separate images. Then these two images should be stitched together
        horizontally (depth-rgb on the left, maps on the right).

        The combined map image contains two rows of images and at least one column.
        First, the 'map' argument is at the top left, then the first element of the
        'vis_map_imgs' argument is at the bottom left. If there are more than one
        element in 'vis_map_imgs', then the second element is at the top right, the
        third element is at the bottom right, and so on.

        Args:
            depth: The depth image (H, W, 3).
            rgb: The rgb image (H, W, 3).
            map: The map image, a 3-channel rgb image, but can have different shape from
                depth and rgb.
            vis_map_imgs: A list of other map images. Each are 3-channel rgb images, but
                can have different sizes.
            text: A list of strings to be rendered above the images.
        Returns:
            np.ndarray: The combined frame image.
        """
        # 基准高度为 rgb 图像的高度
        base_height = rgb.shape[0]

        # 定义一个辅助函数，用于调整图像高度
        def resize_to_height(image, target_height):
            h, w = image.shape[:2]
            if h == target_height:
                return image
            scale_factor = target_height / h
            new_width = int(w * scale_factor)
            resized_image = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
            return resized_image

        # 调整所有图像的高度到 base_height
        depth_resized = resize_to_height(depth, base_height)
        rgb_resized = resize_to_height(rgb, base_height)
        # seg_map_resized = resize_to_height(seg_map, base_height)
        third_rgb_resized = resize_to_height(third_rgb, base_height)
        
        # 调整 vis_map_imgs 的高度
        obstacle_map = vis_map_imgs[0]
        value_map = vis_map_imgs[1]
        obstacle_map_resized = resize_to_height(obstacle_map, base_height)
        value_map_resized = resize_to_height(value_map, base_height)
        
        # 调整 map 的高度
        map_resized = resize_to_height(map, base_height)

        # 第一列: Depth (上) 和 obs_map (下)
        obstacle_map_width = obstacle_map_resized.shape[1]
        depth_width = depth_resized.shape[1]
        if obstacle_width != depth_width:
            # 调整底部 vis_map_img 的宽度以匹配顶部
            obstacle_width = cv2.resize(obstacle_map_width, (depth_width, base_height), interpolation=cv2.INTER_AREA)
        first_column = np.vstack((depth_resized, obstacle_map_resized))

        # 第二列: RGB (上) 和 value_map (下)
        rgb_width = rgb_resized.shape[1]
        value_map_width = value_map_resized.shape[1]
        if value_map_width != rgb_width:
            # 调整底部 vis_map_img 的宽度以匹配顶部
            value_map_width = cv2.resize(obstacle_width, (rgb_width, base_height), interpolation=cv2.INTER_AREA)
        second_column = np.vstack((rgb_resized, value_map_resized))

        # 第三列： Third RGB （上）和 top down map 下
        third_rgb_width = third_rgb_resized.shape[1]
        map_width = map_resized.shape[1]
        if map_width != third_rgb_width:
            # 调整底部 vis_map_img 的宽度以匹配顶部
            map_width = cv2.resize(map_width, (third_rgb_width, base_height), interpolation=cv2.INTER_AREA)
        third_column = np.vstack((third_rgb_resized, map_width))

        # 多个 vis_map_imgs: vis_map_imgs[0] (上), vis_map_imgs[1] (下)

        main_frame = np.hstack((first_column, second_column, third_column))

        # 添加文本注释到图像顶部
        for t in text[::-1]:
            main_frame = add_text_to_image(main_frame, t, top=True)

        return main_frame

def sim_xy_to_grid_xy(
    upper_bound: Tuple[int, int],
    lower_bound: Tuple[int, int],
    grid_resolution: Tuple[int, int],
    sim_xy: np.ndarray,
    remove_duplicates: bool = True,
) -> np.ndarray:
    """Converts simulation coordinates to grid coordinates.

    Args:
        upper_bound (Tuple[int, int]): The upper bound of the grid.
        lower_bound (Tuple[int, int]): The lower bound of the grid.
        grid_resolution (Tuple[int, int]): The resolution of the grid.
        sim_xy (np.ndarray): A numpy array of 2D simulation coordinates.
        remove_duplicates (bool): Whether to remove duplicate grid coordinates.

    Returns:
        np.ndarray: A numpy array of 2D grid coordinates.
    """
    grid_size = np.array(
        [
            abs(upper_bound[1] - lower_bound[1]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        ]
    )
    grid_xy = ((sim_xy - lower_bound[::-1]) / grid_size).astype(int)

    if remove_duplicates:
        grid_xy = np.unique(grid_xy, axis=0)

    return grid_xy


def color_point_cloud_on_map(infos: List[Dict[str, Any]], policy_info: List[Dict[str, Any]], env: int = 0) -> None:
    if policy_info is None or "target_point_cloud" not in policy_info[env] or len(policy_info[env]["target_point_cloud"]) == 0:
        return

    upper_bound = infos[env]["top_down_map"]["upper_bound"]
    lower_bound = infos[env]["top_down_map"]["lower_bound"]
    grid_resolution = infos[env]["top_down_map"]["grid_resolution"]
    tf_episodic_to_global = infos[env]["top_down_map"]["tf_episodic_to_global"]

    cloud_episodic_frame = policy_info[env]["target_point_cloud"][:, :3]
    cloud_global_frame_xyz = transform_points(tf_episodic_to_global, cloud_episodic_frame)
    cloud_global_frame_habitat = xyz_to_habitat(cloud_global_frame_xyz)
    cloud_global_frame_habitat_xy = cloud_global_frame_habitat[:, [2, 0]]

    grid_xy = sim_xy_to_grid_xy(
        upper_bound,
        lower_bound,
        grid_resolution,
        cloud_global_frame_habitat_xy,
        remove_duplicates=True,
    )

    new_map = infos[env]["top_down_map"]["map"].copy()
    try: 
        new_map[grid_xy[:, 0], grid_xy[:, 1]] = MAP_TARGET_POINT_INDICATOR 
    except: 
        print("Seems reach to a wrong floor, the goal is not on the map.")
    # new_map[grid_xy[:, 0], grid_xy[:, 1]] = MAP_TARGET_POINT_INDICATOR # 可能报错，到了错误的楼层但是目标不在这一层。

    infos[env]["top_down_map"]["map"] = new_map


def overlay_frame(frame: np.ndarray, info: Dict[str, Any], additional: Optional[List[str]] = None) -> np.ndarray:
    """
    Renders text from the `info` dictionary to the `frame` image.
    """

    lines = []
    flattened_info = flatten_dict(info)
    for k, v in flattened_info.items():
        if isinstance(v, str):
            lines.append(f"{k}: {v}")
        else:
            try:
                lines.append(f"{k}: {v:.2f}")
            except TypeError:
                pass
    if additional is not None:
        lines.extend(additional)

    frame = overlay_text_to_image(frame, lines, font_size=0.25)

    return frame
