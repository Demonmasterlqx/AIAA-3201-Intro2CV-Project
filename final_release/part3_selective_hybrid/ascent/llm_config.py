from __future__ import annotations

from dataclasses import dataclass

from model_api.llm_backends import DEFAULT_DEEPSEEK_MODEL

DEFAULT_QWEN_VL_MODEL_PATH = "pretrained_weights/Qwen2.5-VL-7B-Instruct"


@dataclass
class LLMBackendConfig:
    backend: str = "local_qwen"
    model_name: str = "Qwen2.5-7B-Instruct"
    base_url: str = ""
    api_key: str = ""
    timeout_seconds: int = 20
    max_retries: int = 3
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass
class VisionLLMBackendConfig:
    mode: str = "blip2"
    backend: str = "local_qwen_vl"
    model_name: str = "Qwen2.5-VL-7B-Instruct"
    model_path: str = DEFAULT_QWEN_VL_MODEL_PATH
    port_env: str = "QWEN2_5_VL_PORT"
    timeout_seconds: int = 60
    max_retries: int = 2
    max_tokens: int = 256
    temperature: float = 0.0
    blip2_confidence_threshold: float = 0.15
    confidence_threshold: float = 0.70
    strong_confidence_threshold: float = 0.99
    confirmation_confidence_threshold: float = 0.95
    confirmation_window_steps: int = 8
    confirmation_iou_threshold: float = 0.70
    hybrid_high_risk_categories: str = "tv,tv_monitor,couch,sofa,bed,potted_plant,plant,toilet"
    hybrid_blip_low_threshold: float = 0.16
    hybrid_blip_high_threshold: float = 0.34
    hybrid_accept_on_qwen_reject_if_blip_high: bool = True
    hybrid_accept_tv_ambiguous_uncertain_if_blip_accept: bool = True
    tristate_enabled: bool = True
    uncertain_confidence_threshold: float = 0.55
    tv_ambiguous_false_to_uncertain_enabled: bool = True
    category_thresholds_json: str = (
        '{"tv": {"confidence": 0.82, "strong": 0.97, "confirmation": 0.92}, '
        '"tv_monitor": {"confidence": 0.82, "strong": 0.97, "confirmation": 0.92}, '
        '"potted_plant": {"confidence": 0.86, "strong": 0.98, "confirmation": 0.94}, '
        '"plant": {"confidence": 0.86, "strong": 0.98, "confirmation": 0.94}, '
        '"toilet": {"confidence": 0.64, "strong": 0.94, "confirmation": 0.86}, '
        '"couch": {"confidence": 0.76, "strong": 0.96, "confirmation": 0.90}, '
        '"sofa": {"confidence": 0.76, "strong": 0.96, "confirmation": 0.90}, '
        '"bed": {"confidence": 0.74, "strong": 0.96, "confirmation": 0.90}, '
        '"chair": {"confidence": 0.78, "strong": 0.96, "confirmation": 0.90}}'
    )
    cache_pose_decay: float = 0.75
    cache_heading_decay_deg: float = 45.0
    cache_max_age_steps: int = 10
    qwen_approach_standoff: float = 0.55
    qwen_close_stop_radius: float = 0.60
    qwen_stalled_stop_radius: float = 0.82
    qwen_stall_steps: int = 8
    qwen_approach_recovery_enabled: bool = False
    suppress_distant_budget_stop: bool = False
    suppress_rejected_patch_enabled: bool = False
    suppress_rejected_patch_requires_blip_accept: bool = True
    suppress_rejected_patch_max_patches: int = 3
    suppress_rejected_patch_min_points: int = 20
    suppress_rejected_patch_dilation_px: int = 2


@dataclass
class FrontierRecoveryConfig:
    stuck_recovery_enabled: bool = False


@dataclass
class StairBudgetConfig:
    floor_budget_enabled: bool = False
