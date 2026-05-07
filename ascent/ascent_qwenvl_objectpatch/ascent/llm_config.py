from __future__ import annotations

from dataclasses import dataclass

from model_api.llm_backends import DEFAULT_DEEPSEEK_MODEL

DEFAULT_QWEN_VL_MODEL_PATH = "pretrained_weights/Qwen2.5-VL-7B-Instruct"


@dataclass
class LLMBackendConfig:
    backend: str = "deepseek_api"
    model_name: str = DEFAULT_DEEPSEEK_MODEL
    base_url: str = ""
    api_key: str = ""
    timeout_seconds: int = 20
    max_retries: int = 3
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass
class VisionLLMBackendConfig:
    backend: str = "local_qwen_vl"
    model_name: str = "Qwen2.5-VL-7B-Instruct"
    model_path: str = DEFAULT_QWEN_VL_MODEL_PATH
    port_env: str = "QWEN2_5_VL_PORT"
    timeout_seconds: int = 60
    max_retries: int = 2
    max_tokens: int = 256
    temperature: float = 0.0
    confidence_threshold: float = 0.70
    strong_confidence_threshold: float = 0.99
    confirmation_confidence_threshold: float = 0.95
    weak_confirmation_confidence_threshold: float = 0.90
    weak_confirmation_min_supports: int = 3
    confirmation_window_steps: int = 8
    confirmation_iou_threshold: float = 0.70
