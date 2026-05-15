from __future__ import annotations

from dataclasses import dataclass

from model_api.llm_backends import DEFAULT_DEEPSEEK_MODEL


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
