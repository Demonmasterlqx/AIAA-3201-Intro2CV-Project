from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from urllib.parse import urlparse

import requests

from model_api.qwen25_out import Qwen2_5Client
from model_api.qwen25_vl_out import Qwen2_5_VLClient


LOGGER = logging.getLogger(__name__)
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


class BaseLLMClient:
    backend_name: str = "unknown"

    def chat(self, txt: str) -> str:
        raise NotImplementedError

    def get_logging_info(self) -> Dict[str, Any]:
        raise NotImplementedError


class BaseVisionLLMClient:
    backend_name: str = "unknown_vision"

    def chat_image(self, image, txt: str) -> str:
        raise NotImplementedError

    def get_logging_info(self) -> Dict[str, Any]:
        raise NotImplementedError


def load_env_file(env_path: Optional[str] = None) -> None:
    target_path = Path(env_path) if env_path is not None else Path(__file__).resolve().parents[1] / ".env"
    if not target_path.exists():
        return
    for raw_line in target_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def create_llm_client(llm_config: Any) -> BaseLLMClient:
    load_env_file()
    backend = getattr(llm_config, "backend", "deepseek_api")
    if backend == "deepseek_api":
        return DeepSeekAPILLMClient(
            base_url=getattr(llm_config, "base_url", None),
            api_key=getattr(llm_config, "api_key", None),
            model_name=getattr(llm_config, "model_name", DEFAULT_DEEPSEEK_MODEL),
            timeout_seconds=getattr(llm_config, "timeout_seconds", 20),
            max_retries=getattr(llm_config, "max_retries", 3),
            max_tokens=getattr(llm_config, "max_tokens", 512),
            temperature=getattr(llm_config, "temperature", 0.0),
        )
    if backend == "local_qwen":
        return LocalQwenLLMClient(
            port=int(os.environ.get("QWEN2_5_PORT", "13181")),
            host=os.environ.get("OUT_HOST", "localhost"),
            model_name=getattr(llm_config, "model_name", "Qwen2.5-7b"),
        )
    raise ValueError(f"Unsupported LLM backend: {backend}")


def create_vision_llm_client(llm_config: Any) -> BaseVisionLLMClient:
    load_env_file()
    backend = getattr(llm_config, "backend", "local_qwen_vl")
    if backend == "local_qwen_vl":
        port_env = getattr(llm_config, "port_env", "QWEN2_5_VL_PORT")
        port = int(os.environ.get(port_env, "13187"))
        return LocalQwenVLClient(
            port=port,
            host=os.environ.get("OUT_HOST", "localhost"),
            model_name=getattr(llm_config, "model_name", "Qwen2.5-VL-7B-Instruct"),
            timeout_seconds=getattr(llm_config, "timeout_seconds", 60),
        )
    raise ValueError(f"Unsupported vision LLM backend: {backend}")


class LocalQwenLLMClient(BaseLLMClient):
    backend_name = "local_qwen"

    def __init__(self, port: int, host: str, model_name: str) -> None:
        self._client = Qwen2_5Client(port=port)
        self._host = host
        self._port = port
        self._model_name = model_name
        self._last_retry_count = 0
        self._last_error = ""
        self._health_url = f"http://{host}:{port}/qwen2_5"

    def chat(self, txt: str) -> str:
        try:
            requests.get(self._health_url, timeout=1)
        except Exception as exc:
            self._last_error = f"local_qwen_unavailable: {exc}"
            self._last_retry_count = 0
            LOGGER.warning("Local Qwen backend is unavailable at %s:%s", self._host, self._port)
            return "-1"
        response = self._client.chat(txt)
        if response == "-1":
            self._last_error = "local_qwen_request_failed"
        else:
            self._last_error = ""
        self._last_retry_count = 0
        return response

    def get_logging_info(self) -> Dict[str, Any]:
        return {
            "llm_backend": self.backend_name,
            "llm_model": self._model_name,
            "llm_endpoint_host": self._host,
            "llm_endpoint_port": self._port,
            "llm_retry_count": self._last_retry_count,
            "llm_last_error": self._last_error,
        }


class DeepSeekAPILLMClient(BaseLLMClient):
    backend_name = "deepseek_api"

    def __init__(
        self,
        *,
        base_url: Optional[str],
        api_key: Optional[str],
        model_name: str,
        timeout_seconds: int,
        max_retries: int,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self._base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._model_name = os.environ.get("DEEPSEEK_MODEL", model_name or DEFAULT_DEEPSEEK_MODEL)
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._chat_url = _resolve_deepseek_chat_url(self._base_url)
        self._last_retry_count = 0
        self._last_error = ""
        if not self._api_key:
            raise ValueError("DEEPSEEK_API_KEY is required when backend=deepseek_api")

    def chat(self, txt: str) -> str:
        payload = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI assistant with advanced spatial reasoning capabilities. Your task is to choose the optimal option to find the target object.",
                },
                {"role": "user", "content": txt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._last_retry_count = 0
        self._last_error = ""
        for attempt in range(self._max_retries):
            try:
                response = requests.post(
                    self._chat_url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                content = _extract_openai_style_content(data)
                self._last_retry_count = attempt
                return content
            except Exception as exc:
                self._last_retry_count = attempt + 1
                self._last_error = str(exc)
                LOGGER.warning(
                    "DeepSeek request failed on attempt %s/%s: %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if attempt + 1 < self._max_retries:
                    time.sleep(min(2 * (attempt + 1), 5))
        return "-1"

    def get_logging_info(self) -> Dict[str, Any]:
        return {
            "llm_backend": self.backend_name,
            "llm_model": self._model_name,
            "llm_endpoint_host": urlparse(self._base_url).netloc,
            "llm_endpoint_port": "",
            "llm_retry_count": self._last_retry_count,
            "llm_last_error": self._last_error,
        }


class LocalQwenVLClient(BaseVisionLLMClient):
    backend_name = "local_qwen_vl"

    def __init__(self, port: int, host: str, model_name: str, timeout_seconds: int) -> None:
        self._client = Qwen2_5_VLClient(port=port)
        self._host = host
        self._port = port
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._last_error = ""
        self._last_retry_count = 0
        self._health_url = f"http://{host}:{port}/qwen2_5_vl"
        os.environ.setdefault("QWEN2_5_VL_TIMEOUT", str(timeout_seconds))

    def chat_image(self, image, txt: str) -> str:
        try:
            requests.get(self._health_url, timeout=2)
        except Exception as exc:
            self._last_error = f"local_qwen_vl_unavailable: {exc}"
            self._last_retry_count = 0
            LOGGER.warning("Local Qwen-VL backend is unavailable at %s:%s", self._host, self._port)
            return "-1"
        response = self._client.chat_image(image, txt)
        if response == "-1":
            self._last_error = "local_qwen_vl_request_failed"
        else:
            self._last_error = ""
        self._last_retry_count = 0
        return response

    def get_logging_info(self) -> Dict[str, Any]:
        return {
            "llm_backend": self.backend_name,
            "llm_model": self._model_name,
            "llm_endpoint_host": self._host,
            "llm_endpoint_port": self._port,
            "llm_retry_count": self._last_retry_count,
            "llm_last_error": self._last_error,
        }


def _resolve_deepseek_chat_url(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/chat/completions"


def _extract_openai_style_content(data: Dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return "-1"
    first_choice = choices[0]
    message = first_choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(parts) or "-1"
    return content or "-1"
