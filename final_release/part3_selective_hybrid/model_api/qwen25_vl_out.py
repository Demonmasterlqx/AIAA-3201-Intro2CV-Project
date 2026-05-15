import os
import random
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from model_api.server_wrapper_out import ServerMixin, host_model, send_request_vlm, str_to_image

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover - runtime dependency probe
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None


DEFAULT_QWEN_VL_MODEL_PATH = os.environ.get(
    "QWEN2_5_VL_MODEL_PATH",
    "pretrained_weights/Qwen2.5-VL-7B-Instruct",
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class QWen2_5_VL:
    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_VL_MODEL_PATH,
        device: Optional[str] = None,
        seed: int = 2025,
    ) -> None:
        set_seed(seed)
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_error = ""
        self._generate_kwargs = {
            "max_new_tokens": int(os.environ.get("QWEN2_5_VL_MAX_NEW_TOKENS", "256")),
            "do_sample": False,
        }

        if AutoProcessor is None or Qwen2_5_VLForConditionalGeneration is None:
            self._load_error = "transformers Qwen2.5-VL classes are unavailable"
            return
        if not os.path.exists(model_name):
            self._load_error = f"model path not found: {model_name}"
            return

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map={"": self.device},
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception as exc:  # pragma: no cover - runtime model init
            self._load_error = str(exc)
            self.model = None
            self.processor = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.processor is not None

    def chat_image(self, image: np.ndarray, txt: str) -> str:
        if not self.is_ready:
            raise RuntimeError(self._load_error or "Qwen-VL model is not ready")

        pil_img = Image.fromarray(image.astype(np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": txt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt],
            images=[pil_img],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        generated_ids = self.model.generate(**inputs, **self._generate_kwargs)
        trimmed_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return response[0] if response else ""


class Qwen2_5_VLClient:
    def __init__(self, port: int = 13187) -> None:
        host = os.getenv("OUT_HOST", "localhost")
        self.url = f"http://{host}:{port}/qwen2_5_vl"
        self.timeout_seconds = int(os.getenv("QWEN2_5_VL_TIMEOUT", "60"))

    def chat_image(self, image: np.ndarray, txt: str) -> str:
        try:
            response = send_request_vlm(
                self.url,
                timeout=self.timeout_seconds,
                image=image,
                txt=txt,
            )
            return str(response["response"])
        except Exception as exc:
            print(f"Qwen2_5_VL request failed: {exc}")
            return "-1"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("QWEN2_5_VL_PORT", "13187")))
    parser.add_argument("--model-path", type=str, default=DEFAULT_QWEN_VL_MODEL_PATH)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    class Qwen2_5_VLServer(ServerMixin, QWen2_5_VL):
        def __init__(self) -> None:
            QWen2_5_VL.__init__(self, model_name=args.model_path, device=args.device)
            ServerMixin.__init__(self)

        def process_payload(self, payload: dict) -> dict:
            if payload is None:
                return {
                    "status": "ok" if self.is_ready else "degraded",
                    "message": "Qwen2.5-VL service",
                    "model_path": self.model_name,
                    "ready": self.is_ready,
                    "error": self._load_error,
                }

            image = str_to_image(payload.get("image"))
            return {"response": self.chat_image(image, payload["txt"])}

    qwen_vl = Qwen2_5_VLServer()
    print(f"Qwen2.5-VL ready={qwen_vl.is_ready} path={qwen_vl.model_name} device={qwen_vl.device}")
    host_model(qwen_vl, name="qwen2_5_vl", port=args.port)
