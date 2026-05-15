from dataclasses import dataclass
from typing import Iterable, List, Optional

import clip
import numpy as np
import torch

from zson.transforms import get_transform


@dataclass
class DenseImageFeatures:
    global_feature: np.ndarray
    patch_features: np.ndarray


class ClipDenseEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        input_size: int = 224,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model_name = model_name
        self.model, _ = clip.load(model_name, device=self.device, jit=False)
        self.model.eval()
        self.transform = get_transform("clip", size=input_size)
        if not hasattr(self.model.visual, "conv1"):
            raise ValueError(
                "Plan A expects a ViT-style CLIP image encoder with patch tokens"
            )
        grid_size = getattr(self.model.visual, "grid_size", None)
        if isinstance(grid_size, tuple):
            self.patch_grid = int(grid_size[0])
        elif grid_size is not None:
            self.patch_grid = int(grid_size)
        else:
            self.patch_grid = input_size // 32
        self.feature_dim = int(self.model.visual.output_dim)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    @torch.no_grad()
    def encode_texts(self, prompts: Iterable[str]) -> np.ndarray:
        prompts = list(prompts)
        if len(prompts) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        tokens = clip.tokenize(prompts, context_length=77).to(self.device)
        embeddings = self.model.encode_text(tokens).float()
        embeddings = self._normalize(embeddings)
        return embeddings.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, prompt: str) -> np.ndarray:
        return self.encode_texts([prompt])[0]

    @torch.no_grad()
    def encode_image_patches(self, rgb: np.ndarray) -> DenseImageFeatures:
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        tensor = torch.from_numpy(rgb[None]).to(self.device)
        tensor = self.transform(tensor)
        tensor = tensor.to(dtype=self.model.visual.conv1.weight.dtype)
        visual = self.model.visual

        x = visual.conv1(tensor)  # N, C, H, W
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(
            (x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj

        x = self._normalize(x.float())
        global_feature = x[:, 0, :]
        patch_features = x[:, 1:, :].reshape(
            x.shape[0], self.patch_grid, self.patch_grid, x.shape[-1]
        )
        return DenseImageFeatures(
            global_feature=global_feature[0].cpu().numpy().astype(np.float32),
            patch_features=patch_features[0].cpu().numpy().astype(np.float32),
        )
