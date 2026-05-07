import json
import os
from typing import Dict

import numpy as np


def normalize_category(category: str) -> str:
    return category.replace("_", " ").strip()


def build_prompt(prompt_template: str, category: str) -> str:
    return prompt_template.format(cat=normalize_category(category))


class TextGoalCache:
    def __init__(self, encoder, prompt_template: str, cache_dir: str):
        self.encoder = encoder
        self.prompt_template = prompt_template
        self.cache_path = os.path.join(cache_dir, "text_goal_cache.json")
        self._entries: Dict[str, Dict[str, object]] = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as file:
                raw_entries = json.load(file)
            for category, payload in raw_entries.items():
                self._entries[category] = {
                    "prompt": payload["prompt"],
                    "embedding": np.asarray(payload["embedding"], dtype=np.float32),
                }

    def get_embedding(self, category: str) -> np.ndarray:
        if category not in self._entries:
            prompt = build_prompt(self.prompt_template, category)
            embedding = self.encoder.encode_text(prompt)
            self._entries[category] = {
                "prompt": prompt,
                "embedding": embedding.astype(np.float32),
            }
        return self._entries[category]["embedding"]  # type: ignore[return-value]

    def dump(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        serializable = {}
        for category, payload in self._entries.items():
            serializable[category] = {
                "prompt": payload["prompt"],
                "embedding": np.asarray(payload["embedding"]).tolist(),
            }
        with open(self.cache_path, "w") as file:
            json.dump(serializable, file, indent=2)
