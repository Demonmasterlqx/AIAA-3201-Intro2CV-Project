from __future__ import annotations

import json
import math
from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional


class ASEpisodeLogger:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.episode_log_path = self.run_dir / "episode_events.jsonl"
        self.summary_log_path = self.run_dir / "episode_summary.jsonl"

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._append_jsonl(
            self.episode_log_path,
            {
                "event_type": event_type,
                **self._serialize(payload),
            },
        )

    def log_summary(self, payload: Dict[str, Any]) -> None:
        self._append_jsonl(self.summary_log_path, self._serialize(payload))

    def dump_artifact(
        self, artifact_name: str, payload: Dict[str, Any], subdir: Optional[str] = None
    ) -> Path:
        target_dir = self.run_dir if subdir is None else (self.run_dir / subdir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / artifact_name
        target_path.write_text(
            json.dumps(self._serialize(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return target_path

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._serialize(payload), sort_keys=True))
            handle.write("\n")

    def _serialize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {key: _normalize(value) for key, value in payload.items()}


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value
    return value
