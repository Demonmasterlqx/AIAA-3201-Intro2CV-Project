from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Dict


class GEpisodeLogger:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "episode_events.jsonl"
        self.summary_path = self.run_dir / "episode_summary.jsonl"
        self.snapshot_path = self.run_dir / "region_graph_snapshot.json"

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._append(self.events_path, {"event_type": event_type, **payload})

    def log_summary(self, payload: Dict[str, Any]) -> None:
        self._append(self.summary_path, payload)

    def save_snapshot(self, payload: Dict[str, Any]) -> None:
        self.snapshot_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _append(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
