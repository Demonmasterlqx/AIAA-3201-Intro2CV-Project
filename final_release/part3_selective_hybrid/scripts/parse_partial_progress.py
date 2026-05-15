#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


PROGRESS_RE = re.compile(
    r"Till Now Average Success rate:\s*([0-9.]+)%\s*\(([0-9.]+)\s+out of\s+([0-9]+)\)"
)
SPL_RE = re.compile(r"Till Now Average Spl:\s*([0-9.]+)%")
DTG_RE = re.compile(r"Till Now Average Dtg:\s*([-+0-9.eE]+)")
FAIL_RE = re.compile(r"failed due to '([^']+)'")
FINAL_RE = re.compile(
    r"Final check mode=([^ ]+) confidence=([-+0-9.eE]+), decision=([^,]+), tristate=([^,]+), patches=([0-9.]+)"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize an in-progress ASCENT run.log.")
    parser.add_argument("run_log", type=Path)
    args = parser.parse_args()

    text = args.run_log.read_text(encoding="utf-8", errors="replace") if args.run_log.exists() else ""
    progress = PROGRESS_RE.findall(text)
    spl = SPL_RE.findall(text)
    dtg = DTG_RE.findall(text)
    failures = FAIL_RE.findall(text)
    final_events = FINAL_RE.findall(text)

    final_modes = Counter(event[0] for event in final_events)
    final_decisions = Counter(event[2].strip() for event in final_events)
    final_tristates = Counter(event[3].strip() for event in final_events)

    summary = {
        "running_log": str(args.run_log),
        "progress": None,
        "spl_percent": float(spl[-1]) if spl else None,
        "dtg": float(dtg[-1]) if dtg else None,
        "failure_buckets": dict(Counter(failures)),
        "num_failures": len(failures),
        "final_modes": dict(final_modes),
        "final_decisions": dict(final_decisions),
        "final_tristates": dict(final_tristates),
    }
    if progress:
        sr_percent, successes, episodes = progress[-1]
        summary["progress"] = {
            "sr_percent": float(sr_percent),
            "successes": float(successes),
            "episodes": int(episodes),
        }

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
