#!/usr/bin/env python3
from __future__ import annotations

import run_as_ascent_eval_1pct as runner


runner.DEFAULT_EPISODE_COUNT = 2
runner.RESULT_ROOT = runner.REPO_ROOT / "results" / "as_ascent" / "eval_10pct"


if __name__ == "__main__":
    raise SystemExit(runner.main())
