#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_metrics(result_dir: Path) -> Dict[str, Any]:
    path = result_dir / "metrics_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare fixed-subset ASCENT baseline and variant metrics.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    baseline = load_metrics(args.baseline)
    variant = load_metrics(args.variant)
    b_sr = float(baseline.get("sr") or 0.0)
    v_sr = float(variant.get("sr") or 0.0)
    b_spl = float(baseline.get("spl") or 0.0)
    v_spl = float(variant.get("spl") or 0.0)
    b_steps = float(baseline.get("steps") or 0.0)
    v_steps = float(variant.get("steps") or 0.0)

    passed = v_sr > b_sr or (v_sr == b_sr and v_spl > b_spl and v_steps <= b_steps * 1.05)
    report = {
        "passed": passed,
        "baseline": baseline,
        "variant": variant,
        "delta": {
            "sr": v_sr - b_sr,
            "sr_percent_points": (v_sr - b_sr) * 100.0,
            "spl": v_spl - b_spl,
            "spl_percent_points": (v_spl - b_spl) * 100.0,
            "dtg": (variant.get("dtg") or 0.0) - (baseline.get("dtg") or 0.0),
            "steps": v_steps - b_steps,
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
