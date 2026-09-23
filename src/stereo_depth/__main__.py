from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .pipeline import StereoConfig, StereoDepthPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate disparity, metric depth and obstacle outputs.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--config", default=None)
    parser.add_argument("--baseline", type=float, default=None)
    parser.add_argument("--focal-length", type=float, default=None)
    args = parser.parse_args()
    values = {}
    if args.config:
        values = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if args.baseline is not None:
        values["baseline_m"] = args.baseline
    if args.focal_length is not None:
        values["focal_length_px"] = args.focal_length
    pipeline = StereoDepthPipeline(StereoConfig(**values))
    left, right = pipeline.read_pair(args.left, args.right)
    result = pipeline.run(left, right)
    pipeline.save_result(result, args.output)
    nearest = result["nearest_m"]
    print(f"Saved outputs to {args.output}; nearest valid depth: {nearest:.2f} m" if nearest else f"Saved outputs to {args.output}; no valid depth")


if __name__ == "__main__":
    main()
