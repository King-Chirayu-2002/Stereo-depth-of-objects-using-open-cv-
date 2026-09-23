from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class StereoConfig:
    baseline_m: float = 0.075
    focal_length_px: float = 520.0
    min_disparity: int = 0
    num_disparities: int = 160
    block_size: int = 5
    min_depth_m: float = 0.20
    max_depth_m: float = 8.0
    roi_fraction: float = 0.35

    def __post_init__(self) -> None:
        self.num_disparities = max(16, int(self.num_disparities // 16 * 16))
        self.block_size = max(3, int(self.block_size) | 1)


class StereoDepthPipeline:
    """Compute disparity, metric depth, obstacle mask and a compact point cloud."""

    def __init__(self, config: StereoConfig | None = None) -> None:
        self.config = config or StereoConfig()

    @staticmethod
    def read_pair(left_path: str | Path, right_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise FileNotFoundError("Both --left and --right images must exist and be readable.")
        if left.shape[:2] != right.shape[:2]:
            raise ValueError("Left and right images must have the same dimensions.")
        return left, right

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    def compute_disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        matcher = cv2.StereoSGBM_create(
            minDisparity=self.config.min_disparity,
            numDisparities=self.config.num_disparities,
            blockSize=self.config.block_size,
            P1=8 * self.config.block_size**2,
            P2=32 * self.config.block_size**2,
            uniquenessRatio=8,
            speckleWindowSize=80,
            speckleRange=2,
            disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        return matcher.compute(self._gray(left), self._gray(right)).astype(np.float32) / 16.0

    def depth_from_disparity(self, disparity: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = self.config.focal_length_px * self.config.baseline_m / disparity
        return np.where(
            (disparity > 0) & (depth >= self.config.min_depth_m) & (depth <= self.config.max_depth_m),
            depth,
            0.0,
        ).astype(np.float32)

    def obstacle_mask(self, depth: np.ndarray) -> np.ndarray:
        h, w = depth.shape
        width = max(1, int(w * self.config.roi_fraction))
        x0, x1 = (w - width) // 2, (w + width) // 2
        roi = depth[:, x0:x1]
        valid = (roi > 0) & (roi <= self.config.max_depth_m)
        mask = np.zeros_like(depth, dtype=np.uint8)
        mask[:, x0:x1] = (valid * 255).astype(np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    def nearest_distance(self, depth: np.ndarray) -> float | None:
        values = depth[depth > 0]
        return float(values.min()) if values.size else None

    def point_cloud(self, left: np.ndarray, depth: np.ndarray, stride: int = 4) -> np.ndarray:
        h, w = depth.shape
        f, b = self.config.focal_length_px, self.config.baseline_m
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        z = depth[::stride, ::stride]
        valid = z > 0
        x = (xs - cx) * z / f
        y = (ys - cy) * z / f
        colors = left[::stride, ::stride][:, :, ::-1]
        return np.column_stack((x[valid], y[valid], z[valid], colors[valid])).astype(np.float32)

    @staticmethod
    def save_point_cloud(points: np.ndarray, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x_m", "y_m", "z_m", "r", "g", "b"])
            writer.writerows(points.tolist())

    def run(self, left: np.ndarray, right: np.ndarray) -> dict:
        disparity = self.compute_disparity(left, right)
        depth = self.depth_from_disparity(disparity)
        return {"disparity": disparity, "depth": depth, "obstacle_mask": self.obstacle_mask(depth), "nearest_m": self.nearest_distance(depth), "points": self.point_cloud(left, depth)}

    def save_result(self, result: dict, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        disparity = result["disparity"]
        normalized = cv2.normalize(np.maximum(disparity, 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(output / "disparity.png"), cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO))
        cv2.imwrite(str(output / "depth_m.png"), np.clip(result["depth"] * 1000, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(output / "obstacle_mask.png"), result["obstacle_mask"])
        self.save_point_cloud(result["points"], output / "point_cloud.csv")
        summary = {"config": asdict(self.config), "nearest_m": result["nearest_m"], "valid_points": int(len(result["points"]))}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
