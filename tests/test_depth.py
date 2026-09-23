import cv2
import numpy as np

from stereo_depth import StereoConfig, StereoDepthPipeline


def test_depth_formula() -> None:
    pipeline = StereoDepthPipeline(StereoConfig(focal_length_px=100, baseline_m=0.1))
    depth = pipeline.depth_from_disparity(np.array([[10.0]], dtype=np.float32))
    assert np.isclose(depth[0, 0], 1.0)


def test_point_cloud_shape() -> None:
    pipeline = StereoDepthPipeline(StereoConfig(focal_length_px=100, baseline_m=0.1))
    left = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    points = pipeline.point_cloud(left, depth, stride=2)
    assert points.shape[1] == 6
    assert points.shape[0] == 16


def test_obstacle_mask_only_center_roi() -> None:
    pipeline = StereoDepthPipeline(StereoConfig(roi_fraction=0.5))
    depth = np.ones((20, 20), dtype=np.float32)
    mask = pipeline.obstacle_mask(depth)
    assert mask[:, 0].sum() == 0
    assert mask[:, 10].sum() > 0
