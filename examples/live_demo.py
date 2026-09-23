from __future__ import annotations

import argparse

import cv2

from stereo_depth import StereoDepthPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview two synchronized cameras and live disparity.")
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=1)
    args = parser.parse_args()
    left_cam, right_cam = cv2.VideoCapture(args.left_camera), cv2.VideoCapture(args.right_camera)
    if not left_cam.isOpened() or not right_cam.isOpened():
        raise RuntimeError("Could not open both cameras. Try --left-camera/--right-camera device IDs.")
    pipeline = StereoDepthPipeline()
    while True:
        ok_l, left = left_cam.read()
        ok_r, right = right_cam.read()
        if not ok_l or not ok_r:
            break
        disparity = pipeline.compute_disparity(left, right)
        view = cv2.normalize(disparity.clip(min=0), None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        cv2.imshow("left", left)
        cv2.imshow("disparity (q to quit)", cv2.applyColorMap(view, cv2.COLORMAP_TURBO))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    left_cam.release(); right_cam.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
