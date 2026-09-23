from pathlib import Path

import cv2
import numpy as np


def make_pair(output: str = "data") -> None:
    """Create a deterministic textured pair with a foreground rectangle."""
    rng = np.random.default_rng(7)
    h, w = 360, 640
    left = (rng.random((h, w)) * 45).astype(np.uint8)
    for x in range(0, w, 24):
        cv2.line(left, (x, 0), (x, h), 80, 1)
    cv2.rectangle(left, (245, 105), (395, 275), 220, -1)
    cv2.circle(left, (320, 190), 45, 30, -1)
    right = np.zeros_like(left)
    shift = 22
    right[:, :-shift] = left[:, shift:]
    Path(output).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{output}/sample_left.png", left)
    cv2.imwrite(f"{output}/sample_right.png", right)
    print(f"Wrote synthetic stereo pair to {output}/")


if __name__ == "__main__":
    make_pair()
