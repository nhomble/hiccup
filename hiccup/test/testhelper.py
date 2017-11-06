import numpy as np
import cv2


def rgb_123(N: int):
    parts = [
        np.full((N, N), 1, dtype=np.uint8),
        np.full((N, N), 2, dtype=np.uint8),
        np.full((N, N), 3, dtype=np.uint8)
    ]
    return cv2.merge(parts)
