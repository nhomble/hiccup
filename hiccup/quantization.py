import enum
from typing import List
import functools

import numpy as np

"""
Handle quantization of the matrices
"""


class QTables(enum.Enum):
    JPEG_LUMINANCE = "jpeg standard luminance"
    JPEG_CHROMINANCE = "jpeg standard chrominance"


table = {
    # taken from wikipedia: https://en.wikipedia.org/wiki/Quantization_(image_processing)
    # JPEG Standard, Annex K (from Bernd Girod)
    QTables.JPEG_LUMINANCE: np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]),
    QTables.JPEG_CHROMINANCE: np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])
}

all_tables = set(table.keys())


def dead_quantize(block: np.ndarray, option: QTables):
    """
    With an 8x8 block, perform dead quantization with a certain table. Dead 'cause we are creating deadzones
    """
    t = table[option]
    dividend = np.divide(block, t)
    quantized = np.round(dividend)
    return quantized


def quality_threshold(imgs: List[np.ndarray], q_factor=.05):
    """
    For wavelet compression, we won't rely on magical tables for quantization, we'll just pick how many coefficients we
    to keep by thresholding a certain percentage as suggested in the literature.
    """
    vals = functools.reduce(lambda x, y: x + list(y), imgs, [])
    s = list(sorted(vals))
    keep_up_too = int(np.ceil(len(vals) * q_factor))
    thresh_index = len(vals) - keep_up_too - 1
    return s[thresh_index]
