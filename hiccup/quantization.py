import enum
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
