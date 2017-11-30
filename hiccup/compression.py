import cv2
import numpy as np

import hiccup.transform as transform
import hiccup.hicimage as hic
import hiccup.quantization as qnt

"""
Houses the entry functions to either compression algorithm
"""


def jpeg_compression(rgb_image: np.ndarray) -> hic.HicImage:
    """
    JPEG compression
    """
    yrcrcb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    [gray, color_1, color_2] = cv2.split(yrcrcb)

    t_gray = transform.dct_channel(gray, qnt.QTables.JPEG_LUMINANCE)
    [t_color_1, t_color_2] = [transform.dct_channel(transform.down_sample(c), qnt.QTables.JPEG_CHROMINANCE) for c in
                              [color_1, color_2]]

    return None


def jpeg_decompression(hic: hic.HicImage) -> np.ndarray:
    """
    Decompress a JPEG image for viewing
    """
    pass
