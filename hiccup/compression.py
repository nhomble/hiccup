import cv2
import numpy as np

import hiccup.codec as codec
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

    encoding = codec.jpeg_encode(t_gray, [t_color_1, t_color_2])

    return None


def jpeg_decompression(hic: hic.HicImage) -> np.ndarray:
    """
    Decompress a JPEG image for viewing
    """
    pass


def wavelet_compression(rgb_image: np.ndarray) -> hic.HicImage:
    yrcrcb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    [gray, color_1, color_2] = cv2.split(yrcrcb)
    [t_gray, t_color_1, t_color2] = [transform.wavelet_split_resolutions(c, transform.Wavelet.DAUBECHIE) for c in
                                     [gray, color_1, color_2]]
    [t_gray, t_color_1, t_color2] = [transform.threshold_channel_by_quality(b) for b in [t_gray, t_color_1, t_color2]]
    return None


def wavelet_decompression(hic: hic.HicImage) -> np.ndarray:
    pass
