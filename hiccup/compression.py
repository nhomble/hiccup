import cv2
import numpy as np

import hiccup.codec as codec
import hiccup.transform as transform
import hiccup.hicimage as hic
import hiccup.quantization as qnt
import hiccup.model as model

"""
Houses the entry functions to either compression algorithm
"""


def jpeg_compression(rgb_image: np.ndarray) -> hic.HicImage:
    """
    JPEG compression
    """
    yrcrcb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    [gray, color_1, color_2] = cv2.split(yrcrcb)

    t_gray = transform.dct_channel(gray, model.QTables.JPEG_LUMINANCE)
    [t_color_1, t_color_2] = [transform.dct_channel(transform.down_sample(c), model.QTables.JPEG_CHROMINANCE) for c in
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
    # Daubechie should be a setting
    [t_gray, t_color_1, t_color_2] = [transform.wavelet_split_resolutions(c, model.Wavelet.DAUBECHIE) for c in
                                      [gray, color_1, color_2]]

    [t_gray, t_color_1, t_color_2] = [transform.linearize_subband(qnt.subband_quantize(transform.subband_view(b))) for b
                                      in [t_gray, t_color_1, t_color_2]]

    [t_gray, t_color_1, t_color_2] = [transform.threshold_channel_by_quality(b) for b in
                                      [t_gray, t_color_1, t_color_2]]
    t_gray = [qnt.round_quantize(b) for b in t_gray]
    t_color_1 = [qnt.round_quantize(b) for b in t_color_1]
    t_color_2 = [qnt.round_quantize(b) for b in t_color_2]
    encoding = codec.wavelet_encode(t_gray, [t_color_1, t_color_2])

    return encoding


def wavelet_decompression(s) -> np.ndarray:
    (t_gray, t_color_1, t_color_2) = codec.wavelet_decode(s)
    channels = [transform.wavelet_merge_resolutions(m, model.Wavelet.DAUBECHIE) for m in [t_gray, t_color_1, t_color_2]]
    yrcrcb = cv2.merge(channels).astype(np.float32)
    return cv2.cvtColor(yrcrcb, cv2.COLOR_YCrCb2RGB)
