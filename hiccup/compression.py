import cv2
import numpy as np

import hiccup.codec as codec
import hiccup.transform as transform
import hiccup.hicimage as hic
import hiccup.quantization as qnt
import hiccup.model as model
import hiccup.settings as settings
import hiccup.utils as utils

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

    channels = {
        "lum": gray,
        "cr": color_1,
        "cb": color_2
    }

    def channel_func(k, v: np.ndarray):
        offset = np.subtract(v.astype(np.int64), np.power(2, 8))
        transformed = transform.wavelet_split_resolutions(offset, settings.WAVELET, settings.WAVELET_NUM_LEVELS)
        subbands = transform.subband_view(transformed)
        quantized = qnt.subband_quantize(subbands, multiplier=settings.WAVELET_SUBBAND_QUANTIZATION_MULTIPLIER)
        r_transformed = transform.linearize_subband(quantized)
        thresholded = transform.threshold_channel_by_quality(r_transformed, q_factor=settings.WAVELET_QUALITY_FACTOR)
        return thresholded

    channels = utils.dict_map(channels, channel_func)

    encoding = codec.wavelet_encode(channels["lum"], [channels["cr"], channels["cb"]])

    utils.debug_msg("The image is %d x %d x 3 which is %d pixels" % (
        rgb_image.shape[0], rgb_image.shape[1], 3 * utils.size(rgb_image.shape)))

    return hic.HicImage.wavelet_image(encoding)


def wavelet_decompression(hic: hic.HicImage) -> np.ndarray:
    hic.apply_settings()
    payload = hic.payload()
    (t_gray, t_color_1, t_color_2) = codec.wavelet_decode(payload)
    channels = {
        "lum": t_gray,
        "cr": t_color_1,
        "cb": t_color_2
    }

    def channel_func(k, v):
        merged = transform.wavelet_merge_resolutions(v, settings.WAVELET)
        offset = np.add(merged, np.power(2, 8)).astype(np.uint8)
        return offset

    channels = utils.dict_map(channels, channel_func)
    yrcrcb = cv2.merge([channels["lum"], channels["cr"], channels["cb"]]).astype(np.uint8)
    return cv2.cvtColor(yrcrcb, cv2.COLOR_YCrCb2RGB)
