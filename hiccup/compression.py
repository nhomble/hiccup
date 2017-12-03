import cv2
import numpy as np

import hiccup.transform as transform
import hiccup.hicimage as hic
import hiccup.quantization as qnt
import hiccup.model as model
import hiccup.settings as settings
import hiccup.utils as utils

"""
Houses the entry functions to either compression algorithm
"""


def jpeg_compression(rgb_image: np.ndarray) -> model.CompressedImage:
    """
    JPEG compression
    """
    yrcrcb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    [gray, color_1, color_2] = cv2.split(yrcrcb)
    channels = {
        "lum": gray,
        "cr": color_1,
        "cb": color_2
    }

    def channel_fun(k, v):
        if k == "lum":
            return transform.dct_channel(v, model.QTables.JPEG_LUMINANCE, block_size=settings.JPEG_BLOCK_SIZE)
        else:
            return transform.dct_channel(transform.down_sample(v), model.QTables.JPEG_CHROMINANCE,
                                         block_size=settings.JPEG_BLOCK_SIZE)

    channels = utils.dict_map(channels, channel_fun)
    return model.CompressedImage.from_dict(channels)


def jpeg_decompression(d: model.CompressedImage) -> np.ndarray:
    """
    Decompress a JPEG image for viewing
    """

    def channel_fun(k, v):
        if k == "lum":
            return transform.inv_dct_channel(v, model.QTables.JPEG_LUMINANCE, block_size=settings.JPEG_BLOCK_SIZE)
        else:
            return transform.up_sample(transform.inv_dct_channel(v, model.QTables.JPEG_CHROMINANCE,
                                                                 block_size=settings.JPEG_BLOCK_SIZE))

    channels = utils.dict_map(d.as_dict, channel_fun)
    y = transform.force_merge(channels["lum"], channels["cr"], channels["cb"])
    return cv2.cvtColor(y, cv2.COLOR_YCrCb2RGB)


def wavelet_compression(rgb_image: np.ndarray) -> model.CompressedImage:
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
        if settings.WAVELET_SUBBAND_QUANTIZATION_MULTIPLIER != 0:
            subbands = qnt.subband_quantize(subbands, multiplier=settings.WAVELET_SUBBAND_QUANTIZATION_MULTIPLIER)
        r_transformed = transform.linearize_subband(subbands)
        thresholded = transform.threshold_channel_by_quality(r_transformed, q_factor=settings.WAVELET_QUALITY_FACTOR)

        if settings.WAVELET_THRESHOLD != 0:
            thresholded = [transform.threshold(part, settings.WAVELET_THRESHOLD) for part in thresholded]
        rounded = [qnt.round_quantize(t) for t in thresholded]
        return rounded

    channels = utils.dict_map(channels, channel_func)

    return model.CompressedImage.from_dict(channels)


def wavelet_decompression(channels: model.CompressedImage) -> np.ndarray:
    def channel_func(k, v):
        subbands = transform.subband_view(v)
        if settings.WAVELET_SUBBAND_QUANTIZATION_MULTIPLIER != 0:
            subbands = qnt.subband_invert_quantize(subbands, settings.WAVELET_SUBBAND_QUANTIZATION_MULTIPLIER)
        li = transform.linearize_subband(subbands)
        merged = transform.wavelet_merge_resolutions(li, settings.WAVELET)
        offset = np.add(merged, np.power(2, 8)).astype(np.uint8)
        return offset

    channels = utils.dict_map(channels.as_dict, channel_func)
    yrcrcb = transform.force_merge(channels["lum"], channels["cr"], channels["cb"]).astype(np.uint8)
    return cv2.cvtColor(yrcrcb, cv2.COLOR_YCrCb2RGB)
