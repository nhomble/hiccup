import enum
from typing import Iterable
import functools

import numpy as np

import hiccup.model as model
import hiccup.utils as utils

"""
Handle quantization of the matrices
"""

table = {
    # taken from wikipedia: https://en.wikipedia.org/wiki/Quantization_(image_processing)
    # JPEG Standard, Annex K (from Bernd Girod)
    model.QTables.JPEG_LUMINANCE: np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]),
    model.QTables.JPEG_CHROMINANCE: np.array([
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


def deadzone_quantize(block, div):
    dividend = np.divide(block, div)
    return round_quantize(dividend)


def jpeg_quantize(block: np.ndarray, option: model.QTables):
    """
    With an 8x8 block, perform dead quantization with a certain table. Dead 'cause we are creating deadzones
    """
    t = table[option]
    return deadzone_quantize(block, t)


def subband_quantize(subbands, multiplier=1):
    """
    Uniformly dead zone
    """
    subbands[0] = round_quantize(subbands[0])
    hfs = subbands[1:]
    for (i, band) in enumerate(hfs):
        subbands[i + 1] = [np.divide(h, multiplier * (i * i + 1)) for h in band]
        subbands[i + 1] = round_quantize(subbands[i + 1])
    return subbands


def subband_invert_quantize(subbands, multiplier=1):
    subbands[0] = round_quantize(subbands[0])
    hfs = subbands[1:]
    for (i, band) in enumerate(hfs):
        subbands[i + 1] = [np.multiply(h, multiplier * (i * i + 1)) for h in band]
    return subbands


def round_quantize(block: np.ndarray):
    return np.round(block).astype(np.int32)


def quality_threshold_value(vals: list, q_factor=1):
    """
    For wavelet compression, we won't rely on magical tables for quantization, we'll just pick how many coefficients we
    to keep by thresholding a certain percentage as suggested in the literature.

    NB we also just uniformly more aggressively on the subbands so this doesn't have to be used.
    """
    s = list(sorted(vals))
    keep_up_too = int(np.ceil(len(vals) * q_factor))
    thresh_index = len(vals) - keep_up_too
    return s[thresh_index]
