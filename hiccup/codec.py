import functools
from typing import List

import numpy as np

import hiccup.model as model
import hiccup.utils as utils
import hiccup.transform as transform

"""
Encoding/Decoding functionality aka
    Run Length encoding
    Huffman Encoding - looking at papers, you can rely on the default Huffman encodings for say jpeg but then for 
    our eventual Wavelet encoding, the same Huffman encodings are definitely not applicable. To be consistent, and 
    avoid having to copy the entire RL Huffman table, I'll generate on the fly and persist. This is expensive for
    smaller images, but for very large images this is a small penalty.
"""

# http://www.globalspec.com/reference/39556/203279/appendix-b-huffman-tables-for-the-dc-and-ac-coefficients-of-the-jpeg-baseline-encoder
huffman = {
    model.Compression.JPEG: {
        "DC_HUFFMAN_CODE": {
            model.Coefficient.DC: {
                0: range(1),
                1: range(1, 2),
                2: range(2, 4),
                3: range(4, 8),
                4: range(8, 16),
                5: range(16, 32),
                6: range(32, 64),
                7: range(64, 128),
                8: range(128, 256),
                9: range(256, 512),
                10: range(512, 1024),
                11: range(1024, 2048),
                12: range(2048, 4096),
                13: range(4096, 8192),
                14: range(8192, 16384),
                15: range(16384, 32768)
            },
            model.Coefficient.AC: {
                0: None,
                1: range(1, 2),
                2: range(2, 4),
                3: range(4, 8),
                4: range(8, 16),
                5: range(16, 32),
                6: range(32, 64),
                7: range(64, 128),
                8: range(128, 256),
                9: range(256, 512),
                10: range(512, 1024),
                11: range(1024, 2048),
                12: range(2048, 4096),
                13: range(4096, 8192),
                14: range(8192, 16384),
                15: None
            }
        }
    }
}


def jpeg_category(val: int, coeff: model.Coefficient):
    """
    Determine JPEG DC category for huffman

    From Gonzalez and Wood
    """
    for k, v in huffman[model.Compression.JPEG]["DC_HUFFMAN_CODE"][coeff].items():
        if v is not None and abs(val) in v:
            return k
    raise RuntimeError("You must have a category for value: " + str(val))


def differential_coding(blocks: np.ndarray):
    """
    Produce differential coding for the DC coefficients
    """
    dc_comps = [transform.dc_component(b) for b in blocks]
    return utils.differences(dc_comps)


def _break_up_rle(code, max_len):
    l = code["zeros"]
    div = l // max_len
    full = {
        "zeros": max_len,
        "value": code["value"]
    }
    return ([full] * div) + [{
        "zeros": l - (div * max_len),
        "value": code["value"]
    }]


def run_length_coding(arr: np.ndarray, max_len=0xF):
    """
    Come up with the run length encoding for a matrix
    """

    def reduction(agg, next):
        if "value" in agg[-1]:
            agg.append({"zeros": 0})

        if next == 0:
            agg[-1]["zeros"] += 1
            return agg

        if "value" not in agg[-1]:
            agg[-1]["value"] = next

        return agg

    rl = functools.reduce(reduction, arr, [{"zeros": 0}])

    # If the last element has no value then it was 0! That is a special tuple, (0,0)
    if "value" not in rl[-1]:
        rl[-1] = {"zeros": 0, "value": 0}

    # the goal of RLE in the case of compression is to contain the first symbol (length, size) within a byte
    # so if the length is too long, then we need to break it up
    rl = [_break_up_rle(code, max_len) for code in rl]
    rl = utils.flatten(rl)

    return [dict(d, bits=utils.num_bits_for_int(d["value"])) for d in rl]


def jpeg_encode(luminance: np.ndarray, chrominances: List[np.ndarray]):
    """
    Do the standard jpeg encoding
    """
    dc_code = differential_coding(luminance)
    return None
