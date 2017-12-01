import functools
import numpy as np

import hiccup.model as model
import hiccup.utils as utils
import hiccup.transform as transform

"""
Encoding/Decoding functionality aka
    Run Length encoding
    Huffman Encoding
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
    ret = []
    i = 0
    for dc in dc_comps:  # ugh
        if len(ret) == 0:
            ret.append(dc)
        else:
            ret.append(dc - dc_comps[i])
            i += 1
    return ret


def run_length_coding(matrix: np.ndarray):
    """
    Come up with the run length encoding for a matrix

    TODO: too long
    """
    zigzag = transform.zigzag(matrix)

    def reduction(agg, next):
        if next == 0:
            agg[-1]["zeros"] += 1
            return agg

        if "value" not in agg[-1]:
            agg[-1]["value"] = next

        agg.append({"zeros": 0})
        return agg

    rl = functools.reduce(reduction, zigzag, [{"zeros": 0}])

    # If the last element has value then it was 0! That is a special tuple, (0,0)
    if "value" not in rl[-1]:
        rl[-1] = {"zeros": 0, "value": 0}

    return [dict(d, bits=utils.num_bits_for_int(d["value"])) for d in rl]


def encode(array: np.ndarray):
    pass


def decode(array: np.ndarray):
    pass
