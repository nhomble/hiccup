import functools
import numpy as np

import hiccup.utils as utils
import hiccup.transform as transform

"""
Encoding/Decoding functionality aka
    Run Length encoding
    Huffman Encoding
"""


def differential_coding(blocks: np.ndarray):
    """
    Produce differential coding for the DC coefficients
    """
    dc_comps = [b[0][0] for b in blocks]
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
