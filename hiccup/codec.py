import numpy as np

import hiccup.transform as transform

"""
Encoding/Decoding functionality aka
    Run Length encoding
    Huffman Encoding
"""

table = {

}


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


def encode(array: np.ndarray):
    pass


def decode(array: np.ndarray):
    pass
