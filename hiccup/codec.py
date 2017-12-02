import functools
from typing import List

import numpy as np

import hiccup.model as model
import hiccup.utils as utils
import hiccup.transform as transform
import hiccup.huffman as huffman

"""
Encoding/Decoding functionality aka
    Run Length encoding
    Huffman Encoding - looking at papers, you can rely on the default Huffman encodings for say jpeg but then for 
    our eventual Wavelet encoding, the same Huffman encodings are definitely not applicable. To be consistent, and 
    avoid having to copy the entire RL Huffman table, I'll generate on the fly and persist. This is expensive for
    smaller images, but for very large images this is a small penalty.
"""

# http://www.globalspec.com/reference/39556/203279/appendix-b-huffman-tables-for-the-dc-and-ac-coefficients-of-the-jpeg-baseline-encoder
jpeg_categories = {
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
    for k, v in jpeg_categories[model.Compression.JPEG]["DC_HUFFMAN_CODE"][coeff].items():
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


def decode_run_length(rles: list, length: int):
    arr = utils.flatten([([0] * d["zeros"]) + [d["value"]] for d in rles])
    fill = length - len(arr)
    arr += ([0] * fill)
    return arr


def recover_run_length_coding(zeros, values):
    tups = zip(zeros, values)
    return [{"zeros": tup[0], "value": tup[1]} for tup in tups]


def jpeg_rle(rle):
    """
    JPEG defines an encoding for the each run length code since the DCT coefficients are maxes at a certain range. The
    wavelet basis functions should also be within the
    """
    pass


def encode_shape(shape):
    return "%d.%d" % shape


def decode_shape(s):
    [x, y] = s.split(".")
    return int(x), int(y)


def wavelet_encode(luminance: list, chrominances: List[list]):
    """
    In brief reading of literature, Huffman coding is still considered for wavelet image compression.
    """

    def lin(L):
        return utils.flatten([utils.img_as_list(i) for i in L])

    full_lum_data = lin(luminance)
    utils.debug_msg("Full wavelet luminance data length: %d" % len(full_lum_data))
    utils.debug_msg("Full wavelet luminance data\n%s" % " ".join([str(i) for i in full_lum_data]))
    rl_lum_data = run_length_coding(full_lum_data)
    zlum_huff = huffman.HuffmanTree.construct_from_data(rl_lum_data, key_func=lambda rl: rl["zeros"])
    vlum_huff = huffman.HuffmanTree.construct_from_data(rl_lum_data, key_func=lambda rl: rl["value"])

    rl_ch_1 = run_length_coding(lin(chrominances[0]))
    rl_ch_2 = run_length_coding(lin(chrominances[1]))
    rl_chr_data = rl_ch_1 + rl_ch_2
    zch_huff = huffman.HuffmanTree.construct_from_data(rl_chr_data, key_func=lambda rl: rl["zeros"])
    vch_huff = huffman.HuffmanTree.construct_from_data(rl_chr_data, key_func=lambda rl: rl["value"])

    master_array = "\n".join([
        # huffman tables
        zlum_huff.encode_table(),
        vlum_huff.encode_table(),
        zch_huff.encode_table(),
        vch_huff.encode_table(),

        zlum_huff.encode_data(),
        vlum_huff.encode_data(),

        zch_huff.encode_data(data=rl_ch_1),
        vch_huff.encode_data(data=rl_ch_1),
        zch_huff.encode_data(data=rl_ch_2),
        vch_huff.encode_data(data=rl_ch_2),

        encode_shape(luminance[0].shape),  # encode the smallest shape
        encode_shape(luminance[-1].shape)
    ])
    return master_array


def wavelet_decode_pull_subbands(data, shapes):
    offset = utils.size(shapes[0])
    subbands = [np.array(data[:offset]).reshape(shapes[0])]

    for i in range(len(shapes)):
        subbands.append(np.array(data[offset:offset + utils.size(shapes[i])]).reshape(shapes[i]))
        offset += utils.size(shapes[i])

        subbands.append(np.array(data[offset:offset + utils.size(shapes[i])]).reshape(shapes[i]))
        offset += utils.size(shapes[i])

        subbands.append(np.array(data[offset:offset + utils.size(shapes[i])]).reshape(shapes[i]))
        offset += utils.size(shapes[i])
    return subbands


def wavelet_decoded_subbands_shapes(min_shape, max_shape):
    levels = int(np.sqrt(max_shape[0] // min_shape[0]))
    shapes = [(min_shape[0] * (np.power(2, i)), min_shape[1] * (np.power(2, i))) for i in range(0, levels + 1)]
    return shapes


def wavelet_decoded_length(min_shape, max_shape):
    shapes = wavelet_decoded_subbands_shapes(min_shape, max_shape)
    length = functools.reduce(lambda agg, s: agg + (3 * (s[0] * s[1])), shapes, 0)
    length += (min_shape[0] * min_shape[1])
    return length


def wavelet_decode(bit_string):
    sections = bit_string.split("\n")
    zlum_huff = huffman.HuffmanTree.construct_from_coding(sections[0])
    vlum_huff = huffman.HuffmanTree.construct_from_coding(sections[1])
    zch_huff = huffman.HuffmanTree.construct_from_coding(sections[2])
    vch_huff = huffman.HuffmanTree.construct_from_coding(sections[3])

    zlum_data = zlum_huff.decode_data(sections[4])
    vlum_data = vlum_huff.decode_data(sections[5])

    zch1_data = zch_huff.decode_data(sections[6])
    vch1_data = vch_huff.decode_data(sections[7])

    zch2_data = zch_huff.decode_data(sections[8])
    vch2_data = vch_huff.decode_data(sections[9])

    min_shape = decode_shape(sections[10])
    max_shape = decode_shape(sections[11])

    length = wavelet_decoded_length(min_shape, max_shape)

    lum_rle = recover_run_length_coding(zlum_data, vlum_data)
    ch1_rle = recover_run_length_coding(zch1_data, vch1_data)
    ch2_rle = recover_run_length_coding(zch2_data, vch2_data)

    lum = decode_run_length(lum_rle, length)
    ch1 = decode_run_length(ch1_rle, length)
    ch2 = decode_run_length(ch2_rle, length)

    utils.debug_msg("Recovered luminance from RLE\n%s" % " ".join([str(l) for l in lum]))

    shapes = wavelet_decoded_subbands_shapes(min_shape, max_shape)
    return wavelet_decode_pull_subbands(lum, shapes), \
           wavelet_decode_pull_subbands(ch1, shapes), \
           wavelet_decode_pull_subbands(ch2, shapes)


def jpeg_encode(luminance: np.ndarray, chrominances: List[np.ndarray]):
    """
    Generally follow JPEG encoding. Since for the wavelet work I am don't have some standard huffman tree to work with
    I might as well be consistent between the two implementations and just encode the entire array with custom
    Huffman trees. To attempt to be honest with the implementation though, I'll still treat the DC components
    separately by doing the differences and again applying a custom Huffman. A mean feature of DCT on each block is the
    meaning of the DC component.
    """
    dc_lum = differential_coding(luminance)
    dc_chs = [differential_coding(m) for m in chrominances]

    ac_lum = run_length_coding(transform.ac_components(luminance))
    ac_chs = [run_length_coding(transform.ac_components(m)) for m in chrominances]

    dc_lu_huffman = huffman.HuffmanTree.construct_from_data(dc_lum)
    dc_ch_huffman = huffman.HuffmanTree.construct_from_data(utils.flatten(dc_chs))
    ac_lu_huffman = huffman.HuffmanTree.construct_from_data(ac_lum, key_func=lambda rl: rl["zeros"])
    ac_ch_huffman = huffman.HuffmanTree.construct_from_data(utils.flatten(ac_chs), key_func=lambda rl: rl["zeros"])

    master_string = "\n".join([
        # table data
        dc_lu_huffman.encode_table(),
        dc_ch_huffman.encode_table(),
        ac_lu_huffman.encode_table(),
        ac_ch_huffman.encode_table(),
        # huffman encoding
        dc_lu_huffman.encode_data(),
        dc_ch_huffman.encode_data(chrominances[0]),
        dc_ch_huffman.encode_data(chrominances[1]),
        ac_lu_huffman.encode_data(),
        ac_ch_huffman.encode_data(chrominances[0]),
        ac_ch_huffman.encode_data(chrominances[1]),
        # auxiliary
        [s["size"] for s in ac_lum],
        [s["value"] for s in ac_lum],

        [s["size"] for s in ac_chs[0]],
        [s["value"] for s in ac_chs[0]],

        [s["size"] for s in ac_chs[1]],
        [s["value"] for s in ac_chs[1]]

    ])
    return master_string
