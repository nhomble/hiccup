import functools
from typing import List

import numpy as np

import hiccup.settings as settings
import hiccup.model as model
import hiccup.utils as utils
import hiccup.transform as transform
import hiccup.huffman as huffman
import hiccup.hicimage as hic

"""
Encoding/Decoding functionality aka
    Run Length encoding
    Huffman Encoding - looking at papers, you can rely on the default Huffman encodings for say jpeg but then for 
    our eventual Wavelet encoding, the same Huffman encodings are definitely not applicable. To be consistent, and 
    avoid having to copy the entire RL Huffman table, I'll generate on the fly and persist. This is expensive for
    smaller images, but for very large images this is a small penalty.
"""


class RunLength:
    @classmethod
    def from_dict(cls, d):
        return cls(d["value"], d["zeros"])

    def __init__(self, value, length):
        self.value = value
        self.length = length

    def __eq__(self, other):
        return type(self) == type(other) and self.value == other.value and self.length == other.length


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


def run_length_coding(arr: np.ndarray, max_len=0xF) -> List[RunLength]:
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

    return [RunLength.from_dict(r) for r in rl]


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

    utils.debug_msg("Constructed luminance wavelet huffmans")

    rl_ch_1 = run_length_coding(lin(chrominances[0]))
    rl_ch_2 = run_length_coding(lin(chrominances[1]))
    rl_chr_data = rl_ch_1 + rl_ch_2
    zch_huff = huffman.HuffmanTree.construct_from_data(rl_chr_data, key_func=lambda rl: rl["zeros"])
    vch_huff = huffman.HuffmanTree.construct_from_data(rl_chr_data, key_func=lambda rl: rl["value"])

    utils.debug_msg("Constructed chrominance wavelet huffmans")

    master_array = [
        # huffman tables
        hic.PlainString(zlum_huff.encode_table()),
        hic.PlainString(vlum_huff.encode_table()),
        hic.PlainString(zch_huff.encode_table()),
        hic.PlainString(vch_huff.encode_table()),

        hic.BitString(zlum_huff.encode_data()),
        hic.BitString(vlum_huff.encode_data()),

        hic.BitString(zch_huff.encode_data(data=rl_ch_1)),
        hic.BitString(vch_huff.encode_data(data=rl_ch_1)),
        hic.BitString(zch_huff.encode_data(data=rl_ch_2)),
        hic.BitString(vch_huff.encode_data(data=rl_ch_2)),

        hic.PlainString(encode_shape(luminance[0].shape)),
        hic.PlainString(encode_shape(luminance[-1].shape))
    ]
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


def wavelet_decode(sections):
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


def huffman_encode(huff: huffman.HuffmanTree) -> hic.Payload:
    leaves = huff.encode_table()
    return hic.PayloadStringP([hic.IntegerStringP([t[0], t[1]]) for t in leaves])


def huffman_data_encode(huff: huffman.HuffmanTree) -> hic.Payload:
    data = huff.encode_data()
    return hic.BitStringP(data)


def jpeg_encode(compressed: model.CompressedImage) -> hic.HicImage:
    """
    Generally follow JPEG encoding. Since for the wavelet work I am don't have some standard huffman tree to work with
    I might as well be consistent between the two implementations and just encode the entire array with custom
    Huffman trees. To attempt to be honest with the implementation though, I'll still treat the DC components
    separately by doing the differences and again applying a custom Huffman. A main feature of DCT on each block is the
    meaning of the DC component.

    For RL it's also easier implementation-wise to split up the length from the value and not try to optimize and weave
    them together. Yes, the encoding will suffer bloat, but we are trying to highlight the transforms anyway.
    """
    dc_comps = utils.dict_map(compressed.as_dict,
                              lambda _, v: differential_coding(transform.split_matrix(v, settings.JPEG_BLOCK_SIZE)))

    ac_comps = utils.dict_map(compressed.as_dict, lambda _, v: run_length_coding(
        transform.ac_components(transform.split_matrix(v, settings.JPEG_BLOCK_SIZE))))

    dc_huffs = utils.dict_map(dc_comps, lambda _, v: huffman.HuffmanTree.construct_from_data(v))
    ac_value_huffs = utils.dict_map(ac_comps,
                                    lambda _, v: huffman.HuffmanTree.construct_from_data(v, key_func=lambda s: s.value))
    ac_length_huffs = utils.dict_map(ac_comps,
                                     lambda _, v: huffman.HuffmanTree.construct_from_data(v,
                                                                                          key_func=lambda s: s.length))

    def encode_huff(d):
        huffs = [t[1] for t in d.items()]
        return [huffman_encode(h) for h in huffs]

    def encode_data(d):
        huffs = [t[1] for t in d.items()]
        return [huffman_data_encode(h) for h in huffs]

    payloads = utils.flatten([
        encode_huff(dc_huffs),
        encode_huff(ac_value_huffs),
        encode_huff(ac_length_huffs),

        encode_data(dc_huffs),
        encode_data(ac_value_huffs),
        encode_data(ac_length_huffs)
    ])
    return hic.HicImage.wavelet_image(payloads)
