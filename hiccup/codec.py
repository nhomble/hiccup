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

    def __init__(self, value=0, length=0):
        self.value = value
        self.length = length

    def __eq__(self, other):
        return type(self) == type(other) and self.value == other.value and self.length == other.length

    def __str__(self):
        return "(%d, %d)" % (self.length, self.value)

    @property
    def segment(self):
        return [0] * self.length + [self.value]

    @property
    def is_trailing(self):
        return self.value == 0 and self.length == 0


def differential_coding(blocks: np.ndarray):
    """
    Produce differential coding for the DC coefficients
    """
    dc_comps = [transform.dc_component(b) for b in blocks]
    return utils.differences(dc_comps)


def run_length_coding(arr: np.ndarray, max_len=0xF) -> List[RunLength]:
    """
    Come up with the run length encoding for a matrix
    """

    def _break_up_rle(code, max_len):
        l = code["zeros"]
        div = l // max_len
        full = {
            "zeros": max_len - 1,  # minus 1 because we get another for free from the value
            "value": 0
        }
        return ([full] * div) + [{
            "zeros": l - (div * max_len),
            "value": code["value"]
        }]

    def reduction(agg, next):
        if "value" in agg[-1]:
            agg.append({"zeros": 0})

        if next == 0:
            agg[-1]["zeros"] += 1
            return agg

        if "value" not in agg[-1]:
            agg[-1]["value"] = next

        return agg
    utils.debug_msg("Going to determine RLE for %d size array" % len(arr))
    rl = functools.reduce(reduction, arr, [{"zeros": 0}])
    utils.debug_msg("%d long RLE created" % len(rl))
    # If the last element has no value then it was 0! That is a special tuple, (0,0)
    if "value" not in rl[-1]:
        rl[-1] = {"zeros": 0, "value": 0}

    # the goal of RLE in the case of compression is to contain the first symbol (length, size) within a byte
    # so if the length is too long, then we need to break it up
    if max_len is not None:
        utils.debug_msg("Breaking up RLE lengths that are larger than %d" % max_len)
        rl = [_break_up_rle(code, max_len) for code in rl]
        rl = utils.flatten(rl)

    utils.debug_msg("Make RLE objects")
    return [RunLength.from_dict(r) for r in rl]


def decode_run_length(rles: List[RunLength], length: int):
    arr = []
    for (i, d) in enumerate(rles):
        arr.append(d.segment)
    arr = utils.flatten(arr)
    # arr = utils.flatten([d.segment for d in rles])

    if rles[-1].is_trailing:
        fill = length - len(arr)
        arr += ([0] * fill)

    return arr


def wavelet_encode(compressed: model.CompressedImage):
    """
    In brief reading of literature, Huffman coding is still considered for wavelet image compression. There are other
    more effective (and complicated schemes) that I think are out of scope of this project which is just to introduce
    the concepts.
    """

    def collapse_subbands(k, v):
        out = [transform.zigzag(l) for l in v]
        out = utils.flatten(out)
        return out

    utils.debug_msg("Starting Wavelet encoding")
    lin_subbands = utils.dict_map(compressed.as_dict, collapse_subbands)
    utils.debug_msg("Have completed linearizing the subbands")
    rles = utils.dict_map(lin_subbands, lambda _, v: run_length_coding(v))
    utils.debug_msg("Have completed the run length encodings")

    values_huffs = utils.dict_map(rles,
                                  lambda _, v: huffman.HuffmanTree.construct_from_data(v, key_func=lambda t: t.value))
    length_huffs = utils.dict_map(rles,
                                  lambda _, v: huffman.HuffmanTree.construct_from_data(v, key_func=lambda t: t.length))
    utils.debug_msg("Huffman trees are constructed")

    def encode_huff(d):
        huffs = [t[1] for t in d.items()]
        return [huffman_encode(h) for h in huffs]

    def encode_data(d):
        huffs = [t[1] for t in d.items()]
        return [huffman_data_encode(h) for h in huffs]

    smallest = compressed.luminance_component[0].shape
    biggest = compressed.luminance_component[-1].shape

    payloads = utils.flatten([
        encode_huff(values_huffs),
        encode_huff(length_huffs),

        encode_data(values_huffs),
        encode_data(length_huffs),

        [
            hic.TupP(smallest[0], smallest[1]),
            hic.TupP(biggest[0], biggest[1])
        ]
    ])
    return hic.HicImage.wavelet_image(payloads)


def wavelet_decode_pull_subbands(data, shapes):
    offset = utils.size(shapes[0])
    subbands = [transform.izigzag(np.array(data[:offset]), shapes[0])]

    for i in range(len(shapes)):
        subbands.append(transform.izigzag(np.array(data[offset:offset + utils.size(shapes[i])]), shapes[i]))
        offset += utils.size(shapes[i])

        subbands.append(transform.izigzag(np.array(data[offset:offset + utils.size(shapes[i])]), shapes[i]))
        offset += utils.size(shapes[i])

        subbands.append(transform.izigzag(np.array(data[offset:offset + utils.size(shapes[i])]), shapes[i]))
        offset += utils.size(shapes[i])
    return subbands


def wavelet_decoded_subbands_shapes(min_shape, max_shape):
    """
    We just do Haar or Daubechie, assume power of 2
    """

    levels = int(np.sqrt(max_shape[0] // min_shape[0]))
    shapes = [(min_shape[0] * (np.power(2, i)), min_shape[1] * (np.power(2, i))) for i in range(0, levels + 1)]
    return shapes


def wavelet_decoded_length(min_shape, max_shape):
    shapes = wavelet_decoded_subbands_shapes(min_shape, max_shape)
    length = functools.reduce(lambda agg, s: agg + (3 * (s[0] * s[1])), shapes, 0)
    length += (min_shape[0] * min_shape[1])
    return length


def wavelet_decode(hic: hic.HicImage) -> model.CompressedImage:
    utils.debug_msg("Wavelet decode")
    assert hic.hic_type == model.Compression.HIC
    payloads = hic.payloads
    utils.debug_msg("Decoding Huffman trees")
    value_huffs = {
        "lum": huffman_decode(payloads[0]),
        "cr": huffman_decode(payloads[1]),
        "cb": huffman_decode(payloads[2])
    }
    length_huffs = {
        "lum": huffman_decode(payloads[3]),
        "cr": huffman_decode(payloads[4]),
        "cb": huffman_decode(payloads[5])
    }

    utils.debug_msg("Decode RLE values")
    value_comps = {
        "lum": huffman_data_decode(payloads[6], value_huffs["lum"]),
        "cr": huffman_data_decode(payloads[7], value_huffs["cr"]),
        "cb": huffman_data_decode(payloads[8], value_huffs["cb"]),
    }
    utils.debug_msg("Decode RLE lengths")
    length_comps = {
        "lum": huffman_data_decode(payloads[9], length_huffs["lum"]),
        "cr": huffman_data_decode(payloads[10], length_huffs["cr"]),
        "cb": huffman_data_decode(payloads[11], length_huffs["cb"]),
    }
    min_shape = payloads[12].numbers
    max_shape = payloads[13].numbers

    utils.debug_msg("Unloaded all of the data")
    # ====
    rles = utils.dict_map(value_comps,
                          lambda k, v: [RunLength(value=t[1], length=t[0]) for t in list(zip(length_comps[k], v))])
    length = wavelet_decoded_length(min_shape, max_shape)

    data = utils.dict_map(rles, lambda _, v: decode_run_length(v, length))
    shapes = wavelet_decoded_subbands_shapes(min_shape, max_shape)
    channels = utils.dict_map(data, lambda _, v: wavelet_decode_pull_subbands(v, shapes))
    return model.CompressedImage.from_dict(channels)


def huffman_encode(huff: huffman.HuffmanTree) -> hic.Payload:
    """
    Encode huffman in payload
    """

    leaves = huff.encode_table()
    return hic.PayloadStringP(hic.TupP, [hic.TupP(t[0], t[1]) for t in leaves])


def huffman_decode(data: hic.PayloadStringP) -> huffman.HuffmanTree:
    """
    Decode huffman from payload
    """
    number_string = data.payloads
    leaves = [p.numbers for p in number_string]
    return huffman.HuffmanTree.construct_from_coding(leaves)


def huffman_data_encode(huff: huffman.HuffmanTree) -> hic.Payload:
    """
    Encode huffman data into payload
    """
    data = huff.encode_data()
    return hic.BitStringP(data)


def huffman_data_decode(data: hic.BitStringP, huffman: huffman.HuffmanTree) -> list:
    """
    Decode huffman data from payload with huffman tree
    """
    return huffman.decode_data(data.payload)


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
    utils.debug_msg("Starting JPEG encoding")
    dc_comps = utils.dict_map(compressed.as_dict,
                              lambda _, v: differential_coding(transform.split_matrix(v, settings.JPEG_BLOCK_SIZE)))

    utils.debug_msg("Determined differences DC components")

    def ac_comp_fun(k, v):
        utils.debug_msg("Determining AC components for: " + k)
        splits = transform.split_matrix(v, settings.JPEG_BLOCK_SIZE)
        acs = transform.ac_components(splits)
        utils.debug_msg("Calculating RLE for: " + k)
        out = run_length_coding(acs)
        return out

    # on each transformed channel, run RLE on the AC components of each block
    ac_comps = utils.dict_map(compressed.as_dict, ac_comp_fun)

    utils.debug_msg("Determined RLEs for AC components")
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
        encode_data(ac_length_huffs),

        [
            hic.TupP(compressed.shape[0][0], compressed.shape[0][1]),
            hic.TupP(compressed.shape[1][0], compressed.shape[1][1])
        ]
    ])

    return hic.HicImage.jpeg_image(payloads)


def jpeg_decode(hic: hic.HicImage) -> model.CompressedImage:
    """
    Reverse jpeg_encode()
    payloads = utils.flatten([
        encode_huff(dc_huffs),
        encode_huff(ac_value_huffs),
        encode_huff(ac_length_huffs),

        encode_data(dc_huffs),
        encode_data(ac_value_huffs),
        encode_data(ac_length_huffs)
    ])
    """
    utils.debug_msg("JPEG decode")
    assert hic.hic_type == model.Compression.JPEG
    payloads = hic.payloads
    utils.debug_msg("Decoding Huffman trees")
    dc_huffs = {
        "lum": huffman_decode(payloads[0]),
        "cr": huffman_decode(payloads[1]),
        "cb": huffman_decode(payloads[2])
    }
    ac_value_huffs = {
        "lum": huffman_decode(payloads[3]),
        "cr": huffman_decode(payloads[4]),
        "cb": huffman_decode(payloads[5])
    }
    ac_length_huffs = {
        "lum": huffman_decode(payloads[6]),
        "cr": huffman_decode(payloads[7]),
        "cb": huffman_decode(payloads[8])
    }

    utils.debug_msg("Decode DC differences")
    dc_comps = {
        "lum": huffman_data_decode(payloads[9], dc_huffs["lum"]),
        "cr": huffman_data_decode(payloads[10], dc_huffs["cr"]),
        "cb": huffman_data_decode(payloads[11], dc_huffs["cb"]),
    }

    utils.debug_msg("Decode RLE values")
    ac_values = {
        "lum": huffman_data_decode(payloads[12], ac_value_huffs["lum"]),
        "cr": huffman_data_decode(payloads[13], ac_value_huffs["cr"]),
        "cb": huffman_data_decode(payloads[14], ac_value_huffs["cb"]),
    }
    utils.debug_msg("Decode RLE lengths")
    ac_lengths = {
        "lum": huffman_data_decode(payloads[15], ac_length_huffs["lum"]),
        "cr": huffman_data_decode(payloads[16], ac_length_huffs["cr"]),
        "cb": huffman_data_decode(payloads[17], ac_length_huffs["cb"]),
    }
    shapes = {
        "lum": payloads[18].numbers,
        "cr": payloads[19].numbers,
        "cb": payloads[19].numbers
    }
    utils.debug_msg("Unloaded all of the data")
    # ====

    sub_length = utils.size(settings.JPEG_BLOCK_SHAPE()) - 1
    utils.debug_msg("Calculating AC RLEs")
    ac_rle = utils.dict_map(ac_values,
                            lambda k, v: [RunLength(t[1], t[0]) for t in list(zip(ac_lengths[k], v))])

    def ac_mat_fun(k, v):
        utils.debug_msg("Determining deficient AC matricies for: " + k)
        ac_length = utils.size(shapes[k]) - len(dc_comps[k])
        out = decode_run_length(v, ac_length)
        if k == "lum":
            s = [str(i) for i in out]
            print(" ".join(s))
        return out

    ac_mats = utils.dict_map(ac_rle, ac_mat_fun)
    ac_mats = utils.dict_map(ac_mats, lambda _, v: utils.group_tuples(v, sub_length))
    dc_comps = utils.dict_map(dc_comps, lambda _, v: utils.invert_differences(v))

    def merge_comps(dc_key, dc_values):
        utils.debug_msg("Merging: " + dc_key)
        tuples = ac_mats[dc_key]  # there are all of the AC zigzag arrays missing their DC component
        assert len(tuples) == len(dc_values)
        zipped = zip(dc_values, tuples)  # combine them to be mapped later
        lin_mats = [[t[0], *t[1]] for t in zipped]  # create the linearized block
        mats = [transform.izigzag(np.array(m), settings.JPEG_BLOCK_SHAPE()) for m in lin_mats]
        return mats

    compressed = utils.dict_map(dc_comps, merge_comps)
    merged = utils.dict_map(compressed, lambda k, v: transform.merge_blocks(np.array(v), shapes[k]))
    return model.CompressedImage.from_dict(merged)
