from typing import Tuple, List
import functools
import numpy as np
import cv2
import scipy
import scipy.fftpack
import pywt

import hiccup.model as model
import hiccup.utils as utils
import hiccup.quantization as qz

"""
Helpful transformation functions from sampling the image, to applying dwt and dct 
"""


def pad_matrix(matrix: np.ndarray, N):
    """
    So that we can cleanly split the matrix into blocks
    """
    x, y = matrix.shape
    if x % N == 0 and y % N == 0:
        return matrix
    new_shape = (
        x + ((N - x % N) % N),
        y + ((N - y % N) % N)
    )
    result = np.zeros(new_shape, dtype=matrix.dtype)
    result[:x, :y] = matrix
    return result


def split_matrix(matrix, N):
    """
    Divide matrix intro an array of NxN blocks
    """
    matrix = pad_matrix(matrix, N)
    h, w = matrix.shape
    horizontal_segments = matrix.reshape(h // N, N, -1, N)
    interleave_segments = horizontal_segments.swapaxes(1, 2)
    form_blocks = interleave_segments.reshape(-1, N, N)
    return form_blocks


def merge_blocks(blocks: np.ndarray, shape):
    """
    After doing block level transformations, reconstruct a 2d matrix
    """
    y, x = shape
    num, N, _ = blocks.shape
    if y * x != num * N * N:
        # I must have done some padding
        assert y * x < num * N * N  # otherwise we somehow dropped data
        padded = pad_matrix(np.zeros(shape), N)
        y, x = padded.shape

    d4 = blocks.reshape(y // N, x // N, N, N)
    block_mats = [
        [np.matrix(d4[y][x]) for x in range(d4[y].shape[0])]
        for y in range(d4.shape[0])
    ]
    out = np.array(np.bmat(block_mats))
    sanitized = out[:shape[0], :shape[1]]
    return sanitized


def dct2(matrix: np.ndarray):
    """
    Computed dct type II

    Credits for dct2() comes from Mark Newman <mejn@umich.edu>
    """

    M = matrix.shape[0]
    N = matrix.shape[1]
    a = np.empty([M, N], float)
    b = np.empty([M, N], float)

    for i in range(M):
        a[i, :] = scipy.fftpack.dct(matrix[i, :])
    for j in range(N):
        b[:, j] = scipy.fftpack.dct(a[:, j])

    return b


def idct2(matrix: np.ndarray):
    """
    Inverse dct type II

    Credits for idct2() comes from Mark Newman <mejn@umich.edu>
    """
    M = matrix.shape[0]
    N = matrix.shape[1]
    a = np.empty([M, N], float)
    y = np.empty([M, N], float)

    for i in range(M):
        a[i, :] = scipy.fftpack.idct(matrix[i, :])
    for j in range(N):
        y[:, j] = scipy.fftpack.idct(a[:, j])

    return np.divide(y, 256)


def _zigzag_indices(matrix: np.ndarray):
    """
    Ok not the prettiest.. but I cannot figure how to do this the np-way

    """
    d = {}
    for y, row in enumerate(matrix):
        for x, ele in enumerate(row):
            if x + y in d:
                d[x + y].append((y, x))
            else:
                d[x + y] = [(y, x)]
    combined = []
    for i in range(len(d)):
        if i % 2 == 0:
            combined.append(d[i])
        else:
            combined.append(list(reversed(d[i])))
    return functools.reduce(lambda a, b: a + b, combined)


def zigzag(matrix: np.ndarray):
    """
    Linearize the 2d matrix into a 1d array by diagonals. This way, the bulk of 0 are clumped together at the end and
    compress better with Huffman coding

    """
    indices = _zigzag_indices(matrix)
    result = list(map(lambda t: matrix[t[0]][t[1]], indices))
    return result


def izigzag(arr: np.ndarray, shape: Tuple[int, int]):
    """
    Recreate our matrix from the diagonals
    """
    mat = np.zeros(shape)
    indices = _zigzag_indices(mat)
    zipped = zip(arr, indices)
    for ele in zipped:
        y, x = ele[1]
        mat[y][x] = ele[0]
    return mat


def up_sample(matrix: np.ndarray, factor=2):
    """
    Upsample our array, thankfully opencv does most of the work for us
    """
    x, y = matrix.shape
    new_shape = (x * factor, y * factor)
    return cv2.pyrUp(matrix, dstsize=new_shape, borderType=cv2.BORDER_DEFAULT)


def down_sample(matrix: np.ndarray, factor=2):
    """
    Downsample our array, thankfully opencv does this too
    """
    x, y = matrix.shape
    new_shape = (x // factor, y // factor)
    return cv2.pyrDown(matrix, dstsize=new_shape, borderType=cv2.BORDER_DEFAULT)


def inv_dct_channel(channel: np.ndarray, quantization_table: model.QTables, block_size=8):
    blocks = split_matrix(channel, block_size)
    rev_q_blocks = [qz.invert_jpeg_quantize(b, quantization_table) for b in blocks]
    rev_dct = [idct2(b) for b in rev_q_blocks]
    merged = merge_blocks(np.array(rev_dct), channel.shape)
    offset = np.add(merged, 128).astype(np.uint8)
    return offset


def dct_channel(channel: np.ndarray, quantization_table: model.QTables, block_size=8):
    """
    Apply the Discrete Cosine transform on a channel of our image to later be encoded
    """
    offset = channel.astype(np.int64) - 128

    blocks = split_matrix(offset, block_size)
    transformed_blocks = [dct2(block) for block in blocks]
    qnt_blocks = [qz.jpeg_quantize(block, quantization_table) for block in transformed_blocks]
    quantized = np.array(qnt_blocks)
    ret = merge_blocks(quantized, channel.shape)
    return ret


def wavelet_split_resolutions(channel: np.ndarray, wavelet: model.Wavelet, levels=3):
    """
    Simple wrapper to also flatten the array for convenience
    """
    cascade = pywt.wavedec2(channel, wavelet.value, level=levels)
    return linearize_subband(cascade)


def linearize_subband(subbands):
    """
    Unravel the subbands so I can do general transforms irrespective of the subbands
    """
    subbands[0] = [subbands[0]]
    return functools.reduce(lambda x, y: x + list(y), subbands, [])


def subband_view(pyramid: List[np.ndarray]):
    """
    Reconstruct a subband
    """
    return [pyramid[0]] + utils.group_tuples(pyramid[1:], 3)


def wavelet_merge_resolutions(pyramid: List[np.ndarray], wavelet: model.Wavelet):
    """
    Since hombln just had to unravel the output of .wavedec2(), I need to reconstruct here!
    """
    coeffs = subband_view(pyramid)
    return pywt.waverec2(coeffs, wavelet.value)


def threshold(arr: np.ndarray, thresh, replace=0):
    """
    Supposedly scipy.stats has this, but I must not have the version. I'll search for it later.
    """

    def _vec(ele):
        if abs(ele) < thresh:
            return replace
        else:
            return ele

    vector = np.vectorize(_vec)
    return vector(arr)


def threshold_channel_by_quality(parts: List[np.ndarray], q_factor=1):
    """
    Actually apply the threshold after calculating the value. We take an array in parts since most likely
    our call is from the Wavelet compression where we have a series of images from the filter bank.
    """
    imgs = [utils.img_as_list(img) for img in parts]
    vals = utils.flatten(imgs)
    val = qz.quality_threshold_value(vals, q_factor)
    return [threshold(i, val) for i in parts]


def dc_component(block: np.ndarray):
    """
    Return the DC component for a block (JPEG)
    """
    return block[0][0]


def ac_components(block: np.ndarray):
    """
    Return the AC components for a block (JPEG)
    """
    return zigzag(block)[1:]
