from typing import Tuple

import functools
import numpy as np
import cv2
import scipy
import scipy.fftpack

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
        x + (N - x % N),
        y + (N - y % N)
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
        raise RuntimeError("Incompatible dimensions")

    d4 = blocks.reshape(y // N, x // N, N, N)
    block_mats = [
        [np.matrix(d4[y][x]) for x in range(d4[y].shape[0])]
        for y in range(d4.shape[0])
    ]
    return np.array(np.bmat(block_mats))


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

    return y


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


def dct_channel(self, channel):
    offset = channel.astype(np.int8) - 128

    blocks = split_matrix(offset, self._block_size)
    transformed_blocks = [dct2(block) for block in blocks]
    table = qz.table[self._quantization_table]
    dividend = np.divide(transformed_blocks, table)
    quantized = np.round(dividend)
    result = merge_blocks(quantized, channel.shape).astype(channel.dtype)
    return result


def jpeg_compression(self, rgb_img: np.ndarray):
    yrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
    splits = cv2.split(yrcb)

    splits[0] = self.dct_compress(splits[0])  # 0 is the luminosity
    compressed_img = cv2.merge(splits)
    return compressed_img
