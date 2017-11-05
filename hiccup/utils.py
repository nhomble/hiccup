import numpy as np
import rawpy
import cv2
import scipy
import scipy.fftpack

"""
Helpful functions ranging from image loading to extending signal transforms for sequences into 2d. 

Credits for dct2() comes from Mark Newman <mejn@umich.edu>
"""


def open_raw_img(path):
    """
    open image from path
    :param path: path to raw image
    :return:
    """

    # opencv cannot handle raw images
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess()
    return rgb


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


def debug_img(img):
    while True:
        cv2.imshow("debugging image", img)
        if cv2.waitKey() == ord('q'):
            break


def dct2(matrix: np.ndarray):
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
    M = matrix.shape[0]
    N = matrix.shape[1]
    a = np.empty([M, N], float)
    y = np.empty([M, N], float)

    for i in range(M):
        a[i, :] = scipy.fftpack.idct(matrix[i, :])
    for j in range(N):
        y[:, j] = scipy.fftpack.idct(a[:, j])

    return y
