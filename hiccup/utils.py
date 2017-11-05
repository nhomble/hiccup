import numpy as np
import rawpy
import cv2


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
