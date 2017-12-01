import math
import cv2

"""
Random helpful utils
"""


def debug_img(img):
    while True:
        cv2.imshow("debugging image", img)
        if cv2.waitKey() == ord('q'):
            break


def group_tuples(l, n):
    """
    Group a list of elements into tuples of size n.
    """
    assert len(l) % n == 0
    ret = []
    for i in range(0, len(l), n):
        v = l[i:i + n]
        ret.append(tuple(v))
    return ret


def num_bits_for_int(n: int):
    """
    Calculate the number of bits required to represent integer n
    """
    n = abs(n)
    bits = 0
    while n > 0:
        n >>= 1
        bits += 1
    return bits


def differences(arr):
    """
    Compute differences between elements
    """
    ret = []
    i = 0
    for dc in arr:  # ugh
        if len(ret) == 0:
            ret.append(dc)
        else:
            ret.append(dc - arr[i])
            i += 1
    return ret
