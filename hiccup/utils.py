import functools
from typing import List

import cv2
import numpy as np
import datetime

import hiccup.settings as settings

"""
Random helpful utils
"""


def debug_img(img):
    while True:
        cv2.imshow("debugging image", img)
        if cv2.waitKey() == ord('q'):
            break


def debug_msg(msg):
    if settings.DEBUG:
        print("%s %s" % (datetime.datetime.utcnow(), msg))


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
    n = abs(int(n))
    bits = 0
    while n > 0:
        n >>= 1
        bits += 1
    return bits


def differences(arr: List[int]):
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


def invert_differences(arr: List[int]):
    """
    Invert differences
    """
    ret = [arr[0]]
    for diff in arr[1:]:
        ret.append(ret[-1] + diff)
    return ret


def identity(x):
    """
    Identity function
    """
    return x


def group_by(data, key_func=identity, value_func=identity):
    """
    Quick frequency table for Huffman
    """

    def reduce(dic, ele):
        k = key_func(ele)
        if k in dic:
            dic[k].append(value_func(ele))
        else:
            dic[k] = [value_func(ele)]
        return dic

    return functools.reduce(reduce, data, {})


def first(l: iter, predicate):
    """
    Get first element to satisfy predicate
    """
    for ele in l:
        if predicate(ele):
            return ele
    raise RuntimeError("Found nothing to match predicate")


def flatten(l: iter):
    """
    Simple flatten for my use cases
    """
    return functools.reduce(lambda x, y: x + y, l)


def img_as_list(img: np.ndarray):
    """
    Don't care how, just flatten the ndarray into 1d-list. Helpful if I am doing image wide calcs that don't care about
    positioning.
    """
    rows = img.tolist()
    return flatten(rows)


def size(shape: tuple):
    return shape[0] * shape[1]
