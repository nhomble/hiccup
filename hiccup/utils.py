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
