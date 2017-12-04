import rawpy

"""
IO ickiness
"""


def open_raw_img(path):
    """
    open image from path
    """

    # opencv cannot handle raw images
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess()
    return rgb


def bin_string(i: int):
    """
    Binary string
    """
    return bin(i)[2:]


def bin_string_as_bytes(s):
    """
    Need to represent each bit as an actual bit instead of wasting a byte per char. Also I have to respect the zeros
    """
    assert len(s) > 0 and len(s) % 8 == 0  # if not, I probably forgot to pad upstream in HIC
    i = int(s, 2)
    out = i.to_bytes(len(s) // 8, byteorder='big')
    return out
