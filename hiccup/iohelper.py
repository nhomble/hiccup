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


def bin_string(i: int) -> str:
    """
    Binary string
    """
    return bin(i)[2:]


def bin_string_as_bytes(s: str) -> bytearray:
    """
    Need to represent each bit as an actual bit instead of wasting a byte per char. Also I have to respect the zeros
    """
    assert len(s) > 0 and len(s) % 8 == 0  # if not, I probably forgot to pad upstream in HIC
    i = int(s, 2)
    out = bytearray(i.to_bytes(len(s) // 8, byteorder='big'))
    return out


def padded_bs_2_bytes(s: str) -> bytearray:
    """
    I NEED TO BE BYTE ALIGNED, otherwise how do I know what's going on
    """
    padding = 8 - (len(s) % 8)
    s += ("0" * padding)
    binary = bin_string_as_bytes(s)
    binary.extend(chr(padding).encode('ascii'))
    # so first byte tells me the padding
    return binary


def padded_bytes_2_bs(bites: bytearray) -> str:
    padding = bites[-1]
    b = bites[:-1]
    i = int.from_bytes(b, byteorder='big')
    s = bin_string(i)
    # any diff should be leading
    res = s[:-padding]
    if len(res) + padding % 8 != 0:
        # forward padd
        diff = 8 - (len(res) + padding % 8)
        res = ("0" * diff) + res
    return res
