import rawpy
import bitstring

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
    return bitstring.BitArray('0b' + s).bytes


def padded_bs_2_bytes(s: str) -> bytearray:
    """
    I NEED TO BE BYTE ALIGNED, otherwise how do I know what's going on
    """
    b = bitstring.BitArray('0b' + s)

    padding = 8 - (len(b) % 8)
    # assert we are byte aligned
    b.append(padding)

    to_prep = bitstring.Bits(uint=padding, length=8)
    b.prepend(to_prep)

    return b.tobytes()


def padded_bytes_2_bs(bites: bytearray) -> str:
    b = bitstring.BitArray(bytes=bites)
    padding = b[:8].int
    b <<= 8
    bits = b.bin[:-padding-8]
    return bits
