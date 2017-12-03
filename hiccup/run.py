import os
import cv2
import hiccup.compression as compression
import hiccup.model as model
import hiccup.codec as codec
import hiccup.hicimage as hic

"""
Entry functions for belch
"""


def img_name(path, c: model.Compression):
    f = os.path.split(path)[-1]
    return f + ".%s-hic" % c.value


def compress(path, output, c: model.Compression):
    rgb = cv2.imread(path)
    if c == model.Compression.JPEG:
        compressed = compression.jpeg_compression(rgb)
        hi = codec.jpeg_encode(compressed)
    elif c == model.Compression.HIC:
        compressed = compression.wavelet_compression(rgb)
        hi = codec.wavelet_encode(compressed)
    else:
        raise RuntimeError("Unknown compression type")
    output = os.path.join(output, img_name(path, c))
    hi.write_file(output)


def decompress(path):
    hi = hic.HicImage.from_file(path)
    if hi.hic_type == model.Compression.JPEG:
        compressed = codec.jpeg_decode(hi)
        rgb = compression.jpeg_decompression(compressed)
    elif hi.hic_type == model.Compression.HIC:
        compressed = codec.wavelet_decode(hi)
        rgb = compression.wavelet_decompression(compressed)
    else:
        raise RuntimeError("Unknown compression type")
    cv2.imshow("Result", rgb)
    cv2.waitKey()
