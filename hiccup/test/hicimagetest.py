import cv2
import unittest

import hiccup.hicimage as hic
import hiccup.compression as compression
import hiccup.codec as codec


class HicImageTest(unittest.TestCase):
    def test_plain_string(self):
        rgb = cv2.imread("resources/gh.png")
        out = compression.wavelet_compression(rgb)
        h = codec.wavelet_encode(out)

        bites = h.byte_stream

        retrieve = h.from_bytes(bites)
        print(bites)

