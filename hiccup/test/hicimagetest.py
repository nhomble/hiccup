import cv2
import unittest

import hiccup.compression as compression
import hiccup.codec as codec


class HicImageTest(unittest.TestCase):
    def test_gh_wavelet(self):
        rgb = cv2.imread("resources/gh.png")
        out = compression.wavelet_compression(rgb)
        h = codec.wavelet_encode(out)

        bites = h.byte_stream()

        retrieve = h.from_bytes(bites)
        loads = zip(h.payloads, retrieve.payloads)
        evals = []
        # inspect the cases
        for (i, z) in enumerate(loads):
            evals.append((z[0] == z[1], z[0], z[1]))

        for e in evals:
            self.assertTrue(e[0])

    def test_gh_jpg(self):
        rgb = cv2.imread("resources/gh.png")
        out = compression.jpeg_compression(rgb)
        h = codec.jpeg_encode(out)

        bites = h.byte_stream()

        retrieve = h.from_bytes(bites)
        loads = zip(h.payloads, retrieve.payloads)
        for (i, z) in enumerate(loads):
            self.assertEqual(z[0], z[1], msg="Case %d" % i)
