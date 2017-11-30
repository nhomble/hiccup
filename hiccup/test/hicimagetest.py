import os
import unittest

from hiccup import hicimage

test_dir = "resources/"


class HicImageTest(unittest.TestCase):
    def test_read_header(self):
        path = os.path.join(os.path.dirname(__file__), test_dir + "foo.hic")
        result = hicimage.HicImage.load_file(path)
        self.assertEqual(100, result.rows)
        self.assertEqual(200, result.columns)

        self.assertEqual(result.style, "Fourier")
