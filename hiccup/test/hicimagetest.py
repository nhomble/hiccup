from pydoc import locate
import os
import unittest

from hiccup import hicimage

import numpy as np

test_dir = "resources/"


class HicImageTest(unittest.TestCase):
    def test_read_header(self):
        path = os.path.join(os.path.dirname(__file__), test_dir + "foo.hic")
        result = hicimage.HicImage.load_file(path)
        self.assertEqual(100, result.rows)
        self.assertEqual(200, result.columns)

        transform = result.transform
        self.assertEqual(transform.__class__.__name__, "FourierTransform")

    def test_rw_header(self):
        test_file = test_dir + "_tmp_write.hic"
        try:
            test_image = hicimage.HicImage(1, 2, hicimage.WaveletTransform([]), np.ndarray([1, 2, 3]))
            test_image.write_file(test_file)
            read = hicimage.HicImage.load_file(test_file)
            self.assertEqual(test_image.rows, read.rows)
            self.assertEqual(test_image.columns, read.columns)
            self.assertEqual(test_image.transform.__class__.__name__, read.transform.__class__.__name__)
        finally:
            os.remove(test_file)
