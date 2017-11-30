import unittest
import numpy as np

import hiccup.transform as transform
import hiccup.codec as codec


class CodecTest(unittest.TestCase):
    def test_differential_coding(self):
        matrix = np.array([[i + k for k in range(8)] for i in range(8)])
        blocks = transform.split_matrix(matrix, 4)
        dc = codec.differential_coding(blocks)
        self.assertEqual([0, 4, 0, 4], dc)
