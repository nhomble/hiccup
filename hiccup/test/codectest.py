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

    def test_run_length(self):
        matrix = np.array([
            [99, -59, 0, 7, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [12, -2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        rl = codec.run_length_coding(matrix)
        expected = [{'zeros': 0, 'value': 99, 'bits': 7}, {'zeros': 1, 'value': -59, 'bits': 6},
                    {'zeros': 6, 'value': 7, 'bits': 3}, {'zeros': 4, 'value': 12, 'bits': 4},
                    {'zeros': 1, 'value': -2, 'bits': 2}, {'zeros': 0, 'value': 0, 'bits': 0}]
        self.assertEqual(expected, rl)