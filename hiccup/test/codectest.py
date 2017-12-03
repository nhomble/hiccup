import unittest
import numpy as np

import hiccup.model as model
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
        rl = codec.run_length_coding(transform.zigzag(matrix))
        expected = [{'zeros': 0, 'value': 99, 'bits': 7}, {'zeros': 1, 'value': -59, 'bits': 6},
                    {'zeros': 6, 'value': 7, 'bits': 3}, {'zeros': 4, 'value': 12, 'bits': 4},
                    {'zeros': 1, 'value': -2, 'bits': 2}, {'zeros': 0, 'value': 0, 'bits': 0}]
        self.assertEqual([codec.RunLength.from_dict(e) for e in expected], rl)

    def test_dc_category(self):
        cases = [
            (0, model.Coefficient.DC, 0),
            (-1, model.Coefficient.DC, 1),
            (1, model.Coefficient.DC, 1),
            (511, model.Coefficient.DC, 9)
        ]
        for case in cases:
            self.assertEqual(codec.jpeg_category(case[0], case[1]), case[2], msg="jpeg_category( %d, %s ) == %d" % case)

    def test_ac_category(self):
        cases = [
            (-1, model.Coefficient.DC, 1),
            (1, model.Coefficient.DC, 1),
            (511, model.Coefficient.DC, 9)
        ]
        for case in cases:
            self.assertEqual(codec.jpeg_category(case[0], case[1]), case[2], msg="jpeg_category( %d, %s ) == %d" % case)

        self.assertRaises(RuntimeError, codec.jpeg_category, 0, model.Coefficient.AC)
        self.assertRaises(RuntimeError, codec.jpeg_category, 16385, model.Coefficient.AC)

    def test_rle_too_long(self):
        l = ([0] * 17) + [1]
        arr = np.array(l)
        out = codec.run_length_coding(arr)
        zeros = [s.length for s in out]
        self.assertEqual(zeros, [15, 2])

    def _symbol_0_0(self, has, arr):
        out = codec.run_length_coding(np.array(arr))
        last = out[-1]
        self.assertEqual((last.length == 0 and last.value == 0), has)

    def test_symbol_0_0(self):
        cases = [
            (False, [0, 0, 5]),
            (True, [0, 0, 5, 0, 0])
        ]
        for case in cases:
            self._symbol_0_0(*case)
