import unittest
import numpy as np

import hiccup.model as model
import hiccup.quantization as qnt


class QuantizationTest(unittest.TestCase):
    def test_all_types(self):
        self.assertEqual(2, len(qnt.all_tables))

    def test_quantize(self):
        mat = np.array([
            [16, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ])
        q = qnt.jpeg_quantize(mat, model.QTables.JPEG_LUMINANCE)
        self.assertEqual(q[0][0], 1)
        self.assertEqual(np.sum(q), 1)

    def test_quality_threshold(self):
        i = range(100)
        out = qnt.quality_threshold_value(list(i), q_factor=.05)
        self.assertEqual(out, 94)

    def test_round_quantize(self):
        mat = np.array([
            [.1, .4],
            [.5, .6],
            [.8, 10]
        ])
        out = qnt.round_quantize(mat)
        self.assertTrue(np.array_equiv(out, np.array([
            [0, 0],
            [0, 1],
            [1, 10]
        ])))