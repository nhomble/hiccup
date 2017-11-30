import unittest
import numpy as np

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
        q = qnt.dead_quantize(mat, qnt.QTables.JPEG_LUMINANCE)
        self.assertEqual(q[0][0], 1)
        self.assertEqual(np.sum(q), 1)

    def test_quality_threshold(self):
        i = [
            np.array(range(50)),
            np.array(range(50, 100))
        ]
        out = qnt.quality_threshold(i, q_factor=.05)
        self.assertEqual(out, 94)