import unittest
import numpy as np

import hiccup.utils as utils


class UtilsTest(unittest.TestCase):
    def test_padding(self):
        one = np.ones((1, 1), dtype=np.int)
        padded = utils.pad_matrix(one, 3)
        self.assertEqual((3, 3), padded.shape)
        self.assertEqual(1, np.sum(padded))

    def test_padding_unecessary(self):
        zeros = np.zeros((2, 2))
        padded = utils.pad_matrix(zeros, 2)
        self.assertEqual(padded.shape, (2, 2))

    def test_block_nicely(self):
        square = np.array([
            [1, 2],
            [3, 4]
        ])
        splits = utils.split_matrix(square, 1)
        self.assertEqual(len(splits), 4)
        self.assertEqual((1, 1), splits[0].shape)
        self.assertEqual(1, splits[0][0][0])
        self.assertEqual(2, splits[1][0][0])
        self.assertEqual(3, splits[2][0][0])
        self.assertEqual(4, splits[3][0][0])

    def test_block_must_pad(self):
        square = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        splits = utils.split_matrix(square, 2)
        self.assertEqual(len(splits), 4)
        self.assertEqual(splits[0].shape, (2, 2))
        self.assertEqual(np.sum(splits[0]), 12)
        self.assertEqual(np.sum(splits[3]), 9)

    def test_dct_null_signal(self):
        zeros = np.zeros((5, 5))
        dct = utils.dct2(zeros)
        self.assertEqual(np.sum(dct), 0)

    def test_identity(self):
        random = np.random.randint(-128, 128, (8, 8))
        dct = utils.dct2(random)
        idct = utils.idct2(dct)
        scalar = idct[0][0] / random[0][0]
        opposite = random * scalar * -1
        zeros = np.add(opposite, idct)
        self.assertTrue(np.sum(zeros) < 1e-10)

    def test_merge_blocks(self):
        blocks = np.array([
            [[1, 2], [7, 8]], [[3, 4], [9, 10]], [[5, 6], [11, 12]],
            [[13, 14], [0, 0]], [[15, 16], [0, 0]], [[17, 18], [0, 0]]
        ])
        merged = utils.merge_blocks(blocks, (4, 6))
        self.assertTrue(np.array_equiv(np.array([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [0, 0, 0, 0, 0, 0]
        ]), merged))

    def test_split_merge(self):
        matrix = np.random.randint(0, 256, (8, 12))
        blocks = utils.split_matrix(matrix, 4)
        merged = utils.merge_blocks(blocks, matrix.shape)
        self.assertTrue(np.array_equiv(matrix, merged))

    def test_zigzag(self):
        matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        d = utils.zigzag(matrix)
        self.assertTrue(np.array_equiv([
            1, 4, 2, 3, 5, 7, 8, 6, 9
        ], d))

    def test_izigzag(self):
        arr = np.array([1, 4, 2, 3, 5, 7, 8, 6, 9])
        m = utils.izigzag(arr, (3, 3))
        self.assertTrue(np.array_equiv(np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]), m))

    def test_zigzag_inverse(self):
        matrix = np.random.randint(0, 256, (17, 19))
        ziggy = utils.zigzag(matrix)
        imat = utils.izigzag(ziggy, matrix.shape)
        self.assertTrue(np.array_equiv(matrix, imat))