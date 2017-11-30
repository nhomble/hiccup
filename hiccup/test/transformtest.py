import unittest
import numpy as np

from hiccup.test.testhelper import *
import hiccup.transform as trans


class TransformTest(unittest.TestCase):
    def test_padding(self):
        one = np.ones((1, 1), dtype=np.int)
        padded = trans.pad_matrix(one, 3)
        self.assertEqual((3, 3), padded.shape)
        self.assertEqual(1, np.sum(padded))

    def test_padding_unecessary(self):
        zeros = np.zeros((2, 2))
        padded = trans.pad_matrix(zeros, 2)
        self.assertEqual(padded.shape, (2, 2))

    def test_block_nicely(self):
        square = np.array([
            [1, 2],
            [3, 4]
        ])
        splits = trans.split_matrix(square, 1)
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
        splits = trans.split_matrix(square, 2)
        self.assertEqual(len(splits), 4)
        self.assertEqual(splits[0].shape, (2, 2))
        self.assertEqual(np.sum(splits[0]), 12)
        self.assertEqual(np.sum(splits[3]), 9)

    def test_dct_null_signal(self):
        zeros = np.zeros((5, 5))
        dct = trans.dct2(zeros)
        self.assertEqual(np.sum(dct), 0)

    def test_identity(self):
        random = np.random.randint(-128, 128, (8, 8))
        dct = trans.dct2(random)
        idct = trans.idct2(dct)
        scalar = idct[0][0] / random[0][0]
        opposite = random * scalar * -1
        zeros = np.add(opposite, idct)
        self.assertTrue(np.sum(zeros) < 1e-10)

    def test_merge_blocks(self):
        blocks = np.array([
            [[1, 2], [7, 8]], [[3, 4], [9, 10]], [[5, 6], [11, 12]],
            [[13, 14], [0, 0]], [[15, 16], [0, 0]], [[17, 18], [0, 0]]
        ])
        merged = trans.merge_blocks(blocks, (4, 6))
        self.assertTrue(np.array_equiv(np.array([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [0, 0, 0, 0, 0, 0]
        ]), merged))

    def test_split_merge(self):
        matrix = np.random.randint(0, 256, (8, 12))
        blocks = trans.split_matrix(matrix, 4)
        merged = trans.merge_blocks(blocks, matrix.shape)
        self.assertTrue(np.array_equiv(matrix, merged))

    def test_zigzag(self):
        matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        d = trans.zigzag(matrix)
        self.assertTrue(np.array_equiv([
            1, 4, 2, 3, 5, 7, 8, 6, 9
        ], d))

    def test_izigzag(self):
        arr = np.array([1, 4, 2, 3, 5, 7, 8, 6, 9])
        m = trans.izigzag(arr, (3, 3))
        self.assertTrue(np.array_equiv(np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]), m))

    def test_zigzag_inverse(self):
        matrix = np.random.randint(0, 256, (17, 19))
        ziggy = trans.zigzag(matrix)
        imat = trans.izigzag(ziggy, matrix.shape)
        self.assertTrue(np.array_equiv(matrix, imat))

    def test_up_sample(self):
        matrix = np.array([
            [2, 2],
            [2, 2],
        ]).astype(np.uint8)
        out = trans.up_sample(matrix, factor=2)
        self.assertTrue(np.array_equiv(out, np.array([
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]).astype(np.uint8)))

    def test_down_sample(self):
        matrix = np.array([
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]).astype(np.uint8)
        out = trans.down_sample(matrix, factor=2)
        self.assertTrue(np.array_equiv(out, np.array([
            [2, 2],
            [2, 2],
        ]).astype(np.uint8)))

    def test_dct_null(self):
        img = np.zeros((120, 80)).astype(np.uint8)
        out = trans.dct_channel(img)
        self.assertEqual(0, np.sum(out))

    def test_dct_ones(self):
        img = np.ones((120, 80)).astype(np.uint8)
        out = trans.dct_channel(img)
        self.assertTrue(np.sum(out) > 0)
