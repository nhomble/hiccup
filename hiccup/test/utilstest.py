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
        self.assertEqual(padded.size, (2, 2))

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
