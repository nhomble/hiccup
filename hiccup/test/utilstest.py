import unittest
import hiccup.utils as utils


class UtilsTest(unittest.TestCase):
    def test_grouping(self):
        out = utils.group_tuples([1, 2, 3, 4], 2)
        self.assertEqual(out, [(1, 2), (3, 4)])

    def test_bad_size_grouping(self):
        self.assertRaises(AssertionError, utils.group_tuples, [1, 2, 3], 2)

    def test_bit_test(self):
        cases = [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 2],
            [4, 3],
            [100, 7],
            [-60, 6],
            [-1, 1]
        ]
        for case in cases:
            self.assertEqual(utils.num_bits_for_int(case[0]), case[1],
                             msg="utils.num_bits_for_int(%d) == %d" % (case[0], case[1]))

    def test_differences(self):
        arr = [1, 5, 12, 0]
        self.assertEqual(utils.differences(arr), [
            1, 4, 7, -12
        ])