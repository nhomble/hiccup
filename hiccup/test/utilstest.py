import random
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

    def test_group_by(self):
        vals = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
        table = utils.group_by(vals)
        for i in range(1, 5):
            self.assertEqual(table[i], [i] * i)

    def test_group_by_complex(self):
        vals = [("A", 1), ("B", 1), ("A", 2)]
        table = utils.group_by(vals, lambda t: t[0], lambda t: t[1])
        self.assertEqual(table["A"], [1, 2])
        self.assertEqual(table["B"], [1])

    def test_first(self):
        arr = [1, 2, 3, 4]
        self.assertEqual(utils.first(arr, lambda x: x > 1), 2)
        self.assertRaises(RuntimeError, utils.first, arr, lambda x: x < 0)

    def test_flatten(self):
        self.assertEqual(utils.flatten([
            [1, 2],
            [3, 4]
        ]), [1, 2, 3, 4])

    def test_invert_differences(self):
        diffs = [1, 1, 1, 1]
        self.assertEqual(utils.invert_differences(diffs), [
            1, 2, 3, 4
        ])

    def test_differences_random(self):
        arr = [random.randint(0, 10) for _ in range(100)]
        diffs = utils.differences(arr)
        invert = utils.invert_differences(diffs)
        self.assertEqual(arr, invert)