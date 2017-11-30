import unittest
import hiccup.utils as utils


class UtilsTest(unittest.TestCase):
    def test_grouping(self):
        out = utils.group_tuples([1, 2, 3, 4], 2)
        self.assertEqual(out, [(1, 2), (3, 4)])

    def test_bad_size_grouping(self):
        self.assertRaises(AssertionError, utils.group_tuples, [1, 2, 3], 2)
