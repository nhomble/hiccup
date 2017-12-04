import unittest

import hiccup.iohelper as io


class IOHelperTest(unittest.TestCase):
    def test_bin_string(self):
        self.assertEqual("11", io.bin_string(3))

    def test_s2b(self):
        out = io.bin_string_as_bytes("00000001")
        self.assertEqual(out, b'\x01')

    def test_s2b_assertion(self):
        self.assertRaises(AssertionError, io.bin_string_as_bytes, '0')
        self.assertRaises(AssertionError, io.bin_string_as_bytes, '')