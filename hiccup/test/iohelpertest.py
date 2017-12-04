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

    def _check_pb(self, s):
        out = io.padded_bs_2_bytes(s)
        invert = io.padded_bytes_2_bs(out)
        self.assertEqual(s, invert, msg=s)

    def test_padding_and_back(self):
        cases = [
            "01",
            "0000",
            "1010000",
            "00000",
            "000111"
        ]
        for case in cases:
            self._check_pb(case)
