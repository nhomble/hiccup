import unittest

import hiccup.quantization as qnt


class QuantizationTest(unittest.TestCase):
    def test_all_types(self):
        self.assertSetEqual({"common"}, qnt.all_tables)
