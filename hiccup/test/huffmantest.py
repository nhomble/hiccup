import random
import unittest

import hiccup.huffman as huffman


class HuffmanTest(unittest.TestCase):
    def test_singleton_encoding(self):
        data = [0, 0, 0, 0, 0]
        tree = huffman.HuffmanTree.construct_from_data(data)
        self.assertEqual(len(tree.leaves), 1)
        self.assertEqual(tree.encode_data(), "11111")

    def test_2_nodes(self):
        data = [0, 0, 0, 1]
        tree = huffman.HuffmanTree.construct_from_data(data)
        self.assertEqual(tree.root.frequency, 4)
        self.assertFalse(tree.root.is_leaf)
        self.assertEqual(len(tree.leaves), 2)
        self.assertTrue(tree.root.left in tree.leaves)
        self.assertTrue(tree.root.right in tree.leaves)

        self.assertEqual(tree.root.left.frequency, 1)
        self.assertEqual(tree.root.left.value, 1)
        self.assertTrue(tree.root.left.is_leaf)

        self.assertEqual(tree.root.right.frequency, 3)
        self.assertEqual(tree.root.right.value, 0)
        self.assertTrue(tree.root.right.is_leaf)

    def test_2_complex(self):
        data = [("A", 0), ("B", 1), ("B", 0)]
        tree = huffman.HuffmanTree.construct_from_data(data, key_func=lambda t: t[0])
        self.assertEqual(tree.root.frequency, 3)
        self.assertFalse(tree.root.is_leaf)
        self.assertEqual(len(tree.leaves), 2)
        self.assertTrue(tree.root.left in tree.leaves)
        self.assertTrue(tree.root.right in tree.leaves)

        self.assertEqual(tree.root.left.frequency, 1)
        self.assertEqual(tree.root.left.value, "A")
        self.assertTrue(tree.root.left.is_leaf)

        self.assertEqual(tree.root.right.frequency, 2)
        self.assertEqual(tree.root.right.value, "B")
        self.assertTrue(tree.root.right.is_leaf)

    def test_encode_decode(self):
        data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
        tree = huffman.HuffmanTree.construct_from_data(data)
        out = tree.encode_data()
        self.assertEqual(out, "001" + ("000" * 2) + ("01" * 3) + ("1" * 4))

        inverse = tree.decode_data(out)
        self.assertEqual(data, inverse)

    def _reconstruction_case(self, msg, data):
        tree = huffman.HuffmanTree.construct_from_data(data)

        data_encoded = tree.encode_data()
        table_encoded = tree.encode_table()

        new_tree = huffman.HuffmanTree.construct_from_coding(table_encoded)
        invert = new_tree.decode_data(data_encoded)

        self.assertEqual(data, invert, msg=msg)

    def test_reconstruction(self):
        cases = {
            "random": [random.randint(0, 20) for _ in range(10000)],
            "equal frequencies": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "single point": [0],
            "singleton tree": [0, 0, 0, 0, 0]
        }
        for case in cases.items():
            self._reconstruction_case(*case)

    def test_encode_foreign_data(self):
        data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
        tree = huffman.HuffmanTree.construct_from_data(data)
        out = tree.encode_data(data=[4, 4, 4, 4])
        self.assertEqual(out, "1111")