import unittest

import hiccup.huffman as huffman


class HuffmanTest(unittest.TestCase):
    def test_singleton_encoding(self):
        data = [0, 0, 0, 0, 0]
        tree = huffman.HuffmanTree.construct_from_data(data)
        self.assertEqual(len(tree.leaves), 1)
        self.assertEqual(tree.encoding(), "11111")

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

    def test_encode(self):
        data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
        tree = huffman.HuffmanTree.construct_from_data(data)
        out = tree.encoding()
        self.assertEqual(out, "001" + ("000" * 2) + ("01" * 3) + ("1" * 4))
