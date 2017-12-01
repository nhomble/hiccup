import unittest

import hiccup.huffman as huffman


class HuffmanTest(unittest.TestCase):
    def test_singleton(self):
        data = [0, 0, 0, 0, 0]
        tree = huffman.HuffmanTree.construct_from_data(data)
        self.assertEqual(tree.root.value, 0)
        self.assertEqual(tree.root.frequency, 5)
        self.assertTrue(tree.root.is_leaf)

    def test_2_nodes(self):
        data = [0, 0, 0, 1]
        tree = huffman.HuffmanTree.construct_from_data(data)
        self.assertEqual(tree.root.frequency, 4)
        self.assertFalse(tree.root.is_leaf)

        self.assertEqual(tree.root.left.frequency, 1)
        self.assertEqual(tree.root.left.value, 1)
        self.assertTrue(tree.root.left.is_leaf)

        self.assertEqual(tree.root.right.frequency, 3)
        self.assertEqual(tree.root.right.value, 0)
        self.assertTrue(tree.root.right.is_leaf)