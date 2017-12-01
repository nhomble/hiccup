import heapq

import hiccup.utils as utils

"""
While we use the defaults given by the specs for encoding, we need to realize a Huffman tree to decode
"""


class HuffmanTree:
    @classmethod
    def construct_from_data(cls, data, key_func=utils.identity):
        """
        Public constructor from data
        """
        groups = utils.group_by(data, key_func=key_func)
        root, leaves = cls._construct(groups)
        return cls(root, leaves)

    @classmethod
    def _construct(cls, groups):
        """
        From the groups, construct the Huffman tree and return root for reference
        """
        leaves = [cls.Node.leaf(t[0], len(t[1])) for t in groups.items()]
        node_heap = list(leaves)
        heapq.heapify(node_heap)
        while len(node_heap) > 1:
            l = heapq.heappop(node_heap)
            r = heapq.heappop(node_heap)
            parent = cls.Node.combine(l, r)
            heapq.heappush(node_heap, parent)

        root = heapq.heappop(node_heap)

        return root, leaves

    def __init__(self, root, leaves):
        self.root = root
        self.leaves = leaves

    class Node:
        GROUND = None

        @classmethod
        def combine(cls, l, r):
            return cls(l, r, None, l.frequency + r.frequency) \
                .inherit(l) \
                .inherit(r)

        @classmethod
        def leaf(cls, value, frequency):
            return cls(cls.GROUND, cls.GROUND, value, frequency)

        def __init__(self, left, right, value, frequency):
            self.parent = None
            self.left = left
            self.right = right
            self.value = value
            self.frequency = frequency

        def inherit(self, child):
            """
            Point the child back to the parent for ease in encoding
            """
            child.parent = self
            return self

        @property
        def is_leaf(self):
            return self.left is self.GROUND and self.right is self.GROUND

        def __eq__(self, other):
            return self.value == other.value

        def __lt__(self, other):
            return self.frequency < other.frequency

        def __gt__(self, other):
            return self.frequency > other.frequency

        def __le__(self, other):
            return self < other or self == other

        def __ge__(self, other):
            return self > other or self == other
