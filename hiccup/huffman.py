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
        return cls(root, leaves, data, key_func)

    @classmethod
    def _construct(cls, groups):
        """
        From the groups, construct the Huffman tree and return root for reference
        """
        leaves = [cls.Node.leaf(t[0], len(t[1])) for t in groups.items()]

        # in the stupid case you just have 1 element repeated over and over
        if len(leaves) == 1:
            return cls.Node.singleton(leaves[0]), leaves

        node_heap = list(leaves)
        heapq.heapify(node_heap)
        while len(node_heap) > 1:
            l = heapq.heappop(node_heap)
            r = heapq.heappop(node_heap)
            parent = cls.Node.combine(l, r)
            heapq.heappush(node_heap, parent)

        root = heapq.heappop(node_heap)

        return root, leaves

    def __init__(self, root, leaves, data, key_func):
        self.root = root
        self.leaves = leaves
        self.data = data
        self.key_func = key_func

    def get_leaf(self, value):
        """
        Find our ending leaf from value that must exist by construction
        """
        return utils.first(self.leaves, lambda l: l.value == value)

    def encoding(self):
        """
        Construct binary encoding with Huffman tree
        """

        def translate_path(path, str=""):
            if len(path) < 2:
                return str
            child = path[-2]
            parent = path[-1]
            path = path[:-1]
            if parent.left == child:
                return translate_path(path, str=str + "1")
            elif parent.right == child:
                return translate_path(path, str=str + "0")
            raise RuntimeError("Invalid state")

        paths = [self.get_leaf(self.key_func(d)).path() for d in self.data]
        strs = [translate_path(path) for path in paths]
        return "".join(strs)

    class Node:
        GROUND = None
        ROOT = None

        @classmethod
        def singleton(cls, n):
            return cls(n, cls.GROUND, None, n.frequency).inherit(n)

        @classmethod
        def combine(cls, l, r):
            return cls(l, r, None, l.frequency + r.frequency) \
                .inherit(l) \
                .inherit(r)

        @classmethod
        def leaf(cls, value, frequency):
            return cls(cls.GROUND, cls.GROUND, value, frequency)

        def __init__(self, left, right, value, frequency):
            self.id = id(self)
            self.parent = self.ROOT
            self.left = left
            self.right = right
            self.value = value
            self.frequency = frequency

        def path(self, nodes=None):
            """
            From leaf, find the nodes back up to root
            """
            if nodes is None:
                nodes = []
            nodes.append(self)
            if self.is_root:
                return nodes
            else:
                return self.parent.path(nodes=nodes)

        def inherit(self, child):
            """
            Point the child back to the parent for ease in encoding
            """
            child.parent = self
            return self

        @property
        def is_leaf(self):
            return self.left is self.GROUND and self.right is self.GROUND

        @property
        def is_root(self):
            return self.parent is self.ROOT

        def __eq__(self, other):
            return self.id == other.id

        def __lt__(self, other):
            return self.frequency < other.frequency

        def __gt__(self, other):
            return self.frequency > other.frequency

        def __le__(self, other):
            return self < other or self == other

        def __ge__(self, other):
            return self > other or self == other
