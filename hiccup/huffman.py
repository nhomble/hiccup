import heapq
import functools

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
        utils.debug_msg("Groupby has found %d groups" % len(groups))
        leaves = [cls.Node.leaf(t[0], len(t[1])) for t in groups.items()]
        root, leaves = cls._construct(leaves)
        utils.debug_msg("Finished constructing huffman")
        return cls(root, leaves, data, key_func)

    @classmethod
    def construct_from_leaves(cls, segments, key_func=utils.identity):
        leaves = [cls.Node.leaf(*s) for s in segments]
        root, leaves = cls._construct(leaves)
        return cls(root, leaves, None, key_func)

    @classmethod
    def construct_from_coding(cls, segments, key_func=utils.identity):
        """
        When we store in binary it's going to be (value, symbol), from the symbol we can place our value back in the
        correct place of the tree
        """
        d = dict([(t[1], t[0]) for t in segments])

        levels = len(sorted(segments, key=lambda t: len(t[1]))[-1][1])
        root = cls.Node(None, None, None, None)
        leaves = []

        # the side effects are real
        def h_trav(r, depth, s):
            if s in d:
                # I am really a leaf
                leaves.append(r)
                r.value = d[s]
                return
            if depth != levels:
                r.left = cls.Node.leaf(None, None)
                h_trav(r.left, depth + 1, s + "1")

                r.right = cls.Node.leaf(None, None)
                h_trav(r.right, depth + 1, s + "0")
                r.mass_adopt()

        h_trav(root, 0, '')
        return cls(root, leaves, None, key_func)

    @classmethod
    def _construct(cls, leaves):
        """
        From the groups, construct the Huffman tree and return root for reference
        """
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

        # speed things up
        self._cache = {
            self.get_leaf: {}
        }

    def blow_cache(self):
        """
        Control memory somewhat
        """
        self._cache = {
            self.get_leaf: {}
        }

    def in_cache(self, k, v):
        return v in self._cache[k]

    def get_cache(self, k, v):
        return self._cache[k][v]

    def cache(self, k, v, to_cache):
        self._cache[k][v] = to_cache

    def get_leaf(self, value):
        """
        Find our ending leaf from value that must exist by construction
        """
        if self.in_cache(self.get_leaf, value):
            return self.get_cache(self.get_leaf, value)
        out = utils.first(self.leaves, lambda l: l.value == value)
        self.cache(self.get_leaf, value, out)
        return out

    def translate_path(self, path, str=""):
        if len(path) < 2:
            return str
        child = path[-2]
        parent = path[-1]
        path = path[:-1]
        if parent.left == child:
            return self.translate_path(path, str=str + "1")
        elif parent.right == child:
            return self.translate_path(path, str=str + "0")
        raise RuntimeError("Invalid state")

    def encode_data(self, data=None):
        """
        Construct binary encoding with Huffman tree
        """
        if data is None:
            data = self.data

        paths = [self.get_leaf(self.key_func(d)).path() for d in data]
        utils.debug_msg("Got %d paths" % len(paths))
        strs = [self.translate_path(path) for path in paths]
        utils.debug_msg("Got the strs")
        return "".join(strs)

    def encode_table(self):
        s = [n.encoding for n in self.leaves]
        s = [(t[0], self.translate_path(t[1])) for t in s]
        return s

    def decode_data(self, str):
        bits = list(str)

        def reduce(agg, ele):
            if agg[-1]["node"] == self.Node.GROUND:
                agg.append({"node": self.root})
            if ele == "1":
                next_ele = agg[-1]["node"].left
            elif ele == "0":
                next_ele = agg[-1]["node"].right
            else:
                raise RuntimeError("Illegal state")

            if next_ele.is_leaf:
                agg[-1]["value"] = next_ele.value
                agg[-1]["node"] = next_ele.left  # either is fine, it's just an indicator
                return agg
            else:
                agg[-1]["node"] = next_ele
                return agg

        values = functools.reduce(reduce, bits, [{
            "node": self.root
        }])
        # chop last one because the reduction anticipates more values
        return [v["value"] for v in values if "value" in v]

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

        def mass_adopt(self):
            """
            During reconstruction just adopt both
            """
            self.inherit(self.left).inherit(self.right)

        @property
        def encoding(self):
            return self.value, self.path()

        @property
        def is_leaf(self):
            return self.left is self.GROUND and self.right is self.GROUND

        @property
        def is_root(self):
            return self.parent is self.ROOT

        @property
        def depth(self):
            if self.is_leaf:
                return 1
            else:
                return 1 + max(self.left.depth, self.right.depth)

        def __eq__(self, other):
            return type(self) == type(other) and self.id == other.id

        def __lt__(self, other):
            return self.frequency < other.frequency

        def __gt__(self, other):
            return self.frequency > other.frequency

        def __le__(self, other):
            return self < other or self == other

        def __ge__(self, other):
            return self > other or self == other
