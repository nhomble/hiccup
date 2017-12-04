import abc
from typing import List
import pickle

import hiccup.utils as utils
import hiccup.model as model
import hiccup.iohelper as io

"""
Wrap the representation of a HIC image to make it easier to write/retrieve from byte stream
"""


class Payload:
    """
    Generic abstract class to define a piece of data in the HIC image
    """

    @classmethod
    @abc.abstractmethod
    def from_bytes(cls, b):
        pass

    @property
    @abc.abstractmethod
    def byte_stream(self):
        pass


class TupP(Payload):
    @classmethod
    def from_bytes(cls, b):
        tup = pickle.loads(b)
        return cls(tup[0], tup[1])

    """
    Tuple, probably shape or Huffman node
    """

    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2

    def __eq__(self, other):
        return type(self) == type(other) and self.n1 == other.n1 and self.n2 == other.n2

    @property
    def numbers(self):
        return self.n1, self.n2

    @property
    def byte_stream(self):
        return pickle.dumps((self.n1, self.n2))


class BitStringP(Payload):
    """
    Bit string, probably the encoded huffman data
    """

    @classmethod
    def from_bytes(cls, b):
        return cls(io.padded_bytes_2_bs(b))

    def __init__(self, string: str):
        self.payload = string

    def __eq__(self, other):
        return type(self) == type(other) and self.payload == other.payload

    @property
    def byte_stream(self):
        return io.padded_bs_2_bytes(self.payload)


class PlainStringP(Payload):
    ENCODING = "ascii"
    """
    Just a string (for readability), probably the image type
    """

    @classmethod
    def from_bytes(cls, b):
        return cls(b.decode(cls.ENCODING))

    def __init__(self, string: str):
        self.payload = string

    def __eq__(self, other):
        return type(self) == type(other) and self.payload == other.payload

    @property
    def byte_stream(self):
        return self.payload.encode(encoding=self.ENCODING)


class PayloadStringP(Payload):
    """
    String of payloads, we'll want these consecutive cause they are grouped. Probably a group
    of Huffman nodes
    """

    @classmethod
    def from_bytes(cls, b):
        d = pickle.loads(b)
        p = [d["type"].from_bytes(b) for b in d["data"]]
        return cls(d["type"], p)

    def __init__(self, t, payloads: List[Payload]):
        self.t = t
        self.payloads = payloads

    def __eq__(self, other):
        return type(self) == type(other) and self.t == other.t and self.payloads == other.payloads

    @property
    def byte_stream(self):
        return pickle.dumps({
            "type": self.t,
            "data": [p.byte_stream for p in self.payloads]
        })


class HicImage:
    @classmethod
    def from_bytes(cls, raw_data):
        t = model.Compression(PlainStringP.from_bytes(raw_data[0]).payload)
        if t == model.Compression.JPEG:
            # huffs
            # tree per data per channel == 9
            huffs = [PayloadStringP.from_bytes(b) for b in raw_data[1:10]]
            # data
            data = [BitStringP.from_bytes(b) for b in raw_data[10:19]]
            # shapes
            shapes = [TupP.from_bytes(b) for b in raw_data[19:21]]
            return cls.jpeg_image(huffs + data + shapes)
        else:
            assert t == model.Compression.HIC
            huffs = [PayloadStringP.from_bytes(b) for b in raw_data[1:7]]
            data = [BitStringP.from_bytes(b) for b in raw_data[7:13]]
            shapes = [TupP.from_bytes(b) for b in raw_data[13:15]]
            return cls.wavelet_image(huffs + data + shapes)

    @classmethod
    def wavelet_image(cls, payloads):
        settings_list = [
            PlainStringP(model.Compression.HIC.value),
        ]

        return cls(model.Compression.HIC, settings_list, payloads)

    @classmethod
    def jpeg_image(cls, payloads):
        settings_list = [
            PlainStringP(model.Compression.JPEG.value),
        ]
        return cls(model.Compression.JPEG, settings_list, payloads)

    @classmethod
    def from_file(cls, path):
        with open(path, 'rb') as f:
            raw_data = pickle.load(f)
        assert raw_data is not None
        return cls.from_bytes(raw_data)

    def __init__(self, hic_type: model.Compression, settings: List[Payload], payloads: List[Payload]):
        self.hic_type = hic_type
        self.settings = settings
        self._payloads = payloads

    def write_file(self, path):
        utils.debug_msg("Writing HIC file to: " + path)
        with open(path, 'wb') as f:
            pickle.dump(self.byte_stream(), f)

    @property
    def payloads(self):
        return self._payloads

    def byte_stream(self):
        data = self.settings + self.payloads
        data = [p.byte_stream for p in data]
        return data
