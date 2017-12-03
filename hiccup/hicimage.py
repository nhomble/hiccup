import abc
from typing import List
import pickle

import hiccup.utils as utils
import hiccup.model as model
import hiccup.settings as settings

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
        i = int(b, 2)
        return utils.bin_string(i)

    def __init__(self, string: str):
        self.payload = string

    @property
    def byte_stream(self):
        return bin(int(self.payload, 2))


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

    @property
    def byte_stream(self):
        return pickle.dumps({
            "type": self.t,
            "data": [p.byte_stream for p in self.payloads]
        })


class HicImage(Payload):
    @classmethod
    def from_bytes(cls, b):
        payloads = pickle.loads(b)
        t = PlainStringP.from_bytes(payloads[0]).payload
        if t == model.Compression.JPEG.value:
            pass
        elif t == model.Compression.HIC.value:
            pass
        else:
            raise RuntimeError("Illegal type: " + t)

    @property
    def byte_stream(self):
        full = self.settings + self.payload
        return pickle.dumps([p.byte_stream for p in full])

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

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __init__(self, hic_type: model.Compression, settings: List[Payload], payloads: List[Payload]):
        self.hic_type = hic_type
        self.settings = settings
        self._payloads = payloads

    def write_file(self, path):
        utils.debug_msg("Writing HIC file to: " + path)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @property
    def payload(self):
        return self._payloads
