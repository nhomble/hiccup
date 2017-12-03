import abc
from typing import List

import hiccup.model as model
import hiccup.settings as settings

"""
Wrap the representation of a HIC image to make it easier to write/retrieve from byte stream
"""


class Payload:
    @property
    @abc.abstractmethod
    def byte_stream(self):
        pass


class IntegerStringP(Payload):
    def __init__(self, numbers: List[int]):
        self.numbers = numbers

    @property
    def byte_stream(self):
        return bytearray()


class BitStringP(Payload):
    def __init__(self, string: str):
        self.payload = string

    @property
    def byte_stream(self):
        return bytearray()


class PlainStringP(Payload):
    def __init__(self, string: str):
        self.payload = string

    @property
    def byte_stream(self):
        return self.payload.encode(encoding="ascii")


class PayloadStringP(Payload):
    def __init__(self, payloads: List[Payload]):
        self.payloads = payloads

    @property
    def byte_stream(self):
        return bytearray()


class HicImage:
    @classmethod
    def wavelet_image(cls, payloads):
        settings_list = [
            PlainStringP(settings.WAVELET),
        ]

        return cls(model.Compression.HIC, settings_list, payloads)

    @classmethod
    def jpeg_image(cls, payloads):
        settings_list = [

        ]
        return cls(model.Compression.JPEG, settings_list, payloads)

    def __init__(self, hic_type, settings: List[Payload], payloads: List[Payload]):
        self.hic_type = hic_type
        self.settings = settings
        self.payloads = payloads

    @property
    def bytes_stream(self):
        return bytearray(b"\x10\x32\xFF")

    def write_file(self, path):
        with open(path, 'wb') as f:
            f.write(self.bytes_stream)

    def apply_settings(self):
        pass

    def payload(self):
        # fix
        return [p.payload for p in self.payloads]
