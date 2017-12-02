from typing import List

import hiccup.model as model
import hiccup.settings as settings

"""
Wrap the representation of a HIC image to make it easier to write/retrieve from byte stream
"""


class Payload:
    pass


class BitString(Payload):
    def __init__(self, string: str):
        self.payload = string

    @property
    def byte_stream(self):
        pass


class PlainString(Payload):
    def __init__(self, string: str):
        self.payload = string

    @property
    def byte_stream(self):
        return self.payload.encode(encoding="ascii")


class HicImage:
    @classmethod
    def wavelet_image(cls, payloads):
        settings_list = [
            PlainString(settings.WAVELET),
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
