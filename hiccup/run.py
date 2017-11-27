import numpy as np
import hiccup.iohelper as iohelper
import hiccup.hicimage as hic


def compress(path, output, style):
    if style is None:
        print("Style cannot be none")
        return
    c = Compressor.load(path)
    c.shrink(style, output)


def decompress(path, output):
    d = Decompressor.load(path)
    d.explode(output)


class Compressor:
    @staticmethod
    def load(img_path: str):
        img = iohelper.open_raw_img(img_path)
        return Compressor(img)

    def __init__(self, img: np.ndarray):
        self._img = img

    def shrink(self, style: str, output_path: str):
        pass


class Decompressor:
    @staticmethod
    def load(img_path: str):
        # read binary
        return Decompressor(None)

    def __init__(self, hic: hic.HicImage):
        self._hic = hic

    def explode(self, output_path):
        pass
