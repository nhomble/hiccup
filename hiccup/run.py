import numpy as np
import hiccup.iohelper as iohelper
import hiccup.hicimage as hic


def compress(path, output, style):
    if style is None:
        print("Style cannot be none")
        return
    img = iohelper.open_raw_img(path)


def decompress(path, output):
    pass
