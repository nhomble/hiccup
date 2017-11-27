import numpy as np
import hiccup.iohelper as iohelper


class Compressor:
    @staticmethod
    def load(img_path: str):
        img = iohelper.open_raw_img(img_path)
        return Compressor(img)

    def __init__(self, img: np.ndarray, style: str, ):
        self._img = img
        self._style = style

    def shrink(self, output_path: str):
        pass
