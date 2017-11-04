import numpy as np

from hiccup.transform.base import Transform


class WaveletTransform(Transform):
    def __init__(self, params):
        self._params = params

    @staticmethod
    def style():
        return "Wavelet"

    def format_parameters(self):
        return "".join(self._params)

    def encode(self, img: np.ndarray):
        pass
