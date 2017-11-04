import numpy as np

from hiccup.transform.base import Transform


class FourierTransform(Transform):
    def __init__(self, params):
        self._params = params
        self._quantization_table = params[0]

    @staticmethod
    def style():
        return "Fourier"

    def format_parameters(self):
        return "".join(self._params)

    def encode(self, img: np.ndarray):
        pass
