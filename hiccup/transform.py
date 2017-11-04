import abc
import numpy as np


class Transform:
    """
    Specifies the compression algorithm - this also helps determine the
    """

    @staticmethod
    def from_string(s: str, params: list):
        """
        Helpful map when we are reading the .hic file
        """
        s = s.lower()
        if s == FourierTransform.style().lower():
            return FourierTransform(params)
        elif s == WaveletTransform.style().lower():
            return WaveletTransform(params)
        else:
            raise RuntimeError(s + " is not a valid HIC transformation")

    @abc.abstractmethod
    def format_parameters(self):
        """
        When we save our parameters into the file header, handle formatting our transformation parameters here
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def style():
        """
        The image header should state the style of transform Fourier or Wavelet or blah
        """
        pass

    @abc.abstractmethod
    def encode(self, img: np.ndarray):
        """
        Perform image compression which should end in a character string for the Huffman Coding to deal with
        """
        pass


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
