import numpy as np
from hiccup.hicimage import HicImage
import abc


class Transform:
    """
    Specifies the compression algorithm - this also helps determine the
    """

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
    def encode(self, img: np.ndarray) -> HicImage:
        """
        Perform image compression with the chosen transformation
        """
        pass

    def compress(self, data: np.ndarray):
        raise NotImplemented("need to use huffman coding here")
