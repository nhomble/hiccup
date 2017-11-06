import cv2
import numpy as np

from hiccup.transform.base import Transform
import hiccup.utils as utils
import hiccup.transform.quantization as quantization


class FourierTransform(Transform):
    def __init__(self, quantization_table_type="common", block_size=8):
        self._quantization_table = quantization_table_type
        self._block_size = block_size

        assert (self._quantization_table in quantization.all_tables)

    @staticmethod
    def style():
        return "Fourier"

    def format_parameters(self):
        return "{} {}".format(self._quantization_table, self._block_size)

    def dct_compress(self, luminosity):
        offseted = luminosity.astype(np.int8) - 128

        blocks = utils.split_matrix(offseted, self._block_size)
        transformed_blocks = [utils.dct2(block) for block in blocks]
        table = quantization.table[self._quantization_table]
        dividend = np.divide(transformed_blocks, table)
        quantized = np.round(dividend)
        result = utils.merge_blocks(quantized, luminosity.shape).astype(luminosity.dtype)
        return result

    def compress(self, rgb_img: np.ndarray):
        yrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
        splits = cv2.split(yrcb)

        splits[0] = self.dct_compress(splits[0])  # 0 is the luminosity
        compressed_img = cv2.merge(splits)
        return compressed_img
