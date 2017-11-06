from typing import List

from hiccup.transform.fourier import FourierTransform
from hiccup.transform.wavelet import WaveletTransform


def from_string(s: str, params: List[str]):
    """
    Helpful map when we are reading the .hic file
    """
    s = s.lower()
    if s == FourierTransform.style().lower():
        quantization_table = params[0]
        block_size = int(params[1])
        return FourierTransform(quantization_table_type=quantization_table, block_size=block_size)
    elif s == WaveletTransform.style().lower():
        return WaveletTransform(params)
    else:
        raise RuntimeError(s + " is not a valid HIC transformation")
