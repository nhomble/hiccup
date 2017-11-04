from hiccup.transform.fourier import FourierTransform
from hiccup.transform.wavelet import WaveletTransform


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
