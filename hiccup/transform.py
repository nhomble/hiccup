import abc


class Transform:
    """
    Specifies the compression algorithm - this also helps determine the
    """

    @staticmethod
    def from_string(s: str, params: list):
        s = s.lower()
        if s == FourierTransform.__name__.lower():
            return FourierTransform(params)
        elif s == WaveletTransform.__name__.lower():
            return WaveletTransform(params)
        else:
            raise RuntimeError(s + " is not a valid HIC transformation")

    @abc.abstractmethod
    def format_parameters(self):
        pass

    @property
    @abc.abstractmethod
    def style(self):
        pass


class FourierTransform(Transform):
    def __init__(self, params):
        self._params = params
        self._quantization_table = params[0]
        self._style = "Fourier"

    @property
    def style(self):
        return self._style

    def format_parameters(self):
        return "".join(self._params)


class WaveletTransform(Transform):
    def __init__(self, params):
        self._style = "Wavelet"
        self._params = params

    @property
    def style(self):
        return self._style

    def format_parameters(self):
        return "".join(self._params)
