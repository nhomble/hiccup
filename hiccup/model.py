import numpy as np
import enum

"""
Basically house the enumerations here and other broad modelling components
"""


class Compression(enum.Enum):
    """
    The broad image compression schemes implements in hiccup
    """
    JPEG = "JPEG"
    HIC = "HIC"  # basically wavelet


class Coefficient(enum.Enum):
    """
    Applicable in JPEG compression with the Discrete Fourier transform. The DC component represents the coefficient of
    the sinusoid with no frequency. The AC components refer to coefficients of sinusoids with different frequency.
    """
    DC = "DC"
    AC = "AC"


class QTables(enum.Enum):
    JPEG_LUMINANCE = "jpeg standard luminance"
    JPEG_CHROMINANCE = "jpeg standard chrominance"


class Wavelet(enum.Enum):
    DAUBECHIE = "db1"
    HAAR = "haar"
    COIF = "coif1"
    SYM = "sym2"


class CompressedImage:
    """
    Better for typing
    """

    @classmethod
    def from_dict(cls, d):
        assert len(d) == 3
        return cls(d["lum"], d["cr"], d["cb"])

    def __init__(self, lum, cr, cb):
        self.luminance_component = lum
        self.red_chrominance_component = cr
        self.blue_chrominance_component = cb

    @property
    def shape(self):
        return self.luminance_component.shape, self.red_chrominance_component.shape

    @property
    def as_dict(self):
        """
        Nice for iterating
        """
        return {
            "lum": self.luminance_component,
            "cr": self.red_chrominance_component,
            "cb": self.blue_chrominance_component
        }

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return \
            np.array_equiv(self.luminance_component, other.luminance_component) and np.array_equiv(
                self.red_chrominance_component, other.red_chrominance_component) and np.array_equiv(
                self.blue_chrominance_component, other.blue_chrominance_component)
