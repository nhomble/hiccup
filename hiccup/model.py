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