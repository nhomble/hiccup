import hiccup.model as model

"""
Ok.. it wasn't meant to be like this, but I don't have the time or inclination to properly encapsulate these things. So
here is a global config file that can get modified at runtime when the user is belching.

I suppose once I expose all of the settings it will be easier to organize.
"""

DEBUG = True
WAVELET = model.Wavelet.DAUBECHIE
WAVELET_QUALITY_FACTOR = 1
WAVELET_SUBBAND_QUANTIZATION_MULTIPLIER = 32