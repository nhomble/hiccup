import numpy as np
import cv2
from hiccup.transform import wavelet_split_resolutions
from hiccup.model import Wavelet
rgb = cv2.imread("../../resources/Lenna.png")
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
imgs = wavelet_split_resolutions(gray, Wavelet.DAUBECHIE, levels=2)
for (i, im) in enumerate(imgs):
    cv2.imshow("%d" % i, np.round(im).astype(np.uint8))
cv2.waitKey()