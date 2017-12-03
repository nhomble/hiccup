import cv2
from hiccup.compression import wavelet_compression, wavelet_decompression, jpeg_decompression, jpeg_compression
from hiccup.transform import low_pass

rgb = cv2.imread("../../resources/Lenna.png")
rgb = low_pass(rgb, k=(33,33))

out = wavelet_compression(rgb)
jout = jpeg_compression(rgb)
cv2.imshow("before", rgb)
res = wavelet_decompression(out)
jres = jpeg_decompression(jout)
cv2.imshow("after wave", res)
cv2.imshow("after jpeg", jres)
cv2.waitKey()
