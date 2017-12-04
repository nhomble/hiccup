import datetime
import cv2
import matplotlib.pyplot as plt
import hiccup.compression as compression
import hiccup.codec as codec

MNT = "../../resources/mountain.jpg"
original = cv2.imread(MNT)
original = cv2.resize(original, (original.shape[0] // 4, original.shape[1] // 4))


def jpg_bench(i):
    def f():
        compressed = compression.jpeg_compression(i)
        codec.jpeg_encode(compressed)

    return f


def hic_bench(i):
    def f():
        compressed = compression.wavelet_compression(i)
        codec.wavelet_encode(compressed)

    return f


def my_time(f):
    now = datetime.datetime.utcnow()
    f()
    end = datetime.datetime.utcnow()
    return (end - now).microseconds


import hiccup.settings

hiccup.settings.DEBUG = False
img = original

res = []
for _ in range(70):
    print(_)
    res.append((
        img.shape[0], my_time(jpg_bench(img)), my_time(hic_bench(img))))
    img = cv2.resize(img, (img.shape[0] * 98 // 100, img.shape[1] * 98 // 100))
res = res[14:] # clip the noise
jpeg = [t[1] for t in res]
hic = [t[2] for t in res]
import hiccup.utils as utils

x = [utils.size((t[0], t[0])) for t in res]

plt.plot(jpeg, 'r--')
plt.plot(hic, 'b--')
plt.show()
