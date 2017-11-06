import unittest

import numpy as np
import cv2

from hiccup.test.testhelper import rgb_123
from hiccup.transform.fourier import FourierTransform

test_dir = "resources/"


class FourierTransformTest(unittest.TestCase):
    def test_run(self):
        """
        Really stupid test case just to get some basics out of the way and make the pipes flow.
        """
        img = rgb_123(80)
        transform = FourierTransform()
        out = transform.compress(img)
        splits = cv2.split(out)
        lumin = splits[0]
        self.assertEqual(lumin[0][0], np.max(lumin))

    def test_null(self):
        img = np.zeros((120, 80, 3)).astype(np.uint8)
        transform = FourierTransform()
        out = transform.compress(img)
        lumin = cv2.split(out)[0]
        self.assertEqual(0, np.sum(lumin))