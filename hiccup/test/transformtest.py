import unittest
import functools

from hiccup.test.testhelper import *
import hiccup.transform as trans
import hiccup.model as model


class TransformTest(unittest.TestCase):
    def test_padding(self):
        one = np.ones((1, 1), dtype=np.int)
        padded = trans.pad_matrix(one, 3)
        self.assertEqual((3, 3), padded.shape)
        self.assertEqual(1, np.sum(padded))

    def test_padding_unecessary(self):
        zeros = np.zeros((2, 2))
        padded = trans.pad_matrix(zeros, 2)
        self.assertEqual(padded.shape, (2, 2))

    def test_block_nicely(self):
        square = np.array([
            [1, 2],
            [3, 4]
        ])
        splits = trans.split_matrix(square, 1)
        self.assertEqual(len(splits), 4)
        self.assertEqual((1, 1), splits[0].shape)
        self.assertEqual(1, splits[0][0][0])
        self.assertEqual(2, splits[1][0][0])
        self.assertEqual(3, splits[2][0][0])
        self.assertEqual(4, splits[3][0][0])

    def test_block_must_pad(self):
        square = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        splits = trans.split_matrix(square, 2)
        self.assertEqual(len(splits), 4)
        self.assertEqual(splits[0].shape, (2, 2))
        self.assertEqual(np.sum(splits[0]), 12)
        self.assertEqual(np.sum(splits[3]), 9)

    def test_dct_null_signal(self):
        zeros = np.zeros((5, 5))
        dct = trans.dct2(zeros)
        self.assertEqual(np.sum(dct), 0)

    def test_identity(self):
        random = np.random.randint(-512, 512, (8, 8))
        dct = trans.dct2(random)
        idct = trans.idct2(dct)
        scalar = idct[0][0] / random[0][0]
        opposite = random * scalar * -1
        zeros = np.add(opposite, idct)
        self.assertTrue(np.sum(zeros) < 1e-9)

    def test_merge_blocks(self):
        blocks = np.array([
            [[1, 2], [7, 8]], [[3, 4], [9, 10]], [[5, 6], [11, 12]],
            [[13, 14], [0, 0]], [[15, 16], [0, 0]], [[17, 18], [0, 0]]
        ])
        merged = trans.merge_blocks(blocks, (4, 6))
        self.assertTrue(np.array_equiv(np.array([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [0, 0, 0, 0, 0, 0]
        ]), merged))

    def test_split_merge(self):
        matrix = np.random.randint(0, 256, (8, 12))
        blocks = trans.split_matrix(matrix, 4)
        merged = trans.merge_blocks(blocks, matrix.shape)
        self.assertTrue(np.array_equiv(matrix, merged))

    def test_zigzag(self):
        matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        d = trans.zigzag(matrix)
        self.assertTrue(np.array_equiv([
            1, 4, 2, 3, 5, 7, 8, 6, 9
        ], d))

    def test_izigzag(self):
        arr = np.array([1, 4, 2, 3, 5, 7, 8, 6, 9])
        m = trans.izigzag(arr, (3, 3))
        self.assertTrue(np.array_equiv(np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]), m))

    def test_zigzag_inverse(self):
        matrix = np.random.randint(0, 256, (17, 19))
        ziggy = trans.zigzag(matrix)
        imat = trans.izigzag(ziggy, matrix.shape)
        self.assertTrue(np.array_equiv(matrix, imat))

    def test_up_sample(self):
        matrix = np.array([
            [2, 2],
            [2, 2],
        ]).astype(np.uint8)
        out = trans.up_sample(matrix, factor=2)
        self.assertTrue(np.array_equiv(out, np.array([
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]).astype(np.uint8)))

    def test_down_sample(self):
        matrix = np.array([
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]).astype(np.uint8)
        out = trans.down_sample(matrix, factor=2)
        self.assertTrue(np.array_equiv(out, np.array([
            [2, 2],
            [2, 2],
        ]).astype(np.uint8)))

    def test_dct_middle(self):
        img = np.full((120, 80), 128).astype(np.uint8)
        out = trans.dct_channel(img, model.QTables.JPEG_CHROMINANCE)
        self.assertTrue(np.sum(out) < 1e-10)

    def test_dct_ones(self):
        img = np.ones((120, 80)).astype(np.uint8)
        out = trans.dct_channel(img, model.QTables.JPEG_LUMINANCE)
        self.assertTrue(np.abs(np.sum(out)) > 0)

    def test_dct_null(self):
        img = np.zeros((120, 80)).astype(np.uint8)
        out = trans.dct_channel(img, model.QTables.JPEG_LUMINANCE)
        self.assertTrue(np.sum(out) < 1e-10)

    def test_wavelet_levels(self):
        mat = np.zeros((50, 50))
        out = trans.wavelet_split_resolutions(mat, model.Wavelet.DAUBECHIE, levels=2)
        [self.assertEqual((13, 13), out[i].shape) for i in range(4)]
        [self.assertEqual((25, 25), out[i].shape) for i in range(4, len(out))]
        self.assertEqual(0, functools.reduce(lambda x, y: x + np.sum(y), out, 0))

    def test_wavelet_inverse(self):
        mat = np.random.randint(0, 255, (64, 64))
        pyr = trans.wavelet_split_resolutions(mat, model.Wavelet.DAUBECHIE, levels=3)
        rec = trans.wavelet_merge_resolutions(pyr, model.Wavelet.DAUBECHIE)
        diff = np.sum(np.subtract(mat, rec))
        self.assertTrue(diff < 1e-8)

    def test_threshold(self):
        arr = np.array([1, 2, 3, 4, 5])
        out = trans.threshold(arr, 2, 100)
        self.assertTrue(np.array_equiv(out, np.array([100, 2, 3, 4, 5])))

    def test_threshold_by_quality(self):
        i = [
            np.array([range(50)]),
            np.array([range(50, 100)])
        ]
        out = trans.threshold_channel_by_quality(i, q_factor=.05)
        first = out[0]
        second = out[1]

        self.assertEqual(2, len(out))
        self.assertEqual(np.sum(first), 0)
        self.assertEqual(np.sum(second), sum(range(95, 100)))

    def test_dc_component(self):
        matrix = np.array([
            [1, 2],
            [3, 4]
        ])
        self.assertEqual(1, trans.dc_component(matrix))

    def test_ac_component(self):
        matrix = np.array([
            [1, 2],
            [3, 4]
        ])
        self.assertTrue(np.array_equiv(np.array([3, 2, 4]), trans.ac_components(matrix)))

    def test_subband_view(self):
        l = [
            np.array([
                [1, 2],
                [3, 4]
            ]),
            np.array([
                [5, 6],
                [7, 8]
            ]),
            np.array([
                [9, 10],
                [11, 12]
            ]),
            np.array([
                [13, 14],
                [15, 16]
            ])
        ]
        out = trans.subband_view(l)
        self.assertTrue(np.array_equiv(out[0], l[0]))
        self.assertEquals(len(out), 2)

        self.assertTrue(np.array_equiv(out[1][0], l[1]))
        self.assertTrue(np.array_equiv(out[1][1], l[2]))
        self.assertTrue(np.array_equiv(out[1][2], l[3]))

    def test_linearize_subband(self):
        subbands = [
            np.array([
                [1, 2],
                [3, 4]
            ]),
            (
                np.array([
                    [5, 6],
                    [7, 8]
                ]),
                np.array([
                    [9, 10],
                    [11, 12]
                ]),
                np.array([
                    [13, 14],
                    [15, 16]
                ])
            )
        ]
        out = trans.linearize_subband(subbands)
        l = [
            np.array([
                [1, 2],
                [3, 4]
            ]),
            np.array([
                [5, 6],
                [7, 8]
            ]),
            np.array([
                [9, 10],
                [11, 12]
            ]),
            np.array([
                [13, 14],
                [15, 16]
            ])
        ]
        for (i, ele) in enumerate(l):
            self.assertTrue(np.array_equiv(ele, out[i]))
