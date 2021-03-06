import unittest
import numpy as np

import hiccup.utils as utils
import hiccup.settings as settings
import hiccup.model as model
import hiccup.transform as transform
import hiccup.codec as codec

settings.DEBUG = False


class CodecTest(unittest.TestCase):
    def test_differential_coding(self):
        matrix = np.array([[i + k for k in range(8)] for i in range(8)])
        blocks = transform.split_matrix(matrix, 4)
        dc = codec.differential_coding(blocks)
        self.assertEqual([0, 4, 0, 4], dc)

    def test_run_length(self):
        matrix = np.array([
            [99, -59, 0, 7, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [12, -2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        rl = codec.run_length_coding(transform.zigzag(matrix))
        expected = [{'zeros': 0, 'value': 99, 'bits': 7}, {'zeros': 1, 'value': -59, 'bits': 6},
                    {'zeros': 6, 'value': 7, 'bits': 3}, {'zeros': 4, 'value': 12, 'bits': 4},
                    {'zeros': 1, 'value': -2, 'bits': 2}, {'zeros': 0, 'value': 0, 'bits': 0}]
        self.assertEqual([codec.RunLength.from_dict(e) for e in expected], rl)

    def test_run_length_trivial(self):
        matrix = np.array([
            [1, 2],
            [3, 4]
        ])
        rl = codec.run_length_coding(transform.zigzag(matrix)[1:])
        self.assertEqual(rl, [
            codec.RunLength(3, 0),
            codec.RunLength(2, 0),
            codec.RunLength(4, 0)
        ])

    def test_rle_too_long(self):
        l = ([0] * 17) + [1]
        arr = np.array(l)
        out = codec.run_length_coding(arr, max_len=0xF)
        zeros = [s.length for s in out]
        self.assertEqual(zeros, [14, 2])

    def _symbol_0_0(self, has, arr):
        out = codec.run_length_coding(np.array(arr))
        last = out[-1]
        self.assertEqual((last.length == 0 and last.value == 0), has)

    def test_symbol_0_0(self):
        cases = [
            (False, [0, 0, 5]),
            (True, [0, 0, 5, 0, 0])
        ]
        for case in cases:
            self._symbol_0_0(*case)

    def test_jpeg_encode(self):
        compressed = model.CompressedImage(
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]])
        )
        hic = codec.jpeg_encode(compressed)
        payloads = hic.payloads
        self.assertEqual(len(payloads), 20)
        self.assertEqual(hic.hic_type, model.Compression.JPEG)
        self.assertEqual(payloads[0].payloads[0].numbers, (1, '1'))

    def test_jpeg_inverse(self):
        settings.JPEG_BLOCK_SIZE = 2
        compressed = model.CompressedImage(
            np.array([[1, 2, 11, 22], [3, 4, 33, 44]]),
            np.array([[5, 6, 55, 66], [7, 8, 77, 88]]),
            np.array([[9, 10, 99, 1010], [11, 12, 1111, 1212]])
        )
        hic = codec.jpeg_encode(compressed)
        inverse = codec.jpeg_decode(hic)
        self.assertEqual(compressed, inverse)

    def test_jpeg_inverse_inner_zeros(self):
        settings.JPEG_BLOCK_SIZE = 2
        compressed = model.CompressedImage(
            np.array([[1, 0, 0, 22], [3, 0, 33, 44]]),
            np.array([[5, 6, 0, 66], [7, 8, 77, 88]]),
            np.array([[9, 0, 99, 1010], [11, 12, 1111, 1212]])
        )
        hic = codec.jpeg_encode(compressed)
        inverse = codec.jpeg_decode(hic)
        self.assertEqual(compressed, inverse)

    def test_jpeg_inverse_end_zeros(self):
        settings.JPEG_BLOCK_SIZE = 2
        compressed = model.CompressedImage(
            np.array([[1, 2, 11, 0], [3, 0, 33, 0]]),
            np.array([[5, 6, 55, 0], [7, 8, 77, 0]]),
            np.array([[9, 10, 99, 0], [11, 12, 1111, 0]])
        )
        hic = codec.jpeg_encode(compressed)
        inverse = codec.jpeg_decode(hic)
        self.assertEqual(compressed, inverse)

    def test_zero_blocks(self):
        settings.JPEG_BLOCK_SIZE = 2
        compressed = model.CompressedImage(
            np.array([[0, 0, 11, 0], [0, 0, 0, 0]]),
            np.array([[0, 0, 55, 0], [0, 0, 0, 0]]),
            np.array([[9, 10, 99, 1010], [11, 12, 1111, 1212]])
        )
        hic = codec.jpeg_encode(compressed)
        inverse = codec.jpeg_decode(hic)
        self.assertEqual(compressed, inverse)

    def test_zero_block_rle(self):
        matrix = np.array([
            [0, 0, 11, 0],
            [0, 0, 0, 0]
        ])
        lin = transform.zigzag(matrix)
        out = codec.run_length_coding(lin)
        self.assertEqual(out, [
            codec.RunLength(11, 3),
            codec.RunLength(0, 0)
        ])

    def test_zero_block_reconstruct(self):
        matrix = np.array([
            [0, 0, 11, 0],
            [0, 0, 0, 0]
        ])
        lin = transform.zigzag(matrix)
        out = codec.run_length_coding(lin)
        invert = codec.decode_run_length(out, 8)
        self.assertEqual(lin, invert)

    def test_rle_segment(self):
        rle = codec.RunLength(10, 0)
        self.assertEqual(rle.segment, [10])

        rle = codec.RunLength(9, 3)
        self.assertEqual(rle.segment, [0, 0, 0, 9])

    def test_rle_max_len(self):
        arr = [0, 0, 0, 0, 0, 1]
        rle = codec.run_length_coding(np.array(arr))
        invert = codec.decode_run_length(rle, len(arr))
        self.assertEqual(arr, invert)

    def test_rle_consecutives(self):
        arr = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        rle = codec.run_length_coding(np.array(arr))
        invert = codec.decode_run_length(rle, len(arr))
        self.assertEqual(invert, arr)

    def test_rle_random(self):
        arr = [np.random.randint(-5, 5) for _ in range(10000)]
        rle = codec.run_length_coding(np.array(arr))
        invert = codec.decode_run_length(rle, 10000)
        self.assertEqual(invert, arr)

    def test_rle_break_plus_1(self):
        arr = [0, 0, 0, 0, 0, 1]
        rle = codec.run_length_coding(np.array(arr), max_len=4)
        invert = codec.decode_run_length(rle, len(arr))
        self.assertEqual(invert, arr)

    def test_rle_max_double(self):
        arr = [-1, 0, 0, 0, 0, 1, 2]
        rle = codec.run_length_coding(np.array(arr), max_len=4)
        invert = codec.decode_run_length(rle, len(arr))
        self.assertEqual(arr, invert)

    def test_accidental_combine(self):
        rle = [
            codec.RunLength(value=0, length=14),
            codec.RunLength(value=31, length=0)
        ]
        invert = codec.decode_run_length(rle, 15)
        rle_2 = codec.run_length_coding(invert, max_len=0xF)
        self.assertEqual(rle, rle_2)

    def test_subband_shapes(self):
        out = codec.wavelet_decoded_subbands_shapes((64, 64), (256, 256))
        self.assertEqual([(64, 64), (128, 128), (256, 256)], out)

    def test_pull_subbands(self):
        expected = [
            np.array(range(4)).reshape((2, 2)),
            np.array(range(4, 8)).reshape((2, 2)),
            np.array(range(8, 12)).reshape((2, 2)),
            np.array(range(12, 16)).reshape((2, 2)),

            np.array(range(16, 32)).reshape((4, 4)),
            np.array(range(32, 48)).reshape((4, 4)),
            np.array(range(48, 64)).reshape((4, 4))
        ]
        data = np.array(utils.flatten([transform.zigzag(m) for m in expected]))
        shapes = [(2, 2), (4, 4)]
        out = codec.wavelet_decode_pull_subbands(data, shapes)
        self.assertEqual(len(expected), len(out))
        z = zip(expected, out)
        for e in z:
            self.assertTrue(np.array_equal(e[0], e[1]))
