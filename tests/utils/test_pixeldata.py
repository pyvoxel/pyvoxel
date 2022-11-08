import unittest

import numpy as np
import pydicom

from voxel.utils.pixel_data import apply_rescale, apply_window, invert, pixel_dtype, pixel_range

from .. import util as ututils


class TestPixelData(unittest.TestCase):
    def test_pixel_dtype(self):
        # Test integer types
        for bits in [8, 16, 32, 64]:
            for sign in [0, 1]:
                metadata = {"BitsAllocated": bits, "PixelRepresentation": sign, "PixelData": b""}
                headers = ututils.build_dummy_headers(1, metadata)

                prefix = "int" if sign == 1 else "uint"
                assert pixel_dtype(headers.flat[0]) == np.dtype(f"{prefix}{bits}")

        # Test floats, should not use volume.dtype
        volume = np.random.randint(0, 255, (10, 10, 1), dtype=np.int64)
        metadata = {"FloatPixelData": b""}
        headers = ututils.build_dummy_headers(volume.shape[-1], metadata)
        assert pixel_dtype(headers.flat[0]) == np.float32

        # Test undeterminable types
        with self.assertRaises(ValueError):
            metadata = {"PixelData": b""}
            headers = ututils.build_dummy_headers(1, metadata)
            pixel_dtype(headers.flat[0])

        with self.assertRaises(ValueError):
            metadata = {"BitsAllocated": 7, "PixelRepresentation": 0, "PixelData": b""}
            headers = ututils.build_dummy_headers(1, metadata)
            pixel_dtype(headers.flat[0])

    def test_pixel_range(self):
        # Test integer types
        metadata = {"BitsAllocated": 8, "PixelRepresentation": 1}
        headers = ututils.build_dummy_headers(1, metadata)
        assert pixel_range(headers.flat[0]) == (-128, 127)

        metadata = {"BitsAllocated": 16, "BitsStored": 8, "PixelRepresentation": 0}
        headers = ututils.build_dummy_headers(1, metadata)
        assert pixel_range(headers.flat[0]) == (0, 255)

        # Test floats
        metadata = {"FloatPixelData": b""}
        headers = ututils.build_dummy_headers(1, metadata)
        with self.assertRaises(ValueError):
            pixel_range(headers.flat[0])

    def test_apply_window(self):
        volume = np.arange(9, dtype=np.float32)

        # Test windowing
        windowed = apply_window(volume, 4, 8, output_range=(0, 8), mode="linear_exact")
        assert np.allclose(windowed, np.arange(9))

        # Test linear w/ pixel range, this is similar to the default behaviour of pydicom
        ds = pydicom.Dataset()
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.WindowCenter = 4
        ds.WindowWidth = 20

        output_range = pixel_range(ds)
        windowed = apply_window(volume, 4, 20, output_range=output_range)
        assert np.allclose(
            windowed,
            np.array(
                [
                    -754.84210526,
                    -539.31578947,
                    -323.78947368,
                    -108.26315789,
                    107.26315789,
                    322.78947368,
                    538.31578947,
                    753.84210526,
                    969.36842105,
                ]
            ),
        )

    def test_apply_rescale(self):
        volume = np.arange(9, dtype=np.float32)

        # Test rescaling
        rescaled = apply_rescale(volume, 2, 4)
        assert np.allclose(rescaled, np.arange(9) * 2 + 4)

        # Test inplace
        assert np.shares_memory(apply_rescale(volume, 2, 4, inplace=True), volume)

    def test_invert(self):
        volume = np.arange(9, dtype=np.float32)

        # Test rescaling
        inverted = invert(volume, (0, 8))
        assert np.allclose(inverted, np.arange(9)[::-1])

        # Test inplace
        assert np.shares_memory(invert(volume, (-100, 100), inplace=True), volume)


if __name__ == "__main__":
    unittest.main()
