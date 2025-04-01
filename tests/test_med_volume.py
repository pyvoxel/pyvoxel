import os
import shutil
import unittest

import h5py
import nibabel as nib
import nibabel.testing as nib_testing
import numpy as np
import pydicom
import pydicom.data as pydd
import SimpleITK as sitk

from voxel.device import Device
from voxel.io.dicom import DicomReader
from voxel.io.format_io import ImageDataFormat
from voxel.io.nifti import NiftiReader, NiftiWriter
from voxel.med_volume import MedicalVolume
from voxel.utils import env

from . import util as ututils


class TestMedicalVolume(unittest.TestCase):
    _AFFINE = np.asarray(
        [
            [0.0, 0.0, 0.8, -171.41],
            [0.0, -0.3125, 0.0, 96.0154],
            [-0.3125, 0.0, 0.0, 47.0233],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )  # ('SI', 'AP', 'LR')

    _TEMP_PATH = os.path.join(ututils.TEMP_PATH, __name__)

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls._TEMP_PATH, exist_ok=True)

    @classmethod
    def tearDownCls(cls):
        if os.path.isdir(cls._TEMP_PATH):
            shutil.rmtree(cls._TEMP_PATH)

    def test_reformat(self):
        mv = MedicalVolume(np.random.rand(10, 20, 30), self._AFFINE)
        new_orientation = tuple(x[::-1] for x in mv.orientation[::-1])

        mv2 = mv.reformat(new_orientation)
        assert mv2.orientation == new_orientation
        assert id(mv2) != id(mv)
        assert np.shares_memory(mv2._volume, mv._volume)

        mv2 = mv.reformat(new_orientation, inplace=True)
        assert mv2.orientation == new_orientation
        assert id(mv2) == id(mv)
        assert np.shares_memory(mv2._volume, mv._volume)

        mv2 = mv.reformat(mv.orientation)
        assert id(mv2) != id(mv)
        assert np.shares_memory(mv2._volume, mv._volume)

        mv2 = mv.reformat(mv.orientation, inplace=True)
        assert id(mv2) == id(mv)
        assert np.shares_memory(mv2._volume, mv._volume)

        mv2 = mv.reformat(new_orientation).reformat(mv.orientation)
        assert mv2.is_identical(mv)

        mv2 = mv.reformat(*new_orientation).reformat(*mv.orientation)
        assert mv2.is_identical(mv)

        with self.assertRaises(ValueError):
            mv.reformat(new_orientation, True)

    def test_reformat_as(self):
        mv = MedicalVolume(np.random.rand(10, 20, 30), self._AFFINE)
        mv2 = MedicalVolume(np.random.rand(10, 20, 30), self._AFFINE[:, (0, 2, 1, 3)])
        mv = mv.reformat_as(mv2)
        assert mv.orientation == mv2.orientation

    def test_reformat_header(self):
        volume = np.random.rand(10, 20, 30, 40)
        headers = ututils.build_dummy_headers(volume.shape[2:])
        mv = MedicalVolume(volume, self._AFFINE, headers=headers)
        new_orientation = tuple(x[::-1] for x in mv.orientation[::-1])

        mv2 = mv.reformat(new_orientation)
        assert mv2._headers.shape == (30, 1, 1, 40)

        mv2 = mv.clone()
        mv2.reformat(new_orientation, inplace=True)
        assert mv2._headers.shape == (30, 1, 1, 40)

        volume = np.random.rand(10, 20, 30, 40)
        headers = ututils.build_dummy_headers((volume.shape[2], 1))
        mv = MedicalVolume(volume, self._AFFINE, headers=headers)
        new_orientation = tuple(x[::-1] for x in mv.orientation[::-1])

        mv2 = mv.reformat(new_orientation)
        assert mv2._headers.shape == (30, 1, 1, 1)

    def test_metadata(self):
        field, field_val = "EchoTime", 4.0

        volume = np.random.rand(10, 20, 30, 40)
        headers = ututils.build_dummy_headers(volume.shape[2:], {field: field_val})
        mv_no_headers = MedicalVolume(volume, self._AFFINE)
        mv = MedicalVolume(volume, self._AFFINE, headers=headers)

        assert mv_no_headers.headers() is None
        assert mv_no_headers.headers(flatten=True) is None

        with self.assertRaises((KeyError, ValueError)):
            mv.get_metadata("foobar")
        assert mv.get_metadata("foobar", default=0) == 0

        echo_time = mv.get_metadata(field)
        assert echo_time == field_val

        new_val = 5.0
        mv2 = mv.clone(headers=True)
        mv2.set_metadata(field, new_val)
        assert mv.get_metadata(field, type(field_val)) == field_val
        assert mv2.get_metadata(field, type(new_val)) == new_val
        for h in mv2.headers(flatten=True):
            assert h[field].value == new_val

        new_val = 6.0
        mv2 = mv.clone(headers=True)
        mv2[..., 1].set_metadata(field, new_val)
        assert mv2[..., 0].get_metadata(field) == field_val
        assert mv2[..., 1].get_metadata(field) == new_val
        headers = mv2.headers()
        for h in headers[..., 0].flatten():
            assert h[field].value == field_val
        for h in headers[..., 1].flatten():
            assert h[field].value == new_val

        # Set metadata when volume has no headers.
        mv_nh = MedicalVolume(volume, self._AFFINE, headers=None)
        with self.assertRaises(ValueError):
            mv_nh.set_metadata("EchoTime", 40.0)
        with self.assertWarns(UserWarning):
            mv_nh.set_metadata("EchoTime", 40.0, force=True)
        assert mv_nh._headers.shape == (1,) * len(mv_nh.shape)
        assert mv_nh.get_metadata("EchoTime") == 40.0
        assert mv_nh[:1, :2, :3]._headers.shape == (1,) * len(mv_nh.shape)

    def test_apply_rescale(self):
        metadata = {"RescaleSlope": "2.5", "RescaleIntercept": "1.0"}
        volume = np.arange(9, dtype=np.uint16).reshape(3, 3, 1)
        headers = ututils.build_dummy_headers(volume.shape[2:], metadata)
        mv_no_headers = MedicalVolume(volume, self._AFFINE)
        mv_incorrect_headers = MedicalVolume(volume, self._AFFINE, headers={})
        mv = MedicalVolume(volume, self._AFFINE, headers=headers)

        mv_unchanged = mv_no_headers.apply_rescale()
        assert mv_unchanged.is_identical(mv_no_headers)

        mv_unchanged = mv_incorrect_headers.apply_rescale()
        assert mv_unchanged.is_identical(mv_incorrect_headers)

        assert np.allclose(mv.apply_rescale().volume, volume * 2.5 + 1.0)
        assert np.allclose(mv.apply_rescale(3, 1).volume, volume * 3.0 + 1.0)

        # test sync
        mv2 = mv.apply_rescale(dtype="single", sync=True)
        assert mv.get_metadata("RescaleSlope", dtype=float) == 2.5
        assert mv.get_metadata("RescaleIntercept", dtype=float) == 1.0

        assert mv2.get_metadata("RescaleSlope", default=None) is None
        assert mv2.get_metadata("RescaleIntercept", default=None) is None
        assert mv2.dtype == np.float32

    def test_apply_modality_lut(self):
        ds = pydicom.Dataset()
        lut = pydicom.Dataset()
        lut_data_little_endian = bytearray([10, 0, 11, 0, 12, 0, 13, 0])
        lut_data_big_endian = bytearray([0, 10, 0, 11, 0, 12, 0, 13])

        ds.PixelRepresentation = 1
        ds.ModalityLUTSequence = pydicom.Sequence()
        ds.ModalityLUTSequence.append(lut)

        lut.LUTDescriptor = [4, 2, 16]  # entries, first mapped, bits
        lut.LUTExplanation = "Test"
        ds.RescaleIntercept = "0"
        ds.RescaleSlope = "1"

        volume = np.arange(9, dtype=np.uint16).reshape(3, 3, 1)
        mv_no_headers = MedicalVolume(volume, self._AFFINE)
        mv_incorrect_headers = MedicalVolume(volume, self._AFFINE, headers=[pydicom.Dataset()])
        mv = MedicalVolume(volume, self._AFFINE, headers=[ds])

        mv_unchanged = mv_no_headers.apply_modality_lut()
        assert mv_unchanged.is_identical(mv_no_headers)

        mv_unchanged = mv_incorrect_headers.apply_modality_lut()
        assert mv_unchanged.is_identical(mv_incorrect_headers)

        # test modality lut
        correct = [10, 10, 10, 11, 12, 13, 13, 13, 13]
        correct_u8 = np.array(correct, dtype=np.uint8).reshape(3, 3, 1)

        # test big endian
        lut["LUTData"] = pydicom.DataElement("LUTData", "OW", lut_data_big_endian)
        mv_big_endian = mv.apply_modality_lut()
        assert np.allclose(mv_big_endian._volume, correct_u8)

        # test sync and inplace
        lut["LUTData"] = pydicom.DataElement("LUTData", "OW", lut_data_little_endian)
        mv_inplace = mv.apply_modality_lut(inplace=True, sync=True)
        assert mv_inplace is mv

    def test_apply_window(self):
        volume = np.arange(9, dtype=np.uint8)
        ds = pydicom.Dataset()
        ds.WindowCenter = "4.5"
        ds.WindowWidth = "4.0"
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.VOILUTFunction = "LINEAR"
        mv_no_headers = MedicalVolume(volume, self._AFFINE)
        mv_incorrect_headers = MedicalVolume(volume, self._AFFINE, headers={})
        mv = MedicalVolume(volume, self._AFFINE, headers=[ds])

        mv_unchanged = mv_no_headers.apply_window()
        assert mv_unchanged.is_identical(mv_no_headers)

        mv_unchanged = mv_incorrect_headers.apply_window()
        assert mv_unchanged.is_identical(mv_incorrect_headers)

        # test VOILUTFunction linear
        correct_linear = np.array([0, 0, 0, 42.5, 127.5, 212.5, 255, 255, 255])
        assert np.allclose(mv.apply_window(output_range=(0, 255)).volume, correct_linear)

        # test VOILUTFunction exact
        correct_exact = np.array([0.0, 0.0, 0.0, 42.5, 127.5, 212.5, 255.0, 255.0, 255.0])
        mv.set_metadata("VOILUTFunction", "LINEAR_EXACT", force=True)
        assert np.allclose(mv.apply_window(output_range=(0, 255)).volume, correct_exact)

    def test_apply_voi_lut(self):
        ds = pydicom.Dataset()
        lut1 = pydicom.Dataset()
        lut2 = pydicom.Dataset()
        lut_data_little_endian = bytearray([10, 11, 12, 13])
        lut_data_big_endian = bytearray([0, 10, 0, 11, 0, 12, 0, 13])

        ds.PixelRepresentation = 1
        ds.VOILUTSequence = pydicom.Sequence()
        ds.VOILUTSequence.extend([lut1, lut2])

        # VOILUTSequence can contain multiple LUTs
        lut1.LUTDescriptor = [4, 2, 16]  # entries, first mapped, bits
        lut1.LUTExplanation = "Test1"
        lut2.LUTDescriptor = [4, 2, 16]  # entries, first mapped, bits
        lut2.LUTExplanation = "Test2"

        ds.RescaleIntercept = "0"
        ds.RescaleSlope = "1"

        volume = np.arange(9, dtype=np.uint16).reshape(3, 3, 1)
        mv_no_headers = MedicalVolume(volume, self._AFFINE)
        mv_incorrect_headers = MedicalVolume(volume, self._AFFINE, headers=[pydicom.Dataset()])
        mv = MedicalVolume(volume, self._AFFINE, headers=[ds])

        mv_unchanged = mv_no_headers.apply_voi_lut()
        assert mv_unchanged.is_identical(mv_no_headers)

        mv_unchanged = mv_incorrect_headers.apply_voi_lut()
        assert mv_unchanged.is_identical(mv_incorrect_headers)

        # test modality lut1
        correct = [10, 10, 10, 11, 12, 13, 13, 13, 13]
        correct_u8 = np.array(correct, dtype=np.uint8).reshape(3, 3, 1)

        # test big endian
        lut1["LUTData"] = pydicom.DataElement("LUTData", "OW", lut_data_big_endian)
        mv_big_endian = mv.apply_voi_lut()
        assert np.allclose(mv_big_endian._volume, correct_u8)

        # test modality lut2
        lut2["LUTData"] = pydicom.DataElement("LUTData", "US", lut_data_little_endian)
        mv_lut2 = mv.apply_voi_lut(index=1)
        assert np.allclose(mv_lut2._volume, correct_u8)

    def test_clone(self):
        mv = MedicalVolume(np.random.rand(10, 20, 30), self._AFFINE)
        mv2 = mv.clone()
        assert mv.is_identical(mv2)  # expected identical volumes

        mv = MedicalVolume(
            np.random.rand(10, 20, 30),
            self._AFFINE,
            headers=ututils.build_dummy_headers((1, 1, 30)),
        )
        mv2 = mv.clone(headers=False)
        assert mv.is_identical(mv2)  # expected identical volumes
        assert id(mv.headers(flatten=True)[0]) == id(
            mv2.headers(flatten=True)[0]
        ), "headers not cloned, expected same memory address"

        mv3 = mv.clone(headers=True)
        assert mv.is_identical(mv3)  # expected identical volumes
        assert id(mv.headers(flatten=True)[0]) != id(
            mv3.headers(flatten=True)[0]
        ), "headers cloned, expected different memory address"

    def test_to_nib(self):
        arr = np.random.rand(10, 20, 30)
        mv = MedicalVolume(arr, self._AFFINE)
        nib_img = nib.Nifti1Image(arr, mv.affine)

        nib_from_mv = mv.to_nib()
        assert np.all(nib_from_mv.get_fdata() == nib_img.get_fdata())
        assert np.all(nib_from_mv.affine == nib_img.affine)

    def test_to_sitk(self):
        mv = MedicalVolume(np.random.rand(10, 20, 30), self._AFFINE)
        filepath = os.path.join(ututils.TEMP_PATH, "med_vol_to_sitk.nii.gz")
        NiftiWriter().save(mv, filepath)

        expected = sitk.ReadImage(filepath)

        nr = NiftiReader()
        mv = nr.load(filepath)
        img = mv.to_sitk()

        assert np.allclose(sitk.GetArrayViewFromImage(img), sitk.GetArrayViewFromImage(expected))
        assert img.GetSize() == mv.shape
        assert np.allclose(img.GetOrigin(), expected.GetOrigin())
        assert img.GetSpacing() == img.GetSpacing()
        assert img.GetDirection() == expected.GetDirection()

        mv = MedicalVolume(np.zeros((10, 20, 1, 3)), affine=self._AFFINE)
        img = mv.to_sitk(vdim=-1)
        assert np.all(sitk.GetArrayViewFromImage(img) == 0)
        assert img.GetSize() == (10, 20, 1)

        filepath = pydd.get_testdata_file("MR_small.dcm")
        dr = DicomReader(group_by=None)
        mv = dr.load(filepath)
        mv2 = MedicalVolume.from_sitk(
            mv.to_sitk(transpose_inplane=True), copy=True, transpose_inplane=True
        )
        assert mv2.is_identical(mv)

    @unittest.skipIf(not ututils.is_data_available("qdess"), "unittest data is not available")
    def test_to_from_sitk_dicom_convention(self):
        dp = ututils.get_scan_dirpath("qdess")
        dirpath = ututils.get_read_paths(dp, ImageDataFormat.dicom)[0]

        dr = DicomReader()
        dr = DicomReader(group_by=None, sort_by="InstanceNumber", ignore_ext=False)
        mv: MedicalVolume = dr.load(dirpath)[0]

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dirpath)
        reader.SetFileNames(dicom_names)
        sitk_image = reader.Execute()

        sitk_from_mv = mv.to_sitk(transpose_inplane=True)
        img, expected = sitk_from_mv, sitk_image
        assert np.allclose(sitk.GetArrayViewFromImage(img), sitk.GetArrayViewFromImage(expected))
        assert img.GetSize() == mv.shape
        assert np.allclose(img.GetOrigin(), expected.GetOrigin())
        assert img.GetSpacing() == img.GetSpacing()
        assert img.GetDirection() == expected.GetDirection()

        mv_from_sitk = MedicalVolume.from_sitk(sitk_image, copy=True, transpose_inplane=True)
        assert mv_from_sitk.is_identical(mv)

        mv2 = MedicalVolume.from_sitk(sitk_from_mv, copy=True, transpose_inplane=True)
        assert mv2.is_identical(mv)

    def test_from_nib(self):
        filepath = os.path.join(nib_testing.data_path, "example4d.nii.gz")
        nib_img = nib.load(filepath)

        mv = MedicalVolume.from_nib(nib_img)
        assert np.all(mv.affine == nib_img.affine)
        assert np.all(mv.A == nib_img.get_fdata())

        precision = 4
        mv2 = MedicalVolume.from_nib(
            nib_img, affine_precision=precision, origin_precision=precision
        )
        assert np.allclose(mv2.affine, nib_img.affine, atol=10 ** (-precision))
        assert np.all(mv2.A == nib_img.get_fdata())

    def test_from_sitk(self):
        mv = MedicalVolume(np.random.rand(10, 20, 30), self._AFFINE)
        filepath = os.path.join(ututils.TEMP_PATH, "med_vol_from_sitk.nii.gz")
        NiftiWriter().save(mv, filepath)

        nr = NiftiReader()
        expected = nr.load(filepath)

        img = sitk.ReadImage(filepath)
        mv = MedicalVolume.from_sitk(img)

        assert np.allclose(mv.affine, expected.affine)
        assert mv.shape == expected.shape
        assert np.all(mv.volume == expected.volume)

        img = sitk.Image([10, 20, 1], sitk.sitkVectorFloat32, 3)
        mv = MedicalVolume.from_sitk(img)
        assert np.all(mv.volume == 0)
        assert mv.shape == (10, 20, 1, 3)

    def test_math(self):
        mv1 = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        mv2 = MedicalVolume(2 * np.ones((10, 20, 30)), self._AFFINE)

        out = mv1 + mv2
        assert np.all(mv1._volume == 1)
        assert np.all(mv2._volume == 2)
        assert np.all(out._volume == 3)

        out = mv1 - mv2
        assert np.all(mv1._volume == 1)
        assert np.all(mv2._volume == 2)
        assert np.all(out._volume == -1)

        out = mv1 * mv2
        assert np.all(mv1._volume == 1)
        assert np.all(mv2._volume == 2)
        assert np.all(out._volume == 2)

        out = mv1 / mv2
        assert np.all(mv1._volume == 1)
        assert np.all(mv2._volume == 2)
        assert np.all(out._volume == 0.5)

        out = mv1 // mv2
        assert np.all(mv1._volume == 1)
        assert np.all(mv2._volume == 2)
        assert np.all(out._volume == 0)

        out = mv1**mv2
        assert np.all(mv1._volume == 1)
        assert np.all(mv2._volume == 2)
        assert np.all(out._volume == 1)

        out = mv1.clone()
        out += mv2
        assert np.all(out._volume == 3)

        out = mv1.clone()
        out -= mv2
        assert np.all(out._volume == -1)

        out = mv1.clone()
        out *= mv2
        assert np.all(out._volume == 2)

        out = mv1.clone()
        out /= mv2
        assert np.all(out._volume == 0.5)

        out = mv1.clone()
        out //= mv2
        assert np.all(out._volume == 0)

        out = mv1.clone()
        out **= mv2
        assert np.all(out._volume == 1)

        mv3 = mv1.clone().reformat(mv1.orientation[::-1])
        with self.assertRaises(ValueError):
            mv3 + mv2

    def test_shape_attributes(self):
        vol = np.ones((10, 20, 30, 2))
        mv = MedicalVolume(vol, np.eye(4))
        assert mv.shape == vol.shape
        assert mv.ndim == vol.ndim

    def test_hdf5(self):
        shape = (10, 20, 30)
        volume = np.reshape(list(range(np.product(shape))), shape)
        hdf5_file = os.path.join(self._TEMP_PATH, "unittest.h5")

        with h5py.File(hdf5_file, "w") as f:
            f.create_dataset("volume", data=volume)
        f = h5py.File(hdf5_file, "r")

        mv = MedicalVolume(f["volume"], np.eye(4))
        assert mv.device == Device("cpu")
        assert mv.dtype == f["volume"].dtype

        mv2 = mv[:, :, :1]
        assert np.all(mv2.volume == volume[:, :, :1])
        assert mv2.device == Device("cpu")
        assert mv2.dtype == volume.dtype

    def test_comparison(self):
        mv1 = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        mv2 = MedicalVolume(2 * np.ones((10, 20, 30)), self._AFFINE)

        assert np.all((mv1 == mv1.clone()).volume)
        assert np.all((mv1 != mv2).volume)
        assert np.all((mv1 < mv2).volume)
        assert np.all((mv1 <= mv2).volume)
        assert np.all((mv2 > mv1).volume)
        assert np.all((mv2 >= mv1).volume)

    def test_slice(self):
        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        with self.assertRaises(IndexError):
            mv[4]
        mv_slice = mv[4:5]
        assert mv_slice.shape == (1, 20, 30)

        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        mv[:5, ...] = 2
        assert np.all(mv._volume[:5, ...] == 2) & np.all(mv._volume[5:, ...] == 1)
        assert np.all(mv[:5, ...].volume == 2)

        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        mv2 = mv[:5, ...].clone()
        mv2 += 2
        mv[:5, ...] = mv2
        assert np.all(mv._volume[:5, ...] == 3) & np.all(mv._volume[5:, ...] == 1)
        assert np.all(mv[:5, ...].volume == 3)

    def test_slice_with_headers(self):
        vol = np.stack([np.ones((10, 20, 30)), 2 * np.ones((10, 20, 30))], axis=-1)
        headers = np.stack(
            [
                ututils.build_dummy_headers(vol.shape[2], {"EchoTime": 2}),
                ututils.build_dummy_headers(vol.shape[2], {"EchoTime": 10}),
            ],
            axis=-1,
        )
        mv = MedicalVolume(vol, self._AFFINE, headers=headers)

        mv2 = mv[..., 0]
        assert mv2._headers.shape == (1, 1, 30)
        for h in mv2.headers(flatten=True):
            assert h["EchoTime"].value == 2

        mv2 = mv[..., 1]
        assert mv2._headers.shape == (1, 1, 30)
        for h in mv2.headers(flatten=True):
            assert h["EchoTime"].value == 10

        mv2 = mv[:10, :5, 8:10, :1]
        assert mv2._headers.shape == (1, 1, 2, 1)

        mv2 = mv[:10]
        assert mv2._headers.shape == (1, 1, 30, 2)
        mv2 = mv[:, :10]
        assert mv2._headers.shape == (1, 1, 30, 2)

        mv2 = mv[..., 0:1]
        assert mv2._headers.shape == (1, 1, 30, 1)

        vol = np.stack([np.ones((10, 20, 30)), 2 * np.ones((10, 20, 30))], axis=-1)
        headers = ututils.build_dummy_headers(vol.shape[2], {"EchoTime": 2})[..., np.newaxis]
        mv = MedicalVolume(vol, self._AFFINE, headers=headers)
        mv1 = mv[..., 0]
        mv2 = mv[..., 1]
        assert mv1._headers.shape == (1, 1, 30)
        assert mv2._headers.shape == (1, 1, 30)
        for h1, h2 in zip(mv1.headers(flatten=True), mv2.headers(flatten=True)):
            assert id(h1) == id(h2)

    def test_4d(self):
        vol = np.stack([np.ones((10, 20, 30)), 2 * np.ones((10, 20, 30))], axis=-1)
        mv = MedicalVolume(vol, self._AFFINE)
        assert mv.orientation == ("SI", "AP", "LR")
        assert mv.shape == (10, 20, 30, 2)

        assert np.all(mv[..., 0].volume == 1)
        assert np.all(mv[..., 1].volume == 2)

        ornt = ("AP", "IS", "RL")
        mv2 = mv.reformat(ornt)
        assert mv2.orientation == ornt
        assert mv2.shape == (20, 10, 30, 2)

        mv2 = mv.reformat(ornt).reformat(mv.orientation)
        assert mv2.is_identical(mv)

        fp = os.path.join(self._TEMP_PATH, "test_4d.nii.gz")
        mv.save_volume(fp)
        mv2 = NiftiReader().load(fp)
        assert mv2.is_identical(mv)

    def test_device(self):
        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        assert mv.to(Device(-1)) == mv
        assert mv.cpu() == mv

    @unittest.skipIf(not env.cupy_available(), "cupy not available")
    def test_device_gpu(self):
        import cupy as cp

        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        mv_gpu = mv.to(Device(0))

        assert mv_gpu.device == Device(0)
        assert isinstance(mv_gpu.volume, cp.ndarray)
        assert isinstance(mv_gpu.affine, np.ndarray)

        assert mv_gpu.is_same_dimensions(mv)

        assert cp.all((mv_gpu + 1).volume == 2)
        assert cp.all((mv_gpu - 1).volume == 0)
        assert cp.all((mv_gpu * 2).volume == 2)
        assert cp.all((mv_gpu / 2).volume == 0.5)
        assert cp.all((mv_gpu > 0).volume)
        assert cp.all((mv_gpu >= 0).volume)
        assert cp.all((mv_gpu < 2).volume)
        assert cp.all((mv_gpu <= 2).volume)

        ornt = tuple(x[::-1] for x in mv_gpu.orientation[::-1])
        mv2 = mv_gpu.reformat(ornt)
        assert mv2.orientation == ornt

        mv_cpu = mv_gpu.cpu()
        assert mv_cpu.device == Device(-1)
        assert mv_cpu.is_identical(mv)

        with self.assertRaises(RuntimeError):
            mv_gpu.save_volume(os.path.join(self._TEMP_PATH, "test_device.nii.gz"))

    def test_array_cpu(self):
        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)

        data = np.asarray(mv)
        assert np.shares_memory(data, mv.volume)

    @unittest.skipIf(not env.cupy_available(), "cupy not available")
    def test_array_gpu(self):
        import cupy as cp

        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        mv_gpu = mv.to(Device(0))
        data = cp.asarray(mv_gpu)
        assert cp.shares_memory(data, mv_gpu.volume)

    def test_dtype(self):
        vol = np.ones((10, 20, 30))
        mv = MedicalVolume(vol, self._AFFINE)

        assert mv.volume.dtype == vol.dtype

        mv2 = mv.astype("int32")
        assert id(mv) == id(mv2)
        assert mv2.volume.dtype == np.int32

    def test_repr(self):
        vol = np.ones((10, 20, 30))
        mv = MedicalVolume(vol, self._AFFINE)

        assert mv.__repr__() is not None

    def test_set_volume(self):
        vol = np.ones((10, 20, 30))
        mv = MedicalVolume(vol, self._AFFINE)

        mv.volume += 2
        assert np.all(mv.volume == 3)

    @ututils.requires_packages("torch")
    def test_to_torch(self):
        import torch

        vol = np.ones((10, 20, 30))
        mv = MedicalVolume(vol, self._AFFINE)

        tensor = mv.to_torch()
        assert torch.all(tensor == torch.from_numpy(vol))
        assert tensor.shape == mv.shape

        tensor = mv.to_torch(requires_grad=True, contiguous=True)
        assert tensor.is_contiguous()
        assert tensor.requires_grad
        assert torch.all(tensor == torch.from_numpy(vol))
        assert tensor.shape == mv.shape

        vol = np.ones((10, 20, 30), np.complex)
        mv = MedicalVolume(vol, self._AFFINE)

        tensor = mv.to_torch()
        assert tensor.dtype == torch.complex128
        assert tensor.shape == mv.shape

        tensor = mv.to_torch(view_as_real=True)
        assert tensor.shape == mv.shape + (2,)

    @ututils.requires_packages("torch")
    def test_from_torch(self):
        import torch

        tensor = torch.ones(10, 20, 30)
        mv = MedicalVolume.from_torch(tensor, self._AFFINE)
        assert np.all(tensor.numpy() == mv.A)

        tensor = torch.ones(10, 20, 30)
        mv = MedicalVolume.from_torch(tensor, torch.from_numpy(self._AFFINE))
        assert np.all(tensor.numpy() == mv.A)
        assert isinstance(mv.affine, np.ndarray)

        tensor = torch.ones(10, 20, 30, dtype=torch.complex64)
        mv = MedicalVolume.from_torch(tensor, self._AFFINE)
        assert mv.dtype == np.complex64

        tensor = torch.ones(10, 20, 30, dtype=torch.complex128)
        mv = MedicalVolume.from_torch(tensor, self._AFFINE)
        assert mv.dtype == np.complex128

        tensor = torch.ones(10, 20, 30, 2, dtype=torch.float32)
        mv = MedicalVolume.from_torch(tensor, self._AFFINE, to_complex=True)
        assert mv.dtype == np.complex64
        assert mv.shape == tensor.shape[:3]

        tensor = torch.ones(10, 20, 30, 2, dtype=torch.float64)
        mv = MedicalVolume.from_torch(tensor, self._AFFINE, to_complex=True)
        assert mv.dtype == np.complex128
        assert mv.shape == tensor.shape[:3]

        tensor = torch.ones(10, 20, dtype=torch.float64)
        with self.assertRaises(ValueError):
            mv = MedicalVolume.from_torch(tensor, self._AFFINE)

        tensor = torch.ones(10, 20, 2, dtype=torch.float64)
        with self.assertRaises(ValueError):
            mv = MedicalVolume.from_torch(tensor, self._AFFINE, to_complex=True)

        tensor = torch.ones(10, 20, 30, 3, dtype=torch.float64)
        with self.assertRaises(ValueError):
            mv = MedicalVolume.from_torch(tensor, self._AFFINE, to_complex=True)

    @ututils.requires_packages("zarr")
    def test_to_zarr(self):
        import zarr

        vol = np.ones((10, 20, 30))
        mv = MedicalVolume(vol, self._AFFINE)
        arr = mv.to_zarr(read_only=True)

        assert np.array_equal(arr, vol)
        with self.assertRaises(zarr.errors.ReadOnlyError):
            arr[0, 0, 0] = 2

        aff = np.eye(4) * 2
        mv_aff = MedicalVolume(vol, aff)
        arr = mv_aff.to_zarr(affine_attr="affine")
        assert np.array_equal(np.array(arr.attrs["affine"]), aff)

        arr = mv_aff.to_zarr()
        with self.assertRaises(KeyError):
            arr.attrs["affine"]

        filepath = pydd.get_testdata_file("CT_small.dcm")
        dr = DicomReader()
        mv_hdr = dr.load(filepath)
        arr = mv_hdr.to_zarr(headers_attr="headers")
        assert "headers" in arr.attrs

    @ututils.requires_packages("zarr")
    def test_from_zarr(self):
        import zarr

        filepath = pydd.get_testdata_file("CT_small.dcm")
        dr = DicomReader()
        mv_hdr = dr.load(filepath)

        store = zarr.MemoryStore()
        arr = mv_hdr.to_zarr(store=store, affine_attr="affine", headers_attr="headers")
        assert arr.read_only

        # load without attributes
        mv = MedicalVolume.from_zarr(store=store)
        assert np.array_equal(mv.affine, np.eye(4))
        assert mv.headers() is None

        # load with attributes
        mv = MedicalVolume.from_zarr(store=store, affine_attr="affine", headers_attr="headers")
        assert np.array_equal(mv.affine, mv_hdr.affine)
        assert mv.headers() == mv_hdr.headers()

    def test_indexing(self):
        # Index medical volume with another medical volume.
        mv = MedicalVolume(np.ones((10, 20, 30)), np.eye(4))
        mv_index = MedicalVolume(np.random.rand(10, 20, 30) > 0.5, np.eye(4))
        mv[mv_index] = 0

        assert np.all(mv.A[mv_index.A] == 0)

    def test_mmap_numpy(self):
        shape = (5, 6, 7)

        # Memmap should directly write to the file in r+ mode.
        out_path = os.path.join(self._TEMP_PATH, "test.npy")
        np.save(out_path, np.ones(shape))

        arr = np.load(out_path, mmap_mode="r+")
        mv = MedicalVolume(arr, np.eye(4))
        assert mv.is_mmap

        mv += 1
        assert np.all(mv.A == 2)
        arr2 = np.load(out_path)
        assert np.all(arr2 == 2)

        # Memmap should be read-only.
        out_path = os.path.join(self._TEMP_PATH, "test2.npy")
        np.save(out_path, np.ones(shape))

        arr = np.load(out_path, mmap_mode="r")
        mv = MedicalVolume(arr, np.eye(4))
        assert mv.is_mmap

        orientation = mv.orientation
        mv = mv.reformat(orientation[::-1])
        assert mv.orientation == orientation[::-1]
        assert mv.shape == shape[::-1]

        mv2 = mv[1:3]
        assert np.all(mv2.A == mv.A[1:3])

        with self.assertRaises(ValueError):
            mv += 1

    def test_mmap_nifti(self):
        out_path = os.path.join(self._TEMP_PATH, "test.nii")
        mv = MedicalVolume(np.ones((10, 20, 30)), np.eye(4))
        mv.save_volume(out_path)

        vol = nib.load(out_path)
        assert isinstance(vol.dataobj.__array__(), np.memmap)
        mv2 = MedicalVolume.from_nib(vol, mmap=True)
        assert isinstance(mv2.A, np.memmap) and mv2.A.mode == "c"
        assert mv2.is_mmap
        assert mv2.is_identical(mv)

        mv2 += 1
        assert not mv2.is_mmap
        assert np.all(mv2.A == 2)

        mv2 = MedicalVolume.from_nib(vol, mmap=True)
        assert mv2.is_mmap
        mv2[1:4] = 0
        assert not mv2.is_mmap


if __name__ == "__main__":
    unittest.main()
