import os
import unittest
import zipfile
from io import BytesIO

import nibabel.testing as nib_testing
import requests_mock as rm
from pydicom.data import get_testdata_file

import voxel as vx
from voxel.io.dicom import DicomReader
from voxel.io.format_io import ImageDataFormat
from voxel.io.http import HttpReader
from voxel.io.nifti import NiftiReader


class TestHttpIO(unittest.TestCase):
    nr = NiftiReader()
    dr = DicomReader()

    def test_load_nifti(self):
        """Test with nibabel sample data."""
        base_url = "https://test.com"

        # NIfTI1 file
        fname = "example4d.nii.gz"
        filepath = os.path.join(nib_testing.data_path, fname)
        mv_nifti1 = self.nr.load(filepath)

        with open(filepath, "rb") as f:
            file_nifti1 = f.read()

        with rm.Mocker() as m:
            m.get(f"{base_url}/{fname}", content=file_nifti1)
            with HttpReader() as hr:
                mv = hr.load(f"{base_url}/{fname}")
                assert mv.is_identical(mv_nifti1)

        # NIfTI2 file
        fname = "example_nifti2.nii.gz"
        filepath = os.path.join(nib_testing.data_path, fname)
        mv_nifti2 = self.nr.load(filepath)

        with open(filepath, "rb") as f:
            file_nifti2 = f.read()

        with rm.Mocker() as m:
            m.get(f"{base_url}/{fname}", content=file_nifti2)
            m.get(f"{base_url}/test", content=file_nifti2)

            hr = HttpReader()
            mv = hr.load(f"{base_url}/{fname}")
            assert mv.is_identical(mv_nifti2)

            # Ambiguous file extension
            with self.assertRaises(AttributeError):
                hr.load(f"{base_url}/test")

            with self.assertRaises(ValueError):
                hr.load(f"{base_url}/test", data_format=ImageDataFormat.nifti)

            mv = hr.load(f"{base_url}/test", data_format="nifti", compressed=True)
            assert mv.is_identical(mv_nifti2)
            hr.close()

        # Nifti file with .nii extension
        fname = "functional.nii"
        filepath = os.path.join(nib_testing.data_path, fname)
        mv_nifti = self.nr.load(filepath)

        with open(filepath, "rb") as f:
            file_nifti = f.read()

        with rm.Mocker() as m:
            m.get(f"{base_url}/{fname}", content=file_nifti)
            mv = vx.load(f"{base_url}/{fname}")
            assert mv.is_identical(mv_nifti)

    def test_load_dicom(self):
        """Test with DICOM sample data.

        TODO: add tests for multipart/related
        """
        base_url = "https://test.com"

        # Single DICOM file
        fname = "CT_small.dcm"
        filepath = get_testdata_file(fname)
        mv_dicom = self.dr.load(filepath)

        with open(filepath, "rb") as f:
            file_dicom = f.read()

        with rm.Mocker() as m:
            m.get(f"{base_url}", content=file_dicom)
            m.get(f"{base_url}/{fname}", content=file_dicom)

            with HttpReader() as hr:
                mv = hr.load(f"{base_url}")
                assert mv.is_identical(mv_dicom)

            mv = vx.load(f"{base_url}/{fname}")
            assert mv.is_identical(mv_dicom)

            with self.assertRaises(ValueError):
                vx.load(f"{base_url}/{fname}", data_format="nifti")

        # Zipped DICOM file(s)
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "a") as zf:
            zf.writestr(fname, file_dicom)

        with rm.Mocker() as m:
            headers = {"Content-Type": "application/zip"}
            m.get(f"{base_url}/file.zip", content=zip_buffer.getvalue(), headers=headers)

            mv = vx.load(f"{base_url}/file.zip")
            assert mv.is_identical(mv_dicom)

            m.get(f"{base_url}/bad.zip", content=zip_buffer.getvalue())

            with self.assertRaises(ValueError):
                vx.load(f"{base_url}/bad.zip")


if __name__ == "__main__":
    unittest.main()
