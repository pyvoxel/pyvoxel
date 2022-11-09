"""NIfTI I/O.

This module contains NIfTI input/output helpers.
"""

import gzip
import os
from io import BytesIO
from typing import Collection, Union

import nibabel as nib

import voxel as vx
from voxel.io.format_io import DataReader, DataWriter, ImageDataFormat
from voxel.med_volume import MedicalVolume

__all__ = ["NiftiReader", "NiftiWriter"]


class NiftiReader(DataReader):
    """A class for reading NIfTI files.

    Attributes:
        data_format_code (ImageDataFormat): The supported image data format.
    """

    data_format_code = ImageDataFormat.nifti

    def load(
        self,
        path_or_bytes: Union[str, bytes, os.PathLike, BytesIO],
        mmap: bool = False,
        compressed: bool = False,
    ) -> MedicalVolume:
        """Load volume from NIfTI file path.

        A NIfTI file should only correspond to one volume.

        Args:
            path_or_bytes (Union[str, bytes, os.PathLike, BytesIO]): Path to NIfTI file or
                bytes of NIfTI file.
            mmap (bool): Whether to use memory mapping.
            compressed (bool): Whether to apply gzip decompression. This is only used if
                `path_or_bytes` is a bytes or BytesIO object.

        Returns:
            MedicalVolume: Loaded volume.

        Raises:
            FileNotFoundError: If `file_path` not found.
            ValueError: If `file_path` does not end in a supported NIfTI extension.
        """
        if isinstance(path_or_bytes, bytes):
            path_or_bytes = BytesIO(path_or_bytes)

        if isinstance(path_or_bytes, BytesIO):
            if compressed:
                path_or_bytes = BytesIO(gzip.decompress(path_or_bytes.getvalue()))

            nifti_version = _nifti_version(path_or_bytes)
            fh = nib.FileHolder(fileobj=path_or_bytes)
            if nifti_version == 1:
                nib_img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
            else:
                nib_img = nib.Nifti2Image.from_file_map({"header": fh, "image": fh})

        else:
            if not os.path.isfile(path_or_bytes):
                raise FileNotFoundError("{} not found".format(path_or_bytes))

            if not self.data_format_code.is_filetype(path_or_bytes):
                raise ValueError(
                    "{} must be a file with extension '.nii' or '.nii.gz'".format(path_or_bytes)
                )

            nib_img = nib.load(path_or_bytes)

        return MedicalVolume.from_nib(
            nib_img,
            affine_precision=vx.config.affine_precision,
            origin_precision=vx.config.affine_precision,
            mmap=mmap,
        )

    def __serializable_variables__(self) -> Collection[str]:
        return self.__dict__.keys()

    read = load  # pragma: no cover


class NiftiWriter(DataWriter):
    """A class for writing volumes in NIfTI format.

    Attributes:
        data_format_code (ImageDataFormat): The supported image data format.
    """

    data_format_code = ImageDataFormat.nifti

    def save(self, volume: MedicalVolume, file_path: str):
        """Save volume in NIfTI format,

        Args:
            volume (MedicalVolume): Volume to save.
            file_path (str): File path to NIfTI file.

        Raises:
            ValueError: If `file_path` does not end in a supported NIfTI extension.
        """
        if not self.data_format_code.is_filetype(file_path):
            raise ValueError(
                "{} must be a file with extension '.nii' or '.nii.gz'".format(file_path)
            )

        # Create dir if does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        nib_img = volume.to_nib()
        nib.save(nib_img, file_path)

    def __serializable_variables__(self) -> Collection[str]:
        return self.__dict__.keys()

    write = save  # pragma: no cover


def _nifti_version(buffer: BytesIO) -> int:
    """Get NIfTI version from buffer."""
    nii1_sizeof_hdr = 348
    nii2_sizeof_hdr = 540

    byte_data = buffer.read(4)
    sizeof_hdr = int.from_bytes(byte_data, byteorder="little")
    if sizeof_hdr == nii1_sizeof_hdr:
        return 1
    elif sizeof_hdr == nii2_sizeof_hdr:
        return 2
    else:
        sizeof_hdr = int.from_bytes(byte_data, byteorder="big")
        if sizeof_hdr == nii1_sizeof_hdr:
            return 1
        elif sizeof_hdr == nii2_sizeof_hdr:
            return 2

    raise ValueError(
        "This buffer is not a valid NIfTI file. Pass `compressed=True` if the buffer is gzipped."
    )
