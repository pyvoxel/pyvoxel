"""Voxel."""
from voxel.device import Device, cpu_device, get_array_module, get_device, to_device  # noqa: F401
from voxel.io import load, read, save, write  # noqa: F401
from voxel.io.dicom import DicomReader, DicomWriter  # noqa: F401
from voxel.io.format_io import ImageDataFormat  # noqa: F401
from voxel.io.http import HttpReader  # noqa: F401
from voxel.io.nifti import NiftiReader, NiftiWriter  # noqa: F401
from voxel.med_volume import MedicalVolume  # noqa: F401
from voxel.orientation import to_affine  # noqa: F401
from voxel.utils.collect_env import collect_env_info  # noqa

from .config import config  # noqa: F401

__all__ = [
    "MedicalVolume",
    "DicomReader",
    "DicomWriter",
    "HttpReader",
    "NiftiReader",
    "NiftiWriter",
    "read",
    "save",
    "load",
    "write",
    "ImageDataFormat",
    "Device",
    "get_device",
    "to_device",
    "cpu_device",
    "get_array_module",
    "to_affine",
    "collect_env_info",
]

# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.
__version__ = "0.0.2"
