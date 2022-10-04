"""Voxel."""
from voxel.utils.collect_env import collect_env_info  # noqa

from voxel.med_volume import MedicalVolume  # noqa: F401
from voxel.io.dicom import DicomReader, DicomWriter  # noqa: F401
from voxel.io.nifti import NiftiReader, NiftiWriter  # noqa: F401
from voxel.io import read, save, load, write  # noqa: F401
from voxel.io.format_io import ImageDataFormat  # noqa: F401
from voxel.device import Device, get_device, to_device, cpu_device, get_array_module  # noqa: F401
from voxel.orientation import to_affine  # noqa: F401

from .config import config  # noqa: F401

__all__ = [
    "MedicalVolume",
    "DicomReader",
    "DicomWriter",
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
__version__ = "0.0.1a1"
