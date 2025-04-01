"""The medical image object.

This module defines :class:`MedicalVolume`, which is a wrapper for nD volumes.
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from mmap import mmap
from numbers import Number
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import pydicom
from nibabel.spatialimages import SpatialFirstSlicer as _SpatialFirstSlicerNib
from numpy.lib.mixins import NDArrayOperatorsMixin
from packaging import version
from pydicom import Dataset

import voxel as vx
import voxel.orientation as stdo
from voxel.device import Device, cpu_device, get_array_module, get_device, to_device
from voxel.utils import env
from voxel.utils.pixel_data import apply_rescale, apply_window, invert, invert_window

if env.sitk_available():
    import SimpleITK as sitk
if env.cupy_available():
    import cupy as cp
if env.package_available("h5py"):
    import h5py

if TYPE_CHECKING:
    from voxel.io.format_io import ImageDataFormat


__all__ = ["MedicalVolume"]


# PyTorch version introducing complex tensor support.
_TORCH_COMPLEX_SUPPORT_VERSION = version.Version("1.5.0")


class MedicalVolume(NDArrayOperatorsMixin):
    """The class for medical images.

    Medical volumes use ndarrays to represent medical data. However, unlike standard ndarrays,
    these volumes have inherent spatial metadata, such as pixel/voxel spacing, global coordinates,
    rotation information, all of which can be characterized by an affine matrix following the
    RAS+ coordinate system. The code below creates a random 300x300x40 medical volume with
    scanner origin ``(0, 0, 0)`` and voxel spacing of ``(1,1,1)``:

    >>> mv = MedicalVolume(np.random.rand(300, 300, 40), np.eye(4))

    Medical volumes can also store header information that accompanies pixel data
    (e.g. DICOM headers). These headers are used to expose metadata, which can be fetched
    and set using :meth:`get_metadata()` and :meth:`set_metadata()`, respectively. Headers are
    also auto-aligned, which means that headers will be aligned with the slice(s) of data from
    which they originated, which makes Python slicing feasible. Currently, medical volumes
    support DICOM headers using ``pydicom`` when loaded with :class:`voxel.DicomReader`.

    >>> mv.get_metadata("EchoTime")  # Returns EchoTime
    >>> mv.set_metadata("EchoTime", 10.0)  # Sets EchoTime to 10.0

    Standard math and boolean operations are supported with other ``MedicalVolume`` objects,
    numpy arrays (following standard broadcasting), and scalars. Boolean operations are performed
    elementwise, resulting in a volume with shape as ``self.volume.shape``.
    If performing operations between ``MedicalVolume`` objects, both objects must have
    the same shape and affine matrix (spacing, direction, and origin). Header information
    is not deep copied when performing these operations to reduce computational and memory
    overhead. The affine matrix (``self.affine``) is copied as it is lightweight and
    often modified.

    2D images are also supported when viewed trivial 3D volumes with shape ``(H, W, 1)``:

    >>> mv = MedicalVolume(np.random.rand(10,20,1), np.eye(4))

    Many operations are in-place and modify the instance directly (e.g. `reformat(inplace=True)`).
    To allow chaining operations, operations that are in-place return ``self``.

    >>> mv2 = mv.reformat(ornt, inplace=True)
    >>> id(mv2) == id(mv)
    True

    Medical volumes can interface with the gpu using the :mod:`cupy` library.
    Volumes can be moved between devices (see :class:`Device`) using the ``.to()`` method.
    Only the volume data will be moved to the gpu. Headers and affine matrix will remain on
    the cpu. The following code moves a MedicalVolume to gpu 0 and back to the cpu:

    >>> from voxel import Device
    >>> mv = MedicalVolume(np.random.rand((10,20,30)), np.eye(4))
    >>> mv_gpu = mv.to(Device(0))
    >>> mv_cpu = mv.cpu()

    Note, moving data across devices results in a full copy. Above, ``mv_cpu.volume`` and
    ``mv.volume`` do not share memory. Saving volumes and converting to other images
    (e.g. ``SimpleITK.Image``) are only supported for cpu volumes. Volumes can also only
    be compared when on the same device. For example, both commands below will raise a
    RuntimeError:

    >>> mv_gpu == mv_cpu
    >>> mv_gpu.is_identical(mv_cpu)

    While CuPy requires the current device be set using ``cp.cuda.Device(X).use()`` or inside
    the ``with`` context, ``MedicalVolume`` automatically sets the appropriate context
    for performing operations. This means the CuPy current device need to be the same as the
    ``MedicalVolume`` object. For example, the following still works:

    >>> cp.cuda.Device(0).use()
    >>> mv_gpu = MedicalVolume(cp.ones((3,3,3)), np.eye(4))
    >>> cp.cuda.Device(1).use()
    >>> mv_gpu *= 2

    MedicalVolumes also have a limited NumPy/CuPy-compatible interface.
    Standard numpy/cupy functions that preserve array shapes can be performed
    on MedicalVolume objects:

    >>> log_arr = np.log(mv)
    >>> type(log_arr)
    <class 'voxel.MedicalVolume'>
    >>> exp_arr_gpu = cp.exp(mv_gpu)
    >>> type(exp_arr_gpu)
    <class 'voxel.MedicalVolume'>

    MedicalVolumes are also interoperable with popular image data structures
    with zero-copy, meaning array data will not be copied. Formats currently include the
    SimpleITK Image, Nibabel Nifti1Image, and PyTorch tensors:

    >>> sitk_img = mv.to_sitk()  # Convert to SimpleITK Image
    >>> mv_from_sitk = MedicalVolume.from_sitk(sitk_img)  # Convert back to MedicalVolume
    >>> nib_img = mv.to_nib()  # Convert to nibabel Nifti1Image
    >>> mv_from_nib = MedicalVolume.from_nib(nib_img)
    >>> torch_tensor = mv.to_torch()  # Convert to torch tensor
    >>> mv_from_tensor = MedicalVolume.from_torch(torch_tensor, affine)

    MedicalVolumes can also be used with memmapped arrays.
    This makes loading much faster and allows interaction with larger-than-memory
    arrays. Only when the volume is modified will the volume be loaded
    into memory and modified. If you take a slice of the memmaped array, the underlying
    array will also remain memmapped:

    >>> arr = np.load("/path/to/volume.npy", mmap_mode="r")
    >>> mv = MedicalVolume(arr, np.eye(4))
    >>> mv.is_mmap  # returns True

    We also preserve Nibabel's memmapping of certain file types (e.g. ``.nii``):

    >>> nib_img = nibabel.load("path/to/volume.nii")
    >>> mv = MedicalVolume.from_nib(nib_img, mmap=True)

    Args:
        volume (array-like): nD medical image.
        affine (array-like): 4x4 array corresponding to affine matrix transform in RAS+ coordinates.
            Must be on cpu (i.e. no ``cupy.ndarray``).
        headers (array-like[pydicom.FileDataset]): Headers for DICOM files.
    """

    def __init__(self, volume, affine, headers=None):
        if not isinstance(volume, np.memmap):
            xp = get_array_module(volume)
            volume = xp.asarray(volume)
        self._volume = volume
        self._affine = np.array(affine)
        self._headers = self._validate_and_format_headers(headers) if headers is not None else None

    def save_volume(self, file_path: str, data_format: "ImageDataFormat" = None):
        """Write volumes in specified data format.

        Args:
            file_path (str): File path to save data. May be modified to follow convention
                given by the data format in which the volume will be saved.
            data_format (ImageDataFormat): Format to save data.
        """
        import voxel.io as vio

        device = self.device
        if device != cpu_device:
            raise RuntimeError(f"MedicalVolume must be on cpu, got {self.device}")
        if data_format is None:
            data_format = vx.config.save_format

        writer = vio.get_writer(data_format)
        writer.save(self, file_path)

    def reformat(self, new_orientation: Sequence, *args, inplace: bool = False) -> "MedicalVolume":
        """Reorients volume to a specified orientation.

        Flipping and transposing the volume array (``self.volume``) returns a view if possible.

        Reorientation method:
        ---------------------
        - Axis transpose and flipping are linear operations and therefore can be treated
        independently.
        - working example: ('AP', 'SI', 'LR') --> ('RL', 'PA', 'SI')
        1. Transpose volume and RAS orientation to appropriate column in matrix
            eg. ('AP', 'SI', 'LR') --> ('LR', 'AP', 'SI') - transpose_inds=[2, 0, 1]
        2. Flip volume across corresponding axes
            eg. ('LR', 'AP', 'SI') --> ('RL', 'PA', 'SI') - flip axes 0,1

        Reorientation method implementation:
        ------------------------------------
        1. Transpose: Switching (transposing) axes in volume is the same as switching columns
        in affine matrix

        2. Flipping: Negate each column corresponding to pixel axis to flip (i, j, k) and
        reestablish origins based on flipped axes

        Args:
            new_orientation (Sequence): New orientation.
            inplace (bool, optional): If `True`, do operation in-place and return ``self``.

        Returns:
            MedicalVolume: The reformatted volume. If ``inplace=True``, returns ``self``.
        """
        xp = self.device.xp
        device = self.device
        headers = self._headers

        if len(args):
            if any(isinstance(x, bool) for x in args):
                raise ValueError(
                    "`inplace` is a keyword only argument. Use `inplace=` to specify "
                    "if the operation should be in-place."
                )
            new_orientation = (new_orientation, *args)

        new_orientation = tuple(new_orientation)
        if new_orientation == self.orientation:
            if inplace:
                return self
            return self._partial_clone(volume=self._volume)

        temp_orientation = self.orientation
        temp_affine = np.array(self._affine)

        transpose_inds = stdo.get_transpose_inds(temp_orientation, new_orientation)
        all_transpose_inds = transpose_inds + tuple(range(3, self._volume.ndim))

        with device:
            volume = xp.transpose(self.volume, all_transpose_inds)
        if headers is not None:
            headers = np.transpose(headers, all_transpose_inds)
        for i in range(len(transpose_inds)):
            temp_affine[..., i] = self._affine[..., transpose_inds[i]]

        temp_orientation = tuple([self.orientation[i] for i in transpose_inds])

        flip_axs_inds = list(stdo.get_flip_inds(temp_orientation, new_orientation))
        with device:
            volume = xp.flip(volume, axis=tuple(flip_axs_inds))
        if headers is not None:
            headers = np.flip(headers, axis=tuple(flip_axs_inds))
        a_vecs = temp_affine[:3, :3]
        a_origin = temp_affine[:3, 3]

        # phi is a vector of 1s and -1s, where 1 indicates no flip, and -1 indicates flip
        # phi is used to determine which columns in affine matrix to flip
        phi = np.ones([1, len(a_origin)]).flatten()
        phi[flip_axs_inds] *= -1

        b_vecs = np.array(a_vecs)
        for i in range(len(phi)):
            b_vecs[:, i] *= phi[i]

        # get number of pixels to shift by on each axis.
        # Should be 0 when not flipping - i.e. phi<0 mask
        vol_shape_vec = (
            (np.asarray(volume.shape[:3]) - 1) * (phi < 0).astype(np.float32)
        ).transpose()
        b_origin = np.round(
            a_origin.flatten() - np.matmul(b_vecs, vol_shape_vec).flatten(),
            vx.config.affine_precision,
        )

        temp_affine = np.array(self.affine)
        temp_affine[:3, :3] = b_vecs
        temp_affine[:3, 3] = b_origin
        temp_affine[temp_affine == 0] = 0  # get rid of negative 0s

        if inplace:
            self._affine = temp_affine
            self._volume = volume
            self._headers = headers
            mv = self
        else:
            mv = self._partial_clone(volume=volume, affine=temp_affine, headers=headers)

        assert (
            mv.orientation == new_orientation
        ), f"Orientation mismatch: Expected: {self.orientation}. Got {new_orientation}"
        return mv

    def reformat_as(self, other, inplace: bool = False) -> "MedicalVolume":
        """Reformat this to the same orientation as ``other``. Equivalent to
        ``self.reformat(other.orientation, inplace)``.

        Args:
            other (MedicalVolume): The result volume has the same orientation as ``other``.
            inplace (bool, optional): If `True`, do operation in-place and return ``self``.

        Returns:
            MedicalVolume: The reformatted volume. If ``inplace=True``, returns ``self``.
        """
        return self.reformat(other.orientation, inplace=inplace)

    def is_identical(self, mv):
        """Check if another medical volume is identical.

        Two volumes are identical if they have the same pixel_spacing, orientation,
        scanner_origin, and volume.

        Args:
            mv (MedicalVolume): Volume to compare with.

        Returns:
            bool: `True` if identical, `False` otherwise.
        """
        if not isinstance(mv, MedicalVolume):
            raise TypeError("`mv` must be a MedicalVolume.")

        idevice = self.device
        odevice = mv.device
        if idevice != odevice:
            raise RuntimeError(f"Expected device {idevice}, got {odevice}.")

        with idevice:
            return self.is_same_dimensions(mv) and (mv.volume == self.volume).all()

    def _allclose_spacing(self, mv, precision: int = None, ignore_origin: bool = False):
        """Check if spacing between self and another medical volume is within tolerance.

        Tolerance is `10 ** (-precision)`.

        Args:
            mv (MedicalVolume): Volume to compare with.
            precision (`int`, optional): Number of significant figures after the decimal.
                If not specified, check that affine matrices between two volumes are identical.
                Defaults to `None`.
            ignore_origin (bool, optional): If ``True``, ignore matching origin in the affine
                matrix.

        Returns:
            bool: `True` if spacing between two volumes within tolerance, `False` otherwise.
        """
        if precision is not None:
            tol = 10 ** (-precision)
            return np.allclose(mv.affine[:3, :3], self.affine[:3, :3], atol=tol) and (
                ignore_origin or np.allclose(mv.scanner_origin, self.scanner_origin, rtol=tol)
            )
        else:
            return (mv.affine == self.affine).all() or (
                ignore_origin and (mv.affine[:, :3] == self.affine[:, :3]).all()
            )

    def is_same_dimensions(self, mv, precision: int = None, err: bool = False):
        """Check if two volumes have the same dimensions.

        Two volumes have the same dimensions if they have the same pixel_spacing,
        orientation, and scanner_origin.

        Args:
            mv (MedicalVolume): Volume to compare with.
            precision (`int`, optional): Number of significant figures after the decimal.
                If not specified, check that affine matrices between two volumes are identical.
                Defaults to `None`.
            err (bool, optional): If `True` and volumes do not have same dimensions,
                raise descriptive ValueError.

        Returns:
            bool: ``True`` if pixel spacing, orientation, and scanner origin
                between two volumes within tolerance, ``False`` otherwise.

        Raises:
            TypeError: If ``mv`` is not a MedicalVolume.
            ValueError: If ``err=True`` and two volumes do not have same dimensions.
        """
        if not isinstance(mv, MedicalVolume):
            raise TypeError("`mv` must be a MedicalVolume.")

        is_close_spacing = self._allclose_spacing(mv, precision)
        is_same_orientation = mv.orientation == self.orientation
        is_same_shape = mv.volume.shape == self.volume.shape
        out = is_close_spacing and is_same_orientation and is_same_shape

        if err and not out:
            tol_str = f" (tol: 1e-{precision})" if precision else ""
            if not is_close_spacing:
                raise ValueError(
                    "Affine matrices not equal{}:\n{}\n{}".format(tol_str, self._affine, mv._affine)
                )
            if not is_same_orientation:
                raise ValueError(
                    "Orientations not equal: {}, {}".format(self.orientation, mv.orientation)
                )
            if not is_same_shape:
                raise ValueError(
                    "Shapes not equal: {}, {}".format(self._volume.shape, mv._volume.shape)
                )
            raise AssertionError()  # should not reach here

        return out

    def match_orientation(self, mv):
        """Reorient another MedicalVolume to orientation specified by self.orientation.

        Args:
            mv (MedicalVolume): Volume to reorient.
        """
        warnings.warn(
            "`match_orientation` is deprecated and will be removed in v0.1. "
            "Use `mv.reformat_as(self, inplace=True)` instead.",
            DeprecationWarning,
        )
        if not isinstance(mv, MedicalVolume):
            raise TypeError("`mv` must be a MedicalVolume.")

        mv.reformat(self.orientation, inplace=True)

    def match_orientation_batch(self, mvs):  # pragma: no cover
        """Reorient a collection of MedicalVolumes to orientation specified by self.orientation.

        Args:
            mvs (list[MedicalVolume]): Collection of MedicalVolumes.
        """
        warnings.warn(
            "`match_orientation_batch` is deprecated and will be removed in v0.1. "
            "Use `[x.reformat_as(self, inplace=True) for x in mvs]` instead.",
            DeprecationWarning,
        )
        for mv in mvs:
            self.match_orientation(mv)

    def clone(self, headers=True):
        """Clones the medical volume.

        Args:
            headers (bool, optional): If `True`, clone headers.
                If `False`, headers have shared memory.

        Returns:
            mv (MedicalVolume): A cloned MedicalVolume.
        """
        return MedicalVolume(
            self.volume.copy(),
            self.affine.copy(),
            headers=deepcopy(self._headers) if headers else self._headers,
        )

    def to(self, device):
        """Move to device.

        If on same device, no-op and returns ``self``.

        Args:
            device: The device to move to.

        Returns:
            MedicalVolume
        """
        device = Device(device)
        if self.device == device:
            return self

        return self._partial_clone(volume=to_device(self._volume, device))

    def cpu(self):
        """Move to cpu."""
        return self.to("cpu")

    def astype(self, dtype, **kwargs):
        """Modifies dtype of ``self._volume``.

        Note this operation is done in place. ``self._volume`` is modified, based
        on the ``astype`` implementation of the type associated with ``self._volume``.
        No new MedicalVolume is created - ``self`` is returned.

        Args:
            dtype (str or dtype): Typecode or data-type to which the array is cast.

        Returns:
            self
        """
        if (
            env.package_available("h5py")
            and isinstance(self._volume, h5py.Dataset)
            and version.parse(env.get_version(h5py)) < version.parse("3.0.0")
        ):
            raise ValueError("Cannot cast h5py.Dataset to dtype for h5py<3.0.0")

        self._volume = self._volume.astype(dtype, **kwargs)
        return self

    def to_nib(self):
        """Converts to nibabel Nifti1Image.

        Returns:
            nibabel.Nifti1Image: The nibabel image.

        Raises:
            RuntimeError: If medical volume is not on the cpu.

        Examples:
            >>> mv = MedicalVolume(np.ones((10,20,30)), np.eye(4))
            >>> mv.to_nib()
            <nibabel.nifti1.Nifti1Image>
        """
        device = self.device
        if device != cpu_device:
            raise RuntimeError(f"MedicalVolume must be on cpu, got {self.device}")

        return nib.Nifti1Image(self.A, self.affine.copy())

    def to_sitk(self, vdim: int = None, transpose_inplane: bool = False):
        """Converts to SimpleITK Image.

        SimpleITK Image objects support vector pixel types, which are represented
        as an extra dimension in numpy arrays. The vector dimension can be specified
        with ``vdim``.

        MedicalVolume must be on cpu. Use ``self.cpu()`` to move.

        SimpleITK loads DICOM files as individual slices that get stacked in ``(z, x, y)``
        order. Thus, ``sitk.GetArrayFromImage`` returns an array in ``(y, x, z)`` order.
        To return a SimpleITK Image that will follow this convention, set
        ``transpose_inplace=True``. If you have been using SimpleITK to load DICOM files,
        you will likely want to specify this parameter.

        Args:
            vdim (int, optional): The vector dimension.
            transpose_inplane (bool, optional): If ``True``, transpose inplane axes.
                Recommended to be ``True`` for users who are familiar with SimpleITK's
                DICOM loading convention.

        Returns:
            SimpleITK.Image

        Raises:
            ImportError: If `SimpleITK` is not installed.
            RuntimeError: If MedicalVolume is not on cpu.

        Note:
            Header information is not currently copied.
        """
        if not env.sitk_available():
            raise ImportError("SimpleITK is not installed. Install it with `pip install simpleitk`")
        device = self.device
        if device != cpu_device:
            raise RuntimeError(f"MedicalVolume must be on cpu, got {self.device}")

        arr = self.volume
        ndim = arr.ndim

        if vdim is not None:
            if vdim < 0:
                vdim = ndim + vdim
            axes = tuple(i for i in range(ndim) if i != vdim)[::-1] + (vdim,)
        else:
            axes = range(ndim)[::-1]
        arr = np.transpose(arr, axes)

        affine = self.affine.copy()
        affine[:2] = -affine[:2]  # RAS+ -> LPS+

        origin = tuple(affine[:3, 3])
        spacing = self.pixel_spacing
        direction = affine[:3, :3] / np.asarray(spacing)

        img = sitk.GetImageFromArray(arr, isVector=vdim is not None)
        img.SetOrigin(origin)
        img.SetSpacing(spacing)
        img.SetDirection(tuple(direction.flatten()))

        if transpose_inplane:
            pa = sitk.PermuteAxesImageFilter()
            pa.SetOrder([1, 0, 2])
            img = pa.Execute(img)

        return img

    def to_torch(
        self, requires_grad: bool = False, contiguous: bool = False, view_as_real: bool = False
    ):
        """Zero-copy conversion to torch tensor.

        If torch version supports complex tensors (i.e. torch>=1.5.0), complex MedicalVolume
        arrays will be converted into complex tensors (torch.complex64/torch.complex128).
        Otherwise, tensors will be returned as the real view, where the last dimension has
        two channels (`tensor.shape[-1]==2`). `[..., 0]` and `[..., 1]` correspond to the
        real/imaginary channels, respectively.

        Args:
            requires_grad (bool, optional): Set ``.requires_grad`` for output tensor.
            contiguous (bool, optional): Make output tensor contiguous before returning.
            view_as_real (bool, optional): If ``True`` and underlying array is complex,
                returns a real view of a complex tensor.

        Returns:
            torch.Tensor: The torch tensor.

        Raises:
            ImportError: If ``torch`` is not installed.

        Note:
            This method does not convert affine matrices and headers to tensor types.

        Examples:
            >>> mv = MedicalVolume(np.ones((2,2,2)), np.eye(4))  # zero-copy on CPU
            >>> mv.to_torch()
            tensor([[[1., 1.],
                     [1., 1.]],
                    [[1., 1.],
                     [1., 1.]]], dtype=torch.float64)
            >>> mv_gpu = MedicalVolume(cp.ones((2,2,2)), np.eye(4))  # zero-copy on GPU
            >>> mv.to_torch()
            tensor([[[1., 1.],
                     [1., 1.]],
                    [[1., 1.],
                     [1., 1.]]], device="cuda:0", dtype=torch.float64)
            >>> # view complex array as real tensor
            >>> mv = MedicalVolume(np.ones((3,4,5), dtype=np.complex), np.eye(4))
            >>> tensor = mv.to_torch(view_as_real)
            >>> tensor.shape
            (3, 4, 5, 2)
        """
        if not env.package_available("torch"):
            raise ImportError(  # pragma: no cover
                "torch is not installed. Install it with `pip install torch`. "
                "See https://pytorch.org/ for more information."
            )

        import torch
        from torch.utils.dlpack import from_dlpack

        device = self.device
        array = self.A

        if any(np.issubdtype(array.dtype, dtype) for dtype in (np.complex64, np.complex128)):
            torch_version = env.get_version(torch)
            supports_cplx = version.Version(torch_version) >= _TORCH_COMPLEX_SUPPORT_VERSION
            if not supports_cplx or view_as_real:
                with device:
                    shape = array.shape
                    array = array.view(dtype=array.real.dtype)
                    array = array.reshape(shape + (2,))

        if device == cpu_device:
            tensor = torch.from_numpy(array)
        else:
            tensor = from_dlpack(array.toDlpack())

        tensor.requires_grad = requires_grad
        if contiguous:
            tensor = tensor.contiguous()
        return tensor

    def to_zarr(
        self,
        affine_attr: str = None,
        headers_attr: str = None,
        read_only: bool = True,
        **kwargs,
    ):
        """Converts a `MedicalVolume` to a Zarr array/store.

        Zarr stores can be used to store and access data on disk or in memory. The `store` argument
        can be set to persist the data to disk. See `zarr.open_array` for more information. The
        `affine` and `headers` attributes of `MedicalVolume` are stored as Zarr attributes. To do
        so, the DICOM headers are serialized to DICOM+JSON format.

        The default `mode` is set to `w-` to prevent overwriting existing data. If you want to
        overwrite existing data, set `mode` to `w`.

        Args:
            affine_attr (str, optional): Attribute key of the Zarr Array where the affine matrix
                will be stored in. If `None`, the affine matrix will not be saved.
            headers_attr (str, optional): Attribute key of the Zarr Array where the headers of the
                `MedicalVolume` will be stored in. If `None`, headers will not be saved.
            read_only (bool, optional): If `True`, the returned Zarr store will be read-only.
            **kwargs: Additional parameters passed to `zarr.creation.open_array`.
        Returns:
            zarr.Array
        Examples:
            >>> mv = vx.load("path/to/dicoms")
            >>> store = zarr.DirectoryStore("/path/to/store")
            >>> mv.to_zarr(store=store)

            >>> # Save headers to zarr attributes
            >>> mv.to_zarr(store=store, headers_attr="headers")

            >>> # Load with headers
            >>> MedicalVolume.from_zarr(store, headers_attr="headers")
        """

        if not env.package_available("zarr"):
            raise ImportError(  # pragma: no cover
                "zarr is not installed. Install it with `pip install zarr`. "
            )

        import zarr

        arr = zarr.open_array(**{"mode": "w-", **kwargs, "shape": self.shape, "dtype": self.dtype})
        arr[:] = self._volume

        if affine_attr is not None:
            arr.attrs[affine_attr] = self.affine.tolist()

        if headers_attr is not None:
            json_headers = [h.to_json_dict() for h in self._headers.flatten().tolist()]
            arr.attrs[headers_attr] = json_headers

        arr.read_only = read_only
        return arr

    def headers(self, flatten=False):
        """Returns headers.

        If headers exist, they are currently stored as an array of
        pydicom dataset headers, though this is subject to change.

        Args:
            flatten (bool, optional): If ``True``, flattens header array
                before returning.

        Returns:
            Optional[ndarray[pydicom.dataset.FileDataset]]: Array of headers (if they exist).
        """
        if flatten and self._headers is not None:
            return self._headers.flatten()
        return self._headers

    def get_metadata(self, key, index: int = None, dtype=None, default=np._NoValue):
        """Get metadata value from first header.

        The first header is defined as the first header in ``np.flatten(self._headers)``.
        To extract header information for other headers, use ``self.headers()``.

        Args:
            key (``str`` or pydicom.BaseTag``): Metadata field to access.
            dtype (type, optional): If specified, data type to cast value to.
                By default for DICOM headers, data will be in the value
                representation format specified by pydicom. See
                ``pydicom.valuerep``.
            default (Any): Default value to return if `key`` not found in header.
                If not specified and ``key`` not found in header, raises a KeyError.

        Examples:
            >>> mv.get_metadata("EchoTime")
            '10.0'  # this is a number type ``pydicom.valuerep.DSDecimal``
            >>> mv.get_metadata("EchoTime", dtype=float)
            10.0
            >>> mv.get_metadata("foobar", default=0)
            0

        Raises:
            RuntimeError: If ``self._headers`` is ``None``.
            KeyError: If ``key`` not found and ``default`` not specified.

        Note:
            Currently header information is tied to the ``pydicom.FileDataset`` implementation.
            This function is synonymous to ``dataset.<key>`` in ``pydicom.FileDataset``.
        """
        if self._headers is None:
            raise RuntimeError("No headers found. MedicalVolume must be initialized with `headers`")
        headers = self.headers(flatten=True)

        if key not in headers[0] and default != np._NoValue:
            return default
        else:
            element = headers[0][key]

        val = element.value
        if isinstance(val, list) and index is not None:
            val = val[index]

        if dtype is not None:
            val = dtype(val)

        return val

    def set_metadata(self, key, value, force: bool = False):
        """Sets metadata for all headers.

        Args:
            key (str or pydicom.BaseTag): Metadata field to access.
            value (Any): The value.
            force (bool, optional): If ``True``, force the header to
                set key even if key does not exist in header.

        Raises:
            RuntimeError: If ``self._headers`` is ``None``.
        """
        if self._headers is None:
            if not force:
                raise ValueError(
                    "No headers found. To generate headers and write keys, `force` must be True."
                )
            self._headers = self._validate_and_format_headers([pydicom.Dataset()])
            warnings.warn(
                "Headers were generated and may not contain all attributes "
                "required to save the volume in DICOM format."
            )

        VR_registry = {float: "DS", int: "IS", str: "LS"}
        for h in self.headers(flatten=True):
            if force and key not in h:
                try:
                    setattr(h, key, value)
                except TypeError:
                    h.add_new(key, VR_registry[type(value)], value)
            else:
                h[key].value = value

    def apply_rescale(
        self,
        slope: float = None,
        intercept: float = None,
        dtype: np.dtype = None,
        inplace: bool = False,
        sync: bool = True,
    ) -> "MedicalVolume":
        """Rescales the volume by applying the intercept and slope.

        Args:
            intercept (float, optional): Rescale intercept. If ``None``, will
                use the value in the header.
            slope (float, optional): Rescale slope. If ``None``, will
                use the value in the header.
            dtype (np.dtype, optional): Data type to cast volume to before
                rescale is applied.
            inplace (bool, optional): If ``True``, rescales the volume in place.
            sync (bool, optional): If ``True``, updates the headers.

        Returns:
            MedicalVolume: Volume with rescale applied.
        """
        if self._headers is None and (intercept is None or slope is None):
            return self

        mv = self if inplace else self.clone()
        mv = mv.astype(np.dtype(dtype), copy=False)

        h: pydicom.Dataset = self._headers.flat[0]
        rs = float(h.get("RescaleSlope", 1)) if slope is None else slope
        ri = float(h.get("RescaleIntercept", 0)) if intercept is None else intercept
        apply_rescale(mv._volume, rs, ri, inplace=True)

        if sync:
            mv._delete_metadata("RescaleSlope")
            mv._delete_metadata("RescaleIntercept")

        return mv

    def apply_modality_lut(self, inplace: bool = False, sync: bool = True) -> "MedicalVolume":
        """Applies modality LUT to volume.

        Args:
            inplace (bool, optional): If ``True``, applies modality LUT in place.
            sync (bool, optional): If ``True``, updates the headers.

        Returns:
            MedicalVolume: Volume with modality LUT applied.
        """
        if self._headers is None:
            return self

        return self._apply_lut("ModalityLUTSequence", index=0, inplace=inplace, sync=sync)

    def apply_window(
        self,
        index: int = 0,
        center: float = None,
        width: float = None,
        output_range: Tuple[float, float] = None,
        mode: str = None,
        dtype: np.dtype = None,
        inplace: bool = False,
        sync: bool = True,
    ) -> "MedicalVolume":
        """Windows the volume using the window center and width.
        User supplied values will override the values in the header.

        Args:
            index (int, optional): Index of window to apply.
            center (float, optional): Window center. If ``None``, will
                use the value in the header.
            width (float, optional): Window width. If ``None``, will
                use the value in the header.
            output_range (Tuple[float, float], optional): Output range to
                apply window to. If ``None``, will use the value in the header.
            mode (str, optional): VOI LUT function. If ``None``, will
                use the value in the header.
            dtype (np.dtype, optional): Data type to cast volume to before
                window is applied.
            inplace (bool, optional): If ``True``, applies window in place.
            sync (bool, optional): If ``True``, updates the headers.

        Returns:
            ``MedicalVolume`` with window applied.
        """
        if self._headers is None and (center is None or width is None):
            return self

        # Define the window center, window width, and VOI LUT function
        wc, ww, vlf = center, width, mode
        if self._headers is not None:
            h: pydicom.Dataset = self._headers.flat[0]
            if center is None:
                if "WindowCenter" not in h:
                    return self

                wc = h["WindowCenter"]
                wc = float(wc.value[index]) if wc.VM > 1 else float(wc.value)

            if width is None:
                if "WindowWidth" not in h:
                    return self

                ww = h["WindowWidth"]
                ww = float(ww.value[index]) if ww.VM > 1 else float(ww.value)

            if mode is None:
                vlf = h.get("VOILUTFunction", None)

        # Apply the window transformation
        mv = self if inplace else self.clone()
        mv = self.astype(dtype, copy=False)
        apply_window(mv._volume, wc, ww, output_range, vlf, inplace=True)

        if sync:
            for key in ["WindowCenter", "WindowWidth", "VOILUTFunction"]:
                mv._delete_metadata(key)

        return mv

    def apply_voi_lut(
        self, index: int = 0, inplace: bool = False, sync: bool = True
    ) -> "MedicalVolume":
        """Applies VOI LUT to volume.

        Args:
            index (int, optional): Index of VOI LUT to apply.
            inplace (bool, optional): If ``True``, applies VOI LUT in place.
            sync (bool, optional): If ``True``, updates the headers by removing
                the VOI LUT Sequence and setting the window to the dynamic range.

        Returns:
            ``MedicalVolume`` with VOI LUT applied.
        """
        if self._headers is None:
            return self

        return self._apply_lut("VOILUTSequence", index=index, inplace=inplace, sync=sync)

    def to_grayscale(
        self,
        mode: str = "MONOCHROME2",
        output_range: Tuple[float, float] = None,
        inplace: bool = False,
        sync: bool = True,
    ) -> "MedicalVolume":
        """Converts volume to a grayscale mode."""
        if self._headers is None:
            return self

        options = ["MONOCHROME1", "MONOCHROME2"]
        if mode not in options:
            raise ValueError(f"Photometric Interpretation {mode} is not a grayscale mode.")

        h: pydicom.Dataset = self._headers.flat[0]
        if "PhotometricInterpretation" not in h or h.PhotometricInterpretation == mode:
            return self

        mv = self if inplace else self.clone()
        if sync:
            # Toggle the photometric interpretation
            mv.set_metadata("PhotometricInterpretation", mode)

            # Update the window center and width
            center = np.array(h.get("WindowCenter", [])).flatten()
            width = np.array(h.get("WindowWidth", [])).flatten()

            if len(center) > 0 and len(width) > 0:
                wcs, wws = [], []
                for wc, ww in zip(center, width):
                    new_wc, new_ww = invert_window(mv._volume, wc, ww, output_range)
                    wcs.append(new_wc)
                    wws.append(new_ww)

                if len(wcs) == 1:
                    wcs, wws = wcs[0], wws[0]

                mv.set_metadata("WindowCenter", wcs)
                mv.set_metadata("WindowWidth", wws)
                mv.set_metadata("VOILUTFunction", "LINEAR_EXACT", force=True)

        # Invert the volume
        invert(mv._volume, output_range, inplace=True)
        return mv

    def materialize(self):
        if not self.is_mmap:
            return self[:]
        else:
            xp = self.device.xp
            self._volume = xp.asarray(self._volume)
            return self

    def round(self, decimals=0, affine=False) -> "MedicalVolume":
        """Round array (and optionally affine matrix).

        Args:
            decimals (int, optional): Number of decimals to round to.
            affine (bool, optional): The rounded medical volume.

        Returns:
            MedicalVolume: MedicalVolume with rounded.
        """
        from voxel.numpy_routines import around

        return around(self, decimals, affine)

    def sum(
        self,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=np._NoValue,
        where=np._NoValue,
    ) -> "MedicalVolume":
        """Compute the arithmetic sum along the specified axis. Identical to :meth:`sum_np`.

        See :meth:`sum_np` for more information.

        Args:
            axis: Same as :meth:`sum_np`.
            dtype: Same as :meth:`sum_np`.
            out: Same as :meth:`sum_np`.
            keepdims: Same as :meth:`sum_np`.
            initial: Same as :meth:`sum_np`.
            where: Same as :meth:`sum_np`.

        Returns:
            Union[Number, MedicalVolume]: If ``axis=None``, returns a number or a scalar type of
                the underlying ndarray. Otherwise, returns a medical volume containing sum
                values.
        """
        from voxel.numpy_routines import sum_np

        # `out` is required for cupy arrays because of how cupy calls array.
        if out is not None:
            raise ValueError("`out` must be None")
        return sum_np(self, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)

    def mean(
        self, axis=None, dtype=None, out=None, keepdims=False, where=np._NoValue
    ) -> Union[Number, "MedicalVolume"]:
        """Compute the arithmetic mean along the specified axis. Identical to :meth:`mean_np`.

        See :meth:`mean_np` for more information.

        Args:
            axis: Same as :meth:`mean_np`.
            dtype: Same as :meth:`mean_np`.
            out: Same as :meth:`mean_np`.
            keepdims: Same as :meth:`mean_np`.
            initial: Same as :meth:`mean_np`.
            where: Same as :meth:`mean_np`.

        Returns:
            Union[Number, MedicalVolume]: If ``axis=None``, returns a number or a scalar type of
            the underlying ndarray. Otherwise, returns a medical volume containing mean
            values.
        """
        from voxel.numpy_routines import mean_np

        # `out` is required for cupy arrays because of how cupy calls array.
        if out is not None:
            raise ValueError("`out` must be None")
        return mean_np(self, axis=axis, dtype=dtype, keepdims=keepdims, where=where)

    def std(self, axis=None, dtype=None, out=None, keepdims=False, where=np._NoValue):
        """Compute the standard deviation along the specified axis. Identical to :meth:`std`.

        See :meth:`std` for more information.

        Args:
            axis: Same as :meth:`std`.
            dtype: Same as :meth:`std`.
            out: Same as :meth:`std`.
            keepdims: Same as :meth:`std`.
            initial: Same as :meth:`std`.
            where: Same as :meth:`std`.
        """
        from voxel.numpy_routines import std

        # `out` is required for cupy arrays because of how cupy calls array.
        if out is not None:
            raise ValueError("`out` must be None")
        return std(self, axis=axis, dtype=dtype, keepdims=keepdims, where=where)

    def contiguous(self) -> "MedicalVolume":
        """Returns a MedicalVolume with pixel data contiguous in memory.

        If the pixel data is already contiguous, returns the ``self``
        medical volume.

        Returns:
            MedicalVolume: MedicalVolume with contiguous pixel data.
        """
        from voxel.numpy_routines import ascontiguousarray

        if self.is_contiguous():
            return self
        return ascontiguousarray(self)

    def is_contiguous(self) -> bool:
        """Returns whether the pixel data is contiguous in memory.

        Returns:
            bool: Whether the pixel data is contiguous in memory.
        """
        from voxel.numpy_routines import is_contiguous

        return is_contiguous(self)

    @property
    def A(self):
        """The pixel array. Same as ``self.volume``.

        Examples:
            >>> mv = MedicalVolume([[[1,2],[3,4]]], np.eye(4))
            >>> mv.A
            array([[[1, 2],
                    [3, 4]]])
        """
        return self.volume

    @property
    def volume(self):
        """ndarray: ndarray representing volume values."""
        return self._volume

    @volume.setter
    def volume(self, value):
        """If the volume is of a different shape, the headers are no longer valid, so delete all
        reorientations are done as part of MedicalVolume, so reorientations are permitted.

        However, external setting of the volume to a different shape array is not allowed.
        """
        if value.ndim != self._volume.ndim:
            raise ValueError("New volume must be same as current volume")

        if self._volume.shape != value.shape:
            self._headers = None

        self._volume = value
        self._device = get_device(self._volume)

    @property
    def pixel_spacing(self):
        """tuple[float]: Pixel spacing in order of current orientation."""
        vecs = self._affine[:3, :3]
        ps = tuple(np.sqrt(np.sum(vecs**2, axis=0)))

        assert len(ps) == 3, "Pixel spacing must have length of 3"
        return ps

    @property
    def orientation(self):
        """tuple[str]: Image orientation in standard orientation format.

        See orientation.py for more information on conventions.
        """
        nib_orientation = nib.aff2axcodes(self._affine)
        return stdo.orientation_nib_to_standard(nib_orientation)

    @property
    def scanner_origin(self):
        """tuple[float]: Scanner origin in global RAS+ x,y,z coordinates."""
        return tuple(self._affine[:3, 3])

    @property
    def affine(self):
        """np.ndarray: 4x4 affine matrix for volume in current orientation."""
        return self._affine

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the underlying ndarray."""
        return self._volume.shape

    @property
    def ndim(self) -> int:
        """int: The number of dimensions of the underlying ndarray."""
        return self._volume.ndim

    @property
    def device(self) -> Device:
        """The device the object is on."""
        return get_device(self._volume)

    @property
    def dtype(self):
        """The ``dtype`` of the ndarray.

        Same as ``self.volume.dtype``.
        """
        return self._volume.dtype

    @property
    def is_mmap(self) -> bool:
        """bool: Whether the volume is a memory-mapped array."""
        # important to check if .base is a python mmap object, since a view of a mmap
        # is also a memmap object, but should not be symlinked or copied
        return isinstance(self.A, np.memmap) and isinstance(self.A.base, mmap)

    @classmethod
    def from_nib(
        cls, image, affine_precision: int = None, origin_precision: int = None, mmap: bool = False
    ) -> "MedicalVolume":
        """Constructs MedicalVolume from nibabel images.

        Args:
            image (nibabel.Nifti1Image): The nibabel image to convert.
            affine_precision (int, optional): If specified, rounds the i/j/k coordinate
                vectors in the affine matrix to this decimal precision.
            origin_precision (int, optional): If specified, rounds the scanner origin
                in the affine matrix to this decimal precision.
            mmap (bool, optional): If True, memory map the image.

        Returns:
            MedicalVolume: The medical image.

        Examples:
            >>> import nibabel as nib
            >>> nib_img = nib.Nifti1Image(np.ones((10,20,30)), np.eye(4))
            >>> MedicalVolume.from_nib(nib_img)
            MedicalVolume(
                shape=(10, 20, 30),
                ornt=('LR', 'PA', 'IS')),
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                device=Device(type='cpu')
            )
        """
        affine = np.array(image.affine)  # Make a copy of the affine matrix.
        if affine_precision is not None:
            affine[:3, :3] = np.round(affine[:3, :3], affine_precision)
        if origin_precision:
            affine[:3, 3] = np.round(affine[:3, 3], origin_precision)

        data = image.dataobj.__array__() if mmap else image.get_fdata()
        mv = cls(data, affine)
        if mmap and not mv.is_mmap:
            raise ValueError(
                "Underlying array in the nibabel image is not mem-mapped. " "Please set mmap=False."
            )
        return mv

    @classmethod
    def from_sitk(cls, image, copy=False, transpose_inplane: bool = False) -> "MedicalVolume":
        """Constructs MedicalVolume from SimpleITK.Image.

        Use ``transpose_inplane=True`` if the SimpleITK image was loaded with SimpleITK's
        DICOM reader or if ``transpose_inplace=True`` was used to create the Image
        with :meth:`to_sitk`. See the discussion of SimpleITK's data ordering convention
        in :meth:`to_sitk` for more information.

        If you are getting a segmentation fault, try using ``copy=True``.

        Args:
            image (SimpleITK.Image): The image.
            copy (bool, optional): If ``True``, copies array.
            transpose_inplane (bool, optional): If ``True``, transposes the inplane axes.
                Set this to ``True`` if the SimpleITK image was loaded with SimpleITK's
                DICOM reader. May need to set ``copy=True`` to avoid segmentation fault.

        Returns:
            MedicalVolume

        Note:
            Metadata information is not copied.
        """
        if not env.sitk_available():
            raise ImportError("SimpleITK is not installed. Install it with `pip install simpleitk`")

        if len(image.GetSize()) < 3:
            raise ValueError("`image` must be 3D.")
        is_vector_image = image.GetNumberOfComponentsPerPixel() > 1

        if transpose_inplane:
            pa = sitk.PermuteAxesImageFilter()
            pa.SetOrder([1, 0, 2])
            image = pa.Execute(image)

        if copy:
            arr = sitk.GetArrayFromImage(image)
        else:
            arr = sitk.GetArrayViewFromImage(image)

        ndim = arr.ndim
        if is_vector_image:
            axes = tuple(range(ndim)[-2::-1]) + (ndim - 1,)
        else:
            axes = range(ndim)[::-1]
        arr = np.transpose(arr, axes)

        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = np.asarray(image.GetDirection()).reshape(-1, 3)

        affine = np.zeros((4, 4))
        affine[:3, :3] = direction * np.asarray(spacing)
        affine[:3, 3] = origin
        affine[:2] = -affine[:2]  # LPS+ -> RAS+
        affine[3, 3] = 1

        return cls(arr, affine)

    @classmethod
    def from_torch(cls, tensor, affine, headers=None, to_complex: bool = None) -> "MedicalVolume":
        """Zero-copy construction from PyTorch tensor.

        Args:
            tensor (torch.Tensor): A PyTorch tensor where first three dimensions correspond
                to spatial dimensions.
            affine (np.ndarray): See class parameters.
            headers (np.ndarray[pydicom.FileDataset], optional): See class parameters.
            to_complex (bool, optional): If ``True``, interprets tensor as real view of complex
                tensor and attempts to restructure it as a complex array.

        Returns:
            MedicalVolume: A medical image.

        Raises:
            RuntimeError: If ``affine`` is not on the cpu.
            ValueError: If ``tensor`` does not have at least three spatial dimensions.
            ValueError: If ``to_complex=True`` and shape is not size ``(..., 2)``.
            ImportError: If ``tensor`` on GPU and ``cupy`` not installed.

        Examples:
            >>> import torch
            >>> tensor = torch.ones((2,2,2))
            >>> MedicalVolume.from_torch(tensor, affine=np.eye(4))
            MedicalVolume(
                shape=(2, 2, 2),
                ornt=('LR', 'PA', 'IS')),
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                device=Device(type='cpu')
            )
            >>> tensor = torch.ones((2,2,2), device="cuda")  # zero-copy from GPU 0
            >>> MedicalVolume.from_torch(tensor, affine=np.eye(4))
            MedicalVolume(
                shape=(2, 2, 2),
                ornt=('LR', 'PA', 'IS')),
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                device=Device(type='cuda', index=0)
            )
            >>> tensor = torch.ones((3,4,5,2))  # treat this tensor as view of complex tensor
            >>> mv = MedicalVolume.from_torch(tensor, affine=np.eye(4), to_complex=True)
            >>> print(mv)
            MedicalVolume(
                shape=(3,4,5),
                ornt=('LR', 'PA', 'IS')),
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                device=Device(type='cuda', index=0)
            )
            >>> mv.dtype
            np.complex128
        """
        if not env.package_available("torch"):
            raise ImportError(  # pragma: no cover
                "torch is not installed. Install it with `pip install torch`. "
                "See https://pytorch.org/ for more information."
            )

        import torch
        from torch.utils.dlpack import to_dlpack

        torch_version = env.get_version(torch)
        supports_cplx = version.Version(torch_version) >= _TORCH_COMPLEX_SUPPORT_VERSION
        # Check if tensor needs to be converted to np.complex type.
        # If tensor is of torch.complex64 or torch.complex128 dtype, then from_numpy will take
        # care of conversion to appropriate numpy dtype, and we do not need to do the to_complex
        # logic.
        to_complex = to_complex and (
            not supports_cplx
            or (supports_cplx and tensor.dtype not in (torch.complex64, torch.complex128))
        )

        if isinstance(affine, torch.Tensor):
            if Device(affine.device) != cpu_device:
                raise RuntimeError("Affine matrix must be on the cpu")
            affine = affine.numpy()

        if (not to_complex and tensor.ndim < 3) or (to_complex and tensor.ndim < 4):
            raise ValueError(
                f"Tensor must have three spatial dimensions. Got shape {tensor.shape}."
            )
        if to_complex and tensor.shape[-1] != 2:
            raise ValueError(
                f"tensor.shape[-1] must have shape 2 when to_complex is specified. "
                f"Got shape {tensor.shape}."
            )

        device = Device(tensor.device)
        if device == cpu_device:
            array = tensor.detach().numpy()
        else:
            if env.cupy_available():
                array = cp.fromDlpack(to_dlpack(tensor))
            else:
                raise ImportError(  # pragma: no cover
                    "CuPy is required to convert a GPU torch.Tensor to array. "
                    "Follow instructions at https://docs.cupy.dev/en/stable/install.html to "
                    "install the correct binary."
                )

        if to_complex:
            with get_device(array):
                if array.dtype == np.float32:
                    array = array.view(np.complex64)
                elif array.dtype == np.float64:
                    array = array.view(np.complex128)

                array = array.reshape(array.shape[:-1])

        return cls(array, affine, headers=headers)

    def _partial_clone(self, **kwargs) -> "MedicalVolume":
        """Copies constructor information from ``self`` if not available in ``kwargs``."""
        if kwargs.get("volume", None) is False:
            # special use case to not clone volume
            kwargs["volume"] = self._volume
        for k in ("volume", "affine"):
            if k not in kwargs or (kwargs[k] is True):
                kwargs[k] = getattr(self, f"_{k}").copy()
        if "headers" not in kwargs:
            kwargs["headers"] = self._headers
        elif isinstance(kwargs["headers"], bool) and kwargs["headers"]:
            kwargs["headers"] = deepcopy(self._headers)
        return self.__class__(**kwargs)

    @classmethod
    def from_zarr(
        cls,
        store,
        affine_attr: str = None,
        headers_attr: str = None,
        default_ornt: Tuple[str, str] = np._NoValue,
        **kwargs,
    ) -> "MedicalVolume":
        """Construct a MedicalVolume from a Zarr store.

        This method safely opens a Zarr store and lazily loads the associated volume. The affine
        matrix and headers can be opened through the ``affine_attr`` and ``headers_attr``, if
        applicable.

        Args:
            store (Union[MutableMapping, str]): A zarr store.
            affine_attr (str, optional): Attribute key from the Zarr array where the affine matrix
                is stored in. If `None`, then the affine matrix is assumed to be the identity.
            headers_attr (str, optional): Attribute key to retrieve the headers of the
                `MedicalVolume` from. If `None`, headers will not be retrieved.
            default_ornt (Tuple[str, str], optional): See `MedicalVolume` class parameters.
            **kwargs: Additional parameters are passed along to `zarr.creation.open_array`.

        Returns:
            MedicalVolume: The medical image.

        Examples:
            >>> # load a zarr array from disk
            >>> store = zarr.ZipStore("/path/to/store")
            >>> zarr.save_array(store, np.zeros((10, 10)))
            >>> MedicalVolume.from_zarr(store)

            >>> # load zarr array, affine matrix, and headers
            >>> mv = vx.load("path/to/dicoms")
            >>> mv.to_zarr(store=store, affine_attr="affine", headers_attr="headers")
            >>> MedicalVolume.from_zarr(store=store, affine_attr="affine", headers_attr="headers")
        """

        if not env.package_available("zarr"):
            raise ImportError(  # pragma: no cover
                "zarr is not installed. Install it with `pip install zarr`. "
            )

        import zarr

        arr = zarr.open_array(store, **{"mode": "r", **kwargs})

        # Determine if DICOM+JSON headers are stored in the zarr array. If so, retrieve them and
        # convert them back to a list of pydicom.Dataset instances.
        headers, affine = None, np.eye(4)
        if headers_attr is not None:
            zarr_header = arr.attrs.get(headers_attr, None)
            if zarr_header is None:
                raise KeyError(f"Attribute `{headers_attr}` does not exist on the zarr.Array.")
            headers = [Dataset.from_json(h) for h in zarr_header]

        # Determine if the affine matrix is stored in the zarr array. If so, retrieve it, else
        # try to infer it from the headers. If no headers are available, use the identity matrix.
        if affine_attr is not None:
            if affine_attr not in arr.attrs:
                raise KeyError(f"Attribute `{affine_attr}` does not exist on the zarr.Array.")
            affine = np.array(arr.attrs.get(affine_attr)).reshape(4, 4)
            return cls(arr, affine, headers)

        if headers is not None:
            affine = stdo.to_RAS_affine(headers, default_ornt)

        return cls(arr, affine, headers)

    def _validate_and_format_headers(self, headers):
        """Validate headers are of appropriate shape and format into standardized shape.

        Headers are stored an ndarray of dictionary-like objects with explicit dimensions
        that match the dimensions of ``self._volume``. If header objects are not

        Assumes ``self._volume`` and ``self._affine`` have been set.
        """
        headers = np.asarray(headers)
        if headers.ndim > self._volume.ndim:
            raise ValueError(
                f"`headers` has too many dimensions. "
                f"Got headers.ndim={headers.ndim}, but volume.ndim={self._volume.ndim}"
            )
        for dim in range(-headers.ndim, 0)[::-1]:
            if headers.shape[dim] not in (1, self._volume.shape[dim]):
                raise ValueError(
                    f"`headers` must follow standard broadcasting shape. "
                    f"Got headers.shape={headers.shape}, but volume.shape={self._volume.shape}"
                )

        ndim = self._volume.ndim
        shape = (1,) * (ndim - len(headers.shape)) + headers.shape
        headers = np.reshape(headers, shape)
        return headers

    def _delete_metadata(self, key: Union[str, pydicom.BaseTag]):
        """Deletes metadata for all headers.

        Args:
            key (str or pydicom.BaseTag): Metadata field to access.
        """
        if self._headers is not None:
            for h in self._headers.flat:
                if key in h:
                    del h[key]

    def _apply_lut(
        self,
        key: Union[str, pydicom.BaseTag],
        index: int = 0,
        inplace: bool = False,
        sync: bool = True,
    ) -> "MedicalVolume":
        """Applies a LUT to the volume.

        Args:
            lut (pydicom.Dataset): LUT dataset.
            inplace (bool, optional): If ``True``, applies LUT inplace.

        Returns:
            ``MedicalVolume`` with a LUT applied.
        """
        h: pydicom.Dataset = self._headers.flat[0]
        xp = self.device.xp

        if key not in h or index > len(h.get(key)) - 1:
            return self

        lut = h.get(key)[index]
        mv = self if inplace else self.clone()
        entries = lut.LUTDescriptor[0] if lut.LUTDescriptor[0] > 0 else 2**16
        first_mapped = lut.LUTDescriptor[1]

        bits_stored = int(lut.LUTDescriptor[2])
        bits_allocated = 16 if bits_stored in range(9, 16) else bits_stored
        if bits_allocated not in [8, 16]:
            raise ValueError(f"Unsupported LUT bit depth: {bits_allocated}.")

        sign = "u" if h.PixelRepresentation == 0 else "i"
        lut_dtype = f"<{sign}{bits_allocated // 8}"

        if lut["LUTData"].VR == "OW":
            lut_dtype = "<" if h.is_little_endian else ">" + lut_dtype[1:]
            lut_data = xp.frombuffer(bytearray(lut["LUTData"].value), dtype=lut_dtype)
        else:
            lut_data = xp.array(lut["LUTData"].value, dtype=lut_dtype)

        # Convert all pixel values into LUT indices
        lut_idxs = xp.zeros_like(self._volume, dtype=f"{sign}{bits_allocated // 8}")
        mapped_pixels = self._volume >= first_mapped
        lut_idxs[mapped_pixels] = self._volume[mapped_pixels] - first_mapped
        xp.clip(lut_idxs, 0, entries - 1, out=lut_idxs)
        mv._volume = lut_data[lut_idxs]

        if sync:
            mv._delete_metadata(key)
            mv.set_metadata("BitsStored", bits_stored, force=True)
            mv.set_metadata("BitsAllocated", bits_allocated, force=True)
            mv.set_metadata("HighBit", bits_stored - 1, force=True)

        return mv

    def _extract_input_array_ufunc(self, input, device=None):
        if device is None:
            device = self.device
        device_err = "Expected device {} but got device ".format(device) + "{}"
        if isinstance(input, Number):
            return input
        elif isinstance(input, np.ndarray):
            if device != cpu_device:
                raise RuntimeError(device_err.format(cpu_device))
            return input
        elif env.cupy_available() and isinstance(input, cp.ndarray):
            if device != input.device:
                raise RuntimeError(device_err.format(Device(input.device)))
            return input
        elif isinstance(input, MedicalVolume):
            if device != input.device:
                raise RuntimeError(device_err.format(Device(input.device)))
            assert self.is_same_dimensions(input, err=True)
            return input._volume
        else:
            return NotImplemented

    def _check_reduce_axis(self, axis: Union[int, Sequence[int]]) -> Tuple[int]:
        if axis is None:
            return None
        is_sequence = isinstance(axis, Sequence)
        if not is_sequence:
            axis = (axis,)
        axis = tuple(x if x >= 0 else self.volume.ndim + x for x in axis)
        assert all(x >= 0 for x in axis)
        if any(x < 3 for x in axis):
            raise ValueError("Cannot reduce MedicalVolume along spatial dimensions")
        if not is_sequence:
            axis = axis[0]
        return axis

    def _reduce_array(self, func, *inputs, **kwargs) -> "MedicalVolume":
        """Assumes inputs have been verified."""
        device = self.device
        xp = device.xp

        keepdims = kwargs.get("keepdims", False)
        reduce_axis = self._check_reduce_axis(kwargs["axis"])
        kwargs["axis"] = reduce_axis
        if not isinstance(reduce_axis, Sequence):
            reduce_axis = (reduce_axis,)
        with device:
            volume = func(*inputs, **kwargs)

        if xp.isscalar(volume) or volume.ndim == 0:
            return volume

        if self._headers is not None:
            headers_slices = tuple(
                slice(None) if x not in reduce_axis else slice(0, 1) if keepdims else 0
                for x in range(self._headers.ndim)
            )
            headers = self._headers[headers_slices]
        else:
            headers = None
        return self._partial_clone(volume=volume, headers=headers)

    def __getitem__(self, _slice):
        if isinstance(_slice, MedicalVolume):
            _slice = _slice.reformat_as(self).A

        slicer = _SpatialFirstSlicer(self)
        try:
            _slice = slicer.check_slicing(_slice)
        except ValueError as err:
            raise IndexError(*err.args)

        volume = self._volume[_slice]
        if any(dim == 0 for dim in volume.shape):
            raise IndexError("Empty slice requested")

        headers = self._headers
        if headers is not None:
            _slice_headers = []
            for idx, x in enumerate(_slice):
                if headers.shape[idx] == 1 and not isinstance(x, int):
                    _slice_headers.append(slice(None))
                elif headers.shape[idx] == 1 and isinstance(x, int):
                    _slice_headers.append(0)
                else:
                    _slice_headers.append(x)
            headers = headers[tuple(_slice_headers)]

        affine = slicer.slice_affine(_slice)
        return self._partial_clone(volume=volume, affine=affine, headers=headers)

    def __setitem__(self, _slice, value):
        """
        Note:
            When ``value`` is a ``MedicalVolume``, the headers from that value
            are not copied over. This may be changed in the future.
        """
        if isinstance(value, MedicalVolume):
            image = self[_slice]
            assert value.is_same_dimensions(image, err=True)
            value = value._volume
        with self.device:
            self._volume[_slice] = value
        if self.is_mmap and self._volume.mode == "c":
            self._volume = np.asarray(self._volume)

    def __repr__(self) -> str:
        nl = "\n"
        nltb = "\n  "
        return (
            f"{self.__class__.__name__}({nltb}shape={self.shape},{nltb}"
            f"ornt={self.orientation}),{nltb}spacing={self.pixel_spacing},{nltb}"
            f"origin={self.scanner_origin},{nltb}device={self.device}{nl})"
        )

    def __tunnelvision__(
        self, config: Dict[str, Any] = {}, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Only 3D and 5D volumes are supported, as we want to "reformat" for tunnelvision for now
        if self._volume.ndim not in [3, 5]:
            raise ValueError("Ambiguous dims: reshape to [BxZxYxXxC] or a permutation of [ZxYxX]")

        # Reformat and convert Zarr, CuPy, Torch tensors to NumPy
        # TODO: remove reformatting once Voxel supports arbitrary orientations
        clone = self.clone().reformat(("IS", "AP", "RL"))
        clone._volume = np.asarray(clone._volume)

        # Parse the relevant headers
        spacing = clone.pixel_spacing[::-1]

        ri = clone.get_metadata("RescaleIntercept", default=0)
        if isinstance(ri, pydicom.multival.MultiValue):
            ri = ri[0]

        rs = clone.get_metadata("RescaleSlope", default=1)
        if isinstance(rs, pydicom.multival.MultiValue):
            rs = rs[0]

        ma, mi = np.amax(clone._volume) * rs + ri, np.amin(clone._volume) * rs + ri
        dynamic_range = ma - mi

        ww = clone.get_metadata("WindowWidth", default=dynamic_range)
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = ww[0]

        wc = clone.get_metadata("WindowCenter", default=dynamic_range / 2 + mi)
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = wc[0]

        # Display the volume
        x = np.ascontiguousarray(clone._volume)
        if x.ndim == 3:
            x = x[np.newaxis, ..., np.newaxis]

        config = {
            "spacing": spacing,
            "rescale": {"intercept": float(ri), "slope": float(rs)},
            "window": {"center": float(wc), "width": float(ww)},
            **config,
        }

        return (x, {"config": config, **kwargs})

    def _iops(self, other, op):
        """Helper function for i-type ops (__iadd__, __isub__, etc.)"""
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume

        if isinstance(op, str):
            op = getattr(self._volume, op)
        op(other)

        if self.is_mmap and self._volume.mode == "c":
            self._volume = np.asarray(self._volume)
        return self

    def __iadd__(self, other):
        return self._iops(other, self._volume.__iadd__)

    def __ifloordiv__(self, other):
        return self._iops(other, self._volume.__ifloordiv__)

    def __imul__(self, other):
        return self._iops(other, self._volume.__imul__)

    def __ipow__(self, other):
        return self._iops(other, self._volume.__ipow__)

    def __isub__(self, other):
        return self._iops(other, self._volume.__isub__)

    def __itruediv__(self, other):
        return self._iops(other, self._volume.__itruediv__)

    def __array__(self):
        """Wrapper for performing numpy operations on MedicalVolume array.

        Examples:
            >>> a = np.asarray(mv)
            >>> type(a)
            <class 'numpy.ndarray'>

        Note:
            This is not valid when ``self.volume`` is a ``cupy.ndarray``.
            All CUDA ndarrays must first be moved to the cpu.
        """
        try:
            return np.asarray(self.volume)
        except TypeError:
            raise TypeError(
                "Implicit conversion to a NumPy array is not allowed. "
                "Please use `.cpu()` to move the array to the cpu explicitly "
                "before constructing a NumPy array."
            )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        def _extract_inputs(inputs, device):
            _inputs = []
            for input in inputs:
                input = self._extract_input_array_ufunc(input, device)
                if input is NotImplemented:
                    return input
                _inputs.append(input)
            return _inputs

        if method not in ["__call__", "reduce"]:
            return NotImplemented

        device = self.device
        _inputs = _extract_inputs(inputs, device)
        if _inputs is NotImplemented:
            return NotImplemented

        if method == "__call__":
            with device:
                volume = ufunc(*_inputs, **kwargs)
            if volume.shape != self._volume.shape:
                raise ValueError(
                    f"{self.__class__.__name__} does not support operations that change shape. "
                    f"Use operations on `self.volume` to modify array objects."
                )
            return self._partial_clone(volume=volume)
        elif method == "reduce":
            return self._reduce_array(ufunc.reduce, *_inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        from voxel.numpy_routines import _HANDLED_NUMPY_FUNCTIONS

        if func not in _HANDLED_NUMPY_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MedicalVolume objects.
        if not all(issubclass(t, (MedicalVolume, self.__class__)) for t in types):
            return NotImplemented
        return _HANDLED_NUMPY_FUNCTIONS[func](*args, **kwargs)

    @property
    def __cuda_array_interface__(self):
        """Wrapper for performing cupy operations on MedicalVolume array."""
        if self.device == cpu_device:
            raise TypeError(
                "Implicit conversion to a CuPy array is not allowed. "
                "Please use `.to(device)` to move the array to the gpu explicitly "
                "before constructing a CuPy array."
            )
        return self.volume.__cuda_array_interface__


class _SpatialFirstSlicer(_SpatialFirstSlicerNib):
    def __init__(self, img):
        self.img = img

    def __getitem__(self, slicer):
        raise NotImplementedError("Slicing should be done by `MedicalVolume`")
