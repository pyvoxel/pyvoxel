.. _core_api:

Core
================================================================================

MedicalVolume
---------------------------
.. _core_api_medicalvolume:

.. autosummary::
   :toctree: generated
   :nosignatures:

   voxel.MedicalVolume


Numpy Routines
---------------------------
.. _core_api_numpy_routines:

Numpy operations that are supported on MedicalVolumes.

.. autosummary::
   :toctree: generated
   :nosignatures:

   voxel.numpy_routines.all_np
   voxel.numpy_routines.amax
   voxel.numpy_routines.amin
   voxel.numpy_routines.any_np
   voxel.numpy_routines.argmax
   voxel.numpy_routines.argmin
   voxel.numpy_routines.around
   voxel.numpy_routines.clip
   voxel.numpy_routines.concatenate
   voxel.numpy_routines.expand_dims
   voxel.numpy_routines.may_share_memory
   voxel.numpy_routines.mean_np
   voxel.numpy_routines.nan_to_num
   voxel.numpy_routines.nanargmax
   voxel.numpy_routines.nanargmin
   voxel.numpy_routines.nanmax
   voxel.numpy_routines.nanmean
   voxel.numpy_routines.nanmin
   voxel.numpy_routines.nanstd
   voxel.numpy_routines.nansum
   voxel.numpy_routines.ones_like
   voxel.numpy_routines.pad
   voxel.numpy_routines.shares_memory
   voxel.numpy_routines.squeeze
   voxel.numpy_routines.stack
   voxel.numpy_routines.std
   voxel.numpy_routines.sum_np
   voxel.numpy_routines.where
   voxel.numpy_routines.zeros_like

Standard universal functions that act element-wise on the array are also supported.
A (incomplete) list is shown below:

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - numpy.power
     - numpy.sign
     - numpy.remainder
     - numpy.mod
     - numpy.abs
   * - numpy.log
     - numpy.exp
     - numpy.sqrt
     - numpy.square
     - numpy.reciprocal
   * - numpy.sin
     - numpy.cos
     - numpy.tan
     - numpy.bitwise_and
     - numpy.bitwise_or
   * - numpy.isfinite
     - numpy.isinf
     - numpy.isnan
     - numpy.floor
     - numpy.ceil


Image I/O
---------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   voxel.read
   voxel.write
   voxel.NiftiReader
   voxel.NiftiWriter
   voxel.DicomReader
   voxel.DicomWriter
   voxel.HttpReader


Image Orientation
---------------------------
.. automodule::
   voxel.orientation

.. autosummary::
   :toctree: generated
   :nosignatures:

   voxel.orientation.to_affine
   voxel.orientation.get_transpose_inds
   voxel.orientation.get_flip_inds
   voxel.orientation.orientation_nib_to_standard
   voxel.orientation.orientation_standard_to_nib


Device
----------
.. autosummary::
   :toctree: generated
   :nosignatures:

   voxel.Device
   voxel.get_device
   voxel.to_device
