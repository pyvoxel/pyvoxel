.. _introduction:

Introduction
================================================================================
Voxel is an open-source Python library and application for medical image analysis. Voxel is designed
to streamline medical image analysis by standardizing medical image I/O and simplifying array-like
operations on medical images.


Installation
--------------------------------------------------------------------------------
Install Voxel using pip::

   $ pip install pyvoxel


Features
--------------------------------------------------------------------------------

Dynamic Input/Output (I/O)
^^^^^^^^^^^^^^^^^^^^^^^^^^
Reading and writing medical images relies on standardized data formats.
The Digital Imaging and Communications in Medicine (DICOM) format has been the international
standard for medical image I/O. However, header information is memory intensive and
and may not be useful in cases where only volume information is desired.

The Neuroimaging Informatics Technology Initiative (NIfTI) format is useful in these cases.
It stores only volume-specific header information (rotation, position, resolution, etc.) with
the volume.

Voxel supports the use of both formats. However, because NIfTI headers do not contain relevant scan
information, it is not possible to perform quantitative analysis that require this information.
Therefore, we recommend using DICOM inputs, which is the standard output of acquisition systems,
when starting processing with Voxel.


Array-Like Medical Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Medical images are spatially-aware pixel arrays with metadata. Voxel supports array-like
operations (arithmetic, slicing, etc.) on medical images while preserving spatial attributes and
accompanying metadata with the :class:`MedicalVolume` data structure. It also supports intelligent
reformatting, fast low-level computations, and native GPU support.
