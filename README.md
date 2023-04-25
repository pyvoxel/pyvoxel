# Voxel
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/pyvoxel/pyvoxel/ci.yml?branch=main)
[![codecov](https://codecov.io/gh/pyvoxel/pyvoxel/branch/master/graph/badge.svg?token=X2FRQJHV2M)](https://codecov.io/gh/pyvoxel/pyvoxel)
[![Documentation Status](https://readthedocs.org/projects/pyvoxel/badge/?version=latest)](https://pyvoxel.readthedocs.io/en/latest/?badge=latest)

[Documentation](http://pyvoxel.readthedocs.io/) | [Installation](https://pyvoxel.readthedocs.io/en/latest/introduction.html) | [Basic Usage](https://pyvoxel.readthedocs.io/en/latest/user_guide.html)

Voxel provides fast Pythonic data structures and tools for wrangling with medical images.

## Installation
Voxel requires Python 3.7+. The core module depends on numpy, nibabel, pydicom, requests, and tqdm.

To install Voxel, run:

```bash
pip install pyvoxel
```

## Features
### Simplified, Efficient I/O
Voxel provides efficient readers for DICOM and NIfTI formats built on nibabel and pydicom.
Multi-slice DICOM data can be loaded in parallel with multiple workers and structured into
the appropriate 3D volume(s). For example, multi-echo and dynamic contrast-enhanced
(DCE) MRI scans have multiple volumes acquired at different echo times and trigger times,
respectively. These can be loaded into multiple volumes with ease:

```python
import voxel as vx

xray = vx.load("path/to/xray.dcm")
ct_scan = vx.load("path/to/ct/folder/")

multi_echo_scan = vx.load("/path/to/multi-echo/scan", group_by="EchoNumbers")
dce_scan = vx.load("/path/to/dce/scan", group_by="TriggerTime")
```

### Data-Embedded Medical Images
Voxel's `MedicalVolume` data structure supports array-like operations (arithmetic, slicing, etc.) on medical images while preserving spatial
attributes and accompanying metadata. This structure supports NumPy interoperability intelligent reformatting, fast low-level computations, and native GPU support. For example, given MedicalVolumes `mv_a` and `mv_b` we can do the following:

```python
# Reformat image into Superior->Inferior, Anterior->Posterior, Left->Right directions.
mv_a = mv_a.reformat(("SI", "AP", "LR"))

# Get and set metadata
study_description = mv_a.get_metadata("StudyDescription")
mv_a.set_metadata("StudyDescription", "A sample study")

# Perform NumPy operations like you would on image data.
rss = np.sqrt(mv_a**2 + mv_b**2)

# Move to GPU 0 for CuPy operations
mv_gpu = mv_a.to(vx.Device(0))

# Take slices. Metadata will be sliced appropriately.
mv_subvolume = mv_a[10:20, 10:20, 4:6]
```

### Easily Prepare Data for AI Pipelines
Voxel enables you to preprocess DICOM images for deep learning in a few lines of code:

```python
# Load a scan, and prepare it for AI/visualization
mv = (
  vx.load("/dicoms")
  .apply_rescale()
  .apply_window()
  .to_grayscale()
)

# Zero-copy to PyTorch
arr = mv.to_torch()
```

### Connect with PACS
Voxel provides easy access to data stored in a PACS environment through DICOMweb.
This makes loading data from a remote server just as easy as using the local filesystem.

```python
# Download an MRI from a local Orthanc instance
mv = vx.load("http://localhost:8042/dicom-web/studies/x/series/y", params={"Modality": "MR"})

# Re-use the session for multiple requests
with vx.HttpReader(verbose=True) as hr:
  mv_a = hr.load("http://localhost:8042/dicom-web/studies/v/series/w")
  mv_b = hr.load("http://localhost:8042/dicom-web/studies/x/series/y")
```

## Contribute
If you would like to contribute to Voxel, we recommend you clone the repository and
install Voxel with `pip` in editable mode.

```bash
git clone git@github.com:pyvoxel/pyvoxel.git
cd pyvoxel
pip install -e '.[dev,docs]'
make dev
```

To run tests, build documentation and contribute, run
```bash
make autoformat test build-docs
```

## Citation
Voxel is a refactored version of the [DOSMA](https://github.com/ad12/dosma) package that focuses on medical image data structures and I/O.
If you use Voxel in your research, please cite the following work:

```
@inproceedings{desai2019dosma,
  title={DOSMA: A deep-learning, open-source framework for musculoskeletal MRI analysis},
  author={Desai, Arjun D and Barbieri, Marco and Mazzoli, Valentina and Rubin, Elka and Black, Marianne S and Watkins, Lauren E and Gold, Garry E and Hargreaves, Brian A and Chaudhari, Akshay S},
  booktitle={Proc 27th Annual Meeting ISMRM, Montreal},
  pages={1135},
  year={2019}
}
```

In addition to Voxel, please also consider citing the work that introduced the method used for analysis.
