# Voxel
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/voxelimaging/pyvoxel/CI)
[![codecov](https://codecov.io/gh/voxelimaging/pyvoxel/branch/master/graph/badge.svg?token=X2FRQJHV2M)](https://codecov.io/gh/voxelimaging/pyvoxel)
<!-- [![Documentation Status](https://readthedocs.org/projects/dosma/badge/?version=latest)](https://dosma.readthedocs.io/en/latest/?badge=latest) -->

[Documentation](http://dosma.readthedocs.io/) | [DOSMA Basics Tutorial](https://colab.research.google.com/drive/1zY5-3ZyTBrn7hoGE5lH0IoQqBzumzP1i?usp=sharing)

Voxel provides fast Pythonic data structures and tools for wrangling with medical images.

## Installation
Voxel requires Python 3.7+. The core module depends on numpy, nibabel, pydicom, PyYAML, and tqdm.

To install Voxel, run:

```bash
pip install pyvoxel
```

If you would like to contribute to Voxel, we recommend you clone the repository and
install Voxel with `pip` in editable mode.

```bash
git clone git@github.com:voxelimaging/pyvoxel.git
cd pyvoxel
pip install -e '.[dev,docs]'
make dev
```

To run tests, build documentation and contribute, run
```bash
make autoformat test build-docs
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

multi_echo_scan = vx.load("/path/to/multi-echo/scan", group_by="EchoNumbers", num_workers=8, verbose=True)
dce_scan = vx.load("/path/to/dce/scan", group_by="TriggerTime")
```

### Data-Embedded Medical Images
Voxel's [MedicalVolume](https://dosma.readthedocs.io/en/latest/generated/dosma.MedicalVolume.html#dosma.MedicalVolume)
data structure supports array-like operations (arithmetic, slicing, etc.) on medical images while preserving spatial
attributes and accompanying metadata. This structure supports NumPy interoperability, intelligent reformatting,
fast low-level computations, and native GPU support. For example, given MedicalVolumes `mvA` and `mvB` we can do the following:

```python
# Reformat image into Superior->Inferior, Anterior->Posterior, Left->Right directions.
mvA = mvA.reformat(("SI", "AP", "LR"))

# Get and set metadata
study_description = mvA.get_metadata("StudyDescription")
mvA.set_metadata("StudyDescription", "A sample study")

# Perform NumPy operations like you would on image data.
rss = np.sqrt(mvA**2 + mvB**2)

# Move to GPU 0 for CuPy operations
mv_gpu = mvA.to(dosma.Device(0))

# Take slices. Metadata will be sliced appropriately.
mv_subvolume = mvA[10:20, 10:20, 4:6]
```

## Citation
Voxel is a refactored version of the [dosma](https://github.com/ad12/dosma) package that focuses on medical image data structures and I/O.
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

In addition to DOSMA, please also consider citing the work that introduced the method used for analysis.
