import numpy as np

from voxel.device import cpu_device
from voxel.med_volume import MedicalVolume

try:
    from voxelviz.widget import VoxelWidget

    _SUPPORTS_VOXELVIZ = True
except ImportError:
    widget = None
    _SUPPORTS_VOXELVIZ = False

# `True` if the user is running code in an iPython session.
# This is required to be able to use the viewer.
_IS_IPYTHON_SESSION = None


class Axis:
    def __init__(self) -> None:
        if not _SUPPORTS_VOXELVIZ:
            # TODO: I think the user also has to install ipywidgets.
            # TODO: Add figsize
            raise ImportError(
                "`voxelviz` is not installed. To install, use `pip install voxelviz`."
            )
        if not _is_ipython_session():
            raise RuntimeError("The viewer can only be used in an iPython session.")

    def show(self, volume: MedicalVolume, *, seg: MedicalVolume = None):
        """Display a multi-dimensional volume with optional segmentation overlay.

        Args:
            volume (MedicalVolume): The volume to display.
            seg (MedicalVolume, optional): The segmentation to overlay.
                If specified, all entries must be non-negative integers.
                ``0`` corresponds to background.
        """
        if any(x is not None and x.device != cpu_device for x in [volume, seg]):
            raise ValueError("All images to display must be on the CPU.")

        if volume.ndim > 4:
            raise ValueError("Only volumes with 4 or fewer dimensions are supported.")

        if seg is not None:
            seg = seg.reformat_as(volume)
            # Duplicate the segmentation along the extra dimension.
            # TODO: Does the extra dimension must be the same between the base volume
            # and segmentation?
            if volume.ndim > seg.ndim:
                seg = np.stack([seg] * volume.shape[-1], axis=-1)
            if not seg.is_same_dimensions(volume):
                raise ValueError(
                    "The segmentation and volume must have the same dimensions and spacing."
                )
            if np.any(seg.A < 0):
                raise ValueError("The segmentation must be non-negative.")

        spacing = volume.pixel_spacing
        volume = _auto_reshape(volume)

        vw = VoxelWidget()
        vw.vol = volume
        if seg is not None:
            seg = _auto_reshape(seg)
            vw.seg = seg

        vw.cfg = {"spacing": list(spacing)}
        return vw


def show(volume: MedicalVolume, *, seg: MedicalVolume = None):
    """Display a multi-dimensional volume with optional segmentation overlay.

    Args:
        volume (MedicalVolume): The volume to display.
        seg (MedicalVolume, optional): The segmentation to overlay.
            If specified, all entries must be non-negative integers.
            ``0`` corresponds to background.
    """
    ax = Axis()
    return ax.show(volume, seg=seg)


def _auto_reshape(mv: MedicalVolume):
    """

    The first 3 dimensions in a MedicalVolume are always spatial dimensions.
    We assume we are only working with grayscale images for now.
    """
    assert mv.ndim in (3, 4)
    arr = mv.A

    # Add batch dimension
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    else:
        arr = arr.transpose(3, 0, 1, 2)

    # Add channel dimension.
    arr = arr[..., np.newaxis]
    return arr


def _is_ipython_session() -> bool:
    """Returns if the python is an iPython session.

    Adapted from
    https://discourse.jupyter.org/t/find-out-if-my-code-runs-inside-a-notebook-or-jupyter-lab/6935/3
    """
    global _IS_IPYTHON_SESSION
    if _IS_IPYTHON_SESSION is not None:
        return _IS_IPYTHON_SESSION

    is_ipython_session = None
    try:
        from IPython import get_ipython

        ip = get_ipython()
        is_ipython_session = ip is not None
    except ImportError:
        # iPython is not installed
        is_ipython_session = False
    _IS_IPYTHON_SESSION = is_ipython_session
    return _IS_IPYTHON_SESSION
