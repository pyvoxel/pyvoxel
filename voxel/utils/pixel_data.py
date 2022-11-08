from typing import Tuple

import numpy as np
import pydicom

from voxel.device import get_array_module

__all__ = ["apply_rescale", "apply_window", "pixel_dtype", "pixel_range", "invert"]


def pixel_dtype(ds: "pydicom.Dataset") -> np.dtype:
    """Detect the NumPy dtype of the data in a header.
    NB: byteorder can be incorrect, so use with caution (see pydicom pixel_dtype docs).

    Args:
        ds (pydicom.Dataset): DICOM header.

    Returns:
        np.dtype: NumPy dtype of the data.
    """

    if "FloatPixelData" in ds:
        return np.float32

    elif "DoubleFloatPixelData" in ds:
        return np.float64

    # Default to PixelData
    elif "BitsAllocated" in ds:
        bits_allocated = ds.BitsAllocated
        if bits_allocated in [8, 16, 32, 64]:
            prefix = "int" if ds.get("PixelRepresentation", 0) == 1 else "uint"
            return np.dtype(f"{prefix}{bits_allocated}")

        raise ValueError(f"Unsupported bit depth {bits_allocated}")

    raise ValueError("NumPy dtype could not be detected from the headers.")


def pixel_range(ds: "pydicom.Dataset") -> Tuple[float, float]:
    """Get the pixel range of a DICOM dataset.

    Args:
        ds (pydicom.Dataset): DICOM header.

    Returns:
        Tuple[float, float]: Pixel range.
    """
    if "FloatPixelData" in ds or "DoubleFloatPixelData" in ds:
        raise ValueError("Pixel range cannot be determined for float pixel data.")

    if "BitsStored" in ds or "BitsAllocated" in ds:
        bits = ds.get("BitsStored", ds.get("BitsAllocated"))
        pixel_representation = ds.get("PixelRepresentation", 0)
        if pixel_representation == 1:
            return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1

        return 0, 2**bits - 1

    raise ValueError("Pixel range could not be detected from the headers.")


def apply_window(
    volume: np.ndarray,
    center: float,
    width: float,
    output_range: Tuple[float, float] = None,
    mode: str = None,
    inplace: bool = False,
) -> np.ndarray:
    """Apply a window to a volume.

    The output range defaults to the boundaries of the window in contrast to
    the default behaviour of pydicom which is to use the pixel range. Explicitly
    setting the output range to the pixel range will result in the same behaviour.
    Set the output range to (0, 1) to normalize the volume.

    NB: Whether the minimum output value is rendered as black or white may depend
    on the value of Photometric Interpretation (0028,0004).

    Args:
        volume (np.ndarray): Volume to apply the window to.
        center (float): Window center.
        width (float): Window width.
        output_range (Tuple[float, float]): Output range.
        mode (str, optional): VOI LUT function. Defaults to None.

    Returns:
        np.ndarray: Windowed volume.
    """
    mode = mode.lower() if mode is not None else "linear"

    wc, ww = center, width
    if mode == "linear":
        if ww < 1:
            raise ValueError("Window Width must be greater than 1 for LINEAR.")
        wc -= 0.5
        ww -= 1

    if mode == "linear_exact" and ww <= 0:
        raise ValueError("Window Width must be greater than 0 for LINEAR_EXACT.")

    lb, ub = wc - ww / 2, wc + ww / 2
    if output_range is None:
        output_range = (lb, ub)

    y_min, y_max = output_range
    y_range = y_max - y_min

    volume = volume if inplace else volume.copy()
    xp = get_array_module(volume)
    if mode in ["linear", "linear_exact"]:
        xp.clip(volume, lb, ub, out=volume)
        volume[:] = ((volume - wc) / ww + 0.5) * y_range + y_min
    elif mode == "sigmoid":
        if ww <= 0:
            raise ValueError("Window Width must be greater than 0 for SIGMOID.")
        volume[:] = ww / (1 + xp.exp(-4 * (volume - wc) / ww)) + y_min
    else:
        raise ValueError(f"VOI LUT Function {mode} is not supported.")

    return volume


def apply_rescale(
    volume: np.ndarray,
    slope: float,
    intercept: float,
    inplace: bool = False,
) -> np.ndarray:
    """Apply a rescale to a volume.

    Args:
        volume (np.ndarray): Volume to apply the rescale to.
        slope (float): Rescale slope.
        intercept (float): Rescale intercept.
        inplace (bool, optional): Whether to apply the rescale in place. Defaults to False.

    Returns:
        np.ndarray: Rescaled volume.
    """
    volume = volume if inplace else volume.copy()
    if slope != 1:
        volume *= slope
    if intercept != 0:
        volume += intercept
    return volume


def invert(
    volume: np.ndarray,
    pixel_range: Tuple[float, float] = None,
    inplace: bool = False,
) -> np.ndarray:
    """Invert the pixel values in a volume.

    Args:
        volume (np.ndarray): Volume to invert.
        pixel_range (Tuple[float, float], optional): Pixel range. Defaults to None.
        inplace (bool, optional): Whether to apply the rescale in place. Defaults to False.

    Returns:
        np.ndarray: Inverted volume.
    """
    if pixel_range is None:
        pixel_range = volume.min(), volume.max()

    y_min, y_max = pixel_range
    volume = volume if inplace else volume.copy()
    volume[:] = y_max - volume + y_min

    return volume
