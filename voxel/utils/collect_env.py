import os
import sys

import nibabel
import numpy as np
import pydicom
from tabulate import tabulate

__all__ = ["collect_env_info"]


def collect_env_info():
    """Collect environment information for reporting issues.

    Run this function when reporting issues on Github.
    """
    data = []
    data.append(("sys.platform", sys.platform))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))

    try:
        import voxel  # noqa

        data.append(("voxel", voxel.__version__ + " @" + os.path.dirname(voxel.__file__)))
    except ImportError:
        data.append(("voxel", "failed to import"))

    # Required packages
    data.append(("nibabel", nibabel.__version__))
    data.append(("pydicom", pydicom.__version__))

    # Optional packages
    try:
        import cupy

        data.append(("cupy", cupy.__version__))
    except ImportError:
        pass

    try:
        import sigpy

        data.append(("sigpy", sigpy.__version__))
    except ImportError:
        pass

    env_str = tabulate(data)
    return env_str
