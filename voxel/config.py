from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_ENV_VARIABLE = "VOXEL_CONFIG"


@dataclass
class VoxelConfig:

    # The precision to use for the affine matrix.
    affine_precision: int = 4
    # The default format to save MedicalVolume objects.
    save_format: str = "nifti"

    @classmethod
    def from_yaml(cls, path: str = None):
        if path is None:
            path = os.environ.get(
                CONFIG_ENV_VARIABLE,
                os.path.abspath(
                    os.path.join(os.path.join(Path.home(), ".cache", "voxel"), "config.yaml")
                ),
            )
            if not os.path.exists(path):
                # create empty config
                os.makedirs(os.path.dirname(path), exist_ok=True)
                yaml.dump({}, open(path, "w"))
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)

        return cls(**config)


config = VoxelConfig.from_yaml()
