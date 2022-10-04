import os
import unittest

import voxel as vx
from voxel.utils import env


class TestEnv(unittest.TestCase):
    def test_package_available(self):
        assert env.package_available("voxel")
        assert not env.package_available("blah")

    def test_get_version(self):
        assert env.get_version("voxel") == vx.__version__

    def test_debug(self):
        os_env = os.environ.copy()

        # TODO: add test
        env.debug(True)
        env.debug(False)

        os.environ = os_env  # noqa: B003
