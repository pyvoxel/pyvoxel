import logging
import os
import unittest

from voxel.utils import env
from voxel.utils.logger import setup_logger

from .. import util


class TestSetupLogger(unittest.TestCase):
    def test_log_info(self):
        debug_val = env.debug()

        env.debug(False)
        setup_logger(None)
        with self.assertLogs("voxel", level="INFO"):
            logging.getLogger("voxel").info("Sample log at INFO level")

        env.debug(True)
        setup_logger(None)
        with self.assertLogs("voxel", level="DEBUG"):
            logging.getLogger("voxel").debug("Sample log at DEBUG level")

        env.debug(debug_val)

    def test_makes_file(self):
        setup_logger(util.TEMP_PATH)
        assert os.path.isfile(os.path.join(util.TEMP_PATH, "voxel.log"))

    def test_overwrite_handlers(self):
        logger = setup_logger(name="foobar")
        assert len(logger.handlers) == 2

        logger = setup_logger(name="foobar", abbrev_name="foo")
        assert len(logger.handlers) == 4

        logger = setup_logger(name="foobar", overwrite_handlers=True)
        assert len(logger.handlers) == 2
