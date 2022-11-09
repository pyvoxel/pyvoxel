.. _utils_api:

Utilities
================================================================================

Pixel Data
---------------------------
.. _utils_api_pixel_data:

.. autosummary::
   :toctree: generated
   :nosignatures:

   voxel.utils.pixel_data.apply_rescale
   voxel.utils.pixel_data.apply_window
   voxel.utils.pixel_data.invert
   voxel.utils.pixel_data.pixel_dtype
   voxel.utils.pixel_data.pixel_range


Collect Env
---------------------------
.. _utils_api_collect_env:

.. autosummary::
   :toctree: generated
   :nosignatures:

   voxel.utils.collect_env.collect_env_info


Env
---------------------------
.. _utils_api_env:

.. autosummary::
   :toctree: generated
   :nosignatures:

   voxel.utils.env.package_available
   voxel.utils.env.get_version


Logger
---------------------------
.. _utils_api_logger:

.. autosummary::
   :toctree: generated
   :nosignatures:

   voxel.utils.logger.setup_logger

If you do not want logging messages to display on your console (terminal, Jupyter Notebook, etc.),
the code below will only log messages at the ERROR level or higher:

>>> import logging
>>> voxel.utils.logger.setup_logger(stream_lvl=logging.ERROR)
