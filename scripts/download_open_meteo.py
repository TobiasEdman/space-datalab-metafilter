"""Compatibility alias for the public :mod:`metafilter.open_meteo` module."""

import runpy
import sys

from metafilter import open_meteo as _implementation


if __name__ == "__main__":
    runpy.run_module("metafilter.open_meteo", run_name="__main__")
else:
    sys.modules[__name__] = _implementation
