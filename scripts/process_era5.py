"""Compatibility alias for the public :mod:`metafilter.core` module."""

import runpy
import sys

from metafilter import core as _implementation


if __name__ == "__main__":
    runpy.run_module("metafilter.core", run_name="__main__")
else:
    sys.modules[__name__] = _implementation
