#!/usr/bin/env python
"""
Backwards-compatible wrapper script for ASKAP_continuum_validation.
This allows running `python ASKAP_continuum_validation.py ...` directly
from the repository root.
"""
import os
import sys

_repo_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_repo_dir, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import continuum_validation.ASKAP_continuum_validation
