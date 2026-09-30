#!/usr/bin/env python
"""
Backwards-compatible wrapper script for Radio_continuum_validation.
This allows running `python Radio_continuum_validation.py ...` directly
from the repository root.
"""
import os
import sys

# Ensure src/ is on sys.path
_repo_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_repo_dir, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from continuum_validation.Radio_continuum_validation import main

if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])
