#!/usr/bin/env python3
"""Legacy shim — use `scripts/download_benchmark.py` instead."""
import warnings
warnings.warn("download_benchmark.py at root is deprecated, use scripts/download_benchmark.py", DeprecationWarning, stacklevel=2)
import runpy
runpy.run_path("scripts/download_benchmark.py", run_name="__main__")
