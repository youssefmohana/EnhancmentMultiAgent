#!/usr/bin/env python3
"""Legacy shim — use `scripts/benchmark.py` instead."""
import warnings, sys
warnings.warn("benchmark.py at root is deprecated, use scripts/benchmark.py", DeprecationWarning, stacklevel=2)
import runpy
runpy.run_path("scripts/benchmark.py", run_name="__main__")
