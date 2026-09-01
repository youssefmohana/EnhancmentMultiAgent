#!/usr/bin/env python3
"""Legacy shim — use `scripts/batch_restore.py` instead."""
import warnings
warnings.warn("batch_restore.py at root is deprecated, use scripts/batch_restore.py", DeprecationWarning, stacklevel=2)
import runpy
runpy.run_path("scripts/batch_restore.py", run_name="__main__")
