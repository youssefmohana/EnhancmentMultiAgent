#!/usr/bin/env python3
"""Legacy shim — use `src/enhancement_multiagent/pipelines/restoration.py` instead."""
import warnings
warnings.warn("image_restoration.py is deprecated, use enhancement_multiagent.pipelines.restoration", DeprecationWarning, stacklevel=2)
from enhancement_multiagent.pipelines.restoration import restore_image, restore_image_smart  # noqa: F401
__all__ = ["restore_image", "restore_image_smart"]
# Preserve CLI behavior
if __name__ == "__main__":
    import sys
    from enhancement_multiagent.pipelines.restoration import __name__ as _pipename
    # delegate to new module's main
    from enhancement_multiagent.pipelines.restoration import restore_image as _orig
    # import and run original main
    import runpy
    runpy.run_path("src/enhancement_multiagent/pipelines/restoration.py", run_name="__main__")
